"""LeRobot v3 adapter. Official reader/decoder, local data only during training."""
import copy
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def inventory(root):
    root = Path(root).resolve()
    paths = []
    for folder in ('meta', 'data', 'videos', 'images'):
        for p in (root / folder).rglob('*'):
            if p.is_file() and not any(part.startswith('.') for part in p.relative_to(root).parts):
                if p.suffix in ('.json', '.jsonl', '.parquet', '.mp4', '.png', '.jpg', '.jpeg'):
                    if not p.resolve().is_relative_to(root):
                        raise ValueError(f'Dataset path escapes root: {p}')
                    paths.append(p.resolve())
    return sorted(set(paths))


def open_local(root, camera_key, frameskip, revision=None, repo_id='local/dataset'):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    class LocalDataset(LeRobotDataset):
        def download(self, *args, **kwargs):
            raise ValueError('Incomplete local LeRobot data; run explicit download/prepare first')

        def pull_from_repo(self, *args, **kwargs):
            raise ValueError('Implicit downloads are disabled')

    root = Path(root).resolve()
    for name in ('meta/info.json', 'meta/stats.json', 'meta/tasks.parquet'):
        if not (root / name).is_file():
            raise ValueError(f'Missing local LeRobot metadata: {name}')
    if not list((root / 'meta/episodes').rglob('*.parquet')):
        raise ValueError('Missing local LeRobot episode metadata')
    info = json.loads((root / 'meta/info.json').read_text())
    if info.get('codebase_version') != 'v3.0':
        raise ValueError('Only LeRobot Dataset v3.0 is supported')
    feature = info['features'].get(camera_key, {})
    shape = feature.get('shape', [])
    if feature.get('dtype') not in ('video', 'image') or len(shape) != 3 or (shape[0] != 3 and shape[-1] != 3):
        raise ValueError('Select an RGB image/video camera key')
    fps = info['fps']
    if not isinstance(fps, (int, float)) or not np.isfinite(fps) or fps <= 0:
        raise ValueError('Invalid LeRobot FPS')
    ds = LocalDataset(repo_id, root=root, revision=revision or 'v3.0', video_backend='pyav',
                      delta_timestamps={camera_key: [i * frameskip / fps for i in range(4)],
                                        'action': [i / fps for i in range(4 * frameskip)]})
    # Official __getitem__ decodes all cameras by default. Select one camera in
    # memory; no changes to saved info.json or to the other cameras' files.
    ds.meta.info = copy.deepcopy(ds.meta.info)
    ds.meta.info['features'] = {k: v for k, v in ds.meta.info['features'].items()
                                if v['dtype'] not in ('video', 'image') or k == camera_key}
    keep = [k for k in ds.hf_dataset.column_names
            if k not in info['features'] or info['features'][k]['dtype'] not in ('video', 'image')
            or k == camera_key]
    ds.hf_dataset = ds.hf_dataset.select_columns(keep)
    return ds


def episode_metadata(ds):
    rows = list(ds.meta.episodes)
    if [r['episode_index'] for r in rows] != list(range(len(rows))):
        raise ValueError('LeRobot episode IDs must be contiguous and ordered')
    lengths = np.array([r['length'] for r in rows], dtype=np.int64)
    offsets = np.array([r['dataset_from_index'] for r in rows], dtype=np.int64)
    ends = np.array([r['dataset_to_index'] for r in rows], dtype=np.int64)
    if (not len(rows) or (lengths < 1).any() or offsets[0] != 0
            or not np.array_equal(ends, offsets + lengths)
            or not np.array_equal(offsets[1:], ends[:-1]) or ends[-1] != len(ds)):
        raise ValueError('Invalid LeRobot episode boundaries')
    return lengths, offsets


class LeRobotClips(Dataset):
    def __init__(self, manifest, transform):
        self.manifest, self.transform = manifest, transform
        self.root = Path(manifest['dataset']).parent.parent
        recorded = {Path(manifest['dataset']).resolve(),
                    *(Path(f['path']).resolve() for f in manifest['files'])}
        if set(inventory(self.root)) != recorded:
            raise ValueError('LeRobot file inventory changed since preparation')
        self.lengths = np.asarray(manifest['episode_lengths'], dtype=np.int64)
        self.offsets = np.asarray(manifest['episode_offsets'], dtype=np.int64)
        self.frameskip = manifest['frameskip']
        self.span = 4 * self.frameskip
        self.clip_indices = [(ep, start) for ep, length in enumerate(self.lengths)
                             for start in range(max(0, int(length) - self.span + 1))]
        self.reader = None
        self.pid = None

    def __getstate__(self):
        return dict(self.__dict__, reader=None, pid=None)

    def __len__(self):
        return len(self.clip_indices)

    def __getitem__(self, index):
        if self.reader is None or self.pid != os.getpid():
            m = self.manifest
            self.reader = open_local(self.root, m['camera_key'], self.frameskip,
                                     m.get('revision'), m.get('repo_id') or 'local/dataset')
            self.pid = os.getpid()
        ep, start = self.clip_indices[index]
        item = self.reader[int(self.offsets[ep]) + start]
        key = self.manifest['camera_key']
        if int(item['episode_index']) != ep or int(item['frame_index']) != start:
            raise ValueError('LeRobot frame/episode indexing mismatch')
        if item[key + '_is_pad'].any() or item['action_is_pad'].any():
            raise ValueError('Padded transitions are not training samples')
        # Official LeRobot returns RGB float CHW in [0,1]; official LeWM's
        # ToImage expects uint8 here. This exactly restores decoded RGB bytes.
        pixels = item[key]
        if (pixels.ndim != 4 or pixels.shape[:2] != (4, 3)
                or not torch.isfinite(pixels).all() or (pixels < 0).any() or (pixels > 1).any()):
            raise ValueError('Invalid decoded RGB image')
        batch = self.transform({'pixels': (pixels * 255).round().to(torch.uint8),
                                'action': item['action']})
        batch['action'] = batch['action'].reshape(4, -1)
        return batch
