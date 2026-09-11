"""Demonstration-separated, native-rate action chunks for frozen-encoder BC."""
import hashlib
import json
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch
import torch.nn.functional as F

CAMERAS = ['agentview_rgb', 'eye_in_hand_rgb']


def validate_manifest(m):
    if m.get('camera_order') != CAMERAS or m.get('image_size') != 224:
        raise ValueError('BC requires the native LIBERO two-camera, 224px contract')
    if m.get('image_convention') != 'native opengl; do not flip training images independently of simulator':
        raise ValueError('Unknown LIBERO image convention')
    names = m['task_names']
    if not names or len(names) != len(set(names)) or len(names) != len(m['files']):
        raise ValueError('Invalid task-name/file mapping')
    for name, item in zip(names, m['files']):
        if Path(item['path']).name != name + '_demo.hdf5':
            raise ValueError('Task name does not match its demonstration file')
    mean, std = np.asarray(m['action_mean']), np.asarray(m['action_std'])
    if mean.shape != (7,) or std.shape != (7,) or not np.isfinite([mean, std]).all() or (std <= 0).any():
        raise ValueError('Require finite seven-dimensional training action statistics')
    seen = set()
    for split in ('train_demos', 'validation', 'test'):
        tasks = set()
        for row in m[split]:
            task = row['task']
            if type(task) is not int or not 0 <= task < len(names):
                raise ValueError('Invalid manifest task index')
            key = (task, row['demo'])
            if key in seen:
                raise ValueError('Duplicate demonstration or train/validation/test leakage')
            if type(row['length']) is not int or row['length'] < 1:
                raise ValueError('Invalid demonstration length')
            seen.add(key)
            tasks.add(task)
        if tasks != set(range(len(names))):
            raise ValueError(f'Every task must be represented in {split}')


def split_contract(m):
    """Compare a relocated manifest without rewriting historical evidence."""
    validate_manifest(m)
    value = {k: m[k] for k in ('task_names', 'camera_order', 'image_size',
                              'image_convention', 'frameskip', 'history',
                              'action_mean', 'action_std', 'train_demos', 'validation', 'test')}
    # Byte provenance is checked separately. Old manifests can predate hash fields.
    value['files'] = [{'name': Path(f['path']).name, 'size': f['size']} for f in m['files']]
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def preprocess_images(pixels, device):
    """B,2,3,H,W uint8 native RGB -> normalized B,2,3,224,224."""
    if pixels.ndim != 5 or pixels.shape[1:3] != (2, 3) or pixels.dtype != torch.uint8:
        raise ValueError('Expected uint8 two-camera RGB: (batch,2,3,height,width)')
    pixels = pixels.to(device, non_blocking=True)
    b = pixels.shape[0]
    x = F.interpolate(pixels.flatten(0, 1).float() / 255, (224, 224),
                      mode='bilinear', align_corners=False, antialias=True)
    mean = x.new_tensor([.485, .456, .406])[None, :, None, None]
    std = x.new_tensor([.229, .224, .225])[None, :, None, None]
    return ((x - mean) / std).reshape(b, 2, 3, 224, 224)


class LiberoBCClips(torch.utils.data.Dataset):
    """One observed image pair and H consecutive native actions, within one demo.

    No goal, future images, task-specific models, padded actions or frame skipping.
    Validation uses the manifest's fixed start for each held-out demonstration.
    """
    def __init__(self, manifest, horizon=8, split='train_demos'):
        validate_manifest(manifest)
        if split not in ('train_demos', 'validation', 'test') or horizon < 1:
            raise ValueError('Invalid BC split or action horizon')
        self.manifest, self.horizon, self.split = manifest, horizon, split
        self.rows = manifest[split]
        counts = [row['length'] - horizon + 1 for row in self.rows]
        if min(counts) < 1:
            raise ValueError('Demonstration shorter than action horizon')
        if split != 'train_demos':
            for row, count in zip(self.rows, counts):
                if not 0 <= row['start'] < count:
                    raise ValueError('Held-out action chunk crosses demonstration boundary')
        self.ends = np.cumsum(counts)
        self.files = {}

    def __len__(self):
        return int(self.ends[-1]) if self.split == 'train_demos' else len(self.rows)

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self.split == 'train_demos':
            j = int(np.searchsorted(self.ends, index, side='right'))
            row = self.rows[j]
            start = index - (int(self.ends[j - 1]) if j else 0)
        else:
            row = self.rows[index]
            start = row['start']
        task = row['task']
        if task not in self.files:
            self.files[task] = h5py.File(self.manifest['files'][task]['path'], 'r')
        demo = self.files[task][f'data/{row["demo"]}']
        if len(demo['actions']) != row['length']:
            raise ValueError('Demonstration length changed')
        images = np.stack([demo[f'obs/{camera}'][start] for camera in CAMERAS])
        actions = demo['actions'][start:start + self.horizon].astype(np.float32)
        if actions.shape != (self.horizon, 7) or not np.isfinite(actions).all() or (abs(actions) > 1.00001).any():
            raise ValueError('Invalid native LIBERO actions')
        return torch.from_numpy(images).permute(0, 3, 1, 2), torch.from_numpy(actions), task

    def close(self):
        for f in self.files.values():
            f.close()
        self.files.clear()

    def __getstate__(self):
        return dict(self.__dict__, files={})
