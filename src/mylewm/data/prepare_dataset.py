"""Prepare one HDF5 or LeRobot v3 source; never mix storage formats in a run."""
import argparse
import json
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np

from mylewm.data.verification import data_fingerprints
from mylewm.data.contract import validate_contract


def download(repo_id, revision, destination):
    from huggingface_hub import HfApi, snapshot_download
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('Download requires a fresh directory')
    sha = HfApi().dataset_info(repo_id, revision=revision).sha
    snapshot_download(repo_id, repo_type='dataset', revision=sha, local_dir=destination,
                      allow_patterns=['meta/**', 'data/**', 'videos/**', 'images/**'])
    return sha


def prepare(dataset, output, data_format, contract, camera_key=None, purpose='train',
            repo_id=None, revision=None, seed=20260906):
    output, dataset = Path(output), Path(dataset).resolve()
    if output.exists():
        raise FileExistsError(output)
    c = validate_contract(contract)
    dim, span = len(c['action_names']), 4 * c['frameskip']
    if purpose not in ('train', 'inference') or data_format not in ('hdf5', 'lerobot'):
        raise ValueError('Invalid purpose or data format')
    files = []
    if data_format == 'hdf5':
        if dataset.suffix != '.h5':
            raise ValueError('HDF5 input must be a .h5 file')
        with h5py.File(dataset, 'r') as f:
            lengths = np.asarray(f['ep_len'][:], dtype=np.int64)
            offsets = np.asarray(f['ep_offset'][:], dtype=np.int64)
            if (len(f['pixels'].shape) != 4 or f['pixels'].shape[-1] != 3
                    or f['pixels'].dtype != np.uint8 or f['action'].shape != (len(f['pixels']), dim)):
                raise ValueError('HDF5 requires RGB uint8 pixels and matching action dimensions')
            if (not len(lengths) or lengths.shape != offsets.shape or (lengths < 1).any()
                    or offsets[0] != 0 or not np.array_equal(offsets[1:], (offsets + lengths)[:-1])
                    or offsets[-1] + lengths[-1] != len(f['pixels'])):
                raise ValueError('Invalid HDF5 episode boundaries')
        def read_actions(ep):
            with h5py.File(dataset, 'r') as f:
                return f['action'][int(offsets[ep]):int(offsets[ep] + lengths[ep] - 1)]
    else:
        from mylewm.data.lerobot_data import open_local, episode_metadata, inventory
        if not camera_key:
            raise ValueError('--camera-key is required for LeRobot')
        ds = open_local(dataset, camera_key, c['frameskip'], revision, repo_id or 'local/dataset')
        if ds.fps != c['fps'] or tuple(ds.features['action']['shape']) != (dim,):
            raise ValueError('LeRobot FPS/action dimensions disagree with the input contract')
        lengths, offsets = episode_metadata(ds)
        table = ds.hf_dataset.with_format(None)
        # Check row indexing and time alignment without decoding videos. Episode
        # metadata alone is insufficient to certify transition alignment.
        for ep, (offset, length) in enumerate(zip(offsets, lengths)):
            rows = table.select_columns(['index', 'episode_index', 'frame_index', 'timestamp'])[int(offset):int(offset + length)]
            if (not np.array_equal(rows['index'], np.arange(offset, offset + length))
                    or not np.array_equal(rows['frame_index'], np.arange(length))
                    or not np.all(np.asarray(rows['episode_index']) == ep)
                    or not np.allclose(rows['timestamp'], np.arange(length) / c['fps'], rtol=0, atol=1e-4)):
                raise ValueError(f'Invalid LeRobot frame/time alignment in episode {ep}')
        def read_actions(ep):
            return np.asarray(table.select_columns(['action'])[int(offsets[ep]):int(offsets[ep] + lengths[ep] - 1)]['action'])
        all_files = inventory(dataset)
        dataset = dataset / 'meta/info.json'
        files = [{'path': str(p), 'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns}
                 for p in all_files if p != dataset]
    eligible = np.flatnonzero(lengths >= span)
    if purpose == 'train' and len(eligible) < 3:
        raise ValueError('At least three eligible episodes are required for train/validation/test')
    if not len(eligible):
        raise ValueError('No complete unpadded clips')
    rng = np.random.default_rng(seed)
    ids = rng.permutation(eligible)
    if purpose == 'train':
        held = max(1, int(len(ids) * .1))
        train, val, test = ids[:-2 * held], ids[-2 * held:-held], ids[-held:]
    else:
        train, val, test = np.array([], dtype=int), np.array([], dtype=int), eligible
    manifest = dict(schema='trajectory_v1', data_format=data_format, purpose=purpose,
                    dataset=str(dataset), dataset_size=dataset.stat().st_size,
                    dataset_mtime_ns=dataset.stat().st_mtime_ns, files=files,
                    input_contract=c, history=3, frameskip=c['frameskip'],
                    episode_lengths=lengths.tolist(), episode_offsets=offsets.tolist(),
                    train_episodes=train.tolist(), validation_episodes=val.tolist(), test_episodes=test.tolist(),
                    split_seed=seed, camera_key=camera_key, repo_id=repo_id, revision=revision,
                    validation=[{'episode': int(ep), 'start': int(rng.integers(lengths[ep] - span + 1))}
                                for ep in val[:256]])
    # Streaming population moments on training episodes only; exclude terminal
    # actions. Constant controls get scale 1 rather than an undefined division.
    count, mean, m2 = 0, np.zeros(dim), np.zeros(dim)
    for ep in eligible:
        a = np.asarray(read_actions(ep), dtype=np.float64)
        if a.shape != (lengths[ep] - 1, dim) or not np.isfinite(a).all():
            raise ValueError(f'Invalid nonterminal action in episode {ep}')
        if ep not in train:
            continue
        n, avg = len(a), a.mean(0)
        delta = avg - mean
        m2 += ((a - avg) ** 2).sum(0) + delta ** 2 * count * n / (count + n)
        mean += delta * n / (count + n)
        count += n
    if purpose == 'train':
        if count < 2:
            raise ValueError('Too few training actions for sample standard deviation')
        std = np.sqrt(m2 / (count - 1))
        manifest.update(action_mean=mean.tolist(), action_std=np.where(std > 0, std, 1.).tolist(),
                        constant_action_dimensions=np.flatnonzero(std == 0).tolist())
    print('Preparing data SHA-256 (one-time full scan)', flush=True)
    manifest['data_fingerprints'] = data_fingerprints(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--format', choices=['hdf5', 'lerobot'], required=True)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--dataset', type=Path)
    source.add_argument('--repo-id')
    p.add_argument('--revision', default='main')
    p.add_argument('--download-root', type=Path)
    p.add_argument('--camera-key')
    p.add_argument('--contract', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--purpose', choices=['train', 'inference'], default='train')
    args = p.parse_args()
    contract = validate_contract(json.loads(args.contract.read_text()))
    if args.manifest.exists():
        raise FileExistsError(args.manifest)
    revision = None
    if args.repo_id:
        if args.format != 'lerobot' or args.download_root is None:
            p.error('Hub download requires --format lerobot and --download-root')
        revision = download(args.repo_id, args.revision, args.download_root)
        args.dataset = args.download_root
    elif args.download_root is not None:
        p.error('--download-root requires --repo-id')
    m = prepare(args.dataset, args.manifest, args.format, contract, args.camera_key,
                args.purpose, args.repo_id, revision)
    print(json.dumps({'manifest': str(args.manifest), 'format': m['data_format'],
                      'episodes': len(m['episode_lengths']), 'revision': revision}))


if __name__ == '__main__':
    main()
