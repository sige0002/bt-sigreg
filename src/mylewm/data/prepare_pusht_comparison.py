"""Prepare an explicit 90/10 clip split for matched PushT Raw/BT training."""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch
import stable_pretraining as spt
import stable_worldmodel as swm

from mylewm.data.verification import data_fingerprints


def index_hash(indices):
    return hashlib.sha256(np.asarray(indices, dtype='<i8').tobytes()).hexdigest()


def clip_split(dataset, seed):
    return spt.data.random_split(dataset, [.9, .1],
                                 generator=torch.Generator().manual_seed(seed))


def training_action_rows(dataset, train_indices, total_rows):
    """Unique physical rows consumed by training clips, not held-only rows."""
    clips = np.asarray(dataset.clip_indices, dtype=np.int64)[train_indices]
    starts = np.asarray(dataset.offsets)[clips[:, 0]] + clips[:, 1]
    delta = np.zeros(total_rows + 1, dtype=np.int64)
    np.add.at(delta, starts, 1)
    np.add.at(delta, starts + dataset.span, -1)
    return np.cumsum(delta[:-1]) > 0


def prepare(dataset, output, seed=3072):
    dataset, output = Path(dataset).resolve(), Path(output)
    if output.exists():
        raise FileExistsError(output)
    before = dataset.stat()
    ds = swm.data.HDF5Dataset(str(dataset.with_suffix('')), num_steps=4,
                              frameskip=5, keys_to_load=['pixels', 'action'])
    train, val = clip_split(ds, seed)
    if len(train) < 128 or not len(val):
        raise ValueError('Insufficient clips for a full training batch and validation')
    with h5py.File(dataset, 'r') as f:
        actions = np.asarray(f['action'], dtype=np.float64)
        if (actions.ndim != 2 or actions.shape[1] != 2 or
                f['pixels'].shape != (len(actions), 224, 224, 3) or
                f['pixels'].dtype != np.uint8):
            raise ValueError('Expected public PushT RGB224 data and 2D actions')
    rows = training_action_rows(ds, train.indices, len(actions))
    finite = np.isfinite(actions).all(axis=1)
    selected = actions[rows & finite]
    if len(selected) < 2:
        raise ValueError('Insufficient finite training actions')
    mean, std = selected.mean(axis=0), selected.std(axis=0, ddof=1)
    if (std <= 0).any():
        raise ValueError('Constant action dimension')
    manifest = dict(schema='pusht_clip_split_v1', purpose='train', data_format='hdf5',
        dataset=str(dataset), dataset_size=before.st_size, dataset_mtime_ns=before.st_mtime_ns,
        history=3, frameskip=5, split_kind='random_clip_90_10', split_seed=seed,
        total_clips=len(ds), train_clips=len(train), validation_clips=len(val),
        train_indices_sha256=index_hash(train.indices), validation_indices_sha256=index_hash(val.indices),
        clip_order_sha256=index_hash(np.asarray(ds.clip_indices).reshape(-1)),
        action_mean=mean.tolist(), action_std=std.tolist(),
        statistics='unique_finite_physical_action_rows_in_training_clips_sample_std',
        statistics_rows=len(selected), batch_size_reference=128,
        updates_per_epoch_reference=len(train) // 128,
        warning='Clip split permits overlapping frames across train/validation; not episode-held-out evaluation.')
    print(f'Clips: total={len(ds)}, train={len(train)}, validation={len(val)}; '
          f'updates/epoch={len(train)//128}', flush=True)
    print('Preparing SHA-256: one full data scan', flush=True)
    manifest['data_fingerprints'] = data_fingerprints(manifest)
    after = dataset.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('Dataset changed during preparation')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--seed', type=int, default=3072)
    args = p.parse_args()
    prepare(args.dataset, args.manifest, args.seed)


if __name__ == '__main__':
    main()
