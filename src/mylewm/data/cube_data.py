"""Read-only LeWM Cube data preparation, with explicit episode-held-out splits."""
import argparse
import json
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np

from mylewm.data.contract import PREPROCESSING, validate_contract
from mylewm.data.verification import data_fingerprints


def cube_contract():
    return validate_contract(dict(
        domain='ogbench_cube_single', camera='front_pixels',
        action_names=['delta_x', 'delta_y', 'delta_z', 'delta_yaw', 'delta_gripper'],
        action_units=['normalized_-1_1'] * 5,
        action_convention='ogbench_manipspace_relative_ee_xyz_yaw_gripper_v1',
        fps=20, frameskip=5, history=3, preprocessing=PREPROCESSING))


def inspect_schema(dataset):
    """Inspect dimensions and all physical actions without reading all RGB frames."""
    with h5py.File(dataset, 'r') as f:
        required = ('pixels', 'action', 'ep_len', 'ep_offset', 'qpos', 'qvel',
                    'privileged_block_0_pos', 'privileged_block_0_quat')
        if any(k not in f for k in required):
            raise ValueError('Cube dataset missing required image/action/episode/goal columns')
        if 'privileged_block_1_pos' in f or 'privileged_block_1_quat' in f:
            raise ValueError('Expected single-cube data, not a multi-cube variant')
        n = len(f['pixels'])
        if f['pixels'].shape != (n, 224, 224, 3) or f['pixels'].dtype != np.uint8:
            raise ValueError('Expected LeWM Cube uint8 RGB224 images')
        if f['action'].shape != (n, 5) or not np.issubdtype(f['action'].dtype, np.floating):
            raise ValueError('Cube requires five floating-point physical actions')
        lengths, offsets = f['ep_len'][:], f['ep_offset'][:]
        if (lengths.ndim != 1 or offsets.shape != lengths.shape or not len(lengths)
                or not np.issubdtype(lengths.dtype, np.integer)
                or not np.issubdtype(offsets.dtype, np.integer)
                or (lengths < 1).any() or lengths.sum() != n
                or not np.array_equal(offsets, np.r_[0, np.cumsum(lengths)[:-1]])):
            raise ValueError('Invalid episode boundaries')
        for key, width in (('privileged_block_0_pos', 3), ('privileged_block_0_quat', 4)):
            if f[key].shape != (n, width):
                raise ValueError(f'Invalid {key} shape')
        for key in ('qpos', 'qvel'):
            if f[key].ndim != 2 or len(f[key]) != n:
                raise ValueError(f'Invalid {key} shape')
        # Final rows may contain sentinel actions; only transitions are consumed.
        actions = np.asarray(f['action'], dtype=np.float64)
        terminal = np.zeros(n, dtype=bool)
        terminal[offsets + lengths - 1] = True
        if (not np.isfinite(actions[~terminal]).all()
                or (np.abs(actions[~terminal]) > 1.00001).any()):
            raise ValueError('Nonfinite or out-of-range nonterminal physical action')
        for key in ('qpos', 'qvel', 'privileged_block_0_pos', 'privileged_block_0_quat'):
            if not np.isfinite(f[key][:]).all():
                raise ValueError(f'Nonfinite state/goal column: {key}')
        columns = {k: dict(shape=list(f[k].shape), dtype=str(f[k].dtype)) for k in required}
    return lengths, offsets, actions, columns


def prepare(dataset, output, seed=3072, validation_cases=256, confirm_cases=200):
    dataset, output = Path(dataset).resolve(), Path(output)
    if output.exists():
        raise FileExistsError(output)
    if validation_cases < 2 or confirm_cases < 1:
        raise ValueError('Need at least two validation cases and one confirm case')
    before = dataset.stat()
    lengths, offsets, actions, columns = inspect_schema(dataset)
    eligible = np.flatnonzero(lengths >= 41)
    rng = np.random.default_rng(seed)
    ids = rng.permutation(eligible)
    a, b = int(.8 * len(ids)), int(.9 * len(ids))
    train, val, test = ids[:a], ids[a:b], ids[b:]
    if not len(train) or len(val) < validation_cases or len(test) < confirm_cases:
        raise ValueError('Insufficient distinct episodes for requested train/validation/confirm split')
    selected = np.concatenate([actions[int(offsets[e]):int(offsets[e]+lengths[e])-1] for e in train])
    mean, std = selected.mean(0), selected.std(0, ddof=1)
    if not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError('Invalid training-only action statistics')

    def cases(episodes, count):
        return [dict(episode=int(e), start=int(rng.integers(0, int(lengths[e])-40)))
                for e in episodes[:count]]

    manifest = dict(schema='cube_single_episode_split_v1', purpose='train', data_format='hdf5',
        dataset=str(dataset), dataset_size=before.st_size, dataset_mtime_ns=before.st_mtime_ns,
        input_contract=cube_contract(), history=3, frameskip=5, split_seed=seed,
        train_episodes=train.tolist(), validation_episodes=val.tolist(), test_episodes=test.tolist(),
        validation=cases(val, validation_cases), confirm=cases(test, confirm_cases),
        action_mean=mean.tolist(), action_std=std.tolist(), statistics_rows=len(selected),
        statistics='finite_nonterminal_train_episode_rows_sample_std_float64',
        train_clips=int(np.maximum(lengths[train]-19, 0).sum()), validation_clips=validation_cases,
        columns=columns, excluded_short_episodes=int(len(lengths)-len(eligible)),
        expected_source=dict(repo='quentinll/lewm-cube', revision='02a19a67a0dc8c9d6215f89c19e0a597691e152a'),
        comparison='matched Raw/Cayley/norm-preserving; not exact official training reproduction',
        official_differences=['episode 80/10/10 rather than overlapping clip 90/10',
            'train-only statistics rather than full dataset statistics',
            'explicit update-based LR schedule and deterministic epoch shuffle'])
    print('Preparing Cube SHA-256 (one full scan)', flush=True)
    manifest['data_fingerprints'] = data_fingerprints(manifest)
    after = dataset.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('Data changed during preparation')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--validation-cases', type=int, required=True)
    p.add_argument('--confirm-cases', type=int, required=True)
    args = p.parse_args()
    m = prepare(args.dataset, args.manifest, args.seed, args.validation_cases, args.confirm_cases)
    print(json.dumps({k: m[k] for k in ('train_clips', 'validation_clips', 'action_mean', 'action_std')}))


if __name__ == '__main__':
    main()
