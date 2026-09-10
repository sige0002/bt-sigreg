"""Read-only normalization of the supported SWM episode-boundary schema."""
import h5py
import numpy as np
from pathlib import Path
from stable_worldmodel.data import HDF5Dataset


class EvaluationDataset(HDF5Dataset):
    def __init__(self, name, cache_dir=None, **kwargs):
        from stable_worldmodel.data.utils import get_cache_dir
        path = Path(get_cache_dir(cache_dir, sub_folder='datasets'), f'{name}.h5')
        validate_schema(path)
        super().__init__(name, cache_dir=cache_dir, **kwargs)
        with h5py.File(self.h5_path, 'r') as f:
            lengths = np.asarray(f['ep_len'])
            offsets = np.asarray(f['ep_offset'])
            if (lengths.ndim != 1 or offsets.shape != lengths.shape or
                    lengths.dtype.kind not in 'iu' or offsets.dtype.kind not in 'iu' or
                    np.any(lengths <= 0) or
                    not np.array_equal(offsets, np.r_[0, np.cumsum(lengths)[:-1]])):
                raise ValueError(f'Unsupported episode boundaries ep_len/ep_offset; available keys: {list(f)}')
            n = int(lengths.sum())
            if f['action'].shape[0] != n:
                raise ValueError(f'Episode boundaries do not cover action rows; available keys: {list(f)}')
            episode_key = next((key for key in ('episode_idx', 'ep_idx') if key in f), 'episode_idx')
            self._cache[episode_key] = f[episode_key][:] if episode_key in f else np.repeat(np.arange(len(lengths)), lengths)
            self._cache['step_idx'] = f['step_idx'][:] if 'step_idx' in f else np.arange(n) - np.repeat(offsets, lengths)
            for key in (episode_key, 'step_idx'):
                if self._cache[key].shape != (n,):
                    raise ValueError(f'Invalid {key} shape; available keys: {list(f)}')
                if key not in self._keys:
                    self._keys.append(key)

    def get_row_data(self, row_idx):
        self._open()
        return {col: (self._cache[col] if col in self._cache else self.h5_file[col])[row_idx]
                for col in self._keys}


def validate_schema(path):
    with h5py.File(path, 'r') as f:
        missing = {'ep_len', 'ep_offset', 'action'} - set(f)
        if missing:
            raise ValueError(f'Unsupported PushT HDF5: missing {sorted(missing)}; available keys: {list(f)}')
