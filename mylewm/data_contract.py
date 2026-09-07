"""Explicit data/compute contracts; never equate replacement draws with epochs."""
import hashlib
import os
from pathlib import Path


def file_sha256(path):
    sha = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            sha.update(block)
        # On unified-memory GB10, a 44 GiB hash scan exhausted CUDA allocation
        # despite reclaimable host cache. Discard clean cache of this file only;
        # this does not modify the dataset or global kernel cache policy.
        if hasattr(os,'posix_fadvise') and os.fstat(stream.fileno()).st_size >= 1024**3:
            try:
                os.posix_fadvise(stream.fileno(),0,0,os.POSIX_FADV_DONTNEED)
            except OSError:
                pass  # optional memory hint; hashing correctness is unchanged
    return sha.hexdigest()


def data_fingerprints(manifest):
    paths = {str(Path(manifest['dataset']).resolve())}
    paths.update(str(Path(item['path']).resolve()) for item in manifest.get('files', []))
    result = {}
    for name in sorted(paths):
        path = Path(name)
        before = path.stat()
        sha = file_sha256(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f'Data changed during hashing: {path}')
        result[name] = {'sha256': sha, 'size': after.st_size}
    return result


def training_budget(clips, updates, batch_size, world_size=1, accumulation=1):
    if min(clips, updates, batch_size, world_size, accumulation) < 1:
        raise ValueError('All budget quantities must be positive')
    effective = batch_size * world_size * accumulation
    return {'train_clips': int(clips), 'optimizer_updates': updates,
            'batch_per_device': batch_size, 'world_size': world_size,
            'gradient_accumulation': accumulation, 'effective_batch_size': effective,
            'updates_per_full_epoch_drop_last': clips // effective,
            'updates_for_100_full_epochs_drop_last': 100 * (clips // effective),
            'presented_clips': updates * effective,
            'presentation_equivalent_epochs': updates * effective / clips,
            'sampling': 'independent per-update sampling with replacement',
            'warning': 'Presentation ratio is not uniform traversal or official epoch reproduction.'}
