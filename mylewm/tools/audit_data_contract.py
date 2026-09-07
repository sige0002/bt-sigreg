"""Read-only PushT recipe audit. The installed HDF5 adapter is not Lance reproduction."""
import argparse
import importlib.metadata
import inspect
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mylewm.data_contract import data_fingerprints, file_sha256, training_budget


def audit(manifest_path, updates=50000, batch_size=128):
    import h5py
    import hdf5plugin  # noqa: F401
    import numpy as np
    import torch
    from stable_worldmodel.data import load_dataset
    from stable_worldmodel.data.dataset import HDF5Dataset, Dataset
    from stable_pretraining.data import random_split
    from mylewm.train_rbg import Clips

    manifest = json.loads(manifest_path.read_text())
    dataset_path = Path(manifest['dataset']).resolve()
    identity = data_fingerprints(manifest)
    with h5py.File(dataset_path) as f:
        lengths = f['ep_len'][:]
        pixel_shape = list(f['pixels'].shape)
    controlled = Clips(manifest)
    controlled_budget = training_budget(len(controlled), updates, batch_size)
    controlled_hand_count = int(sum(lengths[manifest['train_episodes']] - 15))
    if controlled_hand_count != len(controlled):
        raise AssertionError('Controlled dataset count mismatch')
    try:
        load_dataset('pusht_expert_train.lance', cache_dir=str(dataset_path.parent.parent),
                     num_steps=4, frameskip=5, keys_to_load=['pixels', 'action'])
    except (ValueError, FileNotFoundError) as error:
        configured_loader = {'status':'blocked', 'error': str(error)}
    else:
        raise RuntimeError('Lance became available; audit its native loader rather than assume HDF5')
    # Explicit compatibility adaptation. We do not rewrite the official configuration.
    ds = HDF5Dataset(name=dataset_path.stem, cache_dir=dataset_path.parent.parent,
                     num_steps=4, frameskip=5, keys_to_load=['pixels', 'action'])
    generator = torch.Generator().manual_seed(3072)
    train, val = random_split(ds, [.9, .1], generator=generator)
    loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True,
                                        drop_last=True, generator=generator, num_workers=0)
    expected = int(np.maximum(lengths - 19, 0).sum())
    if expected != len(ds):
        raise AssertionError('Installed HDF5 loader span differs from audited formula')
    upstream = subprocess.check_output(['git','-C',str(ROOT/'external/le-wm'),
                                         'rev-parse','HEAD'],text=True).strip()
    sources = [ROOT/'lewm/train.py', ROOT/'lewm/utils.py', ROOT/'lewm/config/train/data/pusht.yaml',
               ROOT/'lewm/config/train/lewm.yaml', ROOT/'lewm/UPSTREAM.md',
               ROOT/'external/le-wm/config/train/data/pusht.yaml',
               Path(inspect.getfile(Dataset)), Path(inspect.getfile(random_split))]
    return {'schema':'pusht_recipe_audit_v1', 'manifest_sha256':file_sha256(manifest_path),
        'data_identity':identity, 'upstream_commit':upstream,
        'versions':{name:importlib.metadata.version(name) for name in
                    ('torch','stable-pretraining','stable-worldmodel','h5py')},
        'source_sha256':{str(p):file_sha256(p) for p in sources},
        'pixel_array_shape':pixel_shape,
        'official_repro':{'configured_loader':configured_loader,
            'configured_dataset':'pusht_expert_train.lance',
            'checkpoint_training_history':'unknown; no equivalence to installed dependencies asserted',
            'split':'clip-level 90/10 with shared split/shuffle generator',
            'normalization':'entire dataset column, remove NaN rows, torch sample std',
            'images':'ToImage with ImageNet statistics then Resize(224)',
            'schedule':'installed SPT manual update; see mylewm/docs/VALIDATION.ja.md'},
        'installed_hdf5_adaptation':{'status':'measured adaptation, not official_repro',
            'dataset_len':len(ds),'train_len':len(train),'validation_len':len(val),
            'train_loader_len':len(loader),'updates_for_100_epochs':100*len(loader),
            'span':ds.span,'clips_hand_count':expected,
            'train_indices_sha256':hashlib_array(train.indices),
            'validation_indices_sha256':hashlib_array(val.indices),
            'frames':[0,5,10,15], 'loaded_action_steps':20,
            'predictor_action_steps':15},
        'controlled_comparison':{'budget':controlled_budget,
            'split':'episode-level 80/10/10; eligible length >=41; immutable manifest',
            'split_sizes':{key:len(manifest[key]) for key in
                           ('train_episodes','validation_episodes','test_episodes')},
            'action_mean':manifest['action_mean'],'action_std':manifest['action_std'],
            'normalization':'train episodes only, exclude terminal and nonfinite rows, float64 sample std',
            'images':'uint8 / 255 and ImageNet normalization; no Resize in current PushT path',
            'frames':[0,5,10,15],'action_steps':15,
            'sampling':'StepBatches: SeedSequence([seed, update_index]), replacement',
            'same_method_clip_stream':'Raw/RBG with same seed, batch and update budget'},
        'differences':['Lance config vs installed HDF5 loader', 'clip vs episode split',
                       '90/10 vs 80/10/10', '20-frame vs 16-frame clip validity',
                       'all-data vs train-only action statistics', 'Resize vs native image size',
                       'shuffle without replacement vs per-update replacement',
                       '100 epochs vs fixed update budget', 'scheduler update indexing'],
        'content_hash_policy':'SHA256 streamed from actual files; saved for reuse/comparison. Trainer rehashes at every start/resume; size/mtime alone is insufficient.'}


def hashlib_array(indices):
    import hashlib
    import numpy as np
    return hashlib.sha256(np.asarray(indices,dtype='<i8').tobytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path,default=ROOT/'.cache/stable-wm/pusht/rbg_v0/manifest.json')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    report=audit(args.manifest)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream: json.dump(report,stream,indent=2)
    print(json.dumps({'output':str(args.output),
        'controlled':report['controlled_comparison']['budget'],
        'installed_hdf5_adaptation':report['installed_hdf5_adaptation'],
        'official_repro':report['official_repro']['configured_loader']}),flush=True)


if __name__=='__main__':main()
