"""OGBench Cube single: matched Raw, Cayley BT and norm-preserving BT."""
import argparse
import copy
import importlib.metadata
import json
from pathlib import Path
import hydra
import lightning as pl
import numpy as np
from omegaconf import OmegaConf
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from mylewm.paths import ROOT,source_files
from mylewm.data.verification import file_sha256,verify_training_data
from mylewm.data.contract import validate_contract
from mylewm.training.state import capture_rng,tensor_state_hash
from mylewm.training.train import (GaussianBranch,TrainingModule,EpochBatches,Export,datasets,
                                  compile_encoder,compiler_identity,make_trainer)
from mylewm.training.train_pusht_transport import NormPreservingGaussianBranch



def _require_exact(actual, expected, name='config'):
    if type(actual) is not type(expected):
        raise ValueError(f'Cube transport contract: wrong type at {name}')
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ValueError(f'Cube transport contract: missing/unknown keys at {name}: '
                             f'{sorted(set(actual) ^ set(expected))}')
        for key in expected:
            _require_exact(actual[key], expected[key], f'{name}.{key}')
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(f'Cube transport contract: wrong length at {name}')
        for i, (left, right) in enumerate(zip(actual, expected)):
            _require_exact(left, right, f'{name}[{i}]')
    elif actual != expected:
        raise ValueError(f'Cube transport contract: {name} must be {expected!r}, got {actual!r}')


def expected_config(family):
    from mylewm.training.train_pusht_transport import expected_config as pusht_config
    if family not in ('raw', 'cayley', 'norm_preserving'):
        raise ValueError('Explicit raw, cayley or norm_preserving family required')
    cfg = pusht_config('cayley' if family == 'raw' else family)
    cfg['environment'] = 'ogbench_cube_single'
    cfg['algorithm'] = 'cube_' + family + '_sigreg_v1'
    cfg['model']['action_encoder']['input_dim'] = 25
    # Match official LeWM Embedder default explicitly; do not inherit PushT dims.
    cfg['model']['action_encoder']['smoothed_dim'] = 10
    if family == 'raw':
        cfg['regularizer'] = dict(family='raw', dim=192, projections=1024, knots=17)
    return cfg


def load_model_config(path):
    cfg=OmegaConf.to_container(OmegaConf.load(path),resolve=True,throw_on_missing=True)
    _require_exact(cfg,expected_config(cfg.get('regularizer',{}).get('family')))
    return cfg


def validate_manifest(manifest):
    from mylewm.data.cube_data import cube_contract
    if (manifest.get('schema') != 'cube_single_episode_split_v1'
            or manifest.get('purpose') != 'train'
            or manifest.get('data_format') != 'hdf5'
            or manifest.get('input_contract') != cube_contract()
            or manifest.get('history') != 3 or manifest.get('frameskip') != 5
            or len(manifest.get('action_mean', [])) != 5
            or len(manifest.get('action_std', [])) != 5):
        raise ValueError('Cube single requires its prepared manifest and history3/block5/action5')
    mean, std = np.asarray(manifest['action_mean']), np.asarray(manifest['action_std'])
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError('Invalid Cube training action statistics')


def build_model(config):
    _require_exact(config,expected_config(config['regularizer']['family']))
    return hydra.utils.instantiate(OmegaConf.create(config['model']))


def build_regularizer(config,seed):
    c=config['regularizer']
    if c['family']=='raw':
        return GaussianBranch('raw',seed,dim=c['dim'],projections=c['projections'])
    if c['family']=='norm_preserving':
        return NormPreservingGaussianBranch(seed,dim=c['dim'],depth=c['depth'],kappa=c['kappa'],projections=c['projections'])
    return GaussianBranch('bt',seed,dim=c['dim'],depth=c['depth'],hidden=c['hidden'],kappa=c['kappa'],projections=c['projections'])


def run(args):
    if args.steps < 2 or not 0 < args.warmup_steps < args.steps or args.batch_size < 2 or args.workers < 0 or args.save_every < 1 or not np.isfinite(args.lr) or args.lr <= 0:
        raise ValueError('Invalid steps/warmup/batch/workers/save frequency/lr')
    val_every = getattr(args, 'val_every', None)
    if val_every is None:
        val_every = args.save_every
    if val_every < 1:
        raise ValueError('--val-every must be a positive number of updates')
    if args.output.exists():
        raise FileExistsError('Use a fresh output directory, including for resume')
    if args.resume and (not args.resume.is_file() or args.resume.suffix != '.ckpt'):
        raise ValueError('--resume requires a trusted Lightning .ckpt, not legacy resume.pt')
    compiled = getattr(args, 'compile_encoder', False)
    if compiled and args.accelerator != 'gpu':
        raise ValueError('--compile-encoder is supported only with --accelerator gpu')
    config = load_model_config(args.model_config)
    manifest = json.loads(args.manifest.read_text())
    validate_manifest(manifest)
    if manifest.get('purpose', 'train') != 'train':
        raise ValueError('Training requires a training manifest')
    recipe = {k: v for k, v in vars(args).items() if k not in ('output', 'execute', 'resume', 'manifest')}
    recipe.update(schema='cube_transport_comparison_v1', resolved_model_config=config,
                  model_config_sha256=file_sha256(Path(args.model_config)), manifest_sha256=file_sha256(args.manifest),
                  pin_memory=getattr(args, 'pin_memory', True),
                  compile_encoder=compiled,
                  val_every=val_every,
                  loss_weight=.09, sampling='pytorch_epoch_shuffle_drop_last',
                  normalization='frozen_manifest_train_only', statistics_precision='float64',
                  source_sha256={str(p.relative_to(ROOT)): file_sha256(p) for p in
                    [*source_files(), ROOT/'lewm/train.py', ROOT/'lewm/utils.py',
                     ROOT/'lewm/module.py', ROOT/'lewm/jepa.py', ROOT/'lewm/config/train/model/lewm.yaml']},
                  versions={n: importlib.metadata.version(n) for n in
                    ('torch', 'lightning', 'stable-pretraining', 'stable-worldmodel', 'numpy', 'h5py', 'transformers')})
    if manifest.get('input_contract'):
        recipe['input_contract'] = validate_contract(manifest['input_contract'])
    recipe['data_format'] = manifest.get('data_format', 'hdf5')
    # Versions alone do not identify locally modified installed libraries.
    import inspect
    from stable_pretraining.optim.lr_scheduler import LinearWarmupCosineAnnealingLR
    recipe['dependency_sources'] = {name: file_sha256(Path(inspect.getfile(obj))) for name, obj in
        [('spt_module', spt.Module), ('swm_dataset', swm.data.HDF5Dataset),
         ('spt_scheduler', LinearWarmupCosineAnnealingLR), ('lightning_trainer', pl.Trainer)]}
    if compiled:
        recipe['compiler'] = compiler_identity()
    if not args.execute:
        print(json.dumps(recipe, indent=2))
        print('Dry-run only; no model load, GPU initialization, or training.')
        return
    # Explicit device avoids silently falling back to CPU for a long run.
    torch.set_num_threads(4)
    pl.seed_everything(args.seed, workers=True)
    if args.accelerator == 'gpu':
        torch.cuda.init()
    full = getattr(args, 'verify_data', False)
    print('Verifying dataset: ' + ('full SHA-256 scan' if full else 'size and mtime (no full scan)'), flush=True)
    recipe['data_identity'] = verify_training_data(manifest, full=full)
    train_set, val_set = datasets(manifest)
    if (len(train_set) != manifest['train_clips'] or len(val_set) != manifest['validation_clips']):
        raise ValueError('Native Cube clip counts differ from prepared manifest')
    recipe.update(train_clips=len(train_set), validation_clips=len(val_set))
    batches = EpochBatches(train_set, args.batch_size, args.steps, args.seed)
    train_loader = DataLoader(train_set, batch_sampler=batches, num_workers=args.workers,
                              pin_memory=recipe['pin_memory'] and args.accelerator == 'gpu',
                              generator=torch.Generator().manual_seed(args.seed + 100))
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=0,
                            pin_memory=recipe['pin_memory'] and args.accelerator == 'gpu',
                            generator=torch.Generator().manual_seed(args.seed + 101))
    model = build_model(config)
    if manifest.get('input_contract'):
        model.input_contract = copy.deepcopy(recipe['input_contract'])
    recipe['initial_model_sha256'] = tensor_state_hash(model.state_dict())
    for name in ('mean', 'std'):
        model.register_buffer('training_action_' + name, torch.tensor(manifest['action_' + name], dtype=torch.float64))
    if compiled:
        print('Compiling image encoder; the first training/validation batch can take longer.', flush=True)
        compile_encoder(model)
    reg = build_regularizer(config, args.seed + 10000)
    module = TrainingModule(model, reg, recipe, batches)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'config.json').write_text(json.dumps(recipe, indent=2))
    torch.save({'model': model.state_dict(), 'regularizer': reg.state_dict(),
                'random_state': capture_rng()}, args.output/'initialization.pt')
    checkpoint = ModelCheckpoint(dirpath=args.output, filename='step_{step}',
                                  every_n_train_steps=args.save_every, save_top_k=-1,
                                  save_last=True, auto_insert_metric_name=False)
    trainer = make_trainer(default_root_dir=args.output, accelerator=args.accelerator, devices=1,
        max_steps=args.steps, max_epochs=-1, precision=args.precision, deterministic=True,
        num_sanity_val_steps=0, check_val_every_n_epoch=None, val_check_interval=val_every,
        logger=CSVLogger(str(args.output), name='metrics'), log_every_n_steps=1,
        callbacks=[Export(args.output, args.save_every), checkpoint], enable_model_summary=False)
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader,
                ckpt_path=str(args.resume.resolve()) if args.resume else None, weights_only=False)
    trainer.save_checkpoint(args.output/'last.ckpt')
    (args.output/'completed.json').write_text(json.dumps({'step': trainer.global_step,
        'state': 'completed', 'recipe': recipe['schema']}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--model-config',required=True)
    p.add_argument('--output',type=Path,required=True)
    for name in ('steps','warmup-steps','batch-size','workers','save-every','val-every','seed'):
        p.add_argument('--'+name,type=int,required=True)
    p.add_argument('--lr',type=float,required=True)
    p.add_argument('--pin-memory',action=argparse.BooleanOptionalAction,required=True)
    p.add_argument('--compile-encoder',action=argparse.BooleanOptionalAction,required=True)
    p.add_argument('--accelerator',choices=['cpu','gpu'],required=True)
    p.add_argument('--precision',choices=['32-true','bf16-mixed'],required=True)
    p.add_argument('--verify-data',action='store_true')
    p.add_argument('--resume',type=Path)
    p.add_argument('--execute',action='store_true')
    args=p.parse_args()
    cfg=load_model_config(args.model_config)
    args.model_config=str(Path(args.model_config).resolve())
    args.mode='raw' if cfg['regularizer']['family']=='raw' else 'bt';args.algorithm=cfg['algorithm']
    run(args)


if __name__=='__main__':main()
