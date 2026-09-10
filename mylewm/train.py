"""PushT Raw/BT on the official SWM/SPT/Lightning stack (new recipe).

The shared training.py uses a separate recipe; historical artifacts are preserved. No downloads, implicit
resume, or training without --execute. Single-device, deterministic transforms.
"""
import argparse
import copy
from functools import partial
import importlib.metadata
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'lewm'))
sys.path.insert(0, str(ROOT))

import hydra
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
import numpy as np
from omegaconf import OmegaConf
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from torch.utils.data import BatchSampler, DataLoader, DistributedSampler, Subset

from lewm.train import lejepa_forward
from module import SIGReg
from utils import get_img_preprocessor
from mylewm.bt_sigreg import BoundedTransport, add_bt_arguments
from mylewm.data_contract import file_sha256, verify_training_data
from mylewm.training_state import capture_rng, restore_rng, tensor_state_hash


class GaussianBranch(torch.nn.Module):
    """Official SIGReg in FP32; optional T, with an isolated projection stream."""
    def __init__(self, mode, seed, dim=192, depth=2, hidden=192, kappa=.2,
                 projections=1024):
        super().__init__()
        if mode not in ('raw', 'bt'):
            raise ValueError('Only raw and bt are supported by this recipe')
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.transport = (BoundedTransport(dim, hidden, depth, kappa)
                              if mode == 'bt' else torch.nn.Identity())
        self.gaussian = SIGReg(num_proj=projections)
        self.seed = seed
        # A CUDA scalar converted to int synchronizes every forward. Keep this
        # counter on the host and persist it through Module's checkpoint API.
        self.draw_index = 0

    def get_extra_state(self):
        return {'draw_index': self.draw_index}

    def set_extra_state(self, state):
        index = state['draw_index']
        if type(index) is not int or index < 0:
            raise ValueError('Invalid projection draw index')
        self.draw_index = index

    def forward(self, z):
        devices = [z.device.index] if z.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            seed = self.seed + self.draw_index
            torch.random.default_generator.manual_seed(seed)
            if z.is_cuda:
                torch.cuda.default_generators[z.device.index].manual_seed(seed)
            with torch.autocast(device_type=z.device.type, enabled=False):
                loss = self.gaussian(self.transport(z.float()))
        if self.training:
            self.draw_index += 1
        return loss


class Preprocess:
    """Official image transform, frozen train-only action statistics."""
    def __init__(self, manifest):
        self.image = get_img_preprocessor(source='pixels', target='pixels', img_size=224)
        self.mean = torch.tensor(manifest['action_mean'], dtype=torch.float64)
        self.std = torch.tensor(manifest['action_std'], dtype=torch.float64)
        if self.mean.shape != (2,) or self.std.shape != (2,):
            raise ValueError('PushT requires two-dimensional physical actions')
        if not torch.isfinite(self.mean).all() or not torch.isfinite(self.std).all() or (self.std <= 0).any():
            raise ValueError('Invalid frozen action statistics')

    def __call__(self, batch):
        # SWM applies transforms before grouping the 20 physical actions into 4x10.
        batch = self.image(batch)
        if not torch.isfinite(batch['action'][:15]).all():
            raise ValueError('Non-finite action in prediction context')
        batch['action'] = ((batch['action'].double() - self.mean) / self.std).float()
        return batch


def datasets(manifest):
    path = Path(manifest['dataset']).resolve()
    if not path.is_file() or path.suffix != '.h5':
        raise ValueError('Existing local .h5 required; downloads are not supported')
    for key, actual in [('dataset_size', path.stat().st_size),
                        ('dataset_mtime_ns', path.stat().st_mtime_ns)]:
        if key in manifest and manifest[key] != actual:
            raise ValueError('Dataset changed since manifest preparation')
    if manifest.get('history') != 3 or manifest.get('frameskip') != 5:
        raise ValueError('This recipe requires history=3, frameskip=5')
    # Absolute name is intentional: HDF5Dataset joins it without copying data.
    ds = swm.data.HDF5Dataset(name=str(path.with_suffix('')), cache_dir=ROOT / '.cache/stable-wm',
                             num_steps=4, frameskip=5, keys_to_load=['pixels', 'action'],
                             transform=Preprocess(manifest))
    groups = [manifest[k] for k in ('train_episodes', 'validation_episodes', 'test_episodes')]
    flat = [e for g in groups for e in g]
    if len(flat) != len(set(flat)) or any(not isinstance(e, int) or e < 0 or e >= len(ds.lengths) for e in flat):
        raise ValueError('Episode splits overlap, contain duplicates, or invalid IDs')
    subsets = []
    selected_validation = {(c['episode'], c['start']) for c in manifest.get('validation', [])}
    if selected_validation and (len(selected_validation) != len(manifest['validation']) or
                                any(ep not in groups[1] for ep, _ in selected_validation)):
        raise ValueError('Invalid or duplicate validation cases')
    for split, group in enumerate(groups[:2]):
        allowed = set(group)
        ids = [i for i, (ep, start) in enumerate(ds.clip_indices) if ep in allowed and
               (split == 0 or not selected_validation or (ep, start) in selected_validation)]
        if split == 1 and selected_validation and len(ids) != len(selected_validation):
            raise ValueError('Validation cases outside native loader clip range')
        if not ids:
            raise ValueError('Empty train/validation split')
        subsets.append(Subset(ds, ids))
    return tuple(subsets)


class EpochBatches:
    """Library epoch shuffle plus a resume cursor; no custom random extraction.

    A single virtual Lightning epoch contains the requested update budget.
    The underlying PyTorch sampler reshuffles at each complete dataset pass.
    Only the consumed global step is checkpointed, never a prefetched cursor.
    """
    def __init__(self, dataset, batch_size, steps, seed):
        self.sampler = DistributedSampler(dataset, num_replicas=1, rank=0, seed=seed)
        self.batches = BatchSampler(self.sampler, batch_size, drop_last=True)
        if len(self.batches) < 1:
            raise ValueError('Training split must contain a full batch')
        self.steps, self.start = steps, 0

    def __len__(self):
        # Lightning restores its own completed-batch count. Keep the original
        # virtual epoch length or it will end early and replay the resume tail.
        return self.steps

    def __iter__(self):
        step = self.start
        while step < self.steps:
            epoch, offset = divmod(step, len(self.batches))
            self.sampler.set_epoch(epoch)
            for batch in itertools.islice(self.batches, offset, None):
                if step >= self.steps:
                    return
                yield batch
                step += 1


class TrainingModule(spt.Module):
    """SPT owns backward/optimizer/scheduler; hooks preserve the run contract."""
    def __init__(self, model, regularizer, recipe, batches):
        cfg = OmegaConf.create({'history_size': 3, 'num_preds': 1,
                                'loss': {'sigreg': {'weight': .09}}})
        super().__init__(model=model, sigreg=regularizer,
                         forward=partial(lejepa_forward, cfg=cfg), hparams={'recipe': recipe},
                         optim={'main': {'modules': '^(model|sigreg)(\\.|$)',
                            'optimizer': {'type': 'AdamW', 'lr': recipe['lr'], 'weight_decay': .001,
                                          'exclude_bias_norm': False},
                            'scheduler': {'type': 'LinearWarmupCosineAnnealingLR',
                                'warmup_steps': recipe['warmup_steps'], 'max_steps': recipe['steps'],
                                'warmup_start_lr': 0., 'eta_min': 0.}, 'interval': 'step'}})
        self.recipe, self.batches = recipe, batches
        self.resume_rng = None

    def on_before_optimizer_step(self, optimizer):
        # Same independent model/T clipping as the legacy recipe, after unscale.
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1., error_if_nonfinite=True)
        torch.nn.utils.clip_grad_norm_(self.sigreg.parameters(), 1., error_if_nonfinite=True)

    def on_train_batch_start(self, batch, batch_idx):
        self.log('update', float(self.global_step + 1), on_step=True, on_epoch=False)
        self.log('lr_used', self.optimizers().param_groups[0]['lr'], on_step=True, on_epoch=False)

    def on_save_checkpoint(self, checkpoint):
        checkpoint['recipe'] = self.recipe
        checkpoint['random_state'] = capture_rng()

    def on_load_checkpoint(self, checkpoint):
        if checkpoint.get('recipe') != self.recipe:
            raise ValueError('Recipe mismatch; legacy or changed-condition resume refused')
        step = checkpoint['global_step']
        if not 0 <= step < self.recipe['steps']:
            raise ValueError('Checkpoint has no remaining training steps')
        self.batches.start = step
        self.resume_rng = checkpoint['random_state']

    def on_train_start(self):
        super().on_train_start()
        if self.resume_rng is not None:
            restore_rng(self.resume_rng)
            self.resume_rng = None


class Export(pl.Callback):
    """Inference export only; Lightning's ModelCheckpoint owns training state."""
    def __init__(self, output, every):
        self.output, self.every = Path(output), every

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        if trainer.global_step % self.every == 0 or trainer.global_step == trainer.max_steps:
            path = self.output / f'step_{trainer.global_step}_object.ckpt'
            if path.exists():
                raise FileExistsError(path)
            # No T in the exported object. Do not change the training model's mode.
            torch.save(copy.deepcopy(module.model).eval(), path)


def make_trainer(**kwargs):
    # Use the library's public configuration API. No background environment
    # collection, hardware watcher, external tracker, or extra model exports.
    disabled = ('env_dump', 'hf_checkpoint', 'sklearn_checkpoint',
                'wandb_checkpoint', 'trackio_checkpoint', 'swanlab_checkpoint')
    from stable_pretraining._config import get_config
    previous = dict(get_config().default_callbacks)
    spt.set(default_callbacks={name: False for name in disabled})
    try:
        trainer = pl.Trainer(**kwargs)
    finally:
        spt.set(default_callbacks={**{name: True for name in disabled}, **previous})
    # Installed SPT registers this callback but omits its public config key.
    from stable_pretraining.callbacks.hardware_monitor import HardwareMonitor
    trainer.callbacks = [cb for cb in trainer.callbacks if not isinstance(cb, HardwareMonitor)]
    return trainer


def build_model():
    cfg = OmegaConf.create({'embed_dim': 192, 'history_size': 3, 'img_size': 224,
                           'model': OmegaConf.load(ROOT / 'lewm/config/train/model/lewm.yaml')})
    cfg.model.action_encoder.input_dim = 10
    return hydra.utils.instantiate(cfg.model)


def compile_encoder(model):
    # In-place Module.compile keeps checkpoint keys unchanged. Module's pickle
    # state omits the compiled callable, so deepcopy/export remains eager.
    # Keep the predictor (including its dropout) and projection RNG in eager mode.
    model.encoder.compile(backend='inductor', mode='default')


def compiler_identity():
    from triton import knobs
    assembler = knobs.nvidia.ptxas
    path = Path(assembler.path).resolve()
    return dict(backend='inductor', mode='default', triton=importlib.metadata.version('triton'),
                ptxas_path=str(path), ptxas_version=assembler.version,
                ptxas_sha256=file_sha256(path))


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
    manifest = json.loads(args.manifest.read_text())
    recipe = {k: v for k, v in vars(args).items() if k not in ('output', 'execute', 'resume', 'manifest')}
    recipe.update(schema='pusht_spt_v1', manifest_sha256=file_sha256(args.manifest),
                  pin_memory=getattr(args, 'pin_memory', True),
                  compile_encoder=compiled,
                  val_every=val_every,
                  loss_weight=.09, sampling='pytorch_epoch_shuffle_drop_last',
                  normalization='frozen_manifest_train_only', statistics_precision='float32',
                  source_sha256={str(p.relative_to(ROOT)): file_sha256(p) for p in
                    [Path(__file__), ROOT/'mylewm/bt_sigreg.py', ROOT/'mylewm/training_state.py',
                     ROOT/'mylewm/data_contract.py', ROOT/'lewm/train.py', ROOT/'lewm/utils.py',
                     ROOT/'lewm/module.py', ROOT/'lewm/jepa.py', ROOT/'lewm/config/train/model/lewm.yaml']},
                  versions={n: importlib.metadata.version(n) for n in
                    ('torch', 'lightning', 'stable-pretraining', 'stable-worldmodel', 'numpy', 'h5py', 'transformers')})
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
    recipe.update(train_clips=len(train_set), validation_clips=len(val_set))
    batches = EpochBatches(train_set, args.batch_size, args.steps, args.seed)
    train_loader = DataLoader(train_set, batch_sampler=batches, num_workers=args.workers,
                              pin_memory=recipe['pin_memory'] and args.accelerator == 'gpu',
                              generator=torch.Generator().manual_seed(args.seed + 100))
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=0,
                            pin_memory=recipe['pin_memory'] and args.accelerator == 'gpu',
                            generator=torch.Generator().manual_seed(args.seed + 101))
    model = build_model()
    recipe['initial_model_sha256'] = tensor_state_hash(model.state_dict())
    for name in ('mean', 'std'):
        model.register_buffer('training_action_' + name, torch.tensor(manifest['action_' + name], dtype=torch.float64))
    if compiled:
        print('Compiling image encoder; the first training/validation batch can take longer.', flush=True)
        compile_encoder(model)
    reg = GaussianBranch(args.mode, args.seed + 10000, depth=args.bt_depth,
                         hidden=args.bt_hidden, kappa=args.bt_kappa)
    module = TrainingModule(model, reg, recipe, batches)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'config.json').write_text(json.dumps(recipe, indent=2))
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
        'state': 'completed', 'recipe': 'pusht_spt_v1'}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=['raw', 'bt'], required=True)
    p.add_argument('--steps', type=int, default=100000)
    p.add_argument('--warmup-steps', type=int, default=500)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--pin-memory', action=argparse.BooleanOptionalAction, default=True,
                   help='Use pinned DataLoader memory for GPU transfer (default: enabled on GPU)')
    p.add_argument('--compile-encoder', action='store_true',
                   help='Opt in to torch.compile for the image encoder; requires a compatible Triton/CUDA compiler')
    p.add_argument('--save-every', type=int, default=5000,
                   help='Checkpoint interval in updates (default: 5000)')
    p.add_argument('--val-every', type=int,
                   help='Validation interval in updates (default: same as --save-every)')
    p.add_argument('--seed', type=int, default=3072)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--accelerator', choices=['cpu', 'gpu'], default='gpu')
    p.add_argument('--precision', choices=['32-true', 'bf16-mixed'], default='bf16-mixed')
    p.add_argument('--verify-data', action='store_true', help='Scan all dataset bytes and verify prepare SHA-256 when available')
    p.add_argument('--resume', type=Path, help='Trusted new-recipe Lightning checkpoint only')
    p.add_argument('--execute', action='store_true')
    add_bt_arguments(p)
    run(p.parse_args())


if __name__ == '__main__':
    main()
