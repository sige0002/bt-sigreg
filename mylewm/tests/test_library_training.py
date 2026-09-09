"""New-recipe contract tests: no downloads or real environment evaluation."""
import copy
import json
from pathlib import Path

import h5py
import lightning as pl
import numpy as np
import pytest
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from mylewm import train as new
from mylewm.objectives import one_step_objective, GaussianSIGReg
from mylewm.tests.test_training_state import assert_tree_equal


class TinyModel(torch.nn.Module):
    def __init__(self, dim=6):
        super().__init__()
        self.encoder = torch.nn.Sequential(torch.nn.Linear(3, dim), torch.nn.BatchNorm1d(dim),
                                           torch.nn.Dropout(.2))
        self.action_encoder = torch.nn.Linear(10, dim)
        self.predictor = torch.nn.Linear(dim, dim)

    def encode(self, batch):
        x = batch['pixels'].mean((-1, -2))
        batch['emb'] = self.encoder(x.flatten(0, 1)).reshape(*x.shape[:2], -1)
        if 'action' in batch:
            batch['act_emb'] = self.action_encoder(batch['action'])
        return batch

    def predict(self, z, a):
        return self.predictor(z + a)


class TinyData:
    def __len__(self):
        return 19

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(index)
        return {'pixels': torch.randn(4, 3, 2, 2, generator=generator),
                'action': torch.randn(4, 10, generator=generator), 'index': index}


def recipe():
    return {'steps': 6, 'warmup_steps': 2, 'lr': .003, 'mode': 'bt', 'schema': 'test'}


def make_module(mode='bt', depth=2):
    torch.manual_seed(12)
    data = TinyData()
    batches = new.EpochBatches(data, 4, 6, 23)
    reg = new.GaussianBranch(mode, 44, dim=6, hidden=6, depth=depth, projections=16)
    return new.TrainingModule(TinyModel(), reg, dict(recipe(), mode=mode), batches)


def test_raw_and_identity_bt_match_official_loss_gradients_and_legacy_raw():
    raw, bt = make_module('raw'), make_module('bt')
    raw.log_dict = bt.log_dict = lambda *a, **k: None
    batch = next(iter(DataLoader(TinyData(), batch_size=4)))
    rng = torch.get_rng_state()
    out_raw = raw(copy.deepcopy(batch), stage='fit')
    out_raw['loss'].backward()
    torch.set_rng_state(rng)
    out_bt = bt(copy.deepcopy(batch), stage='fit')
    out_bt['loss'].backward()
    assert_tree_equal(out_raw['loss'], out_bt['loss'])
    assert_tree_equal([p.grad for p in raw.model.parameters()], [p.grad for p in bt.model.parameters()])
    # The official forward still supervises the final frame (no stop-gradient).
    bt.zero_grad()
    batch['pixels'].requires_grad_()
    bt(batch, stage='fit')['pred_loss'].backward()
    assert batch['pixels'].grad[:, -1].abs().sum() > 0
    assert all(p.grad is None for p in bt.sigreg.parameters())
    # Compare old Raw formula on exactly the same tensors/projection seed.
    old_model = copy.deepcopy(raw.model).eval()
    raw.eval()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(44 + int(raw.sigreg.draw_index))
        old_loss, _ = one_step_objective(old_model, batch['pixels'], batch['action'][:, :3],
                                          GaussianSIGReg(dim=6, projections=16), .09, 'raw')
    clean_batch = {k: batch[k].detach().clone() for k in ('pixels', 'action')}
    torch.testing.assert_close(raw(clean_batch, stage='validate')['loss'], old_loss)


def test_epoch_shuffle_and_resume_cursor():
    data = TinyData()
    batches = new.EpochBatches(data, 4, 12, 23)
    all_batches = list(batches)
    assert len(set(sum(all_batches[:4], []))) == 16
    assert all_batches[:4] != all_batches[4:8]
    batches.start = 5
    assert list(batches) == all_batches[5:]


@pytest.fixture
def native_manifest(tmp_path):
    path = tmp_path/'tiny.h5'
    with h5py.File(path, 'w') as f:
        f['ep_len'] = [41, 41, 41]
        f['ep_offset'] = [0, 41, 82]
        f['pixels'] = np.zeros((123, 4, 4, 3), dtype=np.uint8)
        f['action'] = np.arange(246, dtype=np.float32).reshape(123, 2)
    return dict(dataset=str(path), history=3, frameskip=5,
                train_episodes=[0], validation_episodes=[1], test_episodes=[2],
                action_mean=[2., 3.], action_std=[2., 4.])


def test_native_loader_episode_isolation_and_preprocessing(native_manifest):
    train, val = new.datasets(native_manifest)
    assert len(train) == len(val) == 22  # official 20-step span, not old 16-step span
    assert all(train.dataset.clip_indices[i][0] == 0 for i in train.indices)
    assert all(val.dataset.clip_indices[i][0] == 1 for i in val.indices)
    item = train[0]
    assert item['pixels'].shape == (4, 3, 224, 224)
    assert item['action'].shape == (4, 10)
    torch.testing.assert_close(item['action'][0, :2], torch.tensor([-1., -.5]))
    bad = dict(native_manifest, test_episodes=[0])
    with pytest.raises(ValueError, match='overlap'):
        new.datasets(bad)
    selected = dict(native_manifest, validation=[{'episode': 1, 'start': 3}])
    _, val = new.datasets(selected)
    assert len(val) == 1
    assert val.dataset.clip_indices[val.indices[0]] == (1, 3)


class StopAt(pl.Callback):
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        if trainer.global_step == 3:
            trainer.should_stop = True


class Trace(pl.Callback):
    def __init__(self):
        self.rows = []

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        self.rows.append((trainer.global_step, batch['index'].tolist(), float(outputs['loss'])))


def fit(tmp_path, module, stop=False, resume=None, workers=0):
    trace = Trace()
    checkpoint = ModelCheckpoint(dirpath=tmp_path, every_n_train_steps=1,
                                  save_top_k=-1, save_last=True)
    callbacks = [trace, new.Export(tmp_path, 3), checkpoint]
    if stop:
        callbacks.insert(0, StopAt())
    trainer = new.make_trainer(default_root_dir=tmp_path, accelerator='cpu', devices=1,
        max_steps=6, max_epochs=-1, logger=False, enable_progress_bar=False,
        enable_model_summary=False, num_sanity_val_steps=0, callbacks=callbacks,
        check_val_every_n_epoch=None, val_check_interval=2, deterministic=True)
    loader = DataLoader(TinyData(), batch_sampler=module.batches, num_workers=workers,
                        generator=torch.Generator().manual_seed(100))
    val = DataLoader(TinyData(), batch_size=4, generator=torch.Generator().manual_seed(101))
    trainer.fit(module, train_dataloaders=loader, val_dataloaders=val,
                ckpt_path=resume, weights_only=False)
    trainer.save_checkpoint(tmp_path/'final.ckpt')
    return trace.rows, torch.load(tmp_path/'final.ckpt', weights_only=False)


@pytest.mark.parametrize('workers', [0, 2])
def test_library_updates_resume_exactly_and_export_excludes_transport(tmp_path, workers):
    full = make_module()
    initial = copy.deepcopy(full.state_dict())
    rows_full, state_full = fit(tmp_path/'full', full, workers=workers)
    part = make_module()
    rows_part, _ = fit(tmp_path/'part', part, stop=True, workers=workers)
    resumed = make_module()
    rows_resume, state_resume = fit(tmp_path/'resume', resumed,
                                     resume=tmp_path/'part/final.ckpt', workers=workers)
    assert rows_full == rows_part + rows_resume
    for key in ('state_dict', 'optimizer_states', 'lr_schedulers'):
        assert_tree_equal(state_full[key], state_resume[key])
    for prefix in ('model.', 'sigreg.transport.'):
        assert any(not torch.equal(v, initial[k]) for k, v in full.state_dict().items()
                   if k.startswith(prefix))
    exported = torch.load(tmp_path/'full/step_6_object.ckpt', weights_only=False)
    assert not any('transport' in k for k in exported.state_dict())
    assert_tree_equal(exported.state_dict(), full.model.state_dict())
    assert state_full['global_step'] == 6
    assert len(state_full['optimizer_states'][0]['state']) == len(list(full.parameters()))
    bad = make_module()
    bad.recipe = dict(bad.recipe, lr=1.)
    with pytest.raises(ValueError, match='Recipe mismatch'):
        bad.on_load_checkpoint(state_full)


def test_projection_rng_does_not_change_model_rng():
    reg = new.GaussianBranch('bt', 44, dim=6, hidden=6, projections=16)
    z = torch.randn(4, 5, 6)
    state = torch.get_rng_state().clone()
    reg(z)
    assert torch.equal(state, torch.get_rng_state())
    assert reg.draw_index == 1
    reg.eval()(z)
    assert reg.draw_index == 1


@pytest.mark.parametrize('mode', ['raw', 'bt'])
def test_execute_entrypoint_native_data_and_fresh_output(native_manifest, tmp_path, monkeypatch, mode):
    import argparse
    path = tmp_path/'manifest.json'
    path.write_text(json.dumps(native_manifest))
    args = argparse.Namespace(manifest=path, output=tmp_path/'run', mode=mode,
        steps=3, warmup_steps=1, batch_size=2, workers=0, save_every=2,
        seed=3072, lr=5e-5, accelerator='cpu', precision='32-true',
        resume=None, execute=False, bt_depth=2, bt_hidden=192, bt_kappa=.2)
    monkeypatch.setattr(new, 'build_model', lambda: TinyModel(dim=192))
    new.run(args)
    assert not args.output.exists()
    args.execute = True
    new.run(args)
    assert json.loads((args.output/'completed.json').read_text())['step'] == 3
    model = torch.load(args.output/'step_3_object.ckpt', weights_only=False)
    assert hasattr(model, 'training_action_mean')
    assert not hasattr(model, 'sigreg')
    state = torch.load(args.output/'last.ckpt', weights_only=False)
    assert state['recipe']['mode'] == mode
    assert len(state['optimizer_states']) == len(state['lr_schedulers']) == 1
    assert list((args.output/'metrics').glob('version_*/metrics.csv'))
    with pytest.raises(FileExistsError):
        new.run(args)


def test_real_pusht_architecture_identity_bt_matches_raw():
    torch.set_num_threads(4)
    torch.manual_seed(31)
    model = new.build_model().eval()
    data = TinyData()
    batches = new.EpochBatches(data, 2, 6, 23)
    raw = new.TrainingModule(model, new.GaussianBranch('raw', 44), recipe(), batches).eval()
    bt = new.TrainingModule(copy.deepcopy(model), new.GaussianBranch('bt', 44), recipe(), batches).eval()
    raw.log_dict = bt.log_dict = lambda *a, **k: None
    batch = {'pixels': torch.randn(2, 4, 3, 28, 28), 'action': torch.randn(2, 4, 10)}
    a = raw(copy.deepcopy(batch), stage='validate')['loss']
    b = bt(copy.deepcopy(batch), stage='validate')['loss']
    a.backward(); b.backward()
    assert_tree_equal(a, b)
    assert_tree_equal([p.grad for p in raw.model.parameters()], [p.grad for p in bt.model.parameters()])
