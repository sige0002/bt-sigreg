import copy
import math
import pytest
import torch
import lightning as pl
from torch.utils.data import DataLoader
from mylewm.training import train as base
from mylewm.training.extend_pusht_budget import extend_completed_checkpoint
from test_pusht_transport import make_module
from test_library_training import TinyData, Trace, fit
from test_training_state import assert_tree_equal


def continue_fit(path, family, state, stop=None):
    path.mkdir()
    resume = path/'input.ckpt'
    torch.save(state, resume)
    m = make_module(family)
    m.recipe = copy.deepcopy(state['recipe'])
    m.batches.steps = m.recipe['steps']
    trace = Trace()
    class Stop(pl.Callback):
        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            if trainer.global_step == stop:
                trainer.should_stop = True
    trainer = base.make_trainer(default_root_dir=path, accelerator='cpu', devices=1,
        max_steps=m.recipe['steps'], max_epochs=-1, logger=False,
        enable_progress_bar=False, enable_model_summary=False,
        enable_checkpointing=False, callbacks=[trace, Stop()], deterministic=True)
    loader = DataLoader(TinyData(), batch_sampler=m.batches, num_workers=0,
                        generator=torch.Generator().manual_seed(100))
    trainer.fit(m, train_dataloaders=loader, ckpt_path=resume, weights_only=False)
    trainer.save_checkpoint(path/'final.ckpt')
    return trace.rows, torch.load(path/'final.ckpt', weights_only=False)


@pytest.mark.parametrize('family', ['cayley', 'norm_preserving'])
def test_budget_extension_preserves_state_cursor_and_exact_second_resume(tmp_path, family):
    parent = make_module(family)
    _, original = fit(tmp_path/'parent', parent)
    extended = extend_completed_checkpoint(original, parent.recipe, 12)
    for key in ('state_dict', 'random_state'):
        assert_tree_equal(original[key], extended[key])
    assert_tree_equal(original['optimizer_states'][0]['state'], extended['optimizer_states'][0]['state'])
    assert original['recipe']['steps'] == 6
    lr = parent.recipe['lr'] * (1 + math.cos(math.pi * 4 / 10)) / 2
    assert extended['optimizer_states'][0]['param_groups'][0]['lr'] == lr
    full_rows, full = continue_fit(tmp_path/'full', family, extended)
    part_rows, part = continue_fit(tmp_path/'part', family, extended, stop=9)
    tail_rows, tail = continue_fit(tmp_path/'tail', family, part)
    assert full_rows == part_rows + tail_rows
    expected_batches = list(base.EpochBatches(TinyData(), 4, 12, 23))[6:]
    assert [r[0] for r in full_rows] == list(range(7, 13))
    assert [r[1] for r in full_rows] == expected_batches
    for key in ('state_dict', 'optimizer_states', 'lr_schedulers'):
        assert_tree_equal(full[key], tail[key])
    assert full['lr_schedulers'][0]['max_steps'] == 12
    assert full['optimizer_states'][0]['param_groups'][0]['lr'] == 0
    assert any(not torch.equal(v, full['state_dict'][k]) for k, v in original['state_dict'].items()
               if k.startswith('model.'))
    with pytest.raises(ValueError, match='recipe'):
        extend_completed_checkpoint(original, dict(parent.recipe, lr=.1), 12)
    with pytest.raises(ValueError, match='larger'):
        extend_completed_checkpoint(original, parent.recipe, 6)
