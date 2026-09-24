import copy
import torch
import pytest
from torch.utils.data import DataLoader
from mylewm.training import train_pusht_transport as trial
from mylewm.training import train as base
from mylewm.training.state import tensor_state_hash
from mylewm.paths import ROOT
from test_library_training import TinyModel,TinyData,fit,recipe
from test_training_state import assert_tree_equal


@pytest.mark.parametrize('family',['cayley','norm_preserving'])
def test_explicit_config_and_original_model_initialization(family):
 cfg=trial.load_model_config(ROOT/f'mylewm/configs/pusht_transport_{family}.yaml')
 torch.manual_seed(3072);expected=base.build_model()
 torch.manual_seed(3072);actual=trial.build_model(cfg)
 assert tensor_state_hash(actual.state_dict())==tensor_state_hash(expected.state_dict())
 assert any(isinstance(m,torch.nn.BatchNorm1d) for m in actual.projector.modules())
 bad=copy.deepcopy(cfg);bad['model']['action_encoder']['smoothed_dim']=25
 with pytest.raises(ValueError):trial.build_model(bad)


def make_module(family):
 torch.manual_seed(12)
 data=TinyData();batches=base.EpochBatches(data,4,6,23)
 reg=(trial.NormPreservingGaussianBranch(44,dim=6,projections=16) if family=='norm_preserving'
      else base.GaussianBranch('bt',44,dim=6,hidden=6,projections=16))
 return base.TrainingModule(TinyModel(),reg,dict(recipe(),family=family),batches)


def test_identity_norm_branch_preserves_official_loss_and_future_target_gradient():
 a=make_module('cayley');b=make_module('norm_preserving')
 assert_tree_equal(a.model.state_dict(),b.model.state_dict())
 a.log_dict=b.log_dict=lambda *args,**kwargs:None
 batch=next(iter(DataLoader(TinyData(),batch_size=4)))
 rng=torch.get_rng_state();left=a(copy.deepcopy(batch),stage='fit')
 torch.set_rng_state(rng);right=b(copy.deepcopy(batch),stage='fit')
 torch.testing.assert_close(left['loss'],right['loss'],rtol=0,atol=0)
 batch['pixels'].requires_grad_();b(batch,stage='fit')['pred_loss'].backward()
 assert batch['pixels'].grad[:,-1].norm()>0
 assert all(p.grad is None for p in b.sigreg.parameters())


def test_nonidentity_transport_norm_and_projection_rng():
 reg=trial.NormPreservingGaussianBranch(44,dim=6,projections=16)
 with torch.no_grad():
  for block in reg.transport.blocks:block.raw_vector.fill_(.15)
 z=torch.randn(4,8,6)
 before=torch.get_rng_state().clone();value=reg(z)
 assert torch.equal(before,torch.get_rng_state())
 u=reg.transport(z);torch.testing.assert_close(u.norm(dim=-1),z.norm(dim=-1),rtol=1e-6,atol=1e-6)
 assert not torch.equal(u,z)
 assert reg.draw_index==1 and torch.isfinite(value)
 value.backward();assert any(p.grad is not None and p.grad.norm()>0 for p in reg.parameters())
 reg.eval()(z);assert reg.draw_index==1


@pytest.mark.parametrize('family',['cayley','norm_preserving'])
def test_exact_resume_optimizer_projection_rng_and_transport_free_export(tmp_path,family):
 full=make_module(family);rows_full,state_full=fit(tmp_path/'full',full)
 part=make_module(family);rows_part,_=fit(tmp_path/'part',part,stop=True)
 resumed=make_module(family);rows_resume,state_resume=fit(tmp_path/'resume',resumed,resume=tmp_path/'part/final.ckpt')
 assert rows_full==rows_part+rows_resume
 for key in ('state_dict','optimizer_states','lr_schedulers'):
  assert_tree_equal(state_full[key],state_resume[key])
 assert full.sigreg.draw_index==resumed.sigreg.draw_index==6
 exported=torch.load(tmp_path/'full/step_6_object.ckpt',weights_only=False)
 assert_tree_equal(exported.state_dict(),full.model.state_dict())
 assert not any('transport' in k for k in exported.state_dict())
