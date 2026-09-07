import torch
from mylewm.tools.calibrate_rbg_bn import calibrate
from mylewm.training_state import capture_rng
from mylewm.tests.test_training_state import assert_tree_equal


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projector=torch.nn.Sequential(torch.nn.Linear(3,5),torch.nn.BatchNorm1d(5),
                                          torch.nn.Dropout(.4),torch.nn.Linear(5,3))
        self.pred_proj=torch.nn.Sequential(torch.nn.Linear(3,5),torch.nn.BatchNorm1d(5),
                                          torch.nn.Dropout(.4),torch.nn.Linear(5,3))
        self.register_buffer('untouched',torch.tensor([123.]))
    def encode(self,batch):
        x=batch['pixels']
        return {'emb':self.projector(x.flatten(0,1)).reshape_as(x)}
    def action_encoder(self,a):return a
    def predict(self,z,a):return self.pred_proj((z+a).flatten(0,1)).reshape_as(z)


class TrainingOnly(torch.utils.data.Dataset):
    def __init__(self):self.seen=[]
    def __len__(self):return 17
    def __getitem__(self,i):
        assert 0<=i<17
        self.seen.append(i)
        return torch.arange(12).reshape(4,3).float()+i,torch.ones(3,3)*i


def test_only_bn_buffers_change_same_training_sequence_and_rng_preserved():
    torch.manual_seed(123)
    model=Model()
    ds=TrainingOnly()
    rng=capture_rng()
    report=calibrate(model,ds,{},lambda batch,m,d:batch,2,4,'cpu')
    assert_tree_equal(rng,capture_rng())
    assert report['parameter_sha256_before']==report['parameter_sha256_after']
    assert set(report['changed_buffers'])<=set(report['allowed_buffers'])
    assert report['changed_buffers']
    assert report['buffer_sha256_before']['untouched']==report['buffer_sha256_after']['untouched']
    assert ds.seen[:8]==ds.seen[8:]
    assert model.projector[1].momentum==.1
    assert model.pred_proj[1].momentum==.1
    assert all(not layer.training for layer in model.modules())
