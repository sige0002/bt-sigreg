import copy
from pathlib import Path
import torch
import pytest
import os
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'lewm'))
from mylewm.planning_action_adapter import PlanningActionAdapter
from mylewm.evaluation_contract import protocol,validate_result


def test_comparison_cli_works_without_pythonpath():
    script = 'compare_paired.py'
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    result = subprocess.run([sys.executable, str(root/'mylewm/tools'/script), '--help'],
                            cwd=root, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert 'usage:' in result.stdout


class Recorder(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.parameter=torch.nn.Parameter(torch.zeros(1))
    def get_cost(self,info,actions):
        self.seen=(info,actions)
        return actions.square().sum((-1,-2))


def test_shared_physical_actions_and_history_are_normalized_for_checkpoint():
    model=Recorder()
    adapter=PlanningActionAdapter(model,[1.,2.],[2.,3.],[-1.,4.],[4.,2.])
    candidates=torch.arange(24).reshape(1,2,3,4).float()/10
    history=candidates[:,:,:1].clone()
    info={'action':history,'pixels':torch.zeros(1)}
    adapter.get_cost(info,candidates)
    received=model.seen[1].reshape(1,2,3,2,2)
    physical=received*torch.tensor([4.,2.])+torch.tensor([-1.,4.])
    reference=candidates.reshape_as(received)*torch.tensor([2.,3.])+torch.tensor([1.,2.])
    torch.testing.assert_close(physical,reference)
    torch.testing.assert_close(model.seen[0]['action'],model.seen[1][:,:,:1])
    assert info['action'] is history
    torch.testing.assert_close(history,candidates[:,:,:1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA mixed-device regression')
def test_adapter_gpu_model_accepts_cpu_history_and_gpu_candidates():
    model=Recorder().cuda()
    adapter=PlanningActionAdapter(model,[1.,2.],[2.,3.],[-1.,4.],[4.,2.])
    candidates=torch.arange(24,device='cuda').reshape(1,2,3,4).float()/10
    history=candidates[:,:,:1].cpu().clone()
    original=history.clone()
    adapter.get_cost({'action':history},candidates)
    assert model.seen[0]['action'].device == candidates.device
    assert model.seen[1].device == candidates.device
    torch.testing.assert_close(model.seen[0]['action'],model.seen[1][:,:,:1])
    physical=model.seen[1].reshape(1,2,3,2,2)*torch.tensor([4.,2.],device='cuda')+torch.tensor([-1.,4.],device='cuda')
    reference=candidates.reshape_as(physical)*torch.tensor([2.,3.],device='cuda')+torch.tensor([1.,2.],device='cuda')
    torch.testing.assert_close(physical,reference)
    torch.testing.assert_close(history,original)
    assert history.device.type == 'cpu'


def test_adapter_matches_direct_actual_jepa_rollout():
    path=Path('.cache/stable-wm/pusht/lewm_object.ckpt')
    if not path.exists():pytest.skip('Official checkpoint is a local integration fixture')
    torch.set_num_threads(4)
    model=torch.load(path,map_location='cpu',weights_only=False).eval()
    adapter=PlanningActionAdapter(model,[2.,2.],[2.,2.],[0.,0.],[1.,1.]).eval()
    pixels=torch.zeros(1,3,1,3,224,224)
    candidate=torch.tensor([-1.,0.,1.]).reshape(1,3,1,1).expand(1,3,4,10).clone()
    info={'pixels':pixels,'goal':pixels.clone(),'action':torch.zeros(1,3,1,10)}
    direct={'pixels':pixels.clone(),'goal':pixels.clone(),'action':torch.full((1,3,1,10),2.)}
    # The three physical candidate values are exactly 0, 2 and 4.
    expected=torch.tensor([0.,2.,4.]).reshape(1,3,1,1).expand_as(candidate).clone()
    with torch.no_grad():
        actual_cost=adapter.get_cost(info,candidate)
        expected_cost=model.get_cost(direct,expected)
    torch.testing.assert_close(actual_cost,expected_cost,atol=1e-5,rtol=1e-5)



def fixture_result():
    cfg={'seed':42,'world':{},'plan_config':{},'dataset':{},'solver':{},
         'eval':dict(goal_offset_steps=25,eval_budget=50,img_size=224,dataset_name='pusht',callables=[])}
    identity={'checkpoint_sha256':'weights','dataset_sha256':'data'}
    result={'provenance':identity,'config':cfg,'episodes':[1,2],'starts':[0,0],
            'successes':[True,False],'checkpoint_sha256':'weights',
            'physical_actions':[{'actions':[[0,0],[0,0]],'mask':[True,True]}],
            'initial_runtime_hashes':[{'state':'s1'},{'state':'s2'}]}
    return result,[{'episode':1,'start':0},{'episode':2,'start':0}],identity,protocol(cfg)


@pytest.mark.parametrize('change',['weight','source','cases','incomplete','actions','protocol'])
def test_cached_evaluation_rejects_mismatch(change):
    result,cases,identity,expected=fixture_result()
    validate_result(result,cases,identity,expected)
    result=copy.deepcopy(result)
    if change=='weight':result['checkpoint_sha256']='wrong'
    if change=='source':result['provenance']['dataset_sha256']='changed'
    if change=='cases':result['episodes']=[1,1]
    if change=='incomplete':result['successes']=[True]
    if change=='actions':result['physical_actions']=[]
    if change=='protocol':result['config']['seed']=43
    with pytest.raises(ValueError):validate_result(result,cases,identity,expected)
