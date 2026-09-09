"""Launcher preflight tests: no CUDA, model loading, or environment evaluation."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import atexit

import pytest


@pytest.fixture
def launch_fixture(tmp_path):
    root=tmp_path/'repo'
    script=root/'mylewm/tools/evaluate_pusht.sh'
    script.parent.mkdir(parents=True)
    shutil.copyfile(Path(__file__).resolve().parents[1]/'tools/evaluate_pusht.sh',script)
    (root/'.venv/bin').mkdir(parents=True)
    (root/'.venv/bin/python').symlink_to(sys.executable)
    dataset=root/'.cache/stable-wm/datasets/pusht_expert_train.h5'
    dataset.parent.mkdir(parents=True)
    dataset.write_bytes(b'fixture, not real HDF5')
    manifest=root/'manifest.json'
    manifest.write_text(json.dumps({'dataset':str(dataset),'confirm':[{'episode':i,'start':0} for i in range(50)]}))
    checkpoint=root/'fixture_object.ckpt'
    checkpoint.write_bytes(b'not a pickle: dry-run must not load this')
    output=root/'output/test'
    command=['bash',str(script),'--checkpoint',str(checkpoint),'--manifest',str(manifest),'--output',str(output)]
    return root,manifest,output,command


def test_dry_run_validates_without_creating_output_or_loading_checkpoint(launch_fixture):
    root,manifest,output,command=launch_fixture
    result=subprocess.run(command+['--gb10-cache-workaround'],cwd=root,capture_output=True,text=True)
    assert result.returncode == 0, result.stderr
    assert 'Dry-run only' in result.stdout
    assert '"cases": 50' in result.stdout
    assert not output.exists()


@pytest.mark.parametrize('extra', [['--offset','1'],['--offset','-1'],['--num-eval','0']])
def test_rejects_invalid_cases(launch_fixture,extra):
    root,manifest,output,command=launch_fixture
    result=subprocess.run(command+extra,cwd=root,capture_output=True,text=True)
    assert result.returncode != 0
    assert not output.exists()


def test_refuses_existing_output(launch_fixture):
    root,manifest,output,command=launch_fixture
    output.mkdir(parents=True)
    sentinel=output/'keep'
    sentinel.write_text('preserve')
    result=subprocess.run(command,cwd=root,capture_output=True,text=True)
    assert result.returncode != 0
    assert sentinel.read_text() == 'preserve'


def test_rejects_other_dataset(launch_fixture):
    root,manifest,output,command=launch_fixture
    data=json.loads(manifest.read_text())
    data['dataset']=str(root/'different.h5')
    manifest.write_text(json.dumps(data))
    result=subprocess.run(command,cwd=root,capture_output=True,text=True)
    assert result.returncode != 0
    assert not output.exists()


@pytest.mark.parametrize('returncode',[0,1])
def test_execute_orchestration_with_mock_evaluator(launch_fixture,monkeypatch,returncode,capsys):
    # Exercise launch/log/result handling without starting a real GPU process.
    root,manifest,output,command=launch_fixture
    source=Path(command[1]).read_text().split("<<'PY'\n",1)[1].rsplit('\nPY',1)[0]
    monkeypatch.setattr(sys,'argv',['-',str(root),*command[2:],'--execute'])
    def fake_run(cmd,**kwargs):
        compile(cmd[2],'<bootstrap>','exec')
        assert cmd[1] == '-c'
        assert '+eval.audit_provenance=true' in cmd
        assert '+eval.shared_physical_search=true' in cmd
        assert kwargs['cwd'] == root
        assert kwargs['env']['STABLEWM_HOME'] == str(root/'.cache/stable-wm')
        assert (output/'checkpoint_object.ckpt').resolve() == root/'fixture_object.ckpt'
        kwargs['stdout'].write('mock evaluator only\n')
        if returncode == 0:
            plan=json.loads((output/'launch.json').read_text())
            cfg={'seed':42,'world':{},'plan_config':{},'dataset':{},'solver':{},
                 'eval':dict(goal_offset_steps=25,eval_budget=50,img_size=224,dataset_name='pusht',callables=[])}
            identity={k:plan[k] for k in ['checkpoint_sha256','manifest_sha256']}
            result={'provenance':identity,'config':cfg,'episodes':list(range(50)),
                    'starts':[0]*50,'successes':[True]*50,'success_rate':100.,
                    'checkpoint_sha256':identity['checkpoint_sha256'],
                    'physical_actions':[{'actions':[[0,0]],'mask':[True]}],
                    'initial_runtime_hashes':[{'state':str(i)} for i in range(50)]}
            (output/'results.txt.json').write_text(json.dumps(result))
        return subprocess.CompletedProcess(cmd,returncode)
    monkeypatch.setattr(subprocess,'run',fake_run)
    namespace={'__name__':'__main__'}
    if returncode:
        with pytest.raises(SystemExit) as error:
            exec(compile(source,'<launcher>','exec'),namespace)
        assert error.value.code == 1
        assert not (output/'results.txt.json').exists()
        assert 'Evaluation failed' in capsys.readouterr().err
    else:
        exec(compile(source,'<launcher>','exec'),namespace)
        assert 'Completed: 50/50 successes' in capsys.readouterr().out
    assert 'mock evaluator only' in (output/'console.log').read_text()
    namespace['unfinished_exit']()
    atexit.unregister(namespace['unfinished_exit'])
    assert json.loads((output/'status.json').read_text())['state'] == ('failed' if returncode else 'succeeded')
