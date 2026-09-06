import json
import sys
from pathlib import Path

import pytest
from mylewm import run_rbg_replicates as queue


def fake_result(benchmark,seed):
    path=queue.result_path(benchmark,seed)
    path.parent.mkdir(parents=True,exist_ok=True)
    names=['rbg','raw','tc','sub','rbg_bn','raw_bn','tc_bn','sub_bn']
    if benchmark=='pusht':names.append('official')
    path.write_text(json.dumps({'results':dict.fromkeys(names,[]),'comparisons':{'checked':True}}))


def test_replicates_run_sequentially_after_prerequisites(tmp_path,monkeypatch):
    monkeypatch.setattr(queue,'ROOT',tmp_path)
    for benchmark in ('pusht','libero10'):fake_result(benchmark,3072)
    def exited(pid,signal):raise ProcessLookupError
    monkeypatch.setattr(queue.os,'kill',exited)
    calls=[]
    def execute(command,**kwargs):
        benchmark='libero10' if 'libero' in Path(command[1]).name else 'pusht'
        seed=int(command[-1]);calls.append((benchmark,seed))
        fake_result(benchmark,seed)
    monkeypatch.setattr(queue.subprocess,'run',execute)
    monkeypatch.setattr(sys,'argv',['queue','--wait-pid','123'])
    queue.main()
    assert calls==[('pusht',3073),('libero10',3073),('pusht',3074),('libero10',3074)]
    report=json.loads((tmp_path/'.cache/stable-wm/rbg_v0_replication_index.json').read_text())
    assert report['seeds']==[3072,3073,3074]


def test_missing_first_seed_does_not_start_replicates(tmp_path,monkeypatch):
    monkeypatch.setattr(queue,'ROOT',tmp_path)
    def exited(pid,signal):raise ProcessLookupError
    monkeypatch.setattr(queue.os,'kill',exited)
    monkeypatch.setattr(queue.subprocess,'run',lambda *a,**k:pytest.fail('Unexpected training'))
    monkeypatch.setattr(sys,'argv',['queue','--wait-pid','123'])
    with pytest.raises(FileNotFoundError):queue.main()
