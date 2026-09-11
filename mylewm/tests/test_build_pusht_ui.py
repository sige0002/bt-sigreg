import json
from pathlib import Path

import av
import h5py
import numpy as np
import pytest
from PIL import Image

from mylewm.evaluation.build_pusht_ui import build
from mylewm.evaluation.contract import array_hash
from mylewm.evaluation.pusht_result_table import append_trial_table


@pytest.fixture
def evaluation(tmp_path):
    evaluation = tmp_path / 'evaluation'
    evaluation.mkdir()
    dataset = tmp_path / 'data.h5'
    pixels = np.zeros((6, 16, 16, 3), dtype=np.uint8)
    for i in range(6):
        pixels[i, :, :, i % 3] = 40 * i
    states = np.zeros((6, 7), dtype=np.float32)
    states[:, :4] = 100
    states[4, :4] = 200
    with h5py.File(dataset, 'w') as f:
        f['pixels'], f['state'] = pixels, states
        f['ep_offset'], f['ep_len'] = [0, 3], [3, 3]
    goals = states[[1, 4]].copy()
    records = []
    for step in range(3):
        current = goals.copy()
        current[:, 0] += 30 if step < 2 else 0
        if step == 1:
            current[0, 0] = goals[0, 0] + 5
        records.append({'state_after': current.tolist(), 'mask': [step < 2, step < 2]})
    hashes = []
    for index in (0, 3):
        hashes.append({key: array_hash(value[None]) for key, value in
                       [('pixels', pixels[index]), ('goal', pixels[index + 1]),
                        ('state', states[index]), ('goal_state', states[index + 1])]})
    result = {'episodes': [0, 1], 'starts': [0, 0], 'successes': [True, False],
              'initial_runtime_hashes': hashes, 'checkpoint_sha256': 'fixture',
              'physical_actions': records, 'config': {'world': {'env_name': 'swm/PushT-v1'},
                                                      'eval': {'goal_offset_steps': 1}},
              'provenance': {'dataset_metadata': {'path': str(dataset), 'size': dataset.stat().st_size,
                                                'mtime_ns': dataset.stat().st_mtime_ns}}}
    (evaluation / 'results.txt.json').write_text(json.dumps(result))
    (evaluation / 'results.txt').write_text('original matrix evidence\n')
    (evaluation / 'status.json').write_text(json.dumps({'state': 'succeeded', 'successes': 1, 'cases': 2}))
    for case in range(2):
        with av.open(str(evaluation / f'env_{case}.mp4'), 'w') as container:
            stream = container.add_stream('libx264', rate=15)
            stream.width = stream.height = 16
            stream.pix_fmt = 'yuv420p'
            for step in range(3):
                frame = av.VideoFrame.from_ndarray(pixels[step], format='rgb24')
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
    return evaluation


def test_goal_identity_masked_success_and_trial_numbers(evaluation, tmp_path):
    output = tmp_path / 'ui'
    report = build(evaluation, output)
    assert report['successes'] == 1
    rows = json.loads((output / 'report.json').read_text())['cases']
    assert rows[0]['first_success_step'] == 2
    assert rows[1]['first_success_step'] is None
    assert rows[1]['scores'][2]['position'] == 0
    assert rows[1]['scores'][2]['passed'] is False  # inactive frame cannot create success
    with h5py.File(tmp_path / 'data.h5') as f:
        np.testing.assert_array_equal(np.asarray(Image.open(output / 'assets/case_1_goal.png')), f['pixels'][4])
    assert '1回目\tenv_0.mp4\t成功' in (output / 'results.txt').read_text()
    append_trial_table(evaluation)
    text = (evaluation / 'results.txt').read_text()
    assert text.startswith('original matrix evidence\n')
    assert '2回目\tenv_1.mp4\t失敗' in text
    append_trial_table(evaluation)
    assert (evaluation / 'results.txt').read_text() == text
    with pytest.raises(FileExistsError):
        build(evaluation, output)


def test_rejects_different_goal_before_writing_ui(evaluation, tmp_path):
    path = evaluation / 'results.txt.json'
    result = json.loads(path.read_text())
    result['initial_runtime_hashes'][1]['goal'] = 'different goal'
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match='goal differs'):
        build(evaluation, tmp_path / 'invalid')
    assert not (tmp_path / 'invalid').exists()


def test_rejects_outcome_that_only_occurs_after_masking(evaluation, tmp_path):
    path = evaluation / 'results.txt.json'
    result = json.loads(path.read_text())
    result['successes'][1] = True
    path.write_text(json.dumps(result))
    (evaluation / 'status.json').write_text(json.dumps({'state': 'succeeded', 'successes': 2, 'cases': 2}))
    with pytest.raises(ValueError, match='reconstructed success'):
        build(evaluation, tmp_path / 'invalid')
    assert not (tmp_path / 'invalid').exists()
