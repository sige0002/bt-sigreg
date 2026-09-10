import json

import numpy as np
import pytest

from mylewm.tools.build_libero_ui import build


def test_builds_static_report_from_complete_evaluation(tmp_path):
    evaluation = tmp_path/'evaluation'
    evaluation.mkdir()
    (evaluation/'config.json').write_text(json.dumps({'checkpoint': 'model.ckpt', 'seed': 42, 'budget': 1}))
    (evaluation/'summary.json').write_text(json.dumps({'episodes': 1, 'macro_success': 0.}))
    row = {'task_id': 0, 'init_id': 0, 'task': 'task <one>', 'success': False, 'steps': 1, 'elapsed': .5}
    (evaluation/'episodes.jsonl').write_text(json.dumps(row)+'\n')
    images = np.zeros((2, 4, 5, 3), dtype=np.uint8)
    images[1, :, :, 1] = 255
    np.savez(evaluation/'task0_init0.npz', initial=images, goal=images, final=images)
    output = tmp_path/'ui'
    report = build(evaluation, output)
    assert report['episodes'] == 1
    page = (output/'index.html').read_text()
    assert 'task &lt;one&gt;' in page and 'failure' in page
    assert (output/'assets/task0_init0_initial.png').is_file()
    with pytest.raises(FileExistsError):
        build(evaluation, output)
