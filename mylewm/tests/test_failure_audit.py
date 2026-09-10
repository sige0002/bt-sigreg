from mylewm.evaluation.audit_pusht_failure_modes import failure_class, summarize


def test_failure_classes():
    assert failure_class(True, True) == 'both_success'
    assert failure_class(False, False) == 'both_failure'
    assert failure_class(True, False) == 'bt_only_success'
    assert failure_class(False, True) == 'official_only_success'


def test_summary_preserves_count_and_distribution():
    value = summarize([{'x': 1.0}, {'x': 3.0}], 'x')
    assert value == {'n': 2, 'mean': 2.0, 'median': 2.0, 'min': 1.0, 'max': 3.0}
