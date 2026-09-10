"""Storage-independent, explicit observation/action semantics for model reuse."""
import copy
import math

import torch


PREPROCESSING = 'lewm_imagenet_resize224_v1'


def validate_contract(value):
    c = copy.deepcopy(value)
    required = {'domain', 'camera', 'action_names', 'action_units', 'action_convention',
                'fps', 'frameskip', 'history', 'preprocessing'}
    if set(c) != required:
        raise ValueError(f'Input contract requires exactly {sorted(required)}')
    for key in ('domain', 'camera', 'action_convention'):
        if not isinstance(c[key], str) or not c[key].strip():
            raise ValueError(f'Empty contract field: {key}')
    names, units = c['action_names'], c['action_units']
    if (not isinstance(names, list) or not names or len(set(names)) != len(names)
            or not isinstance(units, list) or len(names) != len(units)
            or any(not isinstance(s, str) or not s.strip() for s in names + units)):
        raise ValueError('Action names and units must be nonempty, ordered lists of equal length')
    if isinstance(c['fps'], bool) or not isinstance(c['fps'], (int, float)) or not math.isfinite(c['fps']) or c['fps'] <= 0:
        raise ValueError('fps must be finite and positive')
    if type(c['frameskip']) is not int or c['frameskip'] < 1 or type(c['history']) is not int or c['history'] != 3:
        raise ValueError('Requires positive integer frameskip and history=3')
    if c['preprocessing'] != PREPROCESSING:
        raise ValueError(f'Only {PREPROCESSING} is supported')
    return c


def model_contract(model, legacy_contract=None):
    saved = getattr(model, 'input_contract', None)
    if saved is None:
        if legacy_contract is None:
            raise ValueError('Legacy checkpoint has no input contract; supply its verified training contract')
        saved = legacy_contract
    elif legacy_contract is not None and validate_contract(legacy_contract) != saved:
        raise ValueError('Cannot override a checkpoint input contract')
    return validate_contract(saved)


def inference_manifest(model, source, legacy_contract=None):
    """Use the *training* statistics, never fit statistics on inference data."""
    expected = model_contract(model, legacy_contract)
    actual = validate_contract(source['input_contract'])
    if expected != actual:
        differences = [k for k in expected if expected[k] != actual[k]]
        raise ValueError(f'Incompatible input contract: {differences}')
    result = copy.deepcopy(source)
    for name in ('mean', 'std'):
        value = getattr(model, 'training_action_' + name, None)
        if value is None:
            raise ValueError('Checkpoint lacks frozen training action statistics')
        value = torch.as_tensor(value).detach().cpu().double()
        if value.shape != (len(expected['action_names']),) or not torch.isfinite(value).all():
            raise ValueError('Invalid checkpoint action statistics')
        if name == 'std' and (value <= 0).any():
            raise ValueError('Invalid checkpoint action standard deviations')
        result['action_' + name] = value.tolist()
    return result
