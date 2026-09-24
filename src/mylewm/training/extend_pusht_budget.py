"""Explicit budget extension artifact; never modify a parent checkpoint in place."""
import copy
import math


def extend_completed_checkpoint(checkpoint, expected_recipe, target_steps):
    """Preserve weights/moments/RNG/cursor; replan future cosine LR only.

    This is a new continuation experiment, not an unchanged-condition resume.
    The frozen trainer still compares the entire resulting recipe on load.
    """
    if checkpoint['recipe'] != expected_recipe:
        raise ValueError('Parent recipe mismatch')
    step = checkpoint['global_step']
    if step != expected_recipe['steps'] or target_steps <= step:
        raise ValueError('Expected a completed parent and a larger target budget')
    if len(checkpoint['optimizer_states']) != 1 or len(checkpoint['lr_schedulers']) != 1:
        raise ValueError('Only the single-optimizer PushT recipe is supported')
    old = checkpoint['lr_schedulers'][0]
    if (old['last_epoch'] != step or old['max_steps'] != step
            or old['warmup_steps'] != expected_recipe['warmup_steps']
            or old['eta_min'] != 0 or old['warmup_start_lr'] != 0
            or any(x != expected_recipe['lr'] for x in old['base_lrs'])):
        raise ValueError('Parent scheduler mismatch')
    if any(g['lr'] != 0 for g in checkpoint['optimizer_states'][0]['param_groups']):
        raise ValueError('Expected the completed zero-LR parent')
    loop = checkpoint['loops']['fit_loop']
    progress = loop['epoch_loop.batch_progress']
    if any(progress[scope][k] != step for scope in ('total', 'current')
           for k in ('ready', 'started', 'processed', 'completed')):
        raise ValueError('Use final last.ckpt after batch completion')
    out = copy.deepcopy(checkpoint)
    recipe = copy.deepcopy(expected_recipe)
    recipe['steps'] = target_steps
    out['recipe'] = recipe
    out['hyper_parameters']['recipe'] = copy.deepcopy(recipe)
    scheduler = out['lr_schedulers'][0]
    scheduler['max_steps'] = target_steps
    warmup = scheduler['warmup_steps']
    lrs = [lr * (1 + math.cos(math.pi * (step - warmup) / (target_steps - warmup))) / 2
           for lr in scheduler['base_lrs']]
    scheduler['_last_lr'] = lrs
    for group, lr in zip(out['optimizer_states'][0]['param_groups'], lrs, strict=True):
        group['lr'] = lr
    # Reopen the original virtual epoch, keeping the consumed global batch count.
    loop = out['loops']['fit_loop']
    loop['epoch_loop.batch_progress']['is_last_batch'] = False
    for scope in ('total', 'current'):
        loop['epoch_progress'][scope]['processed'] = 0
        loop['epoch_progress'][scope]['completed'] = 0
    return out
