"""Measure trusted PushT world-model rollout latency without environment/CEM time."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[2]
# Serialized LeWM objects retain the upstream top-level ``jepa`` module name.
sys.path.insert(0, str(ROOT / 'lewm'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measure(model, *, predictions, candidates, repeats, warmup, seed, action_dim):
    """Time E+A+F for one CEM-sized candidate batch and ``predictions`` outputs.

    ``JEPA.rollout`` consumes three observed frames, then has one final predictor
    call even when no future action is appended.  Thus history + predictions - 1
    action entries produces exactly ``predictions`` predictor outputs.
    """
    history = 3
    generator = torch.Generator(device='cuda').manual_seed(seed)
    pixels = torch.randn((1, candidates, history, 3, 224, 224), device='cuda',
                         generator=generator)
    actions = torch.randn((1, candidates, history + predictions - 1, action_dim), device='cuda',
                          generator=generator)

    def run():
        # rollout mutates its info dictionary, so each timing receives fresh keys.
        return model.rollout({'pixels': pixels}, actions, history_size=history)['predicted_emb']

    with torch.inference_mode():
        for _ in range(warmup):
            run()
        torch.cuda.synchronize()
        samples = []
        for _ in range(repeats):
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            output = run()
            end.record()
            end.synchronize()
            if output.shape[-2] != history + predictions:
                raise RuntimeError('Unexpected rollout length')
            samples.append(begin.elapsed_time(end))
    ordered = sorted(samples)
    return {'prediction_calls': predictions, 'samples_ms': samples,
            'mean_ms': sum(samples) / len(samples), 'median_ms': ordered[len(ordered) // 2],
            'min_ms': ordered[0], 'max_ms': ordered[-1]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, action='append', required=True,
                        help='Trusted PushT *_object.ckpt; specify twice for BT and official.')
    parser.add_argument('--output', type=Path, required=True, help='New JSON result under output/.')
    parser.add_argument('--candidates', type=int, default=300, help='CEM candidate batch size.')
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=5)
    args = parser.parse_args()
    if args.candidates < 1 or args.repeats < 1 or args.warmup < 0:
        parser.error('candidates/repeats must be positive and warmup non-negative')
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to((ROOT / 'output').resolve()):
        parser.error('--output must be a new file under repository output/')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this benchmark')

    results = {'schema': 'pusht_rollout_latency_v1', 'device': torch.cuda.get_device_name(),
               'candidates': args.candidates, 'repeats': args.repeats, 'warmup': args.warmup,
               'definition': 'E+A+F rollout only; excludes environment, CEM iterations, goal encoding, and I/O.',
               'models': []}
    for index, checkpoint in enumerate(args.checkpoint):
        checkpoint = checkpoint.resolve()
        if not checkpoint.is_file() or not checkpoint.name.endswith('_object.ckpt'):
            parser.error(f'--checkpoint must name an existing *_object.ckpt: {checkpoint}')
        model = torch.load(checkpoint, map_location='cuda', weights_only=False).eval()
        model.requires_grad_(False)
        action_dim = model.action_encoder.patch_embed.in_channels
        torch.cuda.reset_peak_memory_stats()
        record = {'checkpoint': str(checkpoint), 'checkpoint_sha256': digest(checkpoint),
                  'one_step': measure(model, predictions=1, candidates=args.candidates,
                                      repeats=args.repeats, warmup=args.warmup, seed=3000 + index,
                                      action_dim=action_dim),
                  'twenty_step': measure(model, predictions=20, candidates=args.candidates,
                                         repeats=args.repeats, warmup=args.warmup, seed=4000 + index,
                                         action_dim=action_dim),
                  'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                  'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
        results['models'].append(record)
        del model
        torch.cuda.empty_cache()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
