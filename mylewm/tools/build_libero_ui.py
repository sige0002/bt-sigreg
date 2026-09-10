"""Build a local, static visual report from one completed LIBERO evaluation."""
import argparse
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image


def pair_image(images):
    """Put agent and wrist RGB images side-by-side without changing their pixels."""
    if images.ndim != 4 or images.shape[0] != 2 or images.shape[-1] != 3:
        raise ValueError('Expected two RGB images shaped (2, height, width, 3)')
    if images.dtype != np.uint8:
        raise ValueError('Expected uint8 rendered images')
    height, width = images.shape[1:3]
    canvas = Image.new('RGB', (width * 2, height))
    canvas.paste(Image.fromarray(images[0]), (0, 0))
    canvas.paste(Image.fromarray(images[1]), (width, 0))
    return canvas


def build(evaluation, output):
    evaluation, output = Path(evaluation), Path(output)
    if output.exists():
        raise FileExistsError(f'Use a fresh UI output directory: {output}')
    config_path, rows_path, summary_path = (evaluation/'config.json', evaluation/'episodes.jsonl',
                                             evaluation/'summary.json')
    if not all(p.is_file() for p in (config_path, rows_path, summary_path)):
        raise FileNotFoundError('Evaluation requires config.json, episodes.jsonl, and summary.json')
    config, summary = json.loads(config_path.read_text()), json.loads(summary_path.read_text())
    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line]
    if not rows:
        raise ValueError('Evaluation has no episode records')
    expected = {(row['task_id'], row['init_id']) for row in rows}
    if len(expected) != len(rows):
        raise ValueError('Duplicate task/init records')
    output.mkdir(parents=True)
    assets = output/'assets'
    assets.mkdir()
    cards = []
    for row in rows:
        stem = f"task{row['task_id']}_init{row['init_id']}"
        archive = evaluation/f'{stem}.npz'
        if not archive.is_file():
            raise FileNotFoundError(f'Missing visual archive: {archive}')
        with np.load(archive) as data:
            filenames = []
            for label in ('initial', 'goal', 'final'):
                image = pair_image(data[label])
                name = f'{stem}_{label}.png'
                image.save(assets/name)
                filenames.append((label, name))
        status = 'success' if row['success'] else 'failure'
        images = ''.join(f'<figure><img src="assets/{name}" alt="{label} cameras"><figcaption>{label}: agent | wrist</figcaption></figure>'
                         for label, name in filenames)
        cards.append(
            f'<article class="{status}"><h2>task {row["task_id"]}, init {row["init_id"]}: {status}</h2>'
            f'<p>{html.escape(row["task"])} · steps {row["steps"]} · {row["elapsed"]:.2f}s</p><div class="images">{images}</div></article>')
    title = f'LIBERO evaluation: {evaluation.name}'
    document = f'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>
body{{font-family:system-ui,sans-serif;max-width:1500px;margin:2rem auto;padding:0 1rem;background:#f7f7f7;color:#182028}}article{{background:white;margin:1rem 0;padding:1rem;border-left:8px solid #a33}}article.success{{border-color:#287a45}}h1,h2,p{{margin:.25rem 0 .75rem}}.images{{display:flex;gap:1rem;overflow-x:auto}}figure{{margin:0;min-width:384px}}img{{width:100%;display:block;border:1px solid #ccd}}figcaption{{font-size:.85rem;margin-top:.25rem}}code{{white-space:pre-wrap}}</style>
<h1>{html.escape(title)}</h1><p>Fixed native-environment rollout records. Each panel is <strong>agent view | wrist view</strong>; this UI does not rerun CEM or alter results.</p>
<p>Episodes: {summary["episodes"]}; macro success: {summary["macro_success"]:.1%}</p><code>{html.escape(json.dumps({k: config[k] for k in ("checkpoint", "checkpoint_sha256", "manifest", "manifest_sha256", "seed", "budget", "horizon", "samples", "iterations") if k in config}, indent=2))}</code>
{''.join(cards)}</html>'''
    (output/'index.html').write_text(document)
    return {'output': str(output), 'episodes': len(rows), 'index': str(output/'index.html')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Fresh directory under output/')
    args = parser.parse_args()
    print(json.dumps(build(args.evaluation, args.output), indent=2))


if __name__ == '__main__':
    main()
