"""Build a local, static visual report from one completed LIBERO evaluation."""
import argparse
import html
import json
import os
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
    config_path, rows_path, summary_path = (evaluation/'config.json', evaluation/'episodes.jsonl',
                                             evaluation/'summary.json')
    if not all(p.is_file() for p in (config_path, rows_path, summary_path)):
        raise FileNotFoundError('Evaluation requires config.json, episodes.jsonl, and summary.json')
    config, summary = json.loads(config_path.read_text()), json.loads(summary_path.read_text())
    bc = config.get('controller') == 'flow_matching_bc_v1'
    if output.exists() and not (bc and {p.name for p in output.iterdir()} == {'videos'}):
        raise FileExistsError(f'Use a fresh UI output directory: {output}')
    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line]
    if not rows:
        raise ValueError('Evaluation has no episode records')
    expected = {(row['task_id'], row['init_id']) for row in rows}
    if len(expected) != len(rows):
        raise ValueError('Duplicate task/init records')
    output.mkdir(parents=True, exist_ok=bc)
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
                if bc and label == 'goal' and label not in data:
                    continue
                image = pair_image(data[label])
                name = f'{stem}_{label}.png'
                image.save(assets/name)
                caption = 'Successful demo reference (not policy input)' if bc and label == 'goal' else label
                filenames.append((caption, name))
        status = 'success' if row['success'] else 'failure'
        images = ''.join(f'<figure><img src="assets/{name}" alt="{label} cameras"><figcaption>{label}: agent | wrist</figcaption></figure>'
                         for label, name in filenames)
        video = ''
        if bc and row.get('video'):
            video_name = row['video']
            source = (evaluation/video_name).resolve()
            if not source.is_relative_to(evaluation.resolve()) or not source.is_file():
                raise ValueError('Expected a video within the evaluation directory')
            relative = os.path.relpath(source, output)
            video = f'<video controls preload="metadata" style="max-width:100%" src="{html.escape(relative, quote=True)}"></video>'
        cards.append(
            f'<article class="{status}"><h2>Trial {row.get("trial", len(cards)+1)}: task {row["task_id"]}, init {row["init_id"]}: {status}</h2>'
            f'<p>{html.escape(row["task"])} · steps {row["steps"]} · {row["elapsed"]:.2f}s</p>{video}<div class="images">{images}</div></article>')
    title = f'LIBERO evaluation: {evaluation.name}'
    document = f'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>
body{{font-family:system-ui,sans-serif;max-width:1500px;margin:2rem auto;padding:0 1rem;background:#f7f7f7;color:#182028}}article{{background:white;margin:1rem 0;padding:1rem;border-left:8px solid #a33}}article.success{{border-color:#287a45}}h1,h2,p{{margin:.25rem 0 .75rem}}.images{{display:flex;gap:1rem;overflow-x:auto}}figure{{margin:0;min-width:384px}}img{{width:100%;display:block;border:1px solid #ccd}}figcaption{{font-size:.85rem;margin-top:.25rem}}code{{white-space:pre-wrap}}</style>
<h1>{html.escape(title)}</h1><p>Fixed native-environment rollout records. Each panel is <strong>agent view | wrist view</strong>; success uses the native task condition.</p>
<p>{'BC receives current images and task identity. Successful demo references are for display only; exact image matching is not the success criterion. Video shows simulation time, excluding policy computation waits.' if bc else 'Goal images are CEM inputs. This UI does not rerun CEM or alter results.'}</p>
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
