"""Show recorded PushT videos beside the exact dataset goals used for evaluation."""
import argparse
import html
import json
import shutil
from pathlib import Path

import av
import hdf5plugin  # noqa: F401 — register the dataset's HDF5 compression filters
import h5py
import numpy as np
from PIL import Image

from mylewm.evaluation.contract import array_hash
from mylewm.evaluation.pusht_result_table import trial_table


def case_scores(records, case, goal):
    rows = []
    for step, record in enumerate(records, 1):
        state = np.asarray(record['state_after'][case])
        active = bool(record['mask'][case])
        position = float(np.linalg.norm(goal[:4] - state[:4]))
        angle = float(abs(goal[4] - state[4]))
        if not np.isfinite(state).all() or angle > 2 * np.pi:
            raise ValueError('Unsupported/nonfinite recorded state or angle')
        angle = min(angle, 2 * np.pi - angle)
        rows.append({'step': step, 'active': active, 'position': position,
                     'angle_deg': float(np.degrees(angle)),
                     'passed': bool(active and position < 20 and angle < np.pi / 9)})
    if not rows or not any(row['active'] for row in rows):
        raise ValueError('Case has no active physical state records')
    return rows


def build(evaluation, output):
    evaluation, output = Path(evaluation).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Use a fresh output directory: {output}')
    result_path = evaluation / 'results.txt.json'
    result = json.loads(result_path.read_text())
    status = json.loads((evaluation / 'status.json').read_text())
    if status['state'] != 'succeeded':
        raise ValueError('Evaluation must have succeeded')
    if result['config']['world']['env_name'] != 'swm/PushT-v1':
        raise ValueError('This viewer only supports the audited SWM PushT protocol')
    n = len(result['episodes'])
    if not n or any(len(result[key]) != n for key in ('starts', 'successes', 'initial_runtime_hashes')):
        raise ValueError('Incomplete case records')
    if status['successes'] != sum(result['successes']) or status['cases'] != n:
        raise ValueError('Status and result counts differ')
    provenance = result['provenance']
    dataset = Path(provenance['dataset_metadata']['path'])
    stat = dataset.stat()
    if (stat.st_size, stat.st_mtime_ns) != (provenance['dataset_metadata']['size'],
                                         provenance['dataset_metadata']['mtime_ns']):
        raise ValueError('Dataset metadata changed since evaluation')
    cases, pictures, evidence = [], [], []
    offset = result['config']['eval']['goal_offset_steps']
    with h5py.File(dataset, 'r') as data:
        for i, (episode, start) in enumerate(zip(result['episodes'], result['starts'], strict=True)):
            if start < 0 or start + offset >= int(data['ep_len'][episode]):
                raise ValueError('Initial/goal frame is outside its episode')
            index = int(data['ep_offset'][episode]) + start
            initial, goal = data['pixels'][index], data['pixels'][index + offset]
            state, goal_state = data['state'][index], data['state'][index + offset]
            expected = result['initial_runtime_hashes'][i]
            for key, value in [('pixels', initial), ('goal', goal),
                               ('state', state), ('goal_state', goal_state)]:
                if array_hash(value[None]) != expected[key]:
                    raise ValueError(f'Case {i}: {key} differs from evaluation hash')
            scores = case_scores(result['physical_actions'], i, goal_state)
            passed = [row for row in scores if row['passed']]
            if bool(passed) != result['successes'][i]:
                raise ValueError(f'Case {i}: reconstructed success differs from saved result')
            selected = passed[0] if passed else min(
                (row for row in scores if row['active']),
                key=lambda row: max(row['position'] / 20, row['angle_deg'] / 20))
            # The audited SWM video stores one frame after each physical step,
            # including repeated frames for environments that already stopped.
            with av.open(str(evaluation / f'env_{i}.mp4')) as video:
                stream = video.streams.video[0]
                fps = float(stream.average_rate)
                frames = 0
                selected_image = None
                for frame in video.decode(video=0):
                    frames += 1
                    if frames == selected['step']:
                        selected_image = frame.to_image()
                if frames != len(scores) or selected_image is None or fps <= 0:
                    raise ValueError(f'Case {i}: video/state frame count mismatch')
            cases.append({'case': i, 'episode': episode, 'start': start, 'goal_step': start + offset,
                          'success': bool(passed), 'first_success_step': passed[0]['step'] if passed else None,
                          'selected_step': selected['step'], 'fps': fps, 'scores': scores})
            pictures.append((initial, goal))
            evidence.append(selected_image)

    # All case identities and reconstructed outcomes must pass before publishing.
    output.mkdir(parents=True)
    assets = output / 'assets'
    assets.mkdir()
    for row, (initial, goal), selected_image in zip(cases, pictures, evidence, strict=True):
        i = row['case']
        Image.fromarray(initial).save(assets / f'case_{i}_initial.png')
        Image.fromarray(goal).save(assets / f'case_{i}_goal.png')
        selected_image.save(assets / f'case_{i}_evidence.png')
        shutil.copyfile(evaluation / f'env_{i}.mp4', assets / f'case_{i}.mp4')
    payload = {'evaluation': evaluation.name, 'cases': cases, 'successes': sum(result['successes']),
               'checkpoint_sha256': result['checkpoint_sha256'],
               'source_result': str(result_path), 'goal_offset_steps': offset,
               'verification': 'initial/goal image and state hashes; all reconstructed success flags; video frame counts',
               'failure_evidence_selection': 'active step minimizing max(position/20, angle_deg/20)'}
    (output / 'report.json').write_text(json.dumps(payload, indent=2))
    (output / 'results.txt').write_text(trial_table(result))
    encoded = json.dumps(payload).replace('<', '\\u003c')
    title = html.escape(evaluation.name)
    document = PAGE.replace('__TITLE__', title).replace('__DATA__', encoded)
    (output / 'index.html').write_text(document)
    return {'index': str(output / 'index.html'), 'cases': n, 'successes': payload['successes']}


PAGE = '''<!doctype html>
<html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PushT データGoalへの到達 — __TITLE__</title>
<style>
*{box-sizing:border-box}[hidden]{display:none!important}body{margin:0;background:#f3f5f7;color:#152630;font-family:system-ui,sans-serif;line-height:1.6}
main{max-width:1200px;margin:auto;padding:28px 22px}h1{font-size:27px;margin:0}h2{font-size:20px;margin:0}
p{margin:8px 0 16px}.muted{color:#526675;font-size:14px}.panel{background:white;border:1px solid #dbe2e7;border-radius:12px;padding:20px;margin:16px 0}
.metrics,.media,.toolbar{display:flex;gap:16px;flex-wrap:wrap;align-items:center}.media{align-items:start}.media figure{flex:1;min-width:230px;margin:0}
img,video{display:block;width:100%;aspect-ratio:1;object-fit:contain;background:#f1f3f5;border-radius:8px}figcaption{font-size:14px;margin-bottom:8px;font-weight:650}
button,select{font:inherit;padding:7px 12px;border:1px solid #c6d2da;background:white;border-radius:6px;cursor:pointer}button:disabled{opacity:.4;cursor:default}
.good{color:#126647}.bad{color:#a12b36}.badge{font-weight:700}.metrics>div{flex:1;min-width:190px;padding:12px;background:#f4f7f8;border-radius:8px}.metrics strong{display:block;font-size:24px}
#case-list{display:flex;flex-wrap:wrap;gap:6px;margin-top:15px}#case-list button{font-size:13px;min-width:54px}#case-list button[aria-current=true]{outline:3px solid #326eb2}
details{margin-top:16px}code{overflow-wrap:anywhere}.initial{max-width:224px;margin-top:10px}a{color:#245f9e}
.overlay-wrap{position:relative}.overlay-wrap img.overlay{position:absolute;inset:0;opacity:.5;pointer-events:none}.evidence-score{font-size:14px}
</style>
<main><h1>PushT：データGoalへの到達</h1><p id="summary"></p>
<div class="panel"><strong>何が「成功」か</strong>
<p>各ケースの開始から<span id="offset"></span>ステップ後のデータ画像が正解です。評価中の同じ時点で、<b>操作点とT字物体の位置のずれを合わせた距離が20未満</b>、かつ<b>T字物体の角度差が20度未満</b>になると成功です。一度達成すれば、そのケースは成功として記録されます。</p>
<p><b>合わせる対象は、右の正解画像に写る灰色のT字と青い操作点です。緑のT字への重なりは今回の成功条件ではありません。</b></p>
<p class="muted">位置の単位は512×512の環境座標です。距離は √(操作点Δx²＋Δy²＋物体Δx²＋Δy²)。表示画像上の20ピクセルとは異なります。速度は成功条件に含みません。動画内の目標マークはデータ由来Goalと別の描画設定なので、右の正解画像を参照してください。固定T字への95%被覆率による判定ではありません。</p></div>
<div class="panel"><div class="toolbar"><label>表示 <select id="filter"><option value="all">全ケース</option><option value="success">成功のみ</option><option value="failure">失敗のみ</option></select></label><button id="prev">前へ</button><button id="next">次へ</button><span id="counter"></span></div><div id="case-list"></div></div>
<section class="panel" id="viewer"><div class="toolbar"><h2 id="case-title"></h2><span id="outcome" class="badge"></span></div><p id="case-meta" class="muted"></p>
<div class="media"><figure><figcaption>評価時の動画</figcaption><video id="video" controls playsinline preload="metadata"></video></figure>
<figure><figcaption id="evidence-label">判定時点の記録フレーム</figcaption><div class="overlay-wrap"><img id="evidence" alt="成功時点または最も条件に近い時点の動画フレーム"><img id="overlay" class="overlay" alt="半透明の正解画像" hidden></div><p id="evidence-score" class="evidence-score"></p><label><input id="overlay-toggle" type="checkbox"> 正解を半透明で重ねる</label><br><button id="jump">この時点へ移動</button></figure>
<figure><figcaption>正解画像（評価に使ったGoal）</figcaption><img id="goal" alt="データから取得し評価時のhashと照合した正解画像"></figure></div>
<p id="timing" class="muted"></p><h3 id="frame-title">動画の現在位置での判定</h3><div class="metrics"><div>位置のずれ（20未満）<strong id="position"></strong></div><div>角度差（20度未満）<strong id="angle"></strong></div><div>この時点の条件<strong id="frame-outcome"></strong></div></div>
<details><summary>開始画像と記録の照合</summary><img id="initial" class="initial" alt="評価に入力したデータの開始画像"><p class="muted">開始／Goalの画像と状態を評価時のhashに照合し、全ケースの成功判定を保存物理状態から再計算済み。動画は元の記録をコピーしています。失敗例の参考フレームは、位置のずれ÷20と角度差÷20の大きい方が最小の時点です。</p><p class="muted" id="identity"></p><a href="results.txt">試行番号と成否（テキスト）</a> · <a href="report.json">判定数値の記録（JSON）</a></details></section></main>
<script type="application/json" id="data">__DATA__</script>
<script>
const data=JSON.parse(document.getElementById('data').textContent), el=id=>document.getElementById(id);
let visible=data.cases,index=0,current=null;
el('summary').textContent=`${data.evaluation} ｜ Goal到達 ${data.successes}/${data.cases.length}（${(100*data.successes/data.cases.length).toFixed(0)}%）`;
el('offset').textContent=data.goal_offset_steps;
function score(){if(!current)return;const frame=Math.min(current.scores.length-1,Math.floor(el('video').currentTime*current.fps+1e-6)),r=current.scores[frame];
el('frame-title').textContent=`動画の現在位置：step ${r.step}`;
for(const [key,value,unit,ok] of [['position',r.position,'',r.position<20],['angle',r.angle_deg,'°',r.angle_deg<20]]){el(key).textContent=value.toFixed(3)+unit;el(key).className=ok?'good':'bad';}
el('frame-outcome').textContent=!r.active?'終了後の保持表示':r.passed?'成立':'未達';el('frame-outcome').className=r.passed?'good':r.active?'bad':'muted';}
function choose(i){index=i;current=visible[index];if(!current)return;el('video').pause();
el('case-title').textContent=`${current.case+1}回目（env_${current.case}.mp4）`;
el('outcome').textContent=current.success?`成功：step ${current.first_success_step}で達成`:'失敗：予算内に未達';el('outcome').className='badge '+(current.success?'good':'bad');
el('case-meta').textContent=`episode ${current.episode} ｜ 開始 ${current.start} → Goal ${current.goal_step}`;
el('video').src=`assets/case_${current.case}.mp4`;el('video').load();
for(const key of ['goal','initial','evidence'])el(key).src=`assets/case_${current.case}_${key}.png`;
el('overlay').src=el('goal').src;
el('evidence-label').textContent=(current.success?'最初の成功時点':'最も条件に近い時点')+`：step ${current.selected_step}`;
const selected=current.scores[current.selected_step-1];el('evidence-score').textContent=`この静止画：位置 ${selected.position.toFixed(3)}（${selected.position<20?'達成':'未達'}）、角度 ${selected.angle_deg.toFixed(3)}°（${selected.angle_deg<20?'達成':'未達'}）`;
el('timing').textContent=`動画は${current.fps} fpsで保存されています。環境制御は10 Hzなので、15 fps再生は環境時間の1.5倍速です。CEMの計算待ち時間は動画に含まれません。`;
el('counter').textContent=`${index+1} / ${visible.length}`;el('prev').disabled=index===0;el('next').disabled=index===visible.length-1;
el('identity').textContent=`checkpoint SHA-256: ${data.checkpoint_sha256}`;
for(const button of el('case-list').children)button.setAttribute('aria-current',Number(button.dataset.case)===current.case?'true':'false');score();}
function list(){const value=el('filter').value;visible=data.cases.filter(c=>value==='all'||c.success===(value==='success'));el('case-list').replaceChildren();
visible.forEach((c,i)=>{const b=document.createElement('button');b.textContent=`${c.case+1}回目 ${c.success?'成功':'失敗'}`;b.className=c.success?'good':'bad';b.dataset.case=c.case;b.onclick=()=>choose(i);el('case-list').append(b);});
el('viewer').hidden=!visible.length;if(visible.length)choose(0);else{el('video').pause();current=null;el('counter').textContent='0件';el('prev').disabled=true;el('next').disabled=true;}}
el('filter').onchange=list;el('prev').onclick=()=>choose(index-1);el('next').onclick=()=>choose(index+1);
el('jump').onclick=()=>{el('video').currentTime=(current.selected_step-0.5)/current.fps;};
el('overlay-toggle').onchange=()=>{el('overlay').hidden=!el('overlay-toggle').checked;};
for(const event of ['timeupdate','seeked','loadeddata'])el('video').addEventListener(event,score);list();
</script></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New directory for the portable viewer')
    args = parser.parse_args()
    print(json.dumps(build(args.evaluation, args.output), indent=2))


if __name__ == '__main__':
    main()
