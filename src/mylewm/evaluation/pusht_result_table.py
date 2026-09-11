"""Give each saved PushT outcome an explicit trial number and video filename."""
import json
from pathlib import Path

MARKER = '==== TRIAL RESULTS (1-based trial numbers) ===='


def criterion_text(offset):
    return (f'評価対象: 開始から{offset}step後のデータGoalへの到達。緑のT字へのはめ込み成功率ではありません。\n'
            '成功条件: 操作点と物体の位置4成分の距離 < 20、かつ物体の角度差 < 20度。')


def trial_table(result, checkpoint=None):
    rows = list(zip(result['episodes'], result['starts'], result['successes'], strict=True))
    offset = result['config']['eval']['goal_offset_steps']
    lines = [MARKER]
    if checkpoint:
        lines.append(f'Checkpoint: {checkpoint}')
    lines.append(criterion_text(offset))
    lines.extend(['試行番号は1始まり、動画のenv番号は0始まり。',
                  '試行\t動画\t判定\tepisode\t開始step\tGoal step'])
    for i, (episode, start, success) in enumerate(rows):
        lines.append(f'{i + 1}回目\tenv_{i}.mp4\t{"成功" if success else "失敗"}\t'
                     f'{episode}\t{start}\t{start + offset}')
    lines.append(f'合計: {sum(result["successes"])}/{len(rows)} 成功')
    return '\n'.join(lines) + '\n'


def append_trial_table(evaluation):
    evaluation = Path(evaluation)
    result = json.loads((evaluation / 'results.txt.json').read_text())
    launch_path = evaluation / 'launch.json'
    checkpoint = json.loads(launch_path.read_text())['checkpoint'] if launch_path.is_file() else None
    path = evaluation / 'results.txt'
    existing = path.read_text()
    if MARKER not in existing:
        with path.open('a') as stream:
            stream.write('\n' + trial_table(result, checkpoint))
    elif (criterion := criterion_text(result['config']['eval']['goal_offset_steps'])) not in existing:
        with path.open('a') as stream:
            stream.write('\n' + criterion + '\n')
