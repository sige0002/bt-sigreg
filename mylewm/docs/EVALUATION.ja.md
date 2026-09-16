# 世界モデル＋CEMの評価手順

世界モデルで候補行動の未来を予測し、目標画像に近づく行動をCEMで選び、環境で成功するかを測ります。予測・Goal距離には状態zを使い、学習専用変換Tは使いません。学習lossやBCの教師行動誤差とは別の評価です。

<a id="pusht"></a>
## PushT：保存済みモデルから結果確認まで

学習時に作ったPython環境、公開HDF5、`step_N_object.ckpt`が必要です。この節はリポジトリ直下から、Bashの同じ端末で実行します。学習が終わる前でも、保存が完了した途中checkpointは評価できます。学習中の`.venv`へ`uv sync`を実行しません。

### 1. 入力と出力を指定する

```bash
export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PUSHT_HDF5="$PWD/.cache/pusht/pusht_expert_train.h5"
export PUSHT_EVAL_MANIFEST="$PWD/output/manifests/pusht/eval_example/manifest.json"
export PUSHT_CHECKPOINT="$PWD/output/pusht/raw_example/step_140000_object.ckpt"
export PUSHT_EVAL_OUT="$PWD/output/pusht/eval_raw_example"
ls -lh "$PUSHT_HDF5" "$PUSHT_CHECKPOINT"
```

自分の保存先に合わせて変更します。出力はリポジトリ内`output/`配下の未使用ディレクトリにします。checkpoint隣の`config.json`も保持してください。

### 2. 評価ケースを固定する（初回だけ）

**学習用clip90 manifestには評価ケースがないため、そのまま評価へ渡せません。** 別manifestを作り、RawとBTの両評価で使います。

```bash
.venv/bin/python -m mylewm.training.loop prepare \
  --dataset "$PUSHT_HDF5" --manifest "$PUSHT_EVAL_MANIFEST"
```

この操作は学習を開始せず、データ全量hashと固定200ケースの`confirm`を保存します。評価manifestがある場合は再作成しません。旧schema名を持つファイルですが、ここでは評価ケースの固定にだけ使い、現行Raw／BTの学習には渡しません。

clip90学習ではepisode単位の保持分割をしていないため、これを「学習に含まれないepisodeだけの評価」とは呼びません。

### 3. 設定を確認する（dry-run）

```bash
bash scripts/evaluate_pusht.sh \
  --checkpoint "$PUSHT_CHECKPOINT" --manifest "$PUSHT_EVAL_MANIFEST" \
  --output "$PUSHT_EVAL_OUT" --num-eval 50 --offset 0 --seed 42
```

まだ環境評価を実行しません。`--num-eval 50 --offset 0`は固定200ケースの先頭50件です。200件すべてなら`--num-eval 200`にします。

### 4. 環境評価を開始する

```bash
bash scripts/evaluate_pusht.sh \
  --checkpoint "$PUSHT_CHECKPOINT" --manifest "$PUSHT_EVAL_MANIFEST" \
  --output "$PUSHT_EVAL_OUT" --num-eval 50 --offset 0 --seed 42 --execute
```

SSH越しでは、開始前に`tmux new -s pusht-eval`で端末を保持し、その中で第1節の変数を設定します。`Ctrl-b`の後に`d`で離れ、`tmux attach -t pusht-eval`で戻ります。評価端末のCtrl-Cは評価を止めます。

標準CEMは候補300・反復30・上位30、計画5ブロック×5行動です。環境予算50・Goal間隔25を含め、同じ評価器・引数・固定ケースで比較します。モデルごとに探索量を変更しません。

<a id="intermediate"></a>
### 5. 完了・成功率・動画を確認する

終了直後に`echo $?`で0を確認し、以下を読みます。

```bash
cat "$PUSHT_EVAL_OUT/status.json"
cat "$PUSHT_EVAL_OUT/results.txt.json"
tail -n 10 "$PUSHT_EVAL_OUT/console.log"
```

`status.json`の`state=succeeded`と`cases=50`を確認します。`results.txt.json`の`successes`は各試行の成否配列で、長さが50、合計が成功数です。`status.json`の`successes`は成功数の整数です。200試行なら件数は200に読み替えます。`success_rate`は百分率です。途中終了を完了とせず、未実施分を失敗として補いません。

`viewer/index.html`と動画も保存されます。画像・動画の容量を含めて出力先を確保してください。ブラウザで見る場合は次で公開先をlocalhostに限定します。

```bash
.venv/bin/python -m http.server 8000 --bind 127.0.0.1 --directory "$PUSHT_EVAL_OUT"
```

同じPCのブラウザで`http://127.0.0.1:8000/viewer/`を開きます。SSH先なら接続元で`ssh -L 8000:127.0.0.1:8000 接続先`を使います。サーバー停止はCtrl-Cです。

### 6. BTと同じケースで比べる

第1節のcheckpointを`output/pusht/bt_example/step_140000_object.ckpt`、出力を`output/pusht/eval_bt_example`に変え、第3〜5節を実行します。manifest・ケース数・offset・seed・CEM設定は変えません。

```bash
.venv/bin/python -m mylewm.evaluation.compare_paired \
  --baseline output/pusht/eval_raw_example/results.txt.json \
  --candidate output/pusht/eval_bt_example/results.txt.json \
  --output output/pusht/compare_raw_bt_example.json
```

成功数／試行数、同じケースの成否差、学習・推論時間を報告します。単一seedだけで一般的な優位性を断定しません。公式配布重みの参考値と、同予算で再学習したRawとの比較は分けます。

<a id="libero"></a>
## LIBERO-10：元の128pxデータでのCEM評価

学習済み世界モデル・対応manifest・元の10タスクHDF5が必要です。現行CEM評価器は128px経路です。再生成256pxモデルの通しCEM評価は、この手順で対応済みとは扱いません。

### シミュレータ環境を用意する（初回だけ）

学習用のPython環境とは別に、LIBEROソース・OSMesa・設定ファイルが必要です。以下はUbuntu/Debianでの導入例です。リポジトリ直下で実行します。既に評価用環境がある場合は再導入せず、そのソース・依存・描画設定を使います。稼働中の学習環境を同期しないため、初回は評価専用venvにします。

```bash
export UV_PROJECT_ENVIRONMENT="$PWD/.venv-libero-eval"
uv sync --locked --group libero
sudo apt-get update
sudo apt-get install -y libosmesa6
mkdir -p external
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git external/libero
git -C external/libero checkout 8f1084e3132a39270c3a13ebe37270a43ece2a01
export LIBERO_ROOT="$PWD/external/libero"
export LIBERO_OSMESA_DIR="/usr/lib/$(uname -m)-linux-gnu"
export LIBERO_CONFIG_PATH="$PWD/.cache/libero-config-example"
export LIBERO_DATASET="$PWD/.cache/libero-datasets/libero_10"
export LIBERO_MUJOCO_PATH="$UV_PROJECT_ENVIRONMENT/lib/python3.12/site-packages"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

LIBEROが既にcloneされている場合、clone・checkoutを省き、使用する版を確認してください。次は新規設定ファイルを作ります。既存設定には上書きしません。

```bash
uv run --no-sync python - <<'PYCONFIG'
import os
from pathlib import Path
import yaml
root = Path(os.environ['LIBERO_ROOT']) / 'libero/libero'
config = Path(os.environ['LIBERO_CONFIG_PATH'])
config.mkdir(parents=True, exist_ok=True)
paths = dict(benchmark_root=root, bddl_files=root/'bddl_files',
             init_states=root/'init_files', assets=root/'assets',
             datasets=Path(os.environ['LIBERO_DATASET']).parent)
with (config/'config.yaml').open('x') as f:
    yaml.safe_dump({k: str(v) for k, v in paths.items()}, f)
PYCONFIG
bash scripts/run_libero.sh -c 'import mujoco; from libero.libero import benchmark; print(mujoco.__version__, len(benchmark.get_benchmark_dict()["libero_10"]().tasks))'
```

新規マシンへのこの導入例は、描画一致まで一律に保証するものではありません。ローカルの実験で使った追加runtimeとlockfileの環境が異なる場合もあるため、次の画像監査を合格条件とします。単なるimport成功を評価環境の検証完了とはしません。


### 画像の向き・カメラを監査する

```bash
export LIBERO_AUDIT="$PWD/output/libero10/audit_example"
for task in {0..9}; do
  bash scripts/run_libero.sh -m mylewm.evaluation.audit_libero_images \
    --dataset "$LIBERO_DATASET" --task-id "$task" \
    --output "$LIBERO_AUDIT/task_$task" || break
done
```

これはdry-runではなく、実際に描画します。全10個の`task_N/report.json`が存在し、`passed=true`であることを確認します。監査が失敗した場合は評価へ進まず、解像度・描画runtime・データの対応を直します。

### 設定確認・実行・結果確認

```bash
export LIBERO_MANIFEST="$PWD/output/manifests/libero10/example/manifest.json"
export LIBERO_CHECKPOINT="$PWD/output/libero10/bt_example/step_10000_object.ckpt"
export LIBERO_EVAL_OUT="$PWD/output/libero10/eval_bt_example"
bash scripts/run_libero.sh -m mylewm.evaluation.evaluate_libero \
  --checkpoint "$LIBERO_CHECKPOINT" --manifest "$LIBERO_MANIFEST" \
  --output "$LIBERO_EVAL_OUT" --task-ids 0 1 2 3 4 5 6 7 8 9 \
  --episodes 50 --offset 0 --budget 520 --horizon 8 --samples 128 --iterations 5 \
  --seed 42 --device cuda:0 --render-audit-dir "$LIBERO_AUDIT"
```

上は設定確認のみです。実行する場合は、同じコマンドの末尾に`--execute`を追加します。10タスク×50試行で、成功はLIBEROの成功関数で判定します。

```bash
cat "$LIBERO_EVAL_OUT/status.json"
cat "$LIBERO_EVAL_OUT/summary.json"
wc -l "$LIBERO_EVAL_OUT/episodes.jsonl"
```

終了コード0・`state=succeeded`・500試行を照合します。`summary.json`の`task_rates`・`macro_success`は0〜1の比率です。平均だけでなくタスク別の成功率を読み、同条件Raw／TC／BTの比較を行います。
