# LIBEROのBC：追加学習・再開・環境評価

BCはBehavior Cloning（行動模倣）の略です。成功デモの画像から行動を学びます。この実装は世界モデルの画像encoderを凍結し、タスクID付きflow matching方策を学習します。**未来予測器・BT写像Tを使わないため、世界モデル＋CEMの計画能力を測る主評価ではありません。**

## 1. 入力と環境を揃える

世界モデル学習を終えたリポジトリ直下から、Bashの同じ端末で実行します。既存の学習用`.venv`を使い、ここで依存同期は行いません。必要なのは世界モデルの`*_object.ckpt`、隣の`config.json`、その学習時と同じmanifest・HDF5です。

```bash
export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export BC_WM="$PWD/output/libero10/bt_example/step_10000_object.ckpt"
export BC_MANIFEST="$PWD/output/manifests/libero10/example/manifest.json"
export BC_RUN="$PWD/output/libero10/bc_example"
ls -lh "$BC_WM" "$BC_MANIFEST"
.venv/bin/python -m mylewm.training.train_libero_bc --help
```

この例は元128pxデータの世界モデルを使用します。再生成256pxモデルを使う場合は、そのモデル・manifest・データの組を指定し、元データの統計と混ぜません。

## 2. 設定確認と学習

現在の2カメラ画像から34tokenを取り、幅256・4層・8 headsの方策に渡します。教師は連続8行動×7次元、推論は10 Euler stepです。Goal・未来画像・報酬を入力しません。TC-LeWMを参考にしたローカル実装で、公式コードの移植ではありません。

```bash
.venv/bin/python -m mylewm.training.train_libero_bc \
  --checkpoint "$BC_WM" --manifest "$BC_MANIFEST" --output "$BC_RUN" \
  --steps 40000 --batch-size 256 --workers 4 --seed 3072 \
  --horizon 8 --width 256 --depth 4 --heads 8 --euler-steps 10 \
  --lr 2e-4 --weight-decay 0.01 --save-every 1000 --val-every 1000 --device cuda:0
```

上はdry-runです。設定を確認したら次で実学習を開始します。SSHで実行する場合は`tmux new -s libero-bc`の中で第1節の変数を設定します。`Ctrl-b`、`d`で離れ、`tmux attach -t libero-bc`で戻ります。

```bash
.venv/bin/python -m mylewm.training.train_libero_bc \
  --checkpoint "$BC_WM" --manifest "$BC_MANIFEST" --output "$BC_RUN" \
  --steps 40000 --batch-size 256 --workers 4 --seed 3072 \
  --horizon 8 --width 256 --depth 4 --heads 8 --euler-steps 10 \
  --lr 2e-4 --weight-decay 0.01 --save-every 1000 --val-every 1000 --device cuda:0 \
  --execute
```

## 3. 保存と完了を確認する

```bash
tail -n 3 "$BC_RUN/metrics.jsonl"
cat "$BC_RUN/status.json"
```

終了後に終了コード0、`status.json`・`completed.json`の`state=succeeded`、40,000更新、`step_40000_bc.pt`を照合します。学習完了前には最終ファイルはありません。

```bash
cat "$BC_RUN/completed.json"
ls -lh "$BC_RUN/step_40000_bc.pt" "$BC_RUN/resume.pt"
```

`step_N_bc.pt`はencoderと方策を含む評価用、`resume.pt`はoptimizer・乱数等も含む再開用です。検証lossは教師デモでのflow学習誤差で、実際の生成行動誤差・環境成功率とは別です。

## 4. 中断したBCを再開する

元の学習が停止していることを確認し、同じソース・環境・manifest・入力世界モデルを使います。**元と同じ出力先**と総数40,000を指定し、`--resume --execute`を付けます。

```bash
.venv/bin/python -m mylewm.training.train_libero_bc \
  --checkpoint "$BC_WM" --manifest "$BC_MANIFEST" --output "$BC_RUN" \
  --steps 40000 --batch-size 256 --workers 4 --seed 3072 \
  --horizon 8 --width 256 --depth 4 --heads 8 --euler-steps 10 \
  --lr 2e-4 --weight-decay 0.01 --save-every 1000 --val-every 1000 --device cuda:0 \
  --resume --execute
```

再開時は構成・ソース・依存・データ・encoderの同一性を照合します。hash検証を解除して再開しません。

## 5. 環境成功率を測る場合だけ、描画を準備する

「BC評価」には教師デモとの生成行動比較と環境成功率の2種類があります。ここからは後者です。生成行動の診断だけをしたい場合は、環境評価を代わりに実行しません。過去の生成行動診断は実験専用スクリプトで行っており、共通CLIとしては未整備です。[診断記録](reports/LIBERO_BC_ACTION_DIAGNOSTIC.ja.md)に条件と証拠を残しています。

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


```bash
export BC_AUDIT="$PWD/output/libero10/bc_audit_example"
for task in {0..9}; do
  bash scripts/run_libero.sh -m mylewm.evaluation.audit_libero_images \
    --dataset "$LIBERO_DATASET" --task-id "$task" \
    --output "$BC_AUDIT/task_$task" || break
done
```

監査は実際に描画します。全10個の`task_N/report.json`が`passed=true`であることを確認します。再生成データを使う場合は`LIBERO_DATASET`をそのデータへ向け、各監査コマンドに`--regenerated`を付けます。BC評価器は128／256pxをデータから判定しますが、監査・モデル・データの解像度は一致させます。

## 6. BC環境評価を設定確認し、実行する

新規の出力先にします。1タスク50試行×10タスクです。

```bash
export BC_EVAL_OUT="$PWD/output/libero10/eval_bc_example"
bash scripts/run_libero.sh -m mylewm.evaluation.evaluate_libero_bc \
  --checkpoint "$BC_RUN/step_40000_bc.pt" --manifest "$BC_MANIFEST" \
  --output "$BC_EVAL_OUT" --task-ids 0 1 2 3 4 5 6 7 8 9 \
  --episodes 50 --offset 0 --budget 520 --execute-actions 8 --settling-steps 5 \
  --seed 42 --device cuda:0 --render-audit-dir "$BC_AUDIT"
```

これはdry-runです。実評価を開始する場合は、同じコマンドの末尾に`--execute`を追加します。初期状態復元後は5回のゼロ行動を行い、以後8行動ごとに方策を呼びます。成功はLIBEROの成功関数で判定します。

```bash
cat "$BC_EVAL_OUT/status.json"
cat "$BC_EVAL_OUT/summary.json"
wc -l "$BC_EVAL_OUT/episodes.jsonl"
```

終了コード0・`state=succeeded`・500試行の一致を確認します。`summary.json`の`macro_success`・`task_rates`は0〜1の比率です。平均・タスク別成功率を読みます。動画と`viewer/index.html`も保存されるため容量が必要です。途中停止の残りを失敗に数えません。BCが成功しても、世界モデルが未来予測を使って行動を選べた証拠にはしません。
