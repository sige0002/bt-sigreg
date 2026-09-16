# LIBERO-10の世界モデル学習

このページは、元のLIBERO-10デモ取得からprepare・学習・再開までの正本です。10タスクを1つのモデルで学びます。**元データとOpenVLA方式の再生成データは別のレシピ**です。まず元データで完結する手順を示し、再生成する場合の違いを末尾に記します。BCはこの学習に必須ではありません。

## 1. 環境と保存先

Linux、Git、uv、Python 3.12、CUDA 13.0版PyTorchに対応するNVIDIA環境が前提です。未導入のツールがある場合だけ、以下で導入します。既にclone済みならそのルートへ移動し、cloneを省いてください。HDF5から世界モデルを学ぶだけならLIBEROシミュレータ・描画ライブラリは不要です。

```bash
sudo apt-get update
sudo apt-get install -y git curl tmux
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
git clone https://github.com/sige0002/bt-sigreg.git
cd bt-sigreg
```

SSH越しではここで`tmux new -s libero-train`を実行し、その中で続けます。`Ctrl-b`、`d`で離れ、`tmux attach -t libero-train`で戻ります。

```bash
export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
uv python install 3.12
uv sync --locked --group libero
export CUBLAS_WORKSPACE_CONFIG=:4096:8
.venv/bin/python -c 'import torch; assert torch.cuda.is_available()'
.venv/bin/python -m mylewm.training.train_libero --help
export LIBERO_DATA_ROOT="$PWD/.cache/libero-datasets"
export LIBERO_DATASET="$LIBERO_DATA_ROOT/libero_10"
export LIBERO_MANIFEST="$PWD/output/manifests/libero10/example/manifest.json"
export LIBERO_RUN="$PWD/output/libero10/bt_example"
```

環境同期は初回だけで、学習中には行いません。別ストレージを使う場合は`LIBERO_DATA_ROOT`を変更します。データ・manifest・出力を既存実験へ上書きしません。

## 2. 元の10タスクのデモを取得する

取得済みならこの節を飛ばし、`LIBERO_DATASET`を10個のHDF5が直接置かれたディレクトリへ向けます。

```bash
uv run --no-sync hf download yifengzhu-hf/LIBERO-datasets --repo-type dataset \
  --include 'libero_10/*' --local-dir "$LIBERO_DATA_ROOT"
ls -lh "$LIBERO_DATASET"/*.hdf5
```

PushTと違い、`--dataset`には単一ファイルでなくディレクトリを渡します。

## 3. prepareで分割・統計を固定する

```bash
.venv/bin/python -m mylewm.training.train_libero prepare \
  --dataset "$LIBERO_DATASET" --manifest "$LIBERO_MANIFEST" \
  --split-mode ratio_80_10_10
```

各タスクのデモを丸ごと80%／10%／10%に分け、trainだけから行動統計を作ります。50本なら40／5／5です。manifestの隣に`files.json`を作り、実HDF5の場所・サイズ・mtimeを記録します。初回prepareは全量hashも計算します。公開論文と分割が一致するという意味ではありません。

## 4. 設定確認と学習

**この世界モデルCLIにはdry-runがありません。`train`は即学習を開始します。** 実行しない確認は`--help`とmanifestの確認までです。`--execute`を付ける形式ではありません。

```bash
.venv/bin/python -m mylewm.training.train_libero --help
.venv/bin/python -c 'import json, os; m=json.load(open(os.environ["LIBERO_MANIFEST"])); print(m["dataset"])'
```

次は10,000更新の例です。PushTの14万予算をLIBEROへ自動で転用しません。比較するRaw／TC／BTで同じ予算を先に決めます。

```bash
.venv/bin/python -m mylewm.training.train_libero train \
  --manifest "$LIBERO_MANIFEST" --output "$LIBERO_RUN" --mode bt \
  --steps 10000 --batch-size 128 --workers 4 --no-pin-memory \
  --lr 5e-5 --warmup-steps 500 --min-lr 0 --seed 3072 \
  --bt-depth 2 --bt-kappa 0.2 --bt-hidden 192 --save-every 1000
```

Rawは`--mode raw`、TCは`--mode tc`と未使用の出力先に変えます。構成・初期値hash・データ順・予算・LRを一致させて比較します。

先に100更新の接続確認を行いたい場合は、上のコマンドを**別出力先**にし、`--steps 100 --warmup-steps 10 --save-every 50`へ変更して実行します。短期runを本学習へ予算変更して再開せず、本学習は元の条件・新規初期値・別出力で始めます。

## 5. ログ・完了を確認する

```bash
tail -n 3 "$LIBERO_RUN/metrics.jsonl"
```

終了後は終了コード0、`completed.json`のstep10,000、最終objectと再開状態を照合します。

```bash
cat "$LIBERO_RUN/completed.json"
ls -lh "$LIBERO_RUN/step_10000_object.ckpt" "$LIBERO_RUN/resume.pt"
```

`metrics.jsonl`は損失等の記録で、環境成功率ではありません。`step_N_object.ckpt`は評価用、`resume.pt`はT・optimizer・乱数等も含む再開用です。

## 6. 中断後の再開

元の学習が停止していることを確認し、同じソース・環境・manifest・設定を使用します。**LIBEROは元と同じ出力先**で、パスを取らない`--resume`を付けます。

```bash
.venv/bin/python -m mylewm.training.train_libero train \
  --manifest "$LIBERO_MANIFEST" --output "$LIBERO_RUN" --mode bt \
  --steps 10000 --batch-size 128 --workers 4 --no-pin-memory \
  --lr 5e-5 --warmup-steps 500 --min-lr 0 --seed 3072 \
  --bt-depth 2 --bt-kappa 0.2 --bt-hidden 192 --save-every 1000 --resume
```

`--steps`は元の総数のままです。PushTの「未使用出力＋checkpointパス」とは異なります。初回保存前には再開状態がありません。

## 再生成256pxデータを選ぶ場合

元データと混ぜず、別のデータroot・manifest・runを使います。再生成にはLIBERO環境と描画が必要です。元HDF5だけの学習とは前提が異なります。

```bash
export LIBERO_REGENERATED="$PWD/output/libero10/regenerated_example"
bash scripts/run_libero.sh -m mylewm.data.regenerate_libero_openvla \
  --raw-data "$LIBERO_DATASET" --output "$LIBERO_REGENERATED"
```

これは設定確認です。環境構築と画像監査が済んでいる場合に、同じコマンドへ`--execute`を追加して再生成します。`status.json=succeeded`と終了コード0を確認後、第3節の`--dataset`を`$LIBERO_REGENERATED/data`にし、別manifestをprepareします。以降の学習・再開の形式は同じです。再生成用環境が未構築なら、元データでの学習完了と混同せず、この分岐は実行しません。

学習を終えたモデルの環境評価は別作業です。評価手順は`EVALUATION.ja.md`、画像encoderの補助評価であるBCは`BEHAVIOR_CLONING.ja.md`が正本です。
