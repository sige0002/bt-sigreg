# BT-SIGReg

画像と行動の記録から、行動後の変化を予測する小型の世界モデルを学習します。LeWMのモデル・損失を使うRawを比較対象とし、正則化に学習専用の変換を加えるBT-SIGRegの効果を調べます。

## 最初のPushT学習を始める

再開を含む手順の正本は[学習ガイド](mylewm/docs/TRAINING.ja.md)です。ここにも初回の開始までを省略せず載せます。

以下は**Linux＋NVIDIA GPUで、公開データを取得してRawを学習する順序**です。Bashの同じ端末で実行します。Pythonは3.12、リポジトリのlockfileはPyTorch 2.9.1／CUDA 13.0用です。対応ドライバが必要で、クラウドやGPUの種類を問わず動く構成と保証するものではありません。

### 1. ツールとPython環境を用意する

Ubuntu/Debianで未導入のツールを用意します。既にあれば導入行は飛ばしてください。

```bash
sudo apt-get update
sudo apt-get install -y git curl zstd tmux
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
git clone https://github.com/sige0002/bt-sigreg.git
cd bt-sigreg
# SSH越しで続ける場合だけ、ここで tmux new -s pusht-train を実行
export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
uv python install 3.12
uv sync --locked
export CUBLAS_WORKSPACE_CONFIG=:4096:8
.venv/bin/python -c 'import torch, mylewm; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(), "CUDAが利用できません"'
```

既にclone済みならそのルートへ移動します。`uv sync`は初回だけです。学習中の環境へ依存を同期しません。CUDA確認が失敗した場合はGPU・ドライバを確認してから進みます。

### 2. データを取得して展開する

圧縮約13 GB、展開後約46 GBと、モデル保存用の空き容量が必要です。データ取得済みなら`PUSHT_HDF5`を既存ファイルへ向けて、取得・展開を飛ばせます。

```bash
export PUSHT_DATA_ROOT="$PWD/.cache/pusht"
export PUSHT_HDF5="$PUSHT_DATA_ROOT/pusht_expert_train.h5"
export PUSHT_MANIFEST="$PWD/output/manifests/pusht/clip90_example/manifest.json"
export RAW_RUN="$PWD/output/pusht/raw_example"
mkdir -p "$PUSHT_DATA_ROOT"
uv run --no-sync hf download quentinll/lewm-pusht --repo-type dataset \
  --include pusht_expert_train.h5.zst --local-dir "$PUSHT_DATA_ROOT/download"
zstd -d --keep "$PUSHT_DATA_ROOT/download/pusht_expert_train.h5.zst" \
  -o "$PUSHT_HDF5"
```

### 3. 学習・検証の分割を固定する

manifestはデータの場所、分割、行動の正規化統計を固定したJSONです。RawとBTで同じmanifestを使います。初回だけデータ全量のhashを計算します。

```bash
.venv/bin/python -m mylewm.data.prepare_pusht_comparison \
  --dataset "$PUSHT_HDF5" --manifest "$PUSHT_MANIFEST" --seed 3072
```

### 4. 設定を確認し、学習する

最初のコマンドはdry-runで、学習しません。出力先は未使用の名前にします。

```bash
.venv/bin/python -m mylewm.training.train \
  --manifest "$PUSHT_MANIFEST" --output "$RAW_RUN" --mode raw \
  --steps 140000 --batch-size 128 --workers 4 --no-pin-memory \
  --lr 5e-5 --warmup-steps 500 --save-every 20000 --val-every 20000 --seed 3072
```

表示された設定を確認したら、次で**実学習を開始**します。SSH切断後も続ける場合は、環境変数の設定前に`tmux new -s pusht-train`を実行します。`Ctrl-b`の後に`d`で離れ、`tmux attach -t pusht-train`で戻れます。

```bash
.venv/bin/python -m mylewm.training.train \
  --manifest "$PUSHT_MANIFEST" --output "$RAW_RUN" --mode raw \
  --steps 140000 --batch-size 128 --workers 4 --no-pin-memory \
  --lr 5e-5 --warmup-steps 500 --save-every 20000 --val-every 20000 --seed 3072 \
  --execute
```

ここまででRawの学習を始められます。BTは`--mode bt`と別の出力先に変え、他の条件を揃えます。ログはrun内の`metrics/version_0/metrics.csv`、評価用モデルは`step_N_object.ckpt`です。終了コード0・`completed.json`の140,000更新・最終checkpointを確認します。学習誤差は制御成功率ではありません。

## 作業別の正本

必要な作業のページを1つ開き、上から順に進めてください。概要と詳細の往復は不要です。

| 作業 | 手順 |
|---|---|
| PushTの取得・prepare・Raw／BT学習・保存・再開 | [学習手順](mylewm/docs/TRAINING.ja.md) |
| LIBEROの取得・prepare・世界モデル学習・再開 | [LIBERO学習](mylewm/docs/TRAINING_LIBERO.ja.md) |
| 世界モデル＋CEMの環境評価 | [評価手順](mylewm/docs/EVALUATION.ja.md) |
| BCの追加学習・再開・環境評価 | [BC手順](mylewm/docs/BEHAVIOR_CLONING.ja.md) |

仕組みから知りたい場合は[アルゴリズム](mylewm/docs/ALGORITHM.ja.md)と[実装](mylewm/docs/IMPLEMENTATION.ja.md)へ。コードは`src/mylewm/`、比較用LeWMは`lewm/`です。詳しい数学は[mylewm/docs/research](mylewm/docs/research/RESEARCH_REVIEW.ja.md)、実験の証拠は[mylewm/docs/reports](mylewm/docs/reports/README.md)、旧操作手順は削除し、必要時はGit履歴から参照できます。
