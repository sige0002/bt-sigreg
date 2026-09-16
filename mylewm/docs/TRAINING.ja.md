# 学習手順：環境構築からPushTの学習・再開まで

このページだけで、公開PushTデータの取得 → 分割 → 設定確認 → Raw／BTの学習 → 保存結果の確認 → 再開まで進めます。既に終えた工程は飛ばしてください。LIBEROはデータ形式と再開方法が異なるため、別の正本[LIBERO学習](TRAINING_LIBERO.ja.md)にまとめています。

**読み方：1〜7が新規学習、8は中断した場合だけです。** コマンドはBashで、同じ端末から順番に実行します。実験の進捗や過去のrun名はこの手順に含めません。

<a id="setup"></a>
## 1. 共通環境を用意する

必要なものはLinux、Git、uv、Zstandard、Python 3.12、NVIDIA GPUと対応ドライバです。このリポジトリのlockfileはPyTorch 2.9.1／CUDA 13.0用です。GPU名やクラウドの保存先は固定していません。別CUDA版やCPU専用環境はこのlockfileで動作確認済みとは扱いません。

Ubuntu/DebianでGit・curl・展開ツールがない場合だけ導入します。

```bash
sudo apt-get update
sudo apt-get install -y git curl zstd tmux
```

uvがない場合だけ導入します。これはOSのGPUドライバを導入する操作ではありません。

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

まだリポジトリがなければ取得します。既に作業木がある場合は、そのルートに移動してください。

```bash
git clone https://github.com/sige0002/bt-sigreg.git
cd bt-sigreg
```

SSH越しの長時間学習では、ここで`tmux new -s pusht-train`を実行し、その中で以降の変数設定と学習を行います。`Ctrl-b`の後に`d`で接続を離れ、再接続後は`tmux attach -t pusht-train`で戻れます。

初回だけ専用環境を作ります。既存の学習で使用中の環境には同期しません。

```bash
export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
uv python install 3.12
uv sync --locked
export CUBLAS_WORKSPACE_CONFIG=:4096:8
.venv/bin/python -c 'import torch, mylewm; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(), "CUDAが利用できません"'
.venv/bin/python -m mylewm.training.train --help
```

最後の2つが成功したら先へ進みます。CUDAの確認に失敗する場合、対応ドライバとGPUが必要です。データ処理や学習オプションを変更して回避しないでください。以後`.venv/bin/python`と`uv run --no-sync`を使い、起動時の依存更新を避けます。

## 2. 保存先を決め、公開データを取得する

次の変数をこの端末で設定します。別ディスクにデータを置く場合は`PUSHT_DATA_ROOT`だけ変更できます。固定のマウント先は前提にしていません。

```bash
export PUSHT_DATA_ROOT="$PWD/.cache/pusht"
export PUSHT_HDF5="$PUSHT_DATA_ROOT/pusht_expert_train.h5"
export PUSHT_MANIFEST="$PWD/output/manifests/pusht/clip90_example/manifest.json"
export RAW_RUN="$PWD/output/pusht/raw_example"
export BT_RUN="$PWD/output/pusht/bt_example"
mkdir -p "$PUSHT_DATA_ROOT"
```

圧縮ファイル約13 GBと展開後約46 GBに加え、checkpoint用の空き容量が必要です。取得済みならダウンロード・展開は飛ばし、`PUSHT_HDF5`をそのファイルへ向けます。

```bash
uv run --no-sync hf download quentinll/lewm-pusht --repo-type dataset \
  --include pusht_expert_train.h5.zst --local-dir "$PUSHT_DATA_ROOT/download"
zstd -d --keep "$PUSHT_DATA_ROOT/download/pusht_expert_train.h5.zst" \
  -o "$PUSHT_HDF5"
ls -lh "$PUSHT_HDF5"
```

同じ名前の展開済みファイルへ上書きしません。例の`.cache/`はGit対象外です。別の保存先を選んだ場合も、圧縮ファイルやデータをGitへ追加しないでください。

## 3. 学習用manifestを作る（prepare）

manifestはデータの場所・分割・正規化統計を固定するJSONです。RawとBTで**同じファイル**を使います。

```bash
.venv/bin/python -m mylewm.data.prepare_pusht_comparison \
  --dataset "$PUSHT_HDF5" --manifest "$PUSHT_MANIFEST" --seed 3072
```

公開データでは全1,981,721クリップ、学習1,783,549、検証198,172になる想定です。4フレーム・間隔5のクリップを90%／10%に分けます。初回は全データのSHA-256を計算するため時間がかかります。既存manifestには上書きできません。

画像が重なる隣接クリップも別の分割へ入るので、検証誤差を「未知episodeでの成功率」と解釈しません。prepare後はデータとmanifestを移動・編集せずに使います。

## 4. 学習条件を理解する

| 項目 | 両方式に共通の設定 |
|---|---|
| 更新数 | 140,000。1更新は1バッチで重みを更新する処理 |
| バッチ | 128、端数バッチ破棄、単一GPU、勾配蓄積なし |
| 学習率 | 5e-5、最初の500更新で増加し、その後cosineで140,000時点に0 |
| 乱数 | seed3072。初期モデル・データ順を揃える |
| 保存・検証 | 20,000更新ごと |

このデータ条件の10エポック（学習集合10周）は139,330更新です。14万更新は670更新多い予算であり、論文Figure 18の完全再現とは呼びません。RawはLeWMのモデル・損失を共通trainerで学ぶ比較対象です。BTだけ学習専用変換Tが加わります。

## 5. 設定だけを確認する（dry-run）

次の2コマンドは学習せず、出力フォルダも作りません。`--output`はまだ存在しない場所にします。

```bash
.venv/bin/python -m mylewm.training.train \
  --manifest "$PUSHT_MANIFEST" --output "$RAW_RUN" --mode raw \
  --steps 140000 --batch-size 128 --workers 4 --no-pin-memory \
  --lr 5e-5 --warmup-steps 500 --save-every 20000 --val-every 20000 --seed 3072
.venv/bin/python -m mylewm.training.train \
  --manifest "$PUSHT_MANIFEST" --output "$BT_RUN" --mode bt \
  --steps 140000 --batch-size 128 --workers 4 --no-pin-memory \
  --lr 5e-5 --warmup-steps 500 --save-every 20000 --val-every 20000 --seed 3072
```

`Dry-run only`で終わることを確認します。dry-runだけではデータ読込やGPU学習の成功は確認できません。

## 6. 学習を開始する

**以下は実際に学習を開始します。** 末尾の`--execute`がdry-runとの違いです。通常は1本目の終了後に2本目を実行します。十分な資源で併走する場合は2つの端末を使い、それぞれ第1・2節の環境変数を同じ値に設定します。

```bash
.venv/bin/python -m mylewm.training.train \
  --manifest "$PUSHT_MANIFEST" --output "$RAW_RUN" --mode raw \
  --steps 140000 --batch-size 128 --workers 4 --no-pin-memory \
  --lr 5e-5 --warmup-steps 500 --save-every 20000 --val-every 20000 --seed 3072 \
  --execute
```

```bash
.venv/bin/python -m mylewm.training.train \
  --manifest "$PUSHT_MANIFEST" --output "$BT_RUN" --mode bt \
  --steps 140000 --batch-size 128 --workers 4 --no-pin-memory \
  --lr 5e-5 --warmup-steps 500 --save-every 20000 --val-every 20000 --seed 3072 \
  --execute
```

SSH接続を切っても続けるなら、実行前にtmuxなどの端末保持手段を用意します。学習端末のCtrl-Cは学習を中断します。開始したソース・環境を保存し、学習中に依存を同期しないでください。両runの`config.json`で`initial_model_sha256`と共通条件が一致することを確認します。

## 7. 進捗と完了を確認する

別端末では第2節の変数を設定し直し、次でログ末尾を見ます。

```bash
tail -n 3 "$RAW_RUN/metrics/version_0/metrics.csv"
tail -n 3 "$BT_RUN/metrics/version_0/metrics.csv"
```

`update`が更新数、lossは予測・正則化の誤差です。環境の成功率ではありません。

| 保存物 | 意味 |
|---|---|
| `config.json` | 初期値・学習設定・ソースと依存の識別情報 |
| `step_20000_object.ckpt`など | 保存時点の評価用世界モデル。Tを含まない |
| `step_20000.ckpt`など・`last.ckpt` | T・optimizer・学習率・乱数も含む再開用 |
| `completed.json` | 総更新数に到達した記録 |

学習コマンドの終了直後に`echo $?`で0を確認し、`completed.json`の`step=140000`・`state=completed`と最終object checkpointを照合します。

```bash
cat "$RAW_RUN/completed.json"
ls -lh "$RAW_RUN/step_140000_object.ckpt" "$RAW_RUN/last.ckpt"
```

BTも`RAW_RUN`を`BT_RUN`に置き換えて確認します。途中checkpointの存在だけでは学習完了ではありません。

## 8. 中断した学習を再開する（必要な場合だけ）

例としてRawを20,000更新の保存点から再開します。元のプロセスが終了していること、開始時と同じソース・環境・manifestであることを確認します。**新規出力先**と**学習再開用`.ckpt`**を指定します。

```bash
.venv/bin/python -m mylewm.training.train \
  --manifest "$PUSHT_MANIFEST" --output "$PWD/output/pusht/raw_example_resume1" --mode raw \
  --steps 140000 --batch-size 128 --workers 4 --no-pin-memory \
  --lr 5e-5 --warmup-steps 500 --save-every 20000 --val-every 20000 --seed 3072 \
  --resume "$RAW_RUN/step_20000.ckpt" --execute
```

BTは`--mode bt`、入力`$BT_RUN/step_20000.ckpt`、出力`bt_example_resume1`にします。`--steps`は残り回数ではなく元の総数140,000のままです。予算・batch・保存間隔などを変更すると再開照合で拒否されます。推論用`*_object.ckpt`と旧方式の`resume.pt`は使えません。保存前に止まった分は再実行されます。

## つまずいたとき

| 症状 | 確認すること |
|---|---|
| `FileExistsError` | 新規・再開とも出力先は未使用か。既存成果を削除して回避しない |
| データが見つからない | `PUSHT_HDF5`とmanifest内のパス。学習CLIへ渡すのはmanifest |
| CUDAが使えない | 第1節の確認、GPUとドライバ。マシン移行で同じ環境が使えるとは限らない |
| 再開が不一致で停止する | 開始時のコード・依存・元の総更新数と引数を復元する |
| 2万更新前にcheckpointがない | 保存間隔は2万更新。初回保存までは正常 |

ここまでで学習と再開は完結です。学習したモデルの制御能力を測る次の作業は[評価手順](EVALUATION.ja.md)で行います。
