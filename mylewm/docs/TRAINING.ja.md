# PushT／LIBERO-10：初心者向け学習手順

## 新しいPushT経路：公式ライブラリへ委託（2026-09-09）

新規のRaw／BT比較には `mylewm/train.py` を使います。データ読込はstable-worldmodel、画像前処理と一段損失は公式LeWM、逆伝播・optimizer・schedulerはstable-pretraining、訓練ループ・CSVログ・checkpointはLightningへ委託します。BT固有の処理は学習専用Tと正則化分岐です。

これは新レシピ `pusht_spt_v1` です。既存10万更新の再現経路と互換ではありません。重み・manifest・評価結果は保持しています。旧RBG専用処理は撤去し、共有ループを `training.py`、LIBERO入口を `train_libero.py` へ改名しました。過去runの厳密再開は開始時のGit版が必要です。LIBEROは後半の共有経路を使います。本学習・環境評価はユーザーの明示依頼時だけ実行します。

### 変更する条件・維持する条件

| 項目 | 新しいRaw／BTで共通の条件 |
|---|---|
| 分割・行動統計 | 既存manifestのエピソード分離80/10/10とtrain-only統計を維持 |
| 画像・行動の取得 | SWMの4フレーム・frameskip5。20ステップ分を取得し、予測には先頭3行動chunkを使う。旧ローダーの16ステップ分とは末尾の有効クリップ数が異なる |
| 抽出 | PyTorchのepoch単位シャッフル・drop_last。旧経路の更新ごとの復元抽出から変更 |
| 予測損失 | 公式 `lejepa_forward` をそのまま使用。未来教師への勾配を維持 |
| 正則化 | 公式SIGRegを両方式ともFP32で計算。BTだけTを追加。射影乱数はモデルから分離 |
| 更新予算 | 両方式100,000、batch128、seed3072、同一の新規E/A/F初期値 |
| 学習率 | SPTのwarmup500＋cosine、最大5e-5、目標最小0。初回LRは0、最終更新に使うLRはごく小さい正値。旧添字規約とは異なる |
| optimizer | AdamW、全重みweight decay .001。モデルとTの勾配を別々にノルム1でclip |
| validation | manifestにある固定validationケースを使用（既存PushTは256件）。無ければvalidationエピソードの全有効クリップ |
| 推論 | 公式E/A/Fのみ。Tなし。訓練統計bufferを保持し、既存評価シェルで使用可能 |

各runにmanifest・データ・初期モデル・ソース・主要依存ソースのhash、依存版、レシピを記録します。新Rawと新BTを比較し、旧10万BTとの違いをTだけの効果とは解釈しません。公式配布モデルの完全な学習再現とも呼びません。

### 設定確認と実行

このPCの既存環境を前提とします。新規実験用manifestが無い場合だけ、後半の「3. 学習・検証の分割ファイルを作る」を実施してください。`output/manifests/pusht/manifest.json`は新規作成後だけ使えるパスです。存在する場合は内容を変更せずに使い、保存済みBTの評価には使いません。

```bash
cd "$(git rev-parse --show-toplevel)"
uv run python mylewm/train.py --help
# 既定はdry-run。学習・GPU初期化・出力作成は行わない
uv run python mylewm/train.py --mode bt \
  --manifest output/manifests/pusht/manifest.json \
  --output output/pusht/spt_bt --seed 3072
```

dry-runは設定・ソース識別の確認までで、全データhash・重み・学習動作の保証ではありません。出力先は未使用名を指定し、先にmkdirしないでください。

実際に新規学習する場合だけ、同じコマンドに `--execute` を追加します。GPUの決定論設定も指定します。

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run python mylewm/train.py \
  --mode bt --manifest output/manifests/pusht/manifest.json \
  --output output/pusht/spt_bt --steps 100000 --seed 3072 --execute
```

Rawは `--mode raw --output output/pusht/spt_raw` に変え、その他は同じにします。`3072`はここで明示した学習乱数seedで、run名から推測する値ではありません。実行後は各runの`config.json`にも記録されます。同時起動しません。まず短期動作確認をするなら、別出力名で `--steps 100 --warmup-steps 10 --batch-size 16 --workers 0 --save-every 50` を両方式に揃えて指定します。短期checkpointを10万更新へ延長する用途のresumeはできません。

### ログ・保存・再開

- `metrics/version_N/metrics.csv`：LightningのCSVログ。`update`は1始まりの更新番号、`lr_used`はその更新に使った学習率、`fit/loss`・`fit/pred_loss`・`fit/sigreg_loss`が損失。CSVの標準`step`は0始まりなので区別します。
- `step_N.ckpt`／`last.ckpt`：Lightning形式のモデル・T・optimizer・scheduler・乱数・再開条件。既定5,000更新ごとと最終時点に保存。
- `step_N_object.ckpt`：Tなしの推論専用モデル。従来の `evaluate_pusht.sh` に渡す形式です。新重みの実環境成功率はまだ未測定です。
- `completed.json`：訓練ループ正常完了後のみ生成。例外・途中停止を成功扱いしません。

100,000更新を完走した後は、`completed.json`の`state=completed`と`step=100000`、`step_100000_object.ckpt`の両方を確認してから、[PushT評価手順](EVALUATE_PUSHT.ja.md#2-新しいrawbt-runを評価する)へ進みます。`last.ckpt`や途中の`step_N_object.ckpt`を最終成績として評価しないでください。評価は学習が終了してGPUを使っていないときに、別の新規出力先で実行します。

現在の `monitor_training.sh` は旧JSONL形式用で、新CSVのloss表示には使いません。端末ログ、または `tail -f output/pusht/spt_bt/metrics/version_0/metrics.csv` で確認します。定期監視サービス・外部trackerは起動しません。ログ・出力はGit対象外です。

中断後は**同じ設定・総更新数**で、`--resume 元run/last.ckpt --output 新しい未使用run --execute` を指定します。元のrunは上書きせず、ログは新runへ分離します。自己生成した信頼済みcheckpointだけを使ってください。旧 `resume.pt`、完了済みcheckpoint、変更したレシピでの再開は拒否します。

再開位置だけは薄い補助処理で補完します。PyTorchのepochシャッフルから消費済みバッチを読み飛ばし、prefetch位置ではなくLightningの完了更新数で復元します。CPU小型モデル・worker0/2で連続6更新と3更新＋再開が一致しましたが、実LeWM全体のGPU長期再開まで保証したものではありません。

## 共通のデータ準備とLIBERO学習

新規PushTの学習は冒頭の `train.py` に統一します。以下の `training.py prepare` は分割作成だけです。LIBEROは `train_libero.py` から共有ループを使用します。旧RBG、`--blocks`、`--cross-weight`、廃止済み初期値ファイルの読込引数 `--initialization` はありません。Raw/TC/BTは同じseedから初期化し、記録された初期モデルhashで照合します。過去の完全な手順・再開は[整理記録](CLEANUP.ja.md)のGit履歴を参照してください。

公式LeWM側を学習したい場合は[公式PushT学習の説明書](../../lewm/TRAIN_PUSHT.ja.md)を参照してください。公式trainerの経路と、公平なBT比較向けRaw経路を分けています。

この手順は**このPCの既存`.venv`とダウンロード済みデータを使う手順**です。別PCへの環境構築・ダウンロードを自動化したものではありません。以下のコマンドを順番に実行しますが、PushTとLIBEROの学習は一方ずつにしてください。

PushTの100,000更新は完了済みです。以下は新規実験の手順で、文書整理を理由に学習を自動起動しません。既存の重み・出力を上書きしないでください。

### 0. PushT用の`uv`環境を構築する

これまでの手順は既存`.venv`を前提にしており、依存パッケージを導入するコマンドが欠けていました。リポジトリ直下の`pyproject.toml`と`uv.lock`に、PushT用の公式世界モデル、PyTorch、Lightning、Hydra、HDF5周辺を固定しています。GB10/CUDA 13向けPyTorch indexも同ファイルで選択します。

```bash
uv sync --locked
uv run python -c 'import hydra, h5py, lightning, stable_pretraining, stable_worldmodel, torch; print("hydra", hydra.__version__, "torch", torch.__version__)'
uv run python mylewm/train.py --help
```

`ModuleNotFoundError: No module named 'hydra'` の配布名は`hydra-core`であり、上の`train` extraに含まれます。最後の2コマンドが通れば、少なくともPushT trainerのimportとCLIまで確認できています。CUDAを使わないPCはこのlockfileをそのまま使わず、対応するPyTorch backendで別途lockを作成してください。

LIBEROのPython依存は`uv sync --locked --group libero`で追加します。実環境評価にはさらにLIBERO本体（この作業木では`external/libero`）、OSMesa、`LIBERO_CONFIG_PATH`、10個のHDF5データが必要です。公式LIBEROのPython 3.8 / CUDA 11.3手順をこのPython 3.12 / CUDA 13環境へ一般化した完全な構築手順は未検証です。したがって、新しいPCでLIBERO評価まで行う場合は、ここにない依存を推測で導入せず、対応環境を先に検証してください。

### 1. データを任意の保存先へ取得する

データはGitに入れません。次の例では`BT_SIGREG_DATA_ROOT`だけを自分の大容量ストレージへ変更します。PushTの公式データは圧縮済みで約13.1GB、展開後は約46.3GBです。LIBERO-10は10個で約13.7GBです。必要な空き容量とネットワークを確認してから実行してください。

```bash
export BT_SIGREG_DATA_ROOT=/absolute/path/to/bt-sigreg-data
export PUSHT_HDF5="$BT_SIGREG_DATA_ROOT/pusht/pusht_expert_train.h5"
export LIBERO10_DATASET="$BT_SIGREG_DATA_ROOT/libero_10"
mkdir -p "$(dirname "$PUSHT_HDF5")"

# PushT: 公式 LeWM dataset を取得して HDF5 を展開する。
uv run hf download quentinll/lewm-pusht --repo-type dataset \
  --local-dir "$BT_SIGREG_DATA_ROOT/pusht-source"
zstd -d --stdout "$BT_SIGREG_DATA_ROOT/pusht-source/pusht_expert_train.h5.zst" \
  > "$PUSHT_HDF5"

# LIBERO-10: 公式 LIBERO dataset の必要な10ファイルだけを取得する。
uv run hf download yifengzhu-hf/LIBERO-datasets --repo-type dataset \
  --include 'libero_10/*' --local-dir "$BT_SIGREG_DATA_ROOT"
```

`zstd: command not found`ならOS側のZstandardコマンドを導入してから、PushTの展開行だけを再実行します。ダウンロード済みファイルの削除・再取得は自動では行いません。公式LIBERO Python scriptの`--datasets`には`libero_10`の選択肢がないため、この用途では上の公式Hugging Face repositoryを直接指定します。

### 2. 作業フォルダ・環境・データを確認する

全コマンドはリポジトリ直下で実行します。別の場所に置いた場合は、最初の`cd`だけ実際の場所に変更してください。仮想環境のactivateは不要です。

```bash
cd "$(git rev-parse --show-toplevel)"
export PUSHT_HDF5=/absolute/path/to/pusht_expert_train.h5
export LIBERO10_DATASET=/absolute/path/to/libero_10
uv run python --version
uv run python mylewm/training.py --help
uv run python mylewm/train_libero.py --help
nvidia-smi
ls -lh "$PUSHT_HDF5"
ls "$LIBERO10_DATASET"/*.hdf5
```

PushTは約46.3GBのHDF5が1つ、LIBERO-10はタスクごとのHDF5が10個必要です。`/absolute/path/to/...`は実際の任意の保存場所へ置き換えてください。対象benchmarkのデータが見つからなければ先へ進まないでください。`uv run python`やimportが見つからない場合も環境準備が必要です。任意の最新版をまとめてインストールして既存環境を上書きしないでください。

学習では実画像・実行行動から一段先の潜在状態を予測します。LIBEROは2つの実カメラを使い、10タスクで1つのモデルを共有します。HDF5からの学習にはMuJoCo/OSMesaの起動は不要です。シミュレータが必要なのは後述の制御評価です。

### 3. 学習・検証の分割ファイルを作る（最初の1回だけ）

`manifest.json`はデータの場所、学習/検証/テストの分割、行動の正規化統計を記録するファイルです。データ本体を複製・生成する処理ではありません。新しいmanifestを**現在の実フォルダで**作り、旧フォルダ名の一時リンクに依存しないようにします。

PushT用：

```bash
uv run python mylewm/training.py prepare \
  --dataset "$PUSHT_HDF5" \
  --manifest output/manifests/pusht/manifest.json
```

LIBERO-10用：

```bash
uv run python mylewm/train_libero.py prepare \
  --dataset "$LIBERO10_DATASET" \
  --manifest output/manifests/libero10/manifest.json
```

完了すると分割件数が表示されます。manifestにはデータの**絶対パス**、サイズ、更新時刻（PushTはhashも）を記録し、学習・評価はその記録を使います。LIBEROは隣に`files.json`も作ります。既に作成済みならこの工程を飛ばしてください。`FileExistsError`は上書き防止です。学習開始後はmanifest・データ・フォルダ名を変更しないでください。移転後の**新規実験**は新しいmanifestを作れますが、途中再開のために元のmanifestを書き換えると互換性チェックで拒否されます。

### 4. まず100ステップだけ動作確認する

`--steps`はoptimizerの呼出し数、`--batch-size`は1回に使うクリップ数です。出力先は未使用の名前にします。**runフォルダ自体を先にmkdirしないでください**。trainerが作成します。

PushT（BT例。Rawは`--mode raw --output output/pusht/spt_raw_smoke`だけを変更）：

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run python mylewm/train.py \
  --mode bt --manifest output/manifests/pusht/manifest.json \
  --output output/pusht/spt_bt_smoke \
  --steps 100 --batch-size 16 --workers 0 --seed 3072 \
  --warmup-steps 10 --save-every 50 --execute
```

LIBERO-10：

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run python mylewm/train_libero.py train \
  --mode bt --manifest output/manifests/libero10/manifest.json \
  --output output/libero10/bt_smoke \
  --steps 100 --batch-size 16 --workers 0 --seed 3072 \
  --warmup-steps 10 --lr 5e-5 --min-lr 0 \
  --save-every 50 --diagnostics-every 25 --deterministic
```

最初に全データのハッシュを確認するので、すぐにstepが出なくても停止とは限りません。`hashing_data_for_training_contract`の後に`start`、`step`が出ることを確認します。最後にstep100と`resume.pt`があれば短期処理が終了しています。エラーで終了していないことも確認してください。loss低下だけでは制御性能やマルチタスク能力を評価できません。

### 5. 本学習を開始する

まず端末を閉じても実行を保持するため、既存のtmuxで作業用セッションを作ります。このPCではtmuxを確認済みです。

```bash
tmux new -s bt-training
cd "$(git rev-parse --show-toplevel)"
```

短期学習のcheckpointは使わず、新規初期値から始めます。

PushT・10万ステップ（BT例。Rawは`--mode raw --output output/pusht/spt_raw`だけを変更）：

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run python mylewm/train.py \
  --mode bt --manifest output/manifests/pusht/manifest.json \
  --output output/pusht/spt_bt \
  --steps 100000 --batch-size 128 --workers 4 --seed 3072 \
  --warmup-steps 500 --save-every 5000 --execute
```

LIBERO-10・10万ステップ（本学習のこの設定はまだ完走検証していません）：

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run python mylewm/train_libero.py train \
  --mode bt --manifest output/manifests/libero10/manifest.json \
  --output output/libero10/bt_train \
  --steps 100000 --batch-size 128 --workers 4 --seed 3072 \
  --warmup-steps 500 --lr 5e-5 --min-lr 0 \
  --bt-depth 2 --bt-kappa .2 --bt-hidden 192 \
  --save-every 5000 --diagnostics-every 1000 --deterministic
```

`Ctrl-b`を押して離し、次に`d`を押すとtmuxから離れます。戻るには`tmux attach -t bt-training`。学習画面でのCtrl-Cは**学習そのものを中断**します。tmuxは端末切断対策で、OS再起動や停電後に学習を自動再開するものではありません。

これらは単独学習の手順です。Raw/TCとの厳密な比較では同一の未学習初期重み・データ順・batch・予算等をそろえます。旧初期値生成CLIは削除しました。新規PushTの比較は冒頭の新経路で共通seedから初期化し、configの初期モデルhashを照合します。完了済みPushTの開始コマンドは[実行記録](reports/PUSHT_TRAINING_100K.ja.md)に残しています。

### 6. 別の端末から進捗を見る

PushTの新経路は冒頭のCSVを確認します。LIBEROのJSONLログを見る場合は次を使います。

```bash
bash mylewm/tools/monitor_training.sh --run output/libero10/bt_train
```

100ステップ確認を見る場合は`bt_train`を`bt_smoke`に変更。`--once`を付けると1回表示、`--interval 10`で10秒間隔です。`--run`を省略した既定画面は**完了済みPushT run**なので、新規実験では必ず指定してください。tmuxで起動した学習にはsystemd serviceがないため、`--unit`は付けません。

| 表示・ファイル | 意味 |
|---|---|
| Step / loss / Loss mean | 進捗、最新バッチの損失、直近最大100件の平均 |
| prediction / SIGReg | 予測誤差とGaussian正則化。小さいだけで性能が良いとは限らない |
| Grad / T grad / LR | 勾配ノルムと学習率。warmup後にLRが減少し、最終ステップは0 |
| Validation / T diagnostics | 最新の疎な診断値。どのstepで測った値かを併記 |
| `metrics.jsonl` | 全ステップの数値ログ。学習中に編集・削除しない |
| `config.json` / `budget.json` | 設定・データ/コードの識別情報と学習量 |
| `initialization.pt` | 初期状態の記録。再開時の照合にも必要 |
| `resume.pt` | 最新保存時点のE/A/F・T・optimizer・乱数等。途中再開用 |
| `step_5000_object.ckpt`など | その時点の推論モデル。Tを含まず、途中再開には使わない |

出力はすべて`output/`以下でGit対象外です。通常の数値ログは自動保存されますが、このtmux手順は端末の全stdout/stderrを別のconsole.logへ自動保存するものではありません。エラー発生時は端末のtracebackも残してください。

### 7. 中断した学習を再開する

学習プロセスが終了していることと、対象runに`resume.pt`があることを確認します。**開始時と同じコマンドの末尾に`--resume`だけを追加**して実行してください。これは改名後に開始したLIBERO run向けで、`train_libero.py`を使います。PushT新経路は冒頭のLightning再開手順です。改名前のrunは開始時のGit版で再開し、ソース照合を解除しないでください。

- `--output`は元のrunのまま。`--steps`も元の総数のままです（残りステップ数ではありません）。
- ほかの引数・manifest・データ・学習ソース・環境を変えないでください。`resume mismatch`を無理に解除してはいけません。
- 保存は本学習なら5,000ステップごと。保存前の未確定分はやり直しになります。初回保存前に停止した場合は、未使用の出力名で新規学習し直してください。
- 100ステップの確認runを`--steps 100000 --resume`で延長することはできません。短期診断と本学習ではスケジュールが違います。
- `torch.load`を使うため、自分で作成した信頼済みcheckpointだけを使ってください。不明な配布重みを再開ファイルにしないでください。

### よくある問題

| 症状 | 対処 |
|---|---|
| `FileExistsError` | 新規学習なら未使用の出力名にする。再開なら保存済みrunを確認して`--resume`。既存成果を消して回避しない |
| データが見つからない | `--manifest`と内部のパスを確認。稼働中のmanifestを書き換えない。新規実験なら現フォルダでprepareする |
| `warmup_steps`のエラー | warmupは総steps未満が必要。短期確認は100/10、本学習は100000/500 |
| CUDA out of memory | 他の学習を重複起動していないか確認。新規runとしてbatchを下げる。途中resumeのbatchを勝手に変更しない。特にLIBERO batch128はこのPCで長期未検証 |
| NaN/Inf・10分以上進まない | 端末の例外、プロセス、データ読込を確認し報告。自動再起動や設定変更で隠さない |
| lossが短期確認より大きい | SIGRegの値はbatch数にも依存する。batch16と128の生lossを直接比較しない |

学習が終わっても、PushT/LIBEROの**成功率評価は別工程**です。[PushT評価手順](EVALUATE_PUSHT.ja.md)・[LIBERO評価手順](EVALUATE_LIBERO.ja.md)と[検証記録](VALIDATION.ja.md)を参照してください。
