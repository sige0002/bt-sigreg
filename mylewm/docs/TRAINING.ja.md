# PushT／LIBERO-10：初心者向け学習手順

## 新しいPushT経路：公式ライブラリへ委託（2026-09-09）

新規のRaw／BT比較には `mylewm/train.py` を使います。データ読込はstable-worldmodel、画像前処理と一段損失は公式LeWM、逆伝播・optimizer・schedulerはstable-pretraining、訓練ループ・CSVログ・checkpointはLightningへ委託します。BT固有の処理は学習専用Tと正則化分岐です。

これは新レシピ `pusht_spt_v1` です。既存10万更新の再現経路と互換ではありません。既存の `train_rbg.py`、重み・manifest・評価結果は変更していません。LIBEROは後半の従来経路を引き続き使います。今回、本学習は開始していません。

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

このPCの既存環境を前提とします。新規manifestが無い場合だけ、後半の「2. 学習・検証の分割ファイルを作る」を実施してください。既存manifestは書き換えません。

```bash
cd /home/USER/bt-sigreg
.venv/bin/python mylewm/train.py --help
# 既定はdry-run。学習・GPU初期化・出力作成は行わない
.venv/bin/python mylewm/train.py --mode bt \
  --manifest output/manifests/pusht/manifest.json \
  --output output/pusht/spt_bt_s3072
```

dry-runは設定・ソース識別の確認までで、全データhash・重み・学習動作の保証ではありません。出力先は未使用名を指定し、先にmkdirしないでください。

実際に新規学習する場合だけ、同じコマンドに `--execute` を追加します。GPUの決定論設定も指定します。

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python mylewm/train.py \
  --mode bt --manifest output/manifests/pusht/manifest.json \
  --output output/pusht/spt_bt_s3072 --steps 100000 --execute
```

Rawは `--mode raw --output output/pusht/spt_raw_s3072` に変え、その他は同じにします。同時起動しません。まず短期動作確認をするなら、別出力名で `--steps 100 --warmup-steps 10 --batch-size 16 --workers 0 --save-every 50` を両方式に揃えて指定します。短期checkpointを10万更新へ延長する用途のresumeはできません。

### ログ・保存・再開

- `metrics/version_N/metrics.csv`：LightningのCSVログ。`update`は1始まりの更新番号、`lr_used`はその更新に使った学習率、`fit/loss`・`fit/pred_loss`・`fit/sigreg_loss`が損失。CSVの標準`step`は0始まりなので区別します。
- `step_N.ckpt`／`last.ckpt`：Lightning形式のモデル・T・optimizer・scheduler・乱数・再開条件。既定5,000更新ごとと最終時点に保存。
- `step_N_object.ckpt`：Tなしの推論専用モデル。従来の `evaluate_pusht.sh` に渡す形式です。新重みの実環境成功率はまだ未測定です。
- `completed.json`：訓練ループ正常完了後のみ生成。例外・途中停止を成功扱いしません。

現在の `monitor_training.sh` は旧JSONL形式用で、新CSVのloss表示には使いません。端末ログ、または `tail -f output/pusht/spt_bt_s3072/metrics/version_0/metrics.csv` で確認します。定期監視サービス・外部trackerは起動しません。ログ・出力はGit対象外です。

中断後は**同じ設定・総更新数**で、`--resume 元run/last.ckpt --output 新しい未使用run --execute` を指定します。元のrunは上書きせず、ログは新runへ分離します。自己生成した信頼済みcheckpointだけを使ってください。旧 `resume.pt`、完了済みcheckpoint、変更したレシピでの再開は拒否します。

再開位置だけは薄い補助処理で補完します。PyTorchのepochシャッフルから消費済みバッチを読み飛ばし、prefetch位置ではなくLightningの完了更新数で復元します。CPU小型モデル・worker0/2で連続6更新と3更新＋再開が一致しましたが、実LeWM全体のGPU長期再開まで保証したものではありません。

## 従来経路：LIBEROと既存PushT実験の再現

以下のPushTコマンドは旧 `controlled_training_v2` 用です。新しい公式ライブラリ経路と混ぜません。LIBERO移行と旧コードの撤去は、まだ行っていません。

公式LeWM側を学習したい場合は[公式PushT学習の説明書](../../lewm/TRAIN_PUSHT.ja.md)を参照してください。公式trainerの経路と、公平なBT比較向けRaw経路を分けています。

この手順は**このPCの既存`.venv`とダウンロード済みデータを使う手順**です。別PCへの環境構築・ダウンロードを自動化したものではありません。以下のコマンドを順番に実行しますが、PushTとLIBEROの学習は一方ずつにしてください。

PushTの100,000更新は完了済みです。以下は新規実験の手順で、文書整理を理由に学習を自動起動しません。既存の重み・出力を上書きしないでください。

### 1. 作業フォルダ・環境・データを確認する

全コマンドはリポジトリ直下で実行します。別の場所に置いた場合は、最初の`cd`だけ実際の場所に変更してください。仮想環境のactivateは不要です。

```bash
cd /home/USER/bt-sigreg
.venv/bin/python --version
.venv/bin/python mylewm/train_rbg.py --help
.venv/bin/python mylewm/train_rbg_libero.py --help
nvidia-smi
ls -lh .cache/stable-wm/datasets/pusht_expert_train.h5
ls .cache/libero-datasets/libero_10/*.hdf5
```

PushTは約46.3GBのHDF5が1つ、LIBERO-10はタスクごとのHDF5が10個必要です。対象benchmarkのデータが見つからなければ先へ進まないでください。`.venv/bin/python`やimportが見つからない場合も環境準備が必要です。任意の最新版をまとめてインストールして既存環境を上書きしないでください。別PCの完全な再現環境用lockfileは現状ありません。

学習では実画像・実行行動から一段先の潜在状態を予測します。LIBEROは2つの実カメラを使い、10タスクで1つのモデルを共有します。HDF5からの学習にはMuJoCo/OSMesaの起動は不要です。シミュレータが必要なのは後述の制御評価です。

### 2. 学習・検証の分割ファイルを作る（最初の1回だけ）

`manifest.json`はデータの場所、学習/検証/テストの分割、行動の正規化統計を記録するファイルです。データ本体を複製・生成する処理ではありません。新しいmanifestを**現在の実フォルダで**作り、旧フォルダ名の一時リンクに依存しないようにします。

PushT用：

```bash
.venv/bin/python mylewm/train_rbg.py prepare \
  --dataset .cache/stable-wm/datasets/pusht_expert_train.h5 \
  --manifest output/manifests/pusht/manifest.json
```

LIBERO-10用：

```bash
.venv/bin/python mylewm/train_rbg_libero.py prepare \
  --dataset .cache/libero-datasets/libero_10 \
  --manifest output/manifests/libero10/manifest.json
```

完了すると分割件数が表示されます。LIBEROは隣に`files.json`も作ります。既に作成済みならこの工程を飛ばしてください。`FileExistsError`は上書き防止です。学習開始後はmanifest・データ・フォルダ名を変更しないでください。移転後の**新規実験**は新しいmanifestを作れますが、途中再開のために元のmanifestを書き換えると互換性チェックで拒否されます。

### 3. まず100ステップだけ動作確認する

下のどちらかを実行します。`--steps`はoptimizerの呼出し数、`--batch-size`は1回に使うクリップ数です。出力先は未使用の名前にします。**runフォルダ自体を先にmkdirしないでください**。trainerが作成します。

PushT：

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python mylewm/train_rbg.py train \
  --mode bt --manifest output/manifests/pusht/manifest.json \
  --output output/pusht/bt_smoke_s3072 \
  --steps 100 --batch-size 16 --workers 0 --seed 3072 \
  --warmup-steps 10 --lr 5e-5 --min-lr 0 \
  --save-every 50 --diagnostics-every 25 --deterministic
```

LIBERO-10：

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python mylewm/train_rbg_libero.py train \
  --mode bt --manifest output/manifests/libero10/manifest.json \
  --output output/libero10/bt_smoke_s3072 \
  --steps 100 --batch-size 16 --workers 0 --seed 3072 \
  --warmup-steps 10 --lr 5e-5 --min-lr 0 \
  --save-every 50 --diagnostics-every 25 --deterministic
```

最初に全データのハッシュを確認するので、すぐにstepが出なくても停止とは限りません。`hashing_data_for_training_contract`の後に`start`、`step`が出ることを確認します。最後にstep100と`resume.pt`があれば短期処理が終了しています。エラーで終了していないことも確認してください。loss低下だけでは制御性能やマルチタスク能力を評価できません。

### 4. 本学習を開始する

まず端末を閉じても実行を保持するため、既存のtmuxで作業用セッションを作ります。このPCではtmuxを確認済みです。

```bash
tmux new -s bt-training
cd /home/USER/bt-sigreg
```

その中で、**どちらか一方**を実行してください。ここでの出力名`bt_train_s3072`は完了済みrunとは別です。短期学習のcheckpointは使わず、新規初期値から始めます。

PushT・10万ステップ：

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python mylewm/train_rbg.py train \
  --mode bt --manifest output/manifests/pusht/manifest.json \
  --output output/pusht/bt_train_s3072 \
  --steps 100000 --batch-size 128 --workers 4 --seed 3072 \
  --warmup-steps 500 --lr 5e-5 --min-lr 0 \
  --bt-depth 2 --bt-kappa .2 --bt-hidden 192 \
  --save-every 5000 --diagnostics-every 1000 --deterministic
```

LIBERO-10・10万ステップ（本学習のこの設定はまだ完走検証していません）：

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python mylewm/train_rbg_libero.py train \
  --mode bt --manifest output/manifests/libero10/manifest.json \
  --output output/libero10/bt_train_s3072 \
  --steps 100000 --batch-size 128 --workers 4 --seed 3072 \
  --warmup-steps 500 --lr 5e-5 --min-lr 0 \
  --bt-depth 2 --bt-kappa .2 --bt-hidden 192 \
  --save-every 5000 --diagnostics-every 1000 --deterministic
```

`Ctrl-b`を押して離し、次に`d`を押すとtmuxから離れます。戻るには`tmux attach -t bt-training`。学習画面でのCtrl-Cは**学習そのものを中断**します。tmuxは端末切断対策で、OS再起動や停電後に学習を自動再開するものではありません。

これらは単独学習の手順です。Raw/TCとの厳密な比較では同一の未学習初期重み・データ順・batch・予算等をそろえます。`tools/create_shared_initialization.py`は現状**PushT用**で、生成した重みをLIBEROへ渡してはいけません。完了済みPushTの開始コマンドは[実行記録](reports/PUSHT_TRAINING_100K.ja.md)に残しています。

### 5. 別の端末から進捗を見る

```bash
cd /home/USER/bt-sigreg
# 上の手順で新しく始めたPushTを見る
bash mylewm/tools/monitor_training.sh --run output/pusht/bt_train_s3072
```

LIBEROを見る場合は次を使います。

```bash
bash mylewm/tools/monitor_training.sh --run output/libero10/bt_train_s3072
```

100ステップ確認を見る場合は`bt_train_s3072`を`bt_smoke_s3072`に変更。`--once`を付けると1回表示、`--interval 10`で10秒間隔です。`--run`を省略した既定画面は**完了済みPushT run**なので、新規実験では必ず指定してください。tmuxで起動した学習にはsystemd serviceがないため、`--unit`は付けません。

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

### 6. 中断した学習を再開する

学習プロセスが終了していることと、対象runに`resume.pt`があることを確認します。**開始時と同じコマンドの末尾に`--resume`だけを追加**して実行してください。PushTは`train_rbg.py`、LIBEROは`train_rbg_libero.py`を引き続き使います。

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

学習が終わっても、PushT/LIBEROの**成功率評価は別工程**です。以下の評価ツールと[検証記録](VALIDATION.ja.md)を参照してください。
