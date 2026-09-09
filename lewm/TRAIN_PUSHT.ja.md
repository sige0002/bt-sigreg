# 公式LeWMをPushTで学習する：初心者向け手順

この説明書はローカル比較用の補足です。上流の説明は[README](README.md)、参照commitは[UPSTREAM.md](UPSTREAM.md)に記録しています。対象は**既存`.venv`とダウンロード済みPushT HDF5がある、このPC**です。別PCへの完全な環境構築手順・依存lockfileはまだ整備していません。

2026-09-09時点でBTの100,000更新は完了しています。この説明書の更新では公式学習を開始していません。新規実行前にGPU上の既存プロセスを確認し、重複起動しないでください。

## どちらの方法を使うか

2026-09-09追記：新規の同条件Raw／BT比較には、公式ライブラリへ委託した [新PushT経路](../mylewm/docs/TRAINING.ja.md#新しいpusht経路公式ライブラリへ委託2026-09-09) の `mylewm/train.py --mode raw` を使ってください。以下のBは既存 `controlled_training_v2` の再現用として保持しています。Aの公式trainer自体は変更していません。

| 方法 | 使う場面 | 注意 |
|---|---|---|
| A：`lewm/train.py` | 公式trainerの設定・処理に沿って学びたい | この環境へのHDF5指定変更が必要。設定展開まで確認し、学習完走・途中再開は未検証 |
| B：`mylewm/train_rbg.py --mode raw` | BTと同じデータ分割・更新予算で比較したい | 公式E/A/FとRaw SIGRegを使用するが、公式trainer・全レシピの完全再現ではない |

公式LeWMの損失は「次の潜在状態の予測誤差＋SIGReg」です。RawにはBTの追加写像Tはありません。公式配布checkpointの再現評価と、自分で新規学習する実験も別です。

## 共通の準備

リポジトリ直下で実行します。別の場所に置いた場合は`cd`を変更します。

```bash
cd /home/USER/bt-sigreg
.venv/bin/python --version
nvidia-smi
ls -lh .cache/stable-wm/datasets/pusht_expert_train.h5
```

このPCのデータは約46.3GB。見つからなければ学習を始めないでください。新しい学習出力は`output/`へ保存します（Git対象外）。不明なcheckpointを読み込む必要はなく、以下は新規初期値から始めます。

## A. 公式trainerを使う

### A1. この環境で必要な指定を理解する

手元の`config/train/data/pusht.yaml`は`pusht_expert_train.lance`ですが、インストール済みstable-worldmodel 0.0.6はHDF5ローダーです。**`data.dataset.name=pusht_expert_train.h5`を明示**します。拡張子なしの名前も、この版のresolverでは解決できません。公式READMEの記述より実際のコードを優先してください。

データと出力を分けるため、次の環境変数を使います。両方ともその端末で設定してください。

```bash
export LOCAL_DATASET_DIR="$PWD/.cache/stable-wm"
export STABLEWM_HOME="$PWD/output/pusht/official_native_trial"
```

`LOCAL_DATASET_DIR`の下の`datasets/pusht_expert_train.h5`が読み込まれます。`STABLEWM_HOME`は新しいモデル等の出力先です。既存の別実験に同じ出力先を使わないでください。データをこの出力先へコピーする必要はありません。

### A2. 学習せずに設定を確認する

```bash
.venv/bin/python lewm/train.py \
  data=pusht data.dataset.name=pusht_expert_train.h5 \
  subdir=lewm output_model_name=lewm wandb.enabled=false \
  hydra.run.dir=output/pusht/official_native_trial/hydra \
  --cfg job --resolve
```

これは設定表示だけで、学習は始まりません。手元でこの形式の設定展開を確認しました。`action_encoder.input_dim: ???`は学習関数がデータから設定する項目で、設定表示の段階だけでは異常としません。ここではWandBを無効にしているため、WandBアカウントは不要です。

### A3. 短い試運転を行う（未実行の例）

BTの稼働終了後、上の環境変数を設定した端末で実行します。

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python lewm/train.py \
  data=pusht data.dataset.name=pusht_expert_train.h5 \
  subdir=lewm output_model_name=lewm wandb.enabled=false \
  hydra.run.dir=output/pusht/official_native_trial/hydra \
  trainer.max_epochs=1 +trainer.max_steps=10 \
  +trainer.limit_train_batches=10 +trainer.limit_val_batches=2 \
  loader.batch_size=16 num_workers=0 \
  loader.persistent_workers=false loader.prefetch_factor=null
```

このコマンドは**構文・設定展開までの確認で、実学習の動作保証ではありません**。データ全体の統計取得などが先に走るため、10ステップでも起動が瞬時とは限りません。エラーが出た場合はtracebackを確認し、配布checkpointの再現ができたと扱わないでください。

`weights_epoch_1.pt`と`config.json`は、このコードでは`$STABLEWM_HOME/checkpoints/lewm/`へ保存する設計です。これはモデルのstate_dictであり、optimizer・乱数を含む再開用checkpointではありません。公式READMEの古い`_object.ckpt`説明と混同しないでください。

### A4. 本学習の予算を選ぶ

公式の既定は`trainer.max_epochs=100`、batch128、LR5e-5、weight decay0.001、SIGReg係数0.09。**100エポックは10万ステップではありません。** 手元HDF5ローダーの過去の件数監査では100エポック相当は約139万更新でしたが、配布モデルの実際の学習履歴とは断定できません。

試運転が正常に通った後、別の新しい出力先に切り替えて実行します。次は公式trainer側を10万更新で制限する構文例です。

```bash
export STABLEWM_HOME="$PWD/output/pusht/official_native_100k"
CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python lewm/train.py \
  data=pusht data.dataset.name=pusht_expert_train.h5 \
  subdir=lewm output_model_name=lewm wandb.enabled=false \
  hydra.run.dir=output/pusht/official_native_100k/hydra \
  trainer.max_epochs=100 +trainer.max_steps=100000
```

これだけでBTとの公平な比較になるわけではありません。公式経路は90%のクリップ分割・分割前の正規化統計を使い、共有比較trainerのepisode分割・train-only統計とは違います。スケジューラや統計精度も区別してください。

途中再開について：公式コードは`run_dir/lewm_weights.ckpt`がある場合にManagerへ渡しますが、通常のLightning保存先とこの探索名の整合・完全再開をこの環境では未検証です。`weights_epoch_N.pt`をその名前に変更して代用してはいけません。確実な途中再開とBTとの比較を優先する場合はBを使います。

公式経路は共有trainer用の`metrics.jsonl`を出さないため、`monitor_training.sh`ではlossを表示できません。学習端末のLightning表示・出力ログを確認します。途中切断対策には[学習手順のtmux説明](../mylewm/docs/TRAINING.ja.md#4-本学習を開始する)が使えます。

## B. BTと同条件で比較するRaw LeWM

### B1. 分割を準備する

初回だけ実行します。BTにも同じmanifestを使う計画にします。既に作成済みなら再作成しません。

```bash
.venv/bin/python mylewm/train_rbg.py prepare \
  --dataset .cache/stable-wm/datasets/pusht_expert_train.h5 \
  --manifest output/manifests/pusht/manifest.json
```

### B2. 100ステップで確認する

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python mylewm/train_rbg.py train \
  --mode raw --manifest output/manifests/pusht/manifest.json \
  --output output/pusht/raw_smoke_s3072 \
  --steps 100 --batch-size 16 --workers 0 --seed 3072 \
  --warmup-steps 10 --lr 5e-5 --min-lr 0 \
  --save-every 50 --diagnostics-every 25 --deterministic
```

`--mode raw`が公式の全潜在SIGRegで、`--mode bt`が提案側です。RawではTが存在しないため、監視画面のT勾配は`-`で正常です。

### B3. 比較用の新規初期値を共通化し、10万更新する

以下は**新しい比較実験の例**です。今稼働中のBTと比較する場合は、その開始記録にある共有初期値・manifest・すべての設定を照合して使います。新しいmanifestを作っただけで既存runと完全に同条件とはみなしません。

```bash
.venv/bin/python mylewm/tools/create_shared_initialization.py \
  --seed 3072 --output output/pusht/comparison_s3072_initial.pt
CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python mylewm/train_rbg.py train \
  --mode raw --manifest output/manifests/pusht/manifest.json \
  --initialization output/pusht/comparison_s3072_initial.pt \
  --output output/pusht/raw_train_s3072 \
  --steps 100000 --batch-size 128 --workers 4 --seed 3072 \
  --warmup-steps 500 --lr 5e-5 --min-lr 0 \
  --save-every 5000 --diagnostics-every 1000 --deterministic
```

比較するBTにも同じ`--initialization`とmanifestを渡し、出力先だけ分けます。同じseedだけで済ませず、configの`initial_model_sha256`も比較します。公式損失との恒等時loss/勾配一致をテストしていますが、GPU丸め・学習レシピまで無条件に同一という意味ではありません。

### B4. 監視・再開する

別の端末で実行します。

```bash
cd /home/USER/bt-sigreg
bash mylewm/tools/monitor_training.sh --run output/pusht/raw_train_s3072
```

5,000更新ごとに検証・推論重みと`resume.pt`を保存します。中断後は**開始時と同じコマンドの末尾に`--resume`だけ追加**します。総steps・batch・初期値・コード・データを変えず、学習プロセスが二重に動かないよう確認してください。初回保存前の中断は別の未使用出力名で新規開始します。

出力の意味、OOM等の対処は[mylewmの初心者手順](../mylewm/README.md#はじめて学習する方へ)を参照。Rawを学習しただけでは成功率は分かりません。BTとの比較は同じ初期状態・Goal・CEM予算で別途制御評価し、訓練lossだけで順位付けしないでください。
