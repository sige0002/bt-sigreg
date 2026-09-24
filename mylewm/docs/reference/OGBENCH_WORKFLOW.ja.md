# OGBench CubeをLeWMの条件で比較する

**2026-09-24追記：** ユーザー依頼によりRaw・Cayley v2 BT・ノルム保存BTのCube専用prepare／学習入口と明示YAMLを追加。現在の実行手順は[Cubeの1ページ学習ガイド](../TRAINING_CUBE.ja.md)へ統合した。以下は過去の調査記録であり、「未実装」「未取得」は当時の状態を示す。Cubeの制御評価対応は学習準備とは別に確認する。

方針：2026-09-16。OGBenchではまず**LeWMのCube single expert・画像世界モデル＋CEM**を採用する。公式配布モデルの参考評価を確認し、同じ基盤でRaw／TC／BTを比較する。BCの追加学習は不要。[比較全体の手順](COMPARISON_PROTOCOL.ja.md)

今回は公開資料・ローカルコードの照合とHydra設定展開まで確認した。データ・重みのダウンロード、環境起動、学習・制御評価は未実施。以下の実行例は、その後に使う手順である。

2026-09-16訂正：初版では公開コードの既定値を採用条件にしていたが、[論文の付録E](https://arxiv.org/html/2603.19312v1#A5)ではCubeも10 epochs、[付録D](https://arxiv.org/html/2603.19312v1#A4)ではCEMはPushTのみ30反復・他環境10反復。論文へ揃える実行例はこの2点を明示的に上書きする。PushT Figure 18の約18万更新をCubeへ転用しない。

## 何をLeWMに揃えるか

[LeWM公式Cube設定](https://github.com/lucas-maes/le-wm/blob/main/config/eval/cube.yaml)は、データ中の状態を復元して未来の画像をGoalにする評価である。[OGBench標準の5タスク評価](https://github.com/seohongpark/ogbench#usage-for-offline-goal-conditioned-rl)とは評価ケースの定義が異なる。「OGBench全体の標準スコア」として報告しない。

参照するローカルファイルは[学習データ](../../../lewm/config/train/data/ogb.yaml)、[学習共通設定](../../../lewm/config/train/lewm.yaml)、[モデル](../../../lewm/config/train/model/lewm.yaml)、[Cube評価](../../../lewm/config/eval/cube.yaml)、[CEM](../../../lewm/config/eval/solver/cem.yaml)。今回の確認元はGit `569e8ae`。実行時はこれらのhash、解決済み設定、依存版を記録する。`lewm/`にはローカル修正があるため、上流の無改変コピーとは呼ばない。

| 項目 | 採用するLeWM設定 |
|---|---|
| 環境・データ | `swm/OGBCube-v0`、`env_type=single`、`ogbench/cube_single_expert.h5` |
| 画像・履歴 | 224px、ViT tiny・patch14、状態192次元、履歴3、1段予測 |
| 行動 | frameskip=5。連続する5行動を連結して行動encoderへ渡す |
| 予測器 | 深さ6、heads16、dim_head64、MLP2048、dropout0.1 |
| 学習 | batch128、AdamW、LR5e-5、weight decay1e-3、bf16、clip1.0、seed3072 |
| 正則化の基準 | Raw SIGReg係数0.09、投影1024、knots17 |
| 予算・分割 | 論文10 epochs（コード既定100を上書き）、train比率0.9は公開設定。実更新数・分割単位を記録 |
| 計画 | horizon5、receding_horizon5、action_block5 |
| CEM | 候補300、反復10（共通YAML既定30を上書き）、elite30、var_scale1.0、solver batch1 |
| 評価 | 50ケース、seed42、Goalは25行動先、実行予算50、到達時終了 |

**論文の10 epochsとコード既定100 epochsを区別する。** 1 GPU・勾配蓄積なしなら、実際のtrainサンプル数をNとして概ね`10 × floor(N / 128)`更新になる。データの窓抽出とdrop_lastを含めた実DataLoader長で確定する。配布checkpointが正確に何更新学習したかは、設定ファイルだけから断定しない。

モデルへの主入力は画像・行動であり、`ob_type=states`という環境設定を理由に状態ベクトル学習へ変更しない。学習設定は`observation`も読み込むが、どの列がモデルに消費されるかは実経路で確認する。qpos/qvelとブロック位置・姿勢は初期状態／Goal復元に使い、報酬やタスク仕様を世界モデルの教師に加えない。

## 1. データと重みを準備する

配布先は[LeWM Cubeデータ](https://huggingface.co/datasets/quentinll/lewm-cube/tree/main)と[モデル](https://huggingface.co/quentinll/lewm-cube/tree/main)。2026-09-16のメタデータ確認では、データは`cube_single_expert.tar.zst`、モデルは`config.json`と`weights.pt`。OGBenchの別データセットを名前の類似だけで置き換えない。

次は未実行のダウンロード例。リポジトリ直下、HF CLIのある検証環境で使う。既存の学習用`.venv`へ依存を同期せず、必要な依存変更は`UV_PROJECT_ENVIRONMENT`で隔離する。

```bash
export OGB_DATA_ROOT="$PWD/.cache/lewm-ogbench"
export OGB_RUN_ROOT="$PWD/output/ogbench/cube_reference_s3072"
export LOCAL_DATASET_DIR="$OGB_DATA_ROOT"
export STABLEWM_HOME="$OGB_RUN_ROOT"

hf download quentinll/lewm-cube cube_single_expert.tar.zst \
  --repo-type dataset --revision 02a19a67a0dc8c9d6215f89c19e0a597691e152a \
  --local-dir "$OGB_DATA_ROOT/download"
hf download quentinll/lewm-cube config.json weights.pt \
  --revision b0747c5002e86d2ce8f3cd8178004b97524c587d \
  --local-dir "$OGB_DATA_ROOT/hf_cube"
tar --zstd -tf "$OGB_DATA_ROOT/download/cube_single_expert.tar.zst"
```

アーカイブ内部のパスを一覧で確認してから未使用ディレクトリへ展開し、最終的に`$OGB_DATA_ROOT/datasets/ogbench/cube_single_expert.h5`へ配置する。今回アーカイブ本体は取得していないため内部階層は未確認。現在のローダーは`datasets/`を挟むので、上流READMEの保存場所の略記だけに依存しない。

HDF5を読み取り専用で調べ、画像・行動の形状／時刻対応、episode境界、`qpos`、`qvel`、`privileged_block_0_pos`、`privileged_block_0_quat`を確認する。`goal_`付き列が評価時にどの未来行から生成されるかも照合する。欠損列を推測で元ファイルへ書き足さない。初回prepareでSHA-256、サイズ・mtime、配布revision、分割、列契約を記録し、通常起動はサイズ・mtime確認にする。OGBench用prepareは現状未実装。

HF重みはobject checkpointではない。[LeWM READMEの変換例](../../../lewm/README.md)の入力を`hf_cube`、出力をCube専用パスに変更し、配布configから構築して`strict=True`で読み込む。今回確認したCube configは行動encoder入力25次元で、frameskip5と物理行動5次元に対応する。PushTの2次元行動設定を流用しない。変換後は元state_dictとの一致・予測出力・Goal cost・保存再読込を確認してから評価へ進む。変換自体は未検証。

## 2. 公式学習設定を展開し、短期接続を確認する

リポジトリ直下で、次は**設定を表示するだけ**。今回この2つのHydra設定展開は終了コード0だった。

```bash
.venv/bin/python lewm/train.py data=ogb \
  subdir=ogbench_reference output_model_name=cube_lewm \
  wandb.enabled=false trainer.max_epochs=10 --cfg job --resolve
.venv/bin/python lewm/eval.py --config-name=cube \
  policy=ogbench_reference/cube_lewm solver.n_steps=10 --cfg job --resolve
```

`action_encoder.input_dim`と環境の`max_episode_steps`の`???`は実行関数内で埋まる。設定展開だけでは、データ読込・GPU・描画・checkpoint保存の成功は確認できない。

データ確認後、別の未使用出力で少数バッチの学習・保存・復元を確認する。通過後の公式レシピ学習例は次のとおり。`--cfg`を外すと実学習が始まるので、文書追加だけで実行しない。

```bash
.venv/bin/python lewm/train.py data=ogb \
  subdir=ogbench_reference output_model_name=cube_lewm \
  wandb.enabled=false trainer.max_epochs=10 \
  hydra.run.dir="$OGB_RUN_ROOT/hydra_train"
```

`LOCAL_DATASET_DIR`がデータroot、`STABLEWM_HOME`が出力root。現行SaveCkptCallbackはepochごとのstate_dict保存を使うため、配布object checkpointと同じ形式と決め付けない。保存先とoptimizerを含む再開状態は実際のcallback／Manager／依存版で検証する。

公式trainerは分割前に列の正規化統計を計算し、datasetに対してrandom_splitする。エピソード単位のtrain-only統計・保持分割を要求する主比較とは差がある。**公式レシピの参考再現**にはその差を記録し、**方式差の主比較**ではRaw／TC／BT全てに同じ保持分割・train-only統計を適用する。後者を公式trainer完全再現とは呼ばない。

## 3. LeWM方式で制御評価する

評価は初期行のqpos/qvelを`set_state`で復元し、25行動先のブロック位置・姿勢を`set_target_pos(cube_id=0, …)`へ渡す。Goal画像も同じ未来行から取る。horizon5×action_block5は25物理行動の計画範囲で、予算50とは別である。receding_horizonの解釈と実行履歴は使用するSWM版で確認する。

互換性確認後の実行例。`policy`は`$STABLEWM_HOME`からの相対パスで、`_object.ckpt`を付けない。変換済みモデルを`$STABLEWM_HOME/ogbench_reference/cube_lewm_object.ckpt`へ保存した場合を示す。

```bash
.venv/bin/python lewm/eval.py --config-name=cube \
  policy=ogbench_reference/cube_lewm \
  +cache_dir="$OGB_DATA_ROOT" \
  solver.n_steps=10 output.filename=cube_reference_50.txt \
  hydra.run.dir="$OGB_RUN_ROOT/hydra_eval"
```

`+cache_dir`は現行ローカル評価器のデータ参照用で、モデルrootとは分けられる。評価器は動画も出力するため、1ケースの接続確認で保存サイズを測り、正式50ケースの空き容量を見積もる。現在のCLIに未実装の動画抑制オプションを想定しない。再評価ではモデル／結果のディレクトリを分け、JSONや動画を上書きしない。

本実行の前に、以下を順に通す。

1. SWMのOGBCube登録と描画、保存qpos/qvelからの画像・状態復元、Goal設定、成功関数を1ケースで確認。
2. 公式モデルの行動次元・前処理・物理行動範囲、CEMから環境への時刻対応を確認。
3. episode・開始行・Goal行を固定した50ケースで公式参考評価。ケースごとの成功、成功率と区間、時間、設定／重み／データhashを保存。
4. 同じケース・Goal・探索分布でRaw／TC／BTを評価し、学習seed間の変動とTの追加計算量を併記。

ローカル`lewm/eval.py`にはPushT由来の`EvaluationDataset`、`EnvironmentAudit`、独立episode抽出等の変更が入っている。Cubeで必要なHDF5 schema・環境APIが適合するかは未検証。上流と抽出法が異なる場合は、上流ケース再現とローカル共通ケース比較を分ける。`--config-name=cube`が展開できるだけで正式評価対応済みとはしない。

## 4. BT比較に必要な実装と完了条件

現状、OGBench設定は`lewm/`に存在するが、`mylewm.training.train`へ`data=ogb`を渡す共通Raw／TC／BT CLIはない。PushT用prepareやLIBERO用adapterへCubeデータを入れる手順は使わない。

追加する範囲はOGBenchのデータ契約・prepare・共有TrainingAdapter・評価接続。E/P/A/F、1段予測、未来教師encoderへの勾配を共通にし、Raw／TC／BTの正則化だけを変える。BTはCayley v2のTを学習時だけ使い、制御はzで行う。架空の`--mode bt --env ogbench`コマンドは記載しない。

採用順は、データ／公式重みの準備 → 公式経路の接続・参考評価 → 同じ初期値・データ順・予算のRaw／TC／BT学習 → 固定CEMでの比較。実装回帰、保存再開、公式参考成績、方式間比較を別々に記録する。Cube singleの結果だけでLIBEROの10タスク共有改善を主張しない。

今回はこの手順を追加した段階。実行結果が出たら[実験一覧](../reports/EXPERIMENTS.ja.md)と[検証状況](../reports/VALIDATION.ja.md)へ証拠を添えて反映する。
