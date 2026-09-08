# mylewm：学習・比較基盤

成功率を測る場合は[PushT評価の初心者向け手順](docs/EVALUATE_PUSHT.ja.md)を参照してください。提案・公式両方の実行、GB10のメモリ対処、結果の見方、同条件比較を説明しています。

## はじめて学習する方へ

公式LeWM側を学習したい場合は[公式PushT学習の説明書](../lewm/TRAIN_PUSHT.ja.md)を参照してください。公式trainerの経路と、公平なBT比較向けRaw経路を分けています。

この手順は**このPCの既存`.venv`とダウンロード済みデータを使う手順**です。別PCへの環境構築・ダウンロードを自動化したものではありません。以下のコマンドを順番に実行しますが、PushTとLIBEROの学習は一方ずつにしてください。

**現在はPushTの10万ステップ学習が動いています。今すぐ別の学習を重複起動しないでください。** 既存学習を見たいだけなら、次の2行で十分です。

```bash
cd /home/USER/bt-sigreg
bash mylewm/tools/monitor_training.sh
```

Ctrl-Cで監視画面だけ終了します。以下は、現在の学習が終了してから新しい実験を始めるための手順です。文書更新だけでは新規学習を起動しません。

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

その中で、**どちらか一方**を実行してください。ここでの出力名`bt_train_s3072`は現在稼働中のrunとは別です。短期学習のcheckpointは使わず、新規初期値から始めます。

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

これらは単独学習の手順です。Raw/TCとの厳密な比較では同一の未学習初期重み・データ順・batch・予算等をそろえます。`tools/create_shared_initialization.py`は現状**PushT用**で、生成した重みをLIBEROへ渡してはいけません。今回稼働中のPushTの厳密な開始コマンドは下の実行記録に残しています。

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

100ステップ確認を見る場合は`bt_train_s3072`を`bt_smoke_s3072`に変更。`--once`を付けると1回表示、`--interval 10`で10秒間隔です。`--run`を省略した既定画面は**現在稼働中の旧PushT run**なので、新規実験では必ず指定してください。tmuxで起動した学習にはsystemd serviceがないため、`--unit`は付けません。

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

学習が終わっても、PushT/LIBEROの**成功率評価は別工程**です。以下の評価ツールと[検証記録](docs/VALIDATION.ja.md)を参照してください。

---

最新実行指示（2026-09-08）：下記の短期診断後、ユーザー承認でPushTのBT v2・100,000ステップを開始した。LIBEROとRaw/TCの長時間学習は未開始。現在のrunは次節を参照。

## PushT：10万ステップの実行と監視

端末での監視（Python3標準ライブラリのみ、学習には変更を加えない）：

```bash
bash mylewm/tools/monitor_training.sh                 # 5秒ごとに更新、Ctrl-Cで監視だけ終了
bash mylewm/tools/monitor_training.sh --once          # 1回だけ表示
bash mylewm/tools/monitor_training.sh --interval 10   # 10秒ごと
bash mylewm/tools/monitor_training.sh --run /path/to/run --once
```

ステップ数・loss・直近100件平均・予測損失・SIGReg・勾配・LR・速度/概算残時間・最新検証値・T診断を表示する。既定は今回のPushT runとsystemd service。別runのサービス確認は `--unit SERVICE_NAME` を追加する。書込み途中の最終行は次回まで待つ。速度は最近の中央値による概算で、残りの検証・保存コストを正確に見積もるものではない。このshはlunaとは独立した表示ツールで、実行するまで起動しない。

run: `output/pusht/bt_spectral_v2_100k_s3072/`。学習出力・ログは`output/`以下へまとめ、全体をGit対象外にする。`*.log`もGit対象外。稼働中の今回のrunは旧`.cache/stable-wm/pusht/`へのシンボリックリンクで参照し、書込み中の実体や保存済みconfigは変更しない。初期値・console・source snapshot・完了済み短期診断にも`output/`からリンクを用意した。今後の新規runは`--output output/pusht/NEW_RUN`へ直接保存する。

短期診断からresumeせず、新規の共有初期値を別途生成した。後日Raw/TCを比較する場合は同じ初期値・データ順を使い、学習開始には別途指示を得る。以下は開始時の実コマンドで、旧パスを履歴として保持する。既存runへ重複実行しない。

```bash
.venv/bin/python mylewm/tools/create_shared_initialization.py --seed 3072 --output .cache/stable-wm/pusht/bt_spectral_v2_100k_shared_initialization_s3072.pt
CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python mylewm/train_rbg.py train --mode bt --bt-depth 2 --bt-kappa .2 --bt-hidden 192 --steps 100000 --batch-size 128 --warmup-steps 500 --lr 5e-5 --min-lr 0 --workers 4 --save-every 5000 --diagnostics-every 1000 --seed 3072 --deterministic --initialization .cache/stable-wm/pusht/bt_spectral_v2_100k_shared_initialization_s3072.pt --output .cache/stable-wm/pusht/bt_spectral_v2_100k_s3072
```

上記trainerを実際にはsystemd user service `bt-pusht-spectral-v2-100k-s3072.service` で起動し、チャットのコマンド待機と分離した。consoleは同階層の `bt_spectral_v2_100k_s3072.console.log`。生存確認は `systemctl --user status bt-pusht-spectral-v2-100k-s3072.service`。同じコマンドを重複起動しない。自動再起動は設定していない。

1,000ステップごとにT診断、5,000ごとにvalidation・推論checkpoint・再開用checkpointを保存。100,000回目のLRは既存cosine仕様により0。推論モデル18,034,478、訓練専用T147,840パラメータ。12,800,000クリップ提示の予定で、公式100エポック再現ではない。

開始時の未コミットコードも `bt_spectral_v2_100k_s3072.sources.tar.gz` に保存し、個別学習ソースのSHA256はconfigへ記録する。共有初期モデルstate hashは `1c1962d31c04bb65617d468686d5f23efe777058988c032408429d64dc02bc7d`。実行中の学習ソースを編集しない。

lunaサブエージェントが10分間隔でloss・進捗・勾配・T診断・例外を確認する。これはセッション内のサブエージェント監視で、OS再起動やセッション終了後までの自動復帰を保証する仕組みではない。OpenAI Docs skillは[サブエージェント運用](https://learn.chatgpt.com/docs/agent-configuration/subagents)の確認に使用し、学習実装用skillとしては使用していない。

現在の研究案は[BT-SIGReg](docs/BT_SIGREG.ja.md)。**Tの表現力制限を修正し、97テストとPushT／LIBERO-10各100更新の実データ診断を完了。性能改善・成功率比較は未実証。** 研究目的・数学的限界は[総合レビュー](../README_RESEARCH_REVIEW.ja.md)、数値は[検証記録](docs/VALIDATION.ja.md)を参照。短期診断は終了済み、現在の長時間学習は上記PushTの1runのみ。

## 構成

| 場所 | 残す理由 |
|---|---|
| 直下のPython | PushT/LIBERO共有trainer、モデル、損失、データ・再開・評価契約、診断の共通部品 |
| `tools/` | 公式評価、対応付き比較、BN校正、データ・画像・計算経路の監査、表現診断 |
| `tests/` | 残した実装の回帰テスト |
| `docs/BT_SIGREG.ja.md` | 現行アルゴリズム案 |
| `docs/VALIDATION.ja.md` | 学習・評価基盤の監査結果と未解決事項 |
| `docs/CLEANUP.ja.md` | 削除・移動の一覧と復元方法 |
| `run_libero.sh` | LIBERO専用OSMesa環境の起動wrapper |

`train_rbg.py` / `train_rbg_libero.py` / `rbg.py` はRaw/TCでも使う共有基盤なので残す。学習レシピを変更せず構成を整理したもので、RBGの比較モードは残っているが本命ではない。旧RBG実験キュー、混合アブレーション、Sub-JEPA専用trainer、ABC数値検算と重複した旧研究MDは作業ツリーから削除し、Git履歴に保存した。

## 検証と評価ツール

以下はリポジトリルートで実行する。

```bash
.venv/bin/python -m pytest mylewm/tests -q
# dry-runのみ。学習や環境評価は起動しない
.venv/bin/python mylewm/tools/plan_controlled_comparison.py --steps 100000 --methods raw
.venv/bin/python mylewm/tools/evaluate_official_pusht.py
```

公式PushT評価には `evaluate_official_pusht.py --execute`、記録行動の再実行には `replay_pusht_evaluation.py` を使う。過去の200ケース176成功はデータセット由来Goalの回帰評価であり、未使用最終テストや固定T被覆率とは区別する。比較は `compare_paired.py` / `compare_libero.py`、学習状態とBNの監査は `audit_rbg_checkpoint.py` / `calibrate_rbg_bn.py`。各CLIの引数は `--help` を参照。

## 学習基盤（自動実行しない）

PushTの共有trainerは `train_rbg.py`、LIBERO-10は `train_rbg_libero.py`。どちらも `prepare` と `train` を持ち、`raw` / `tc` / `rbg` / `bt` モードを扱う。BT実体は `bt_sigreg.py` にあり、予測・計画用モデルにTを追加しない。

主比較の計画はRaw/TC/BTを同じ新規初期値・データ順で100,000更新。CLI既定は50,000更新なので明示指定が必要。`plan_controlled_comparison.py --methods raw tc bt` はPushTのdry-run計画を出力できる。既定のRaw/RBG選択は互換性のため維持し、全比較・LIBEROの自動パイプラインはまだない。

BT設定は `--bt-depth 2 --bt-kappa .2 --bt-hidden 192`。これは動作確認用の既定値で、性能に基づく推奨ではない。`--bt-depth 0` は固定恒等Tの対照。kappaは `0 <= kappa < 1` とし、変更した設定を同一checkpointのresumeで混ぜることを拒否する。

現行Tは `cayley_spectral_v2`：左右のCayley直交因子と特異値制約、学習bias付き中心化half-linear/half-tanh活性化。旧Frobenius試作とは互換性がなく、新規初期値を使う。数学的条件・残る制限は[提案書 §10](docs/BT_SIGREG.ja.md#10-修正版原点固定非対称変形特異値直接制約)を参照。

### 短期の実データ診断

2026-09-08にユーザーが許可した範囲は修正・相互レビュー・短期診断まで。本比較の10万更新とは別で、以下を各100更新で実行する。出力先が既にある場合は上書きせず、別名を選ぶ。

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python mylewm/train_rbg.py train --mode bt --steps 100 --batch-size 16 --warmup-steps 10 --lr 5e-5 --min-lr 0 --workers 0 --save-every 50 --diagnostics-every 25 --deterministic --output .cache/stable-wm/pusht/bt_spectral_v2_smoke_20260908_s3072
CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python mylewm/train_rbg_libero.py train --mode bt --steps 100 --batch-size 16 --warmup-steps 10 --lr 5e-5 --min-lr 0 --workers 0 --save-every 50 --diagnostics-every 25 --deterministic --output .cache/stable-wm/libero10/bt_spectral_v2_smoke_20260908_s3072
```

`transport_diagnostics_pre_update` は既存T forwardのz/uから採る診断。時刻別バッチ共分散trace・最小/最大固有値、変位RMS、標本距離比を保存する。追加のモデルforwardや乱数抽選は行わず、損失へ加算しない。batch16の共分散ランクは高々15なので、192次元の最小固有値が約0でも崩壊の証拠ではない。丸め由来の微小負値もそのまま記録する。距離比の数値検査は大域的証明ではない。診断・データ取得の時間も学習ループの経過時間に含まれる。

### 本比較の構文例（未実行）

次は**将来、学習を再開すると指示された場合の構文例**で、今回実行したコマンドではない。データmanifestを先に確認し、出力先は未使用のものにする。

```bash
.venv/bin/python mylewm/train_rbg.py train --mode bt --bt-depth 2 --bt-kappa .2 --bt-hidden 192 --total-steps 100000 --deterministic --output output/pusht/bt_s3072
.venv/bin/python mylewm/train_rbg_libero.py train --mode bt --bt-depth 2 --bt-kappa .2 --bt-hidden 192 --total-steps 100000 --deterministic --output output/libero10/bt_s3072
```

Tは学習専用AdamWグループ（同じLR・weight decay、勾配clipはモデルと別）で更新。`resume.pt` にT・optimizer・独立SIGReg乱数を保存し、`step_N_object.ckpt` はTを持たない推論用E/A/F。configにはTの構成、距離境界、追加訓練パラメータ数を記録し、metricsにはTの勾配ノルムを記録する。既存のcheckpoint監査ツールが出すブロックGaussian値はzの診断であり、Tを通したBT訓練損失とは区別する。

共有trainerの既定はbatch128、warmup500、max-lr5e-5、min-lr0。全更新の使用LRと提示数を記録する。`--resume` は同じ実験設定のみ許可し、予算やデータを変える場合は別出力先に新規実験を作る。旧checkpointは読み戻さない。初期値生成は `tools/create_shared_initialization.py`、検証済みの保存・復元処理は `training_state.py` にある。

100,000更新は公式配布checkpointの過去の学習履歴再現ではない。Raw/TC/BTの公平性、追加訓練計算量、seed差と実制御成績を評価する。残る課題は[検証記録](docs/VALIDATION.ja.md)を参照。

## LIBERO環境

データは `.cache/libero-datasets/libero_10` の10 HDF5。2視点は異なる実カメラ画像を使う。ローカルシミュレータは `external/libero`、robosuite1.4.0 / bddl1.0.1、隔離したMuJoCo3.3.7を使用する。

このGB10環境ではEGL画像に異常があったため、評価には検証済みのOSMesa wrapperを使う。

```bash
bash mylewm/run_libero.sh mylewm/tools/smoke_libero.py
bash mylewm/run_libero.sh mylewm/tools/audit_libero_images.py --task-index 0 --output FRESH_AUDIT_DIRECTORY
bash mylewm/run_libero.sh mylewm/tools/eval_rbg_libero.py --checkpoint CHECKPOINT --output FRESH_OUTPUT_DIRECTORY
```

OSMesaはUbuntu24.04 ARM64用の `libosmesa6=24.0.5-1ubuntu1`、`libglapi-mesa=24.0.5-1ubuntu1`、`libllvm17t64=1:17.0.6-9ubuntu1` を `apt download` し、`dpkg-deb -x PACKAGE .cache/libero-osmesa` で展開したもの。システム全体を変更しない。他環境では画像監査を再実行する。

評価には全10タスクの画像監査が必要。既定の監査先は `.cache/stable-wm/libero10/rbg_v0/render_audit/task_N/report.json`。新規監査は別出力先に保存し、評価の `--render-audit-dir` で指定する。学習済み共有LIBEROモデルの制御性能は未検証であり、BC成績とCEM成績を混ぜない。
