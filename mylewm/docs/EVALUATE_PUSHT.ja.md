# PushT成功率評価：初心者向け手順

このPCの既存`.venv`、ダウンロード済みPushTデータ、信頼済みcheckpointを使います。[学習手順](../README.md)とは別工程で、**追加学習はしません**。LIBERO評価にはこのシェルを使わないでください。

## 1. 何を測るのか

世界モデルとCEM計画器で行動を選び、PushT環境で実際に動かして成功数を数えます。lossから成功率を推測するものではありません。既定はmanifestの`confirm`先頭50ケースで、開始状態とデータセット由来Goalを固定します。固定T字目標の95%被覆率とは別の成功判定です。既に使った回帰評価集合であり、未使用の最終テストではありません。

| 比較 | 分かること |
|---|---|
| 自分の5,000・15,000・20,000ステップ | 同じrunの学習進行に伴う制御成績の変化 |
| 途中checkpointと公式配布checkpoint | 学習量が異なるモデルの到達成績。公平な方式比較ではない |
| 同初期値・データ順・設定・100,000更新のRawとBT | 同更新予算での方式比較。追加計算量・seed差も別途評価する |

公式配布checkpointを「公式15,000ステップ」等と呼ばないでください。過去の公式は同50ケース46/50、全200ケース176/200でしたが、評価コード修正後は同じ版で再評価します。PushTだけでマルチタスク達成や任意の場面での非劣化を保証できません。

## 2. 重複起動を避け、ファイルを確認する

```bash
cd /home/USER/bt-sigreg
systemctl --user list-units --all 'bt-pusht*'
nvidia-smi
bash mylewm/tools/monitor_training.sh --once
ls -lh output/pusht/bt_spectral_v2_100k_s3072/step_*_object.ckpt
ls -lh .cache/stable-wm/pusht/lewm_object.ckpt
ls -lh .cache/stable-wm/datasets/pusht_expert_train.h5
```

**2026-09-08時点では学習と評価が動いています。以下を今すぐ重複起動しないでください。** この時点の20,000更新評価サービスは`bt-pusht-eval-20000-50cases-20260908.service`。`active/running`は稼働中という意味で、評価成功ではありません。新規評価は既存評価の終了を待ちます。学習と同時実行すると遅くなる場合があります。他の学習・Qwen・GPUサービスを勝手に停止しないでください。

- 提案側は`step_20000_object.ckpt`なら20,000更新時点の推論モデルです。
- 公式は`.cache/stable-wm/pusht/lewm_object.ckpt`です。
- `resume.pt`は学習再開用なので評価へ渡しません。
- `torch.load(weights_only=False)`を使います。自分で生成したものや信頼確認済み公式重みだけを使用してください。

既定manifestは`.cache/stable-wm/pusht/rbg_v0/manifest.json`です。現run用で、内部に旧フォルダ名があるため一時互換リンクを学習中に削除しないでください。移転後の新しいmanifestを使う場合は、両モデルへ`--manifest output/manifests/pusht/manifest.json`を指定し、同じ分割・ケースで比較します。既存runのmanifestは書き換えません。

## 3. まず設定確認だけを行う

リポジトリ直下で、未使用の出力名を指定します。出力フォルダ自体はまだ作りません。

```bash
bash mylewm/tools/evaluate_pusht.sh \
  --checkpoint output/pusht/bt_spectral_v2_100k_s3072/step_20000_object.ckpt \
  --output output/pusht/eval_bt_20000_50 \
  --gb10-cache-workaround
```

`Dry-run only`なら設定確認だけで終了しています。GPU起動、cache解放、出力作成、pickle読み込みは行いません。ファイル・ハッシュ・ケース数を確認しますが、重み内容や依存環境の実行成功までは保証しません。

`--gb10-cache-workaround`はこのGB10の起動OOM対処です。**実行時だけ**対象PushT HDF5の読み取りcacheへ解放ヒントを出し、データ読込より先にCUDAを初期化します。データ削除、全体の`drop_caches`、モデルや学習条件の変更はしません。学習側の再読込で一時的に遅くなる可能性はあります。通常の別GPUではこのフラグを外せます。

## 4. 評価を実行し、ログを見る

端末切断後も続けたいなら、先に`tmux new -s pusht-eval`を実行し、その中で次を実行します。設定確認と同じコマンドに`--execute`を追加します。

```bash
cd /home/USER/bt-sigreg
bash mylewm/tools/evaluate_pusht.sh \
  --checkpoint output/pusht/bt_spectral_v2_100k_s3072/step_20000_object.ckpt \
  --output output/pusht/eval_bt_20000_50 \
  --gb10-cache-workaround --execute
```

前景実行です。tmuxでは`Ctrl-b`を押して離し、`d`で離脱、`tmux attach -t pusht-eval`で戻れます。systemd serviceは自動作成しません。評価端末でCtrl-Cを押すと**評価を中断**します。

別端末で進捗を見るには：

```bash
cd /home/USER/bt-sigreg
tail -f output/pusht/eval_bt_20000_50/console.log
```

こちらの`tail`のCtrl-Cは表示だけを止めます。起動時には約46.3GBのデータ内容ハッシュ等を確認するため、ログの間隔が空く場合があります。`CEM solve time`は一回の計画処理の終了で、評価全体の完了ではありません。

既定条件は50ケース、seed42、CEM候補300・更新30・上位30・batch1、計画horizon5・行動block5・再計画間隔5、評価予算50、画像224です。既存`lewm/eval.py`と設定を使い、重み・データ・前処理ソース・依存版・初期状態/Goal・実行行動を記録します。片方だけ候補数等を減らして比較しないでください。

## 5. 終了と結果を確認する

正常終了すると`Completed: 成功数/ケース数 successes (...)`を表示します。`echo $?`を実行**直後**に確認すると終了コードで、0が正常です。ログのエラーも確認してください。

```bash
tail -n 12 output/pusht/eval_bt_20000_50/console.log
.venv/bin/python -c 'import json; d=json.load(open("output/pusht/eval_bt_20000_50/results.txt.json")); print(sum(d["successes"]), "/", len(d["successes"]), "successes;", d["success_rate"], "%")'
```

| ファイル | 内容 |
|---|---|
| `launch.json` | 引数、launcher・重み・manifestのSHA256 |
| `console.log` | 起動、CEM時間、例外、正常評価完了時のPyTorchメモリpeak |
| `results.txt.json` | 成功/失敗、成功率、ケース・実行監査・設定 |
| `results.txt` | 人間向け設定と結果 |
| `env_N.mp4` | 実行動画 |
| `checkpoint_object.ckpt` | 元checkpointへのリンク。複製ではない |

`success_rate`は百分率で46.0なら46%。後述の比較ツールの`candidate_rate`等は0–1なので0.46が46%です。途中終了・結果未生成は**未測定**で、0%ではありません。全出力はGit対象外の`output/`。元checkpointを移動・削除するとリンクも使えなくなるので保持してください。

## 6. 公式モデルと比較する

提案側が終わってから、別の未使用名で公式を評価します。同じmanifest、ケース数、offset、コード版を使います。

```bash
bash mylewm/tools/evaluate_pusht.sh \
  --checkpoint .cache/stable-wm/pusht/lewm_object.ckpt \
  --output output/pusht/eval_official_50 \
  --gb10-cache-workaround --execute
```

両方の正常終了後に比較します。比較レポートも未使用名を使ってください（比較CLI自体に上書き拒否機能はありません）。

```bash
.venv/bin/python mylewm/tools/compare_paired.py \
  --baseline output/pusht/eval_official_50/results.txt.json \
  --candidate output/pusht/eval_bt_20000_50/results.txt.json \
  --output output/pusht/comparison_bt20000_official_50.json
```

`difference`は提案−公式の成功率差、`candidate_only_success`は提案だけ成功したケース数。信頼下限と観測差は区別します。`observed_no_degradation=true`でも任意の状況で性能が落ちない保証ではありません。**ツールは学習予算の同一性を検証しません。** 同更新数での方式比較には、別途そろえた学習が必要です。

`evaluation protocols differ`を無視したりJSONを改変して通してはいけません。ソース版・前処理・manifest・ケース・計画条件を確認し、必要ならそろえて再評価します。

## 7. 別ステップ・ケース数とトラブル

別ステップは`--checkpoint`と`--output`を変え、一つずつ評価します。途中再開は未対応。新しい出力名でケース集合を最初からやり直し、失敗ログは残します。

200ケースは、各モデルで`--num-eval 50 --offset 0`、`50`、`100`、`150`の4バッチを、それぞれ別出力名で実行します。比較CLIの`--baseline`と`--candidate`に各4結果を渡せます。別集合の50ケースと200ケースを対応付き比較として混ぜません。

| 症状 | 対処 |
|---|---|
| 出力が既にある | 未使用名にする。既存結果やログを消して回避しない |
| データ・重みがない | パスとmanifest内部を確認。稼働中ファイルを書き換えない |
| CUDA OOM | プロセス使用量と`free -h`を確認。GB10では上記フラグを使い、他サービスを勝手に停止しない |
| CPU/GPU device mismatch | 現行`PlanningActionAdapter`修正版を使う。稼働中評価のソースは変更しない |
| ログが増えない | ハッシュやCEMが長い場合がある。プロセスと例外を確認し、結果なしを成功/0%としない |

既存launcherで完了した50ケースの計画評価部分は約579秒（15,000更新）・694秒（5,000更新）でした。データ確認・初期化は別で、同時実行状況にも依存します。固定の終了時刻は保証できません。新しい初心者用シェルは同じ評価本体を呼びますが、文書作成のために追加50ケースは起動していません。
