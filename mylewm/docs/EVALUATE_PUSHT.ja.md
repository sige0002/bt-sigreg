# PushT成功率評価：初心者向け手順

更新日：2026-09-10。新規評価は明示依頼時のみ実行し、定期監視・自動評価予約は行いません。

このPCの既存`.venv`、ダウンロード済みPushTデータ、信頼済みcheckpointを使います。[学習手順](TRAINING.ja.md)とは別工程で、**追加学習はしません**。LIBERO評価にはこのシェルを使わないでください。

この手順はローカル監査付き `lewm/eval.py` の固定confirm評価用です。上流の無改変evalによるランダム50ケースとは異なります。両者の差と結果は[評価レポート](reports/PUSHT_CHECKPOINT_EVALUATION.ja.md)を参照してください。

## 1. 何を測るのか

対応HDF5はSWMの`ep_len`・`ep_offset`境界形式です。保存された`episode_idx`または`ep_idx`と`step_idx`を利用し、列がない場合は境界からメモリ上で補完します。データ自体は変更しません。必要な境界列がない場合はCUDA初期化前に不足列と利用可能なキーを表示して停止します。

HDF5は`--manifest`の`dataset`パスから読みます。リポジトリの`.cache`内に置く必要はありません。別PCへ移した場合は`--dataset /absolute/path/to/pusht_expert_train.h5`を追加して上書きできます。manifestは編集しません。移転時はmanifestの`dataset_size`が必須で、サイズ不一致を拒否します。mtimeはコピーで変わるため評価では要求しません。通常はサイズ確認のみで全量走査せず、`--verify-data`指定時だけ全量SHA-256を計算し、prepareのhashがあれば照合します。旧manifestにhashが無い場合、サイズ・HDF5構造の確認だけでは同一内容を保証できません。解決したパスはdry-runと`launch.json`の`dataset`に出ます。

世界モデルとCEM計画器で行動を選び、PushT環境で実際に動かして成功数を数えます。lossから成功率を推測するものではありません。既定はmanifestの`confirm`先頭50ケースで、開始状態とデータセット由来Goalを固定します。固定T字目標の95%被覆率とは別の成功判定です。既に使った回帰評価集合であり、未使用の最終テストではありません。

| 比較 | 分かること |
|---|---|
| 自分の5,000・15,000・100,000ステップ | 同じrunの学習進行に伴う制御成績の変化 |
| 途中checkpointと公式配布checkpoint | 学習量が異なるモデルの到達成績。公平な方式比較ではない |
| 同初期値・データ順・設定・100,000更新のRawとBT | 同更新予算での方式比較。追加計算量・seed差も別途評価する |

公式配布checkpointを「公式15,000ステップ」等と呼ばないでください。過去の公式は同50ケース46/50、全200ケース176/200でしたが、修正後の公式50ケース再評価は45/50でした。異なる版の結果を混ぜて200件を集計しません。PushTだけでマルチタスク達成や任意の場面での非劣化を保証できません。

## 2. 新しいRaw/BT runを評価する

この節は最終評価向けです。学習継続中の保存済み`step_N_object.ckpt`は[途中checkpoint評価手順](EVALUATE_INTERMEDIATE.ja.md)で別GPU評価できます。その場合は`completed.json`や本学習の終了を待つ必要はありません。

新レシピ `pusht_spt_v1` のRaw/BTで**100,000更新の最終成績**を測る場合は、正常終了を確認してから評価します。保存済み途中重みの評価は前段の手順で可能です。以下は学習終了後の単独評価例です。RawはSIGRegを直接適用するモデルで、BTでは`raw`を`bt`へ置き換えます。`--execute`なしは安全な設定確認だけで、GPU初期化・出力作成・評価はしません。

```bash
cd "$(git rev-parse --show-toplevel)"
uv run python -c 'import json; d=json.load(open("output/pusht/spt_raw/completed.json")); assert d == {"step": 100000, "state": "completed", "recipe": "pusht_spt_v1"}; print(d)'
ls -lh output/pusht/spt_raw/step_100000_object.ckpt

# まずdry-run。出力名は未使用にし、ディレクトリを先に作らない。
bash mylewm/tools/evaluate_pusht.sh \
  --checkpoint output/pusht/spt_raw/step_100000_object.ckpt \
  --manifest output/manifests/pusht/manifest.json \
  --output output/pusht/eval_spt_raw_confirm50 \
  --seed 42 --gb10-cache-workaround

# dry-runの内容を確認してから、この1回だけを実行する。
bash mylewm/tools/evaluate_pusht.sh \
  --checkpoint output/pusht/spt_raw/step_100000_object.ckpt \
  --manifest output/manifests/pusht/manifest.json \
  --output output/pusht/eval_spt_raw_confirm50 \
  --seed 42 --gb10-cache-workaround --execute
```

`completed.json`がない、内容が一致しない、checkpointがない、または学習プロセスが残っている場合は評価を始めません。`resume.pt`と`last.ckpt`は学習再開用であり、評価へ渡しません。Raw/BT比較では、両方に**同じ新manifest、`--num-eval`、`--offset`、`--seed`、CEM設定**を使い、各評価の`status.json`が`succeeded`になった後で比較します。新しいrunの評価結果を、旧BT runや公式配布checkpointの数値と同じ条件の方式比較として混ぜません。

## 3. 既存checkpointを評価する前の確認

```bash
cd "$(git rev-parse --show-toplevel)"
systemctl --user list-units --all 'bt-pusht*'
tmux list-sessions 2>/dev/null || true
nvidia-smi
bash mylewm/tools/monitor_training.sh --once
ls -lh output/pusht/bt_spectral_v2_100k_s3072/step_*_object.ckpt
ls -lh .cache/stable-wm/pusht/lewm_object.ckpt
ls -lh .cache/stable-wm/datasets/pusht_expert_train.h5
```

`active/running`は稼働中という意味で、評価成功ではありません。実行前に既存評価の終了を確認します。学習と同時実行すると遅くなる場合があります。他の学習・Qwen・GPUサービスを勝手に停止しないでください。

- 提案側は`step_100000_object.ckpt`なら100,000更新時点の推論モデルです。
- 公式は`.cache/stable-wm/pusht/lewm_object.ckpt`です。
- `resume.pt`は学習再開用なので評価へ渡しません。
- `torch.load(weights_only=False)`を使います。自分で生成したものや信頼確認済み公式重みだけを使用してください。

既定manifestは`.cache/stable-wm/pusht/rbg_v0/manifest.json`です。現run用で、内部に旧フォルダ名があるため一時互換リンクを依存整理なしに削除しないでください。新レシピ用の`output/manifests/pusht/manifest.json`がある場合は、上の新run手順どおり明示指定します。比較する二つの新runには同じmanifestを使い、既存runのmanifestは書き換えません。

## 4. 既存checkpointの設定確認だけを行う

リポジトリ直下で、未使用の出力名を指定します。出力フォルダ自体はまだ作りません。

```bash
bash mylewm/tools/evaluate_pusht.sh \
  --checkpoint output/pusht/bt_spectral_v2_100k_s3072/step_100000_object.ckpt \
  --output output/pusht/eval_bt_100000_50 \
  --gb10-cache-workaround
```

`Dry-run only`なら設定確認だけで終了しています。GPU起動、cache解放、出力作成、pickle読み込みは行いません。ファイル・ハッシュ・ケース数を確認しますが、重み内容や依存環境の実行成功までは保証しません。

`--gb10-cache-workaround`はこのGB10の起動OOM対処です。**実行時だけ**対象PushT HDF5の読み取りcacheへ解放ヒントを出し、データ読込より先にCUDAを初期化します。データ削除、全体の`drop_caches`、モデルや学習条件の変更はしません。学習側の再読込で一時的に遅くなる可能性はあります。通常の別GPUではこのフラグを外せます。

## 5. 評価を実行し、ログを見る

起動端末にも`console.log`と同じログを逐次表示します。データ検証方式、HDF5・統計読込、評価開始、CEM計画開始、環境step呼出し完了とactiveケース数が表示されます。step呼出し数は成功ケース数や完了率ではありません。CEM計算中やHDF5読込中は更新間隔が空きます。`status.json=succeeded`まで評価完了とは扱いません。

端末切断後も続けたいなら、先に`tmux new -s pusht-eval`を実行し、その中で次を実行します。設定確認と同じコマンドに`--execute`を追加します。

```bash
cd "$(git rev-parse --show-toplevel)"
bash mylewm/tools/evaluate_pusht.sh \
  --checkpoint output/pusht/bt_spectral_v2_100k_s3072/step_100000_object.ckpt \
  --output output/pusht/eval_bt_100000_50 \
  --gb10-cache-workaround --execute
```

前景実行です。tmuxでは`Ctrl-b`を押して離し、`d`で離脱、`tmux attach -t pusht-eval`で戻れます。systemd serviceは自動作成しません。評価端末でCtrl-Cを押すと**評価を中断**します。

別端末で進捗を見るには：

```bash
cd "$(git rev-parse --show-toplevel)"
tail -f output/pusht/eval_bt_100000_50/console.log
```

こちらの`tail`のCtrl-Cは表示だけを止めます。通常起動ではデータ全量hashを計算しません（`--verify-data`指定時のみ）。HDF5統計読込やCEM計算中はログの間隔が空きます。`CEM solve time`は一回の計画処理の終了で、評価全体の完了ではありません。

既定条件は50ケース、seed42、CEM候補300・更新30・上位30・batch1、計画horizon5・行動block5・再計画間隔5、評価予算50、画像224です。既存`lewm/eval.py`と設定を使い、重み・データ・前処理ソース・依存版・初期状態/Goal・実行行動を記録します。片方だけ候補数等を減らして比較しないでください。

## 6. 終了と結果を確認する

正常終了すると`Completed: 成功数/ケース数 successes (...)`を表示します。`echo $?`を実行**直後**に確認すると終了コードで、0が正常です。ログのエラーも確認してください。

```bash
tail -n 12 output/pusht/eval_bt_100000_50/console.log
uv run python -c 'import json; d=json.load(open("output/pusht/eval_bt_100000_50/results.txt.json")); print(sum(d["successes"]), "/", len(d["successes"]), "successes;", d["success_rate"], "%")'
```

| ファイル | 内容 |
|---|---|
| `launch.json` | 引数、launcher・重み・manifestのSHA256 |
| `status.json` | launcherのrunning/succeeded/failed。succeededは結果の整合性検査後だけ。SIGKILL等では更新できないため、実プロセスの終了と結果も手動確認 |
| `console.log` | 起動、CEM時間、例外、正常評価完了時のPyTorchメモリpeak |
| `results.txt.json` | 成功/失敗、成功率、ケース・実行監査・設定 |
| `results.txt` | 人間向け設定と結果 |
| `env_N.mp4` | 実行動画 |
| `cem_audit.jsonl` | `--cem-audit`指定時の候補正規化前後・選択plan・予測costの診断 |
| `checkpoint_object.ckpt` | 元checkpointへのリンク。複製ではない |

`success_rate`は百分率で46.0なら46%。後述の比較ツールの`candidate_rate`等は0–1なので0.46が46%です。途中終了・結果未生成は**未測定**で、0%ではありません。全出力はGit対象外の`output/`。元checkpointを移動・削除するとリンクも使えなくなるので保持してください。

### 動かない・極端に低い成功率を調べる

同じ起動コマンドに`--cem-audit`を追加します。1ケースの完走は動作確認にすぎません。性能の確認には同じmanifest・offset・seed・標準予算で複数ケースを使い、公式重みと比較してください。

- 起動ログの`reference_mean/reference_std`は参照データの`StandardScaler.mean_/scale_`です。`scale_`は分散でなく標準偏差（母分散の平方根）。checkpointの`training_mean/training_std`は学習manifest由来で、train-only・標本標準偏差を使用します。
- CEM候補は参照統計で標準化した座標です。モデル入力では参照統計で逆変換してから学習統計で正規化し、環境実行時には参照統計で逆変換します。PushTの物理行動はXYの相対指令で、環境が100倍して目標変位へ変換します。絶対画面座標ではありません。
- JSONの`action_path_diagnostics.cases`に実行行動のmin/max/mean/std/mean_abs・near-zero率（絶対値`1e-3`未満）、実step数、開始/終了状態、agent移動距離と累積移動、object移動距離を記録します。`physical_actions`には各stepのactive maskと前後状態も残ります。非ゼロ指令なのにagentが動かない場合は警告し、`performance_interpretation_valid=false`にします。この診断がtrueでも学習品質を保証しません。
- `cem_audit.jsonl`では候補の正規化前後と選択planの分布を確認できます。ゼロ付近の候補・選択planと、非ゼロ行動に環境が反応しない問題を分けてください。動画は評価終了後に出力されます。

Issue #20では、明示HDF5パス経路で画像列を落として指定Goal画像を渡せない問題と、動画が再利用バッファを参照する問題を修正しました。修正前の低成功率をモデル性能の証拠に使わず、既存ログ・結果は残して別出力先で再評価します。

同じケース・設定のランダム行動対照は、完走した公式評価JSONを入力します。CPUで実環境を動かして動画も保存する検証コマンドです。開始/Goal画像・状態のhashを参照結果と照合し、不一致なら停止します。持続する乱数状態を使い、毎step異なる行動を標本化します。

```bash
CUDA_VISIBLE_DEVICES='' uv run python -m mylewm.tools.evaluate_pusht_random \
  --reference-result output/pusht/eval_official_50/results.txt.json \
  --output output/pusht/random_control_50
```

実行した範囲と失敗ログは[Issue #20の修正検証](reports/PUSHT_ISSUE20.ja.md)を参照してください。

## 7. 公式モデルと比較する

提案側が終わってから、別の未使用名で公式を評価します。同じmanifest、ケース数、offset、コード版を使います。

```bash
bash mylewm/tools/evaluate_pusht.sh \
  --checkpoint .cache/stable-wm/pusht/lewm_object.ckpt \
  --output output/pusht/eval_official_50 \
  --gb10-cache-workaround --execute
```

両方の正常終了後に比較します。比較レポートも未使用名を使ってください（比較CLI自体に上書き拒否機能はありません）。

```bash
uv run python mylewm/tools/compare_paired.py \
  --baseline output/pusht/eval_official_50/results.txt.json \
  --candidate output/pusht/eval_bt_100000_50/results.txt.json \
  --output output/pusht/comparison_bt100000_official_50.json
```

`difference`は提案−公式の成功率差、`candidate_only_success`は提案だけ成功したケース数。信頼下限と観測差は区別します。`observed_no_degradation=true`でも任意の状況で性能が落ちない保証ではありません。**ツールは学習予算の同一性を検証しません。** 同更新数での方式比較には、別途そろえた学習が必要です。

`evaluation protocols differ`を無視したりJSONを改変して通してはいけません。ソース版・前処理・manifest・ケース・計画条件を確認し、必要ならそろえて再評価します。

現状の比較CLIは開始/Goal観測hashを照合しますが、実物理初期状態の一致までは検査しません。2026-09-10の実評価ではobject初期位置・角度に微差が見つかりました。`action_path_diagnostics.cases[].initial_state`も比較し、CLI通過だけで厳密に同じ物理初期状態だったと主張しないでください。原因と影響は未解決です。[70k/100kの実測記録](reports/PUSHT_ISSUE20.ja.md)を参照してください。

## 8. 世界モデルrolloutの速度を測る

制御成功率とは別に、環境・CEM反復・Goal encoder・I/Oを除いたE/A/Fのrollout時間を測れます。既定のCEM候補数300を一つのbatchにし、観測3フレームから予測器を1回または20回呼ぶ時間をCUDA eventで測ります。入力は乱数なので、この値は精度・実環境の総制御時間ではありません。

```bash
uv run python mylewm/tools/benchmark_pusht_rollout.py \
  --checkpoint output/pusht/bt_spectral_v2_100k_s3072/step_100000_object.ckpt \
  --checkpoint .cache/stable-wm/pusht/lewm_object.ckpt \
  --output output/pusht/rollout_latency_bt_official.json
```

出力JSONには各checkpoint hash、warm-up後20反復の平均・中央値・最小/最大、peak GPUメモリを保存します。BTの学習専用Tは推論exportに含まれないため、この測定ではE/A/Fの計算量だけを比較します。

## 9. 別ステップ・ケース数とトラブル

別ステップは`--checkpoint`と`--output`を変え、一つずつ評価します。途中再開は未対応。新しい出力名でケース集合を最初からやり直し、失敗ログは残します。

200ケースは、各モデルで`--num-eval 50 --offset 0`、`50`、`100`、`150`の4バッチを、それぞれ別出力名で実行します。比較CLIの`--baseline`と`--candidate`に各4結果を渡せます。別集合の50ケースと200ケースを対応付き比較として混ぜません。

| 症状 | 対処 |
|---|---|
| 出力が既にある | 未使用名にする。既存結果やログを消して回避しない |
| データ・重みがない | パスとmanifest内部を確認。稼働中ファイルを書き換えない |
| CUDA OOM | プロセス使用量と`free -h`を確認。GB10では上記フラグを使い、他サービスを勝手に停止しない |
| CPU/GPU device mismatch | 現行`PlanningActionAdapter`修正版を使う。稼働中評価のソースは変更しない |
| ログが増えない | ハッシュやCEMが長い場合がある。プロセスと例外を確認し、結果なしを成功/0%としない |

所要時間は重み・ケース・同時実行状況に依存します。最終100kの固定50ケース1バッチは評価部分約194〜207秒でしたが、データ確認・初期化は別です。固定の終了時刻は保証できません。
