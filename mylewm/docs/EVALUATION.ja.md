# 評価手順：PushT／LIBERO-10／途中checkpoint

コードは`src/mylewm/`へ移動しました。環境準備でeditableパッケージを導入し、`python -m mylewm.…`で起動します。[構成・環境準備](../README.md#コードの構成と起動)。

| 目的 | 読む章 |
|---|---|
| HDF5のPushTモデルで成功率を測る | [第1章 PushT](#pusht) |
| HDF5のLIBERO-10モデルで成功率を測る | [第2章 LIBERO-10](#libero) |
| LIBEROの凍結ViT＋行動模倣方策を評価する | [BC方策の学習・評価](BEHAVIOR_CLONING.ja.md) |
| 学習終了前の保存済み重みを評価する | [第3章 途中checkpoint](#intermediate) |
| LeRobotの保持軌道で予測・Policyを確認する | [学習手順のLeRobot評価](TRAINING.ja.md#lerobot-evaluation)・[モデル利用](MODEL_USAGE.ja.md) |

LeRobot入力からの環境成功率評価は未対応です。validation・オフライン潜在予測・Policy保存形式の一致確認と、環境での制御成功率を分けて扱います。[文書一覧へ戻る](README.md)。

<a id="pusht"></a>

## 第1章 HDF5（PushT）の環境評価

更新日：2026-09-10。新規評価は明示依頼時のみ実行し、定期監視・自動評価予約は行いません。

このPCの既存`.venv`、ダウンロード済みPushTデータ、信頼済みcheckpointを使います。[学習手順](TRAINING.ja.md)とは別工程で、**追加学習はしません**。LIBERO評価にはこのシェルを使わないでください。

この手順はローカル監査付き `lewm/eval.py` の固定confirm評価用です。上流の無改変evalによるランダム50ケースとは異なります。両者の差と結果は[評価レポート](reports/PUSHT_CHECKPOINT_EVALUATION.ja.md)を参照してください。

<a id="pusht-1-何を測るのか"></a>

### 1. 何を測るのか

対応HDF5はSWMの`ep_len`・`ep_offset`境界形式です。保存された`episode_idx`または`ep_idx`と`step_idx`を利用し、列がない場合は境界からメモリ上で補完します。データ自体は変更しません。必要な境界列がない場合はCUDA初期化前に不足列と利用可能なキーを表示して停止します。

HDF5は`--manifest`の`dataset`パスから読みます。リポジトリの`.cache`内に置く必要はありません。別PCへ移した場合は`--dataset /absolute/path/to/pusht_expert_train.h5`を追加して上書きできます。manifestは編集しません。移転時はmanifestの`dataset_size`が必須で、サイズ不一致を拒否します。mtimeはコピーで変わるため評価では要求しません。通常はサイズ確認のみで全量走査せず、`--verify-data`指定時だけ全量SHA-256を計算し、prepareのhashがあれば照合します。旧manifestにhashが無い場合、サイズ・HDF5構造の確認だけでは同一内容を保証できません。解決したパスはdry-runと`launch.json`の`dataset`に出ます。

世界モデルとCEM計画器で行動を選び、PushT環境で実際に動かして成功数を数えます。lossから成功率を推測するものではありません。既定はmanifestの`confirm`先頭50ケースで、開始状態とデータセット由来Goalを固定します。固定T字目標の95%被覆率とは別の成功判定です。既に使った回帰評価集合であり、未使用の最終テストではありません。

| 比較 | 分かること |
|---|---|
| 自分の5,000・15,000・100,000ステップ | 同じrunの学習進行に伴う制御成績の変化 |
| 途中checkpointと公式配布checkpoint | 学習量が異なるモデルの到達成績。公平な方式比較ではない |
| 同初期値・データ順・設定・100,000更新のRawとBT | 同更新予算での方式比較。追加計算量・seed差も別途評価する |

公式配布checkpointを「公式15,000ステップ」等と呼ばないでください。過去の公式は同50ケース46/50、全200ケース176/200でしたが、修正後の公式50ケース再評価は45/50でした。異なる版の結果を混ぜて200件を集計しません。PushTだけでマルチタスク達成や任意の場面での非劣化を保証できません。

<a id="pusht-2-新しいrawbt-runを評価する"></a>

### 2. 新しいRaw/BT runを評価する

この節は最終評価向けです。学習継続中の保存済み`step_N_object.ckpt`は[途中checkpoint評価手順](#intermediate)で別GPU評価できます。その場合は`completed.json`や本学習の終了を待つ必要はありません。

新レシピ `pusht_spt_v1` のRaw/BTで**100,000更新の最終成績**を測る場合は、正常終了を確認してから評価します。保存済み途中重みの評価は前段の手順で可能です。以下は学習終了後の単独評価例です。RawはSIGRegを直接適用するモデルで、BTでは`raw`を`bt`へ置き換えます。`--execute`なしは安全な設定確認だけで、GPU初期化・出力作成・評価はしません。

```bash
cd "$(git rev-parse --show-toplevel)"
uv run python -c 'import json; d=json.load(open("output/pusht/spt_raw/completed.json")); assert d == {"step": 100000, "state": "completed", "recipe": "pusht_spt_v1"}; print(d)'
ls -lh output/pusht/spt_raw/step_100000_object.ckpt

# まずdry-run。出力名は未使用にし、ディレクトリを先に作らない。
bash scripts/evaluate_pusht.sh \
  --checkpoint output/pusht/spt_raw/step_100000_object.ckpt \
  --manifest output/manifests/pusht/manifest.json \
  --output output/pusht/eval_spt_raw_confirm50 \
  --seed 42 --gb10-cache-workaround

# dry-runの内容を確認してから、この1回だけを実行する。
bash scripts/evaluate_pusht.sh \
  --checkpoint output/pusht/spt_raw/step_100000_object.ckpt \
  --manifest output/manifests/pusht/manifest.json \
  --output output/pusht/eval_spt_raw_confirm50 \
  --seed 42 --gb10-cache-workaround --execute
```

`completed.json`がない、内容が一致しない、checkpointがない、または学習プロセスが残っている場合は評価を始めません。`resume.pt`と`last.ckpt`は学習再開用であり、評価へ渡しません。Raw/BT比較では、両方に**同じ新manifest、`--num-eval`、`--offset`、`--seed`、CEM設定**を使い、各評価の`status.json`が`succeeded`になった後で比較します。新しいrunの評価結果を、旧BT runや公式配布checkpointの数値と同じ条件の方式比較として混ぜません。

<a id="pusht-3-既存checkpointを評価する前の確認"></a>

### 3. 既存checkpointを評価する前の確認

```bash
cd "$(git rev-parse --show-toplevel)"
systemctl --user list-units --all 'bt-pusht*'
tmux list-sessions 2>/dev/null || true
nvidia-smi
bash scripts/monitor_training.sh --once
ls -lh output/pusht/bt_spectral_v2_100k_s3072/step_*_object.ckpt
ls -lh .cache/stable-wm/pusht/lewm_object.ckpt
ls -lh .cache/stable-wm/datasets/pusht_expert_train.h5
```

`active/running`は稼働中という意味で、評価成功ではありません。実行前に既存評価の終了を確認します。学習と同時実行すると遅くなる場合があります。他の学習・Qwen・GPUサービスを勝手に停止しないでください。

- 提案側は`step_100000_object.ckpt`なら100,000更新時点の推論モデルです。
- 公式は`.cache/stable-wm/pusht/lewm_object.ckpt`です。
- `resume.pt`は学習再開用なので評価へ渡しません。
- `torch.load(weights_only=False)`を使います。自分で生成したものや信頼確認済み公式重みだけを使用してください。

既定manifestは`output/manifests/pusht/manifest.json`です。現在の実パスでprepareしたmanifestを使用し、旧フォルダ名の互換リンクは不要です。比較する二つの新runには同じmanifestを使います。過去runの厳密再開には当時のmanifest・コード・環境を復元してください。[旧manifestの退避・復元](reports/CLEANUP.ja.md#旧manifest再作成と互換リンク削除2026-09-11)。

<a id="pusht-4-既存checkpointの設定確認だけを行う"></a>

### 4. 既存checkpointの設定確認だけを行う

リポジトリ直下で、未使用の出力名を指定します。出力フォルダ自体はまだ作りません。

```bash
bash scripts/evaluate_pusht.sh \
  --checkpoint output/pusht/bt_spectral_v2_100k_s3072/step_100000_object.ckpt \
  --output output/pusht/eval_bt_100000_50 \
  --gb10-cache-workaround
```

`Dry-run only`なら設定確認だけで終了しています。GPU起動、cache解放、出力作成、pickle読み込みは行いません。ファイル・ハッシュ・ケース数を確認しますが、重み内容や依存環境の実行成功までは保証しません。

`--gb10-cache-workaround`はこのGB10の起動OOM対処です。**実行時だけ**対象PushT HDF5の読み取りcacheへ解放ヒントを出し、データ読込より先にCUDAを初期化します。データ削除、全体の`drop_caches`、モデルや学習条件の変更はしません。学習側の再読込で一時的に遅くなる可能性はあります。通常の別GPUではこのフラグを外せます。

<a id="pusht-5-評価を実行しログを見る"></a>

### 5. 評価を実行し、ログを見る

起動端末にも`console.log`と同じログを逐次表示します。データ検証方式、HDF5・統計読込、評価開始、CEM計画開始、環境step呼出し完了とactiveケース数が表示されます。step呼出し数は成功ケース数や完了率ではありません。CEM計算中やHDF5読込中は更新間隔が空きます。`status.json=succeeded`まで評価完了とは扱いません。

端末切断後も続けたいなら、先に`tmux new -s pusht-eval`を実行し、その中で次を実行します。設定確認と同じコマンドに`--execute`を追加します。

```bash
cd "$(git rev-parse --show-toplevel)"
bash scripts/evaluate_pusht.sh \
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

<a id="pusht-6-終了と結果を確認する"></a>

### 6. 終了と結果を確認する

通常の`--execute`は、CEM評価と結果検証に続けて正解画像付きの比較画面を自動生成します。端末と`console.log`へ`Visual report: .../viewer/index.html`を表示し、画面の保存後に`Completed: 成功数/ケース数 successes (...)`を表示します。`echo $?`を実行**直後**に確認すると終了コードで、0が正常です。ログのエラーも確認してください。

```bash
tail -n 12 output/pusht/eval_bt_100000_50/console.log
uv run python -c 'import json; d=json.load(open("output/pusht/eval_bt_100000_50/results.txt.json")); print(sum(d["successes"]), "/", len(d["successes"]), "successes;", d["success_rate"], "%")'
```

| ファイル | 内容 |
|---|---|
| `launch.json` | 引数、launcher・重み・manifestのSHA256 |
| `status.json` | launcherのrunning/succeeded/failed。succeededは結果検証と比較画面保存の完了後。画面生成中はrunningかつphase=visual_report。SIGKILL等では更新できないため、実プロセスの終了と結果も手動確認 |
| `console.log` | 起動、CEM時間、例外、正常評価完了時のPyTorchメモリpeak |
| `results.txt.json` | 成功/失敗、成功率、ケース・実行監査・設定 |
| `results.txt` | 人間向け設定と結果 |
| `env_N.mp4` | 実行動画 |
| `viewer/index.html` | 自動生成する正解画像・動画・判定時点・試行番号付き比較画面。ブラウザで開く |
| `viewer/assets/`・`viewer/report.json` | 画像・動画のコピーと各stepの判定数値 |
| `cem_audit.jsonl` | `--cem-audit`指定時の候補正規化前後・選択plan・予測costの診断 |
| `checkpoint_object.ckpt` | 元checkpointへのリンク。複製ではない |

`success_rate`は百分率で46.0なら46%。後述の比較ツールの`candidate_rate`等は0–1なので0.46が46%です。途中終了・結果未生成は**未測定**で、0%ではありません。全出力はGit対象外の`output/`。元checkpointを移動・削除するとリンクも使えなくなるので保持してください。

`results.txt`末尾の試行一覧は**1回目〜50回目**の番号、動画名、成功／失敗、episode、開始step、Goal stepを1行ずつ表示します。1回目は`env_0.mp4`、50回目は`env_49.mp4`です。以前の配列形式の記録も残します。

### 正解画像と成功・失敗を見比べる

今回のSWM PushT評価では、開始から`goal_offset_steps`（通常25）後のデータ画像がGoalです。正解画像に写る**灰色のT字物体と青い操作点**に合わせます。動画内の緑のT字は別の描画用目標で、今回の成功判定に使うデータ由来Goalを示すものではありません。

この「同じ軌道の25step先をGoalにして、最大50行動で到達する」設定は[LeWM v3 Appendix F.1](https://arxiv.org/html/2603.19312v3#A6.SS1)に記載されています。Goal自体が緑のT字にはまっていない途中配置の場合もあり、今回の到達率を緑のT字へのはめ込み成功率とは解釈できません。

各有効stepで、操作点と物体の位置4成分のずれの距離が20未満、かつ物体の角度差が20度未満なら成功です。位置は512×512の環境座標で、表示画像上の20ピクセルという意味ではありません。両条件を同時に一度満たせば成功として記録します。操作点の位置も必要で、速度はこの成功条件に含みません。固定T字への95%被覆率とは別の評価です。

通常の評価コマンドは、正解画像・元動画・判定時点・数値を並べたHTMLを**評価出力先の`viewer/index.html`へ自動保存**します。別コマンドは不要です。CEMや物理環境を再実行せず、開始／Goalの画像・状態を評価時のhashと照合し、全成功フラグを保存状態から再計算します。データ全量のhash走査はしません。

MP4単体でもGoalを確認できるように、**`viewer/videos/trial_001.mp4`から試行順に、左＝実行動画、右＝固定Goalの動画**を生成します。試行番号・現在step・判定数値も動画内に表示します。viewerの再生と「Goal付きMP4を保存」は同じファイルを使い、assetsに動画を複製しません。元の評価記録`env_N.mp4`は評価ディレクトリに1部だけ保持し、元のfps・フレーム数を維持します。固定Goalの対象は灰色のTと青い操作点です。50ケースの動画生成は約12秒で、CEM再実行は不要でした。

HTTPでのシーク、絞り込み時の選択保持、連続切替の読み込み競合も修正し、PlaywrightでHTTP／file URLを検証しました。[保存整理・ブラウザ検証記録](reports/PUSHT_VIEWER.ja.md)。

画面生成だけが失敗した場合は、`status.json`の`phase=visual_report`・`evaluation_verified=true`とエラーを確認します。評価結果は保持されるので、下の単独生成コマンドを未使用の出力先へ実行できます。CEM評価を再実行する必要はありません。

既存の評価へ画面を追加したい場合だけ、次の単独生成コマンドを使います。

```bash
uv run --no-sync python -m mylewm.evaluation.build_pusht_ui \
  --evaluation output/pusht/eval_bt_compiled_step20000_seed42_cached \
  --output output/pusht/ui_bt_compiled_step20000_seed42
```

`--output`は未使用のディレクトリを指定します。作成済みなら`index.html`をブラウザで開きます。「成功のみ／失敗のみ」の絞り込み、試行番号、成功時点への移動、正解画像の半透明重ね合わせが使えます。失敗例には両閾値に最も近い時点を表示し、どの条件が未達かを示します。動画は15 fps保存、環境は10 Hzで、通常再生は環境時間の1.5倍速です。計画の計算待ち時間は含みません。

<a id="pusht-動かない極端に低い成功率を調べる"></a>

#### 動かない・極端に低い成功率を調べる

同じ起動コマンドに`--cem-audit`を追加します。1ケースの完走は動作確認にすぎません。性能の確認には同じmanifest・offset・seed・標準予算で複数ケースを使い、公式重みと比較してください。

- 起動ログの`reference_mean/reference_std`は参照データの`StandardScaler.mean_/scale_`です。`scale_`は分散でなく標準偏差（母分散の平方根）。checkpointの`training_mean/training_std`は学習manifest由来で、train-only・標本標準偏差を使用します。
- CEM候補は参照統計で標準化した座標です。モデル入力では参照統計で逆変換してから学習統計で正規化し、環境実行時には参照統計で逆変換します。PushTの物理行動はXYの相対指令で、環境が100倍して目標変位へ変換します。絶対画面座標ではありません。
- JSONの`action_path_diagnostics.cases`に実行行動のmin/max/mean/std/mean_abs・near-zero率（絶対値`1e-3`未満）、実step数、開始/終了状態、agent移動距離と累積移動、object移動距離を記録します。`physical_actions`には各stepのactive maskと前後状態も残ります。非ゼロ指令なのにagentが動かない場合は警告し、`performance_interpretation_valid=false`にします。この診断がtrueでも学習品質を保証しません。
- `cem_audit.jsonl`では候補の正規化前後と選択planの分布を確認できます。ゼロ付近の候補・選択planと、非ゼロ行動に環境が反応しない問題を分けてください。動画は評価終了後に出力されます。

Issue #20では、明示HDF5パス経路で画像列を落として指定Goal画像を渡せない問題と、動画が再利用バッファを参照する問題を修正しました。修正前の低成功率をモデル性能の証拠に使わず、既存ログ・結果は残して別出力先で再評価します。

同じケース・設定のランダム行動対照は、完走した公式評価JSONを入力します。CPUで実環境を動かして動画も保存する検証コマンドです。開始/Goal画像・状態のhashを参照結果と照合し、不一致なら停止します。持続する乱数状態を使い、毎step異なる行動を標本化します。

```bash
CUDA_VISIBLE_DEVICES='' uv run python -m mylewm.evaluation.evaluate_pusht_random \
  --reference-result output/pusht/eval_official_50/results.txt.json \
  --output output/pusht/random_control_50
```

実行した範囲と失敗ログは[Issue #20の修正検証](reports/PUSHT_ISSUE20.ja.md)を参照してください。

<a id="pusht-7-公式モデルと比較する"></a>

### 7. 公式モデルと比較する

提案側が終わってから、別の未使用名で公式を評価します。同じmanifest、ケース数、offset、コード版を使います。

```bash
bash scripts/evaluate_pusht.sh \
  --checkpoint .cache/stable-wm/pusht/lewm_object.ckpt \
  --output output/pusht/eval_official_50 \
  --gb10-cache-workaround --execute
```

両方の正常終了後に比較します。比較レポートも未使用名を使ってください（比較CLI自体に上書き拒否機能はありません）。

```bash
uv run python -m mylewm.evaluation.compare_paired \
  --baseline output/pusht/eval_official_50/results.txt.json \
  --candidate output/pusht/eval_bt_100000_50/results.txt.json \
  --output output/pusht/comparison_bt100000_official_50.json
```

`difference`は提案−公式の成功率差、`candidate_only_success`は提案だけ成功したケース数。信頼下限と観測差は区別します。`observed_no_degradation=true`でも任意の状況で性能が落ちない保証ではありません。**ツールは学習予算の同一性を検証しません。** 同更新数での方式比較には、別途そろえた学習が必要です。

`evaluation protocols differ`を無視したりJSONを改変して通してはいけません。ソース版・前処理・manifest・ケース・計画条件を確認し、必要ならそろえて再評価します。

現状の比較CLIは開始/Goal観測hashを照合しますが、実物理初期状態の一致までは検査しません。2026-09-10の実評価ではobject初期位置・角度に微差が見つかりました。`action_path_diagnostics.cases[].initial_state`も比較し、CLI通過だけで厳密に同じ物理初期状態だったと主張しないでください。原因と影響は未解決です。[70k/100kの実測記録](reports/PUSHT_ISSUE20.ja.md)を参照してください。

<a id="pusht-8-世界モデルrolloutの速度を測る"></a>

### 8. 世界モデルrolloutの速度を測る

制御成功率とは別に、環境・CEM反復・Goal encoder・I/Oを除いたE/A/Fのrollout時間を測れます。既定のCEM候補数300を一つのbatchにし、観測3フレームから予測器を1回または20回呼ぶ時間をCUDA eventで測ります。入力は乱数なので、この値は精度・実環境の総制御時間ではありません。

```bash
uv run python -m mylewm.evaluation.benchmark_pusht_rollout \
  --checkpoint output/pusht/bt_spectral_v2_100k_s3072/step_100000_object.ckpt \
  --checkpoint .cache/stable-wm/pusht/lewm_object.ckpt \
  --output output/pusht/rollout_latency_bt_official.json
```

出力JSONには各checkpoint hash、warm-up後20反復の平均・中央値・最小/最大、peak GPUメモリを保存します。BTの学習専用Tは推論exportに含まれないため、この測定ではE/A/Fの計算量だけを比較します。

<a id="pusht-9-別ステップケース数とトラブル"></a>

### 9. 別ステップ・ケース数とトラブル

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

<a id="libero"></a>

## 第2章 HDF5（LIBERO-10）の環境評価

更新日：2026-09-10。新規学習・環境評価は明示依頼時だけ実行し、自動実行・定期監視はしない。

<a id="libero-比較の位置付け"></a>

### 比較の位置付け

上流の[公式LeWM](https://github.com/lucas-maes/le-wm)とローカル`lewm/`には、LIBERO用の学習config・評価config・配布checkpointがない。従ってLIBEROの`--mode raw`は公式LeWMのE/A/F・一段予測・全次元SIGRegを使う**ローカルの公式コア比較ベースライン**であり、公式提供のLIBERO実装／結果とは呼ばない。`--mode bt`は同じ基盤に学習専用BTを追加する提案側である。

Raw/BTは同じ新規manifest、seed、batch、workers、更新数、optimizer、固定初期状態、CEM条件で比較する。LIBERO-10の10タスクを一つの共有モデルで学習する。短期smokeのlossや1初期状態の成否は性能比較ではない。

<a id="libero-1-分割を作る"></a>

### 1. 分割を作る

データを複製せず、分割とtrain-only行動統計をmanifestへ記録する。既にmanifestがあれば変更・再作成しない。

```bash
cd "$(git rev-parse --show-toplevel)"
export LIBERO10_DATASET=/absolute/path/to/libero_10
ls "$LIBERO10_DATASET"/*.hdf5
uv run python -m mylewm.training.train_libero prepare \
  --dataset "$LIBERO10_DATASET" \
  --manifest output/manifests/libero10/manifest.json
```

正常ならtrain 400 demonstrations、validation 50、test 50と表示され、隣に`files.json`が作られる。

<a id="libero-2-osmesa画像監査"></a>

### 2. OSMesa画像監査

このGB10では隔離したOSMesa wrapperを使う。評価前に公式LIBERO task ID 0〜9の全てを監査する。`--task-id`はLIBERO task IDでありHDF5のソート番号ではない。出力は必ず対応する`task_N`に保存する。

```bash
for task_id in {0..9}; do
  bash scripts/run_libero.sh -m mylewm.evaluation.audit_libero_images \
    --task-id "$task_id" \
    --output "output/libero10/render_audit/task_${task_id}"
done
```

各`report.json`で`passed: true`、`render_backend: osmesa`、native MAEが縦反転MAEより小さいことを確認する。評価器はtask名、MuJoCo版、backendが一致しない監査を拒否する。

<a id="libero-3-rawbtを新規学習する"></a>

### 3. Raw／BTを新規学習する

学習の通常起動はサイズ・更新時刻の確認です。全量SHA-256の再検証には`--verify-data`を追加します。新規prepareはhashを記録します。評価器自身の検証仕様はこの学習オプションでは変わりません。

先に短期接続確認をする。下はRaw例で、BTは`--mode bt`だけを変え、その他をそろえる。短期checkpointを本学習へ延長しない。

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run python -m mylewm.training.train_libero train \
  --mode raw --manifest output/manifests/libero10/manifest.json \
  --output output/libero10/raw_smoke \
  --steps 100 --batch-size 16 --workers 0 --seed 3072 \
  --warmup-steps 10 --lr 5e-5 --min-lr 0 \
  --save-every 50 --diagnostics-every 25 --deterministic
```

本比較は新規outputで逐次実行する。RawとBTを同時起動しない。

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run python -m mylewm.training.train_libero train \
  --mode raw --manifest output/manifests/libero10/manifest.json \
  --output output/libero10/raw_train \
  --steps 100000 --batch-size 128 --workers 4 --seed 3072 \
  --warmup-steps 500 --lr 5e-5 --min-lr 0 \
  --save-every 5000 --diagnostics-every 1000 --deterministic
```

BTは`--mode bt --output output/libero10/bt_train`に変えるだけにする。各runの`config.json`の`initial_model_sha256`を照合する。最終評価へ進めるのは`completed.json`が`{"step": 100000, "state": "completed", "recipe": "controlled_comparison"}`であり、`step_100000_object.ckpt`がある場合だけである。`resume.pt`は再開専用で評価に渡さない。

<a id="libero-4-実環境cem評価"></a>

### 4. 実環境CEM評価

この節は最終評価向けです。学習継続中の保存済み`step_N_object.ckpt`については[途中checkpoint評価手順](#intermediate)の別GPU実行を使えます。途中評価には`completed.json`は不要です。

既定はdry-runで、checkpointをロードせず環境も動かさずoutputも作らない。下はRaw最終checkpointの固定10タスク・各50初期状態評価である。BTもcheckpointとoutputだけを変え、同じmanifest・監査・CEM条件を使う。

```bash
# 設定・監査・入力だけを確認する。
bash scripts/run_libero.sh -m mylewm.evaluation.evaluate_libero \
  --checkpoint output/libero10/raw_train/step_100000_object.ckpt \
  --manifest output/manifests/libero10/manifest.json \
  --render-audit-dir output/libero10/render_audit \
  --output output/libero10/eval_raw_confirm50

# dry-runを確認した後だけ環境を動かす。
bash scripts/run_libero.sh -m mylewm.evaluation.evaluate_libero \
  --checkpoint output/libero10/raw_train/step_100000_object.ckpt \
  --manifest output/manifests/libero10/manifest.json \
  --render-audit-dir output/libero10/render_audit \
  --output output/libero10/eval_raw_confirm50 --execute
```

評価中に学習を併走しない。成功は`env.check_success()`だけで決め、デモ状態距離やBCスコアと混ぜない。終了時は`status.json=succeeded`、`summary.json`、10×50行の`episodes.jsonl`、全`taskN_initM.npz`を確認する。途中終了は未測定であり0%ではない。

<a id="libero-5-ブラウザの可視ui"></a>

### 5. ブラウザの可視UI

保存済み評価から、初期・goal・最終の2カメラ（agent view | wrist view）を静的HTMLにまとめる。UIは記録を読むだけで、CEM・環境・checkpointを再実行しない。

```bash
uv run python -m mylewm.evaluation.build_libero_ui \
  --evaluation output/libero10/eval_raw_confirm50 \
  --output output/libero10/ui_raw_confirm50

# ローカルブラウザで http://127.0.0.1:8000 を開く。Ctrl-Cは表示だけを停止する。
uv run python -m http.server 8000 --directory output/libero10/ui_raw_confirm50
```

`index.html`を直接開くこともできる。画面のsuccess/failureは固定実行結果であり、goalへの見た目の近さによる判定ではない。

<a id="libero-実行確認した範囲"></a>

### 実行確認した範囲

2026-09-10に、新manifest作成、Raw CPU 2更新（`completed.json`と`step_2_object.ckpt`）、task ID 0のOSMesa画像監査、RawとBT短期checkpointの各1初期状態・budget 1 CEM評価、dry-run、`status.json=succeeded`、HTML UI生成を実行した。これは接続確認のみであり、100,000更新・全10監査・各50初期状態・Raw対BT性能比較は未実施である。

<a id="intermediate"></a>

## 第3章 学習途中のcheckpoint評価

学習済みの`step_N_object.ckpt`は、本学習の終了前でも評価できる。途中評価には`completed.json`は不要。`last.ckpt`や`resume.pt`は再開用なので渡さない。以下は自分で作成した信頼済みcheckpointを前提とする。

<a id="intermediate-実行前の確認"></a>

### 実行前の確認

`nvidia-smi -L`と`nvidia-smi`で学習GPUと評価に使える余力を確認する。以下の`CUDA_VISIBLE_DEVICES=1`は物理GPU 1で評価する例で、実際のGPU番号またはUUIDに変更する。この指定で評価プロセス内では選んだGPUが`cuda:0`になる。別端末で実行し、学習側の設定は変えない。

共有メモリ型GB10では、同じGPUでも、学習と評価を載せるRAMに余裕があれば併走できる。dry-runの後に評価を開始し、`nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader`、`ps -o pid,rss,cmd -p <評価PID>`、`free -h`で評価中のGPU・CPU常駐メモリとavailable RAMを記録する。高速化後のPushT固定50ケースでは、評価プロセスのGPU観測値341 MiB、CPU RSS約2.44〜2.66 GiB、PyTorch peak allocated約106 MiBだった。追加常駐量は数GiB規模だが、共有メモリの各指標は単純加算せず、起動時ピークを含む予約量とも扱わない。条件ごとにavailable RAMの余裕を確認する。計測条件と旧実装との比較は[高速化・メモリ実測](reports/CACHED_CEM.ja.md)を参照する。併走中は計算資源を共有するため速度は変動する。OOMやRAM不足が発生した場合は評価を停止してログを保持する。

checkpointの保存完了と、学習が次の更新へ進んだことをログで確認する。ファイルが存在するだけでは書込み終了の証拠にならない。特にPushTの保存は実行環境によって直接書込みになる。以下の50,000は例で、保存完了した更新数へ変更する。入力checkpointは評価終了まで移動・削除・上書きしない。

評価環境はcheckpointと互換な依存版を事前に準備する。学習中の共有`.venv`に`uv sync`を実行しない。下記の`UV_NO_SYNC=1`は起動時の自動同期も抑えるため、準備済み環境が必要。別環境を使う場合は`UV_PROJECT_ENVIRONMENT`も設定する。別GPUでもCPU・ディスクは共有するため、学習速度への影響はあり得る。

<a id="intermediate-pusht"></a>

### PushT

通常のlauncherは`CachedCEMSolver`を使います。候補300・反復30・elite30・seed・batch1・計画と実行の長さを維持し、同一探索内の観測／Goal埋め込みを再利用します。300候補へ展開した画像は転送前に1サンプルへ絞ります。キャッシュは環境batch・再計画ごとに破棄します。`--cem-audit`は従来の監査solverを使います。ソースとsolverクラスは結果へ記録されるため、旧評価と同一実装という意味ではありません。

HDF5はmanifest内の任意パスを使う。移転先を指定する場合は以下の両コマンドへ`--dataset /absolute/path/to/pusht_expert_train.h5`を追加する。元manifestを書き換えず、サイズを確認し、`--verify-data`指定時は保存済みhash（存在する場合）とも照合する。

リポジトリ直下で、実際のrun名・保存済みmanifestに置き換える。各出力先は未使用名にする。

```bash
export PUSHT_EVAL_CKPT=output/pusht/spt_raw/step_50000_object.ckpt
export PUSHT_EVAL_OUT=output/pusht/eval_raw_step50000_seed42
CUDA_VISIBLE_DEVICES=1 UV_NO_SYNC=1 bash scripts/evaluate_pusht.sh \
  --checkpoint "$PUSHT_EVAL_CKPT" \
  --manifest output/manifests/pusht/manifest.json \
  --output "$PUSHT_EVAL_OUT" --num-eval 50 --seed 42

# 上のdry-runを確認してから実行する。
CUDA_VISIBLE_DEVICES=1 UV_NO_SYNC=1 bash scripts/evaluate_pusht.sh \
  --checkpoint "$PUSHT_EVAL_CKPT" \
  --manifest output/manifests/pusht/manifest.json \
  --output "$PUSHT_EVAL_OUT" --num-eval 50 --seed 42 --execute
```

BTはcheckpointと出力名をBT runに変更する。GB10のcache解放フラグは学習データの再読込を増やし得るため、この同時評価例では付けていない。評価も通常は全量hashを走査せず、`--verify-data`指定時だけ走査する。

`spt_raw`はSIGRegを直接適用するモデルで、公式配布checkpointとは別のローカル学習runである。70k同士を評価する場合は両方の入力を`step_70000_object.ckpt`にし、それぞれ未使用の出力名を指定する。学習ログの最新更新数でなく、実際に保存完了したcheckpointを選ぶ。

終了コード0、出力先の`status.json`が`succeeded`、`results.txt.json`を確認する。詳細は[PushT評価手順](#pusht)の終了確認を参照する。

<a id="intermediate-libero-10"></a>

### LIBERO-10

[LIBERO評価手順](#libero)のOSMesa環境と対象タスクの画像監査を事前に準備する。`scripts/run_libero.sh`は`LIBERO_OSMESA_DIR`等で外部環境の場所を指定できます。[環境設定](../README.md#pcごとの外部環境設定)を参照し、対象PCでOSMesa構築・画像監査を通してから使ってください。

```bash
export LIBERO_EVAL_CKPT=output/libero10/raw_train/step_50000_object.ckpt
export LIBERO_EVAL_OUT=output/libero10/eval_raw_step50000
CUDA_VISIBLE_DEVICES=1 UV_NO_SYNC=1 bash scripts/run_libero.sh -m mylewm.evaluation.evaluate_libero \
  --checkpoint "$LIBERO_EVAL_CKPT" \
  --manifest output/manifests/libero10/manifest.json \
  --render-audit-dir output/libero10/render_audit \
  --output "$LIBERO_EVAL_OUT" --device cuda:0

# 上のdry-runを確認してから実行する。
CUDA_VISIBLE_DEVICES=1 UV_NO_SYNC=1 bash scripts/run_libero.sh -m mylewm.evaluation.evaluate_libero \
  --checkpoint "$LIBERO_EVAL_CKPT" \
  --manifest output/manifests/libero10/manifest.json \
  --render-audit-dir output/libero10/render_audit \
  --output "$LIBERO_EVAL_OUT" --device cuda:0 --execute
```

終了コード0、`status.json=succeeded`、`summary.json`と`episodes.jsonl`を確認する。UI生成はLIBERO評価手順の`--evaluation`に今回の出力先を指定する。

<a id="intermediate-結果の扱いと確認範囲"></a>

### 結果の扱いと確認範囲

結果にはrun名と更新数を付け、「50,000更新時点の途中評価」等と記録する。比較は同じmanifest・ケース・seed・CEM予算で行う。繰り返し見たconfirm集合はモデル選択に影響するため、未使用の最終テスト成績とは呼ばない。

2026-09-10、ユーザー依頼で学習停止後にGB10上で新Raw 10k・70k、旧BT 15k・70kなどの途中checkpointを実評価し、終了コード・status・結果・動画を確認した。70k同士の固定50ケースはSIGReg 45/50、BT 47/50。[実測と比較上の留保](reports/PUSHT_ISSUE20.ja.md)を参照。別GPUでの同時学習・環境評価の完走、RTX側の該当BT 10k、LIBEROのこの途中評価手順の実環境完走は未検証であり、PushTの実績で代用しない。
