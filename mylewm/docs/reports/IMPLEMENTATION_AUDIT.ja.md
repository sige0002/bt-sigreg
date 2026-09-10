# 実装・運用監査の履歴（2026-09-07〜09）

本書は当時の結果・失敗・未完了判断を保持する履歴です。「学習中」「未実装」「監視開始」は現在の指示ではありません。最新状態は[検証状況](../VALIDATION.ja.md)、制御成績は[PushT評価](PUSHT_CHECKPOINT_EVALUATION.ja.md)を参照。2026-09-09に定期監視は廃止され、専用シェルは整理で削除しました。

## 評価の終了検知・独立監視（2026-09-09）

60,000更新の自動評価はcheckpoint保存後に実際に起動したが、CUDA初期化でOOMとなった。失敗を主エージェントが確認したのはユーザーの進捗照会時であり、起動予約とエージェントへの監視依頼だけでは終了検知・通知が完結していなかった。

改善：`evaluate_pusht.sh`が`status.json`へrunning、結果検証後のsucceeded、未完了終了時のfailedを保存する。SIGKILL等では終了処理を実行できないため、別プロセスの`watch_evaluation.sh`が30秒間隔でsystemdの状態・終了コードと検証済み結果を照合する。監視状態は別JSONとjournalへ保存し、監視エラー・時間超過も成功と区別する。無限再試行・他プロセス停止・条件変更は行わない。

検証：launcher/監視の13テスト合格。成功、失敗、status欠損、通知コマンド失敗を模擬確認。全回帰はCPU限定で107合格・5スキップ。実際の失敗済み60,000評価serviceに対する監視もfailedを検知した。ただし、このPCには`org.freedesktop.Notifications`サービスがなく、実通知は失敗した。そのエラーも保存した。**外部通知やチャットへの自動連絡は未成立**であり、失敗を認識できる無人運用が完全に解決したとは報告しない。通知先の選択・許可・送達試験が残る。

OOM再確認では対象データのclean cache解放だけで最小CUDA割当が失敗する回があり、ドライバのcontext確保失敗を確認した。torch import後に解放する順序へ変更した。診断中に`CUDA_DEVICE_MAX_CONNECTIONS=1`を一度試すと成功し、その後は従来の通常設定でも成功したため、単一変更の因果効果を断定しない。[NVIDIAの接続数設定の説明](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/05-appendices/environment-variables.html)はワークキュー設定であり、今回のOOM解消保証ではない。再評価には接続数変更を持ち込まず、通常設定と同じ50ケース・計画予算を使う。

再評価は新規`retry_devicefix/step_60000_retry/`で、旧失敗出力は保存。評価service `bt-pusht-eval-60000-retry-20260909.service`と独立監視`bt-pusht-watch-60000-retry-20260909.service`を起動し、主エージェントも完了まで追う。ここでの記載時点では評価中であり、成功率は別途レポートへ記録する。

完了追記：01:48 JSTに45/50（90%）で終了し、比較条件と結果を照合して[レポート](PUSHT_CHECKPOINT_EVALUATION.ja.md)へ追記した。独立監視の実運用では終了済み一時unitの自動回収が起きたため、追加修正で`result_verified`（launcher検証済み結果あり）と`service_exit_verified=false`（service終了コード再確認不能）を分離した。unit不在で結果もなければmonitor_error。RemainAfterExitでactive/exitedとなるケースも完了として扱う。これらを追加した関連16テストが合格し、実6万結果の監視でもresult_verifiedを確認した。最初の監視エラーと通知失敗記録は別ファイルとして保持している。

追加修正後の全回帰：`CUDA_VISIBLE_DEVICES='' PYTHONPATH=.:lewm .venv/bin/python -m pytest mylewm -q`で110合格・5スキップ。`git diff --check`合格。学習serviceは継続し、学習ソースhashも開始時と一致する。

## 初心者向けPushT評価手順とlauncher（2026-09-08）

[評価手順](../EVALUATION.ja.md#pusht)と`tools/evaluate_pusht.sh`を追加。既定dry-run、実行時だけ新規output作成・CUDA先行初期化・監査付き既存評価を行う。対象データclean cache解放は明示フラグ。チェックポイント/manifest/launcherのhashと起動引数を残し、正常終了後にケース・成功率の整合性を確認する。途中再開・上書きはしない。

検証は`bash -n`、実20,000更新checkpointのdry-run、ヘルプ、仮データによる入力拒否と模擬subprocessでの成功/失敗処理。全回帰は`CUDA_VISIBLE_DEVICES='' PYTHONPATH=.:lewm .venv/bin/python -m pytest mylewm -q`で**102合格・5スキップ**。稼働中の学習・評価との競合を避け、今回の全体再確認はCPUのみ。新launcherから実50ケースは追加起動しておらず、模擬結果を実制御成功率に数えない。先のGPU混在テスト14件合格とは実行範囲を区別する。

同じ評価本体を使う既存ローカルlauncherでは、5,000更新2/50（4%）、15,000更新23/50（46%）を完了し、ケース・protocol・初期/Goal一致とcheckpoint hashを確認した。10,000更新はユーザー指示で途中中断し、20,000更新へ切替。公式の同50ケース再評価も予定。途中checkpointと公式配布checkpointの学習予算は異なるため、方式の優劣の結論にはしない。

## PushT中間checkpoint評価：起動障害の切り分け（2026-09-08）

5,000/10,000/15,000更新を共通confirm先頭50ケースで評価する途中、二つの独立した起動障害を確認した。以下は障害の解消確認で、成功率の結果ではない。

- GB10のCUDA初期化OOM：カーネルログは`kgrctxAllocMainCtxBuffer`のメモリ確保失敗。学習・QwenのGPU使用量は約13.8/32.8 GiBで、評価必要量ではない。PushT HDF5だけにread-onlyで`POSIX_FADV_DONTNEED`を適用すると、システムfree RAMが約2.4→34 GiBへ増え、新規CUDA初期化・最小割当が成功した。NVIDIAも[キャッシュによる容量内OOM](https://nvidia.custhelp.com/app/answers/detail/a_id/5776/kw/hdr/related/1)を説明している。全体cache削除、ドライバ変更、他プロセス停止は行っていない。
- CEM接続のデバイス不一致：SWMはCPUの過去行動とGPUの候補行動を渡すが、`PlanningActionAdapter`がGPU統計とCPU行動を直接演算していた。正規化前に行動をモデルのデバイスへ移すよう修正した。正規化の数式・値・物理探索分布は変更しない。元の入力は変更しない。

`PYTHONPATH=.:lewm .venv/bin/python -m pytest mylewm/tests/test_evaluation_contract.py -q`で14件合格（新規CUDA混在回帰を含む）。最初のテスト起動はPYTHONPATH不足で公式fixtureの`jepa` importが失敗し、正しい経路で全件を再実行した。学習config内の全source SHA256が現行ファイルと一致することも確認した。

再評価はGit対象外の`output/pusht/bt_spectral_v2_checkpoint_eval_50cases/`に隔離。ローカルwrapperで対象データのclean cache解放とCUDA先行初期化を行い、既存`lewm/eval.py`を実行する。CEM予算・50ケース・精度・seedは維持し、監査ソース変更に合わせ公式モデルも同じ50ケースで再評価する。旧失敗ログは保持。評価完了・性能結果は別途確認が必要。

## PushT BT v2・100,000ステップ開始（2026-09-08）

ユーザーの新しい実行指示に基づき、短期診断とは別の新規共有初期値からPushTのみ開始した。batch128、warmup500、maxLR5e-5/minLR0、seed3072、workers4、保存/validation5,000、T/勾配診断1,000ステップごと。BTの構造・損失は短期診断で確認したv2のまま。Raw/TC/LIBEROの長時間学習は起動していない。

出力 `.cache/stable-wm/pusht/bt_spectral_v2_100k_s3072/`、systemd user unit `bt-pusht-spectral-v2-100k-s3072.service`。初期確認時55ステップ、loss2.83839、service active/running。最初のloss3.86009、T勾配非零、GPU peak allocated約13.75GB。batch128のSIGReg値をbatch16短期診断と直接比較しない。12,800,000クリップ提示予定で、7.7787提示周回相当だが均等周回でも公式100エポック再現でもない。

学習時コードのsnapshotは同階層 `bt_spectral_v2_100k_s3072.sources.tar.gz`、SHA256 `15b109c9d44d94f6a3b1b0dffa096d27779c3b20610c6e0a9c9e9719e8358bff`。新規共有初期値・コマンドは[README](PUSHT_TRAINING_100K.ja.md)に記録した。未コミット状態のコードもsnapshot/config hashで識別する。

lunaサブエージェントを10分間隔の監視へ割り当て、NaN/Inf・進捗停止・異常終了を通知するよう指定した。監視はセッション内で継続するもので、セッションやOS再起動後の自動復帰を保証しない。学習サービスに自動再起動は設定していない。以下は開始前の短期診断と監査履歴。

## BT v2修正・相互レビュー・実データ短期診断（2026-09-08）

今回依頼された段階1–3を完了。旧Frobenius試作をCayley特異値構成へ置換し、学習bias付き中心化half-linear/half-tanhを使用する。原点固定・距離境界を維持して旧奇関数・総核ノルム予算制限を外した。[現行数式](../research/BT_SIGREG.ja.md#6-現行実装cayley特異値制約v2)。新しい実装担当と独立数学・安全監査担当の2体が手順・所見を直接交換した。実在Google DeepMind研究者ではなくAIレビュー。今回、利用可能な一覧に学習実装用skillはなく、使っていない。

全回帰 **97件合格**（BT27＋既存69＋追加診断1、13.15秒）。公式forwardの恒等時一致、CPU/CUDA再開、原点・距離・ヤコビアン・逆写像、矩形行列の作用素ノルム、非奇関数性・全方向の変形、gradcheckを確認した。診断hookも再開一致テストに含む。PyTorchのGB10対応範囲・fork・非推奨API等の警告は残る。

### 実行条件と証拠

PushTとLIBERO-10を**それぞれ新規初期値から100回のoptimizer呼出し、batch16、seed3072**で実行し、両方exit0で終了した。warmup10、最大LR5e-5、最小LR0のcosine。最終100回目はLR0なので重み更新量は0（非零LRは99回）。λ=.09、depth2、κ=.2、hidden192、workers0、決定論モード。モデルはGPU bfloat16 autocast、T/SIGRegはFP32。10万更新、本制御比較、環境成功率試験は起動していない。

- PushT：既存HDF5・既存train/validation分割。訓練1,645,509クリップから1,600提示、今回の抽出は1,600個すべて異なる。
- LIBERO-10：既存10実HDF5、各タスク40訓練demo、2つの実カメラ。105,558クリップから1,600提示、異なるクリップは1,591。通常の復元抽出による重複で、画像複製や架空ビューではない。タスク0–9の提示数は132/149/181/248/145/176/151/148/156/114。タスクIDはモデルへ入力しない。
- データ内容SHA256、モデル・コード・依存版・設定は各 `config.json`、提示量は `budget.json`、全100行は `metrics.jsonl`。50/100時点の推論重みと100時点のT/optimizerを保存した。出力は `.cache/stable-wm/{pusht,libero10}/bt_spectral_v2_smoke_20260908_s3072/`。いずれもGit対象外。
- ダウンロード、外部checkpoint読込、他GPUサービスの停止は行っていない。学習後の重み差分は今回自己生成した初期・最終checkpointだけを比較した。任意pickleの安全化をしたわけではない。

### 実測結果

| 指標 | PushT | LIBERO-10 |
|---|---:|---:|
| 訓練loss・先頭10更新平均 → 末尾10更新平均 | 0.63386 → 0.35443 | 0.63032 → 0.27713 |
| 検証予測MSE・50 → 100更新 | 0.29149 → 0.08737 | 0.04769 → 0.04985 |
| 検証SIGReg・50 → 100更新 | 9.67903 → 4.09117 | 6.00946 → 5.44803 |
| 最終T勾配ノルム・clip前 | 0.003376 | 0.002898 |
| 最終z/u分散trace・時刻平均 | 80.66684 / 80.77037 | 91.79527 / 91.88358 |
| 最終T相対変位RMS | 0.0650% | 0.0504% |
| 最終標本距離比min–max | 1.000581–1.000695 | 1.000448–1.000525 |
| 通常更新の時間中央値 | 0.579秒 | 0.240秒 |
| このプロセスのGPU peak allocated | 約2.12 GB | 約3.77 GB |
| 推論モデルのパラメータ | 18,034,478 | 18,427,874 |
| Tの訓練専用パラメータ | 147,840 | 147,840 |

時間中央値はmetricsの経過時間差から、疎な診断更新と直前にvalidation/saveがある更新を除外した値。データ取得・前処理を含み、純粋なGPU演算時間ではない。初期化・データハッシュ・最終validation/saveの時間は含まない。batch128や10万更新の所要時間へそのまま外挿しない。

Tの初期→最終パラメータ差L2は、特異値パラメータ/回転/biasの順にPushT 0.05641/0.09199/0.02745、LIBERO 0.04301/0.16608/0.01752。Tがoptimizerから漏れていたわけではない。両方の再開checkpointがstep100と2optimizerグループを持つ。

**判定：修正・接続・実データでの学習動作は確認できたが、BTの性能効果は未確認。** 訓練loss低下を世界モデルの性能向上と呼ばない。LIBEROでは検証予測MSEが少し悪化した。100更新ではTが近恒等であり、非線形変形能力が有益に使われた証拠はまだない。同条件Raw/TC比較も未実施。

注意：batch16の時刻別共分散ランクは高々15。微小な負の最小固有値は丸めを含むため崩壊認定に使わない。最大固有値は全時刻最大、traceは時刻平均なので、その比を第一成分の寄与率にしない。validationは既存trainerのバッチ平均で、LIBEROの最終小バッチを含むため母集団Gaussian性の尺度でもタスクmacro平均でもない。行動シャッフルで出力が変わっただけでは正しい因果モデルの証拠にならない。

次段階は、本学習条件のbatch・スケジュールでTの学習速度/非線形性と尺度効果を切り分け、同条件Raw/TCとの比較へ進めるか判断すること。今回の100更新を10万更新のcheckpointへ無条件に継ぎ足さない。旧試作の監査履歴は以下に保持する。

## BT実装・接続・テスト（2026-09-08）

ユーザー指定の段階1–3（アルゴリズム実装、両benchmarkへの接続、数式・勾配・保存再開・推論分離の確認）を実施。実データの短期学習・100,000更新学習・環境成功率評価は実行していない。合成入力を用いる数回のoptimizer更新は統合テストとしてのみ実施した。

実装は `mylewm/bt_sigreg.py`。詳細な式と制約の保守性は[提案書 §9](#付録旧frobenius試作)を参照。新規テストは `mylewm/tests/test_bt_sigreg.py` に集約し、ファイルを用途別に増殖させていない。

| 確認内容 | 結果と範囲 |
|---|---|
| 数式・勾配 | FP64で入力とTの重みの有限差分gradcheck合格 |
| 距離境界 | κ=0/.2/.8、3ブロックで大きいLRの4更新後も、重みの実測特異値、入力間距離、ヤコビアン特異値が理論境界内 |
| 可逆性・分散 | テスト内の固定点反復で逆変換を確認、経験分散traceの上下界も確認。数値検査は大域的証明の代替ではない |
| 恒等初期化 | 出力がzと完全一致し、初回W2勾配・次回W1勾配が非零 |
| Raw一致 | depth=0と初期depth=2で共通Raw経路と一致。さらに実際のPushT／2視点LIBERO構成で、公式lejepa_forward＋公式SIGRegとloss・全世界モデル勾配が完全一致。CPU FP32・同一乱数・λ=.09・有限の3行動chunkによる比較で、公式データ前処理全体の再現ではない |
| 予測座標・教師勾配 | MSEが元のzの一段予測であること、Tへ予測損失勾配が流れないこと、最終未来教師フレームへ勾配が流れることを確認 |
| 条件付けの不在 | Tのバッチ順序、時刻のreshape、単独サンプル、train/evalで同じ写像を確認。既存EのBN依存まで解消した主張ではない |
| 共同更新・再開 | 共有trainerを小型合成モデルへ接続し、1視点/2視点・10/28次元行動の各設定についてCPUとCUDAで4更新連続と2更新後resumeが完全一致。E/A/F、T、AdamW、scheduler、乱数・次バッチ位置を比較 |
| 再開拒否 | κ変更をconfig不一致として拒否。全設定の比較処理は既存の再開契約を利用 |
| 実モデル接続 | 本物のPushT／TwoViewJEPA構成へ合成28×28画像を入力し、CPUで各2更新。実画像や実データ学習ではない |
| 推論export | Tを含まないモデルを保存・読み戻し、実モデルのget_costが完全一致。Tの呼出しを禁止しても計画コスト計算が通る |
| CLI | 両trainerのbt引数とRaw/TC/BTのdry-run計画を確認 |

実行：`CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python -m pytest mylewm/tests -q`。**91件合格（既存69件＋BT22件）**。約15秒。PyTorch GPU対応範囲、pin_memory、fork/Lance等の依存警告は残ったが失敗なし。Tは訓練時FP32、数学検算はFP64。区間演算による厳密な浮動小数点保証ではない。

### 2体のAIによる相互監査

研究者役と数学・安全監査役が、実行方針と所見を直接交換した。実在Google DeepMind研究者の査読ではない。新BTの更新・再開・native推論経路に重大な不整合は見つからなかったが、**奇関数制約と核ノルム総変形予算**を両者が独立に確認した。詳しくは[提案書 §9](#付録旧frobenius試作)。接続テスト合格をもって、研究目的を十分検証できる完成アルゴリズムとは扱わない。性能学習前に写像の表現力を再検討する。

安全確認では、新BTに外部送信・秘密収集・動的コード実行は見つからなかった。ただし依存全体の安全保証ではない。既存trainerの `torch.load(weights_only=False)` は信頼済みcheckpoint専用で、ロード後のschema/hash確認は悪意あるpickleの実行防止にならない。今回の追加テストは新規モデルと自己生成の一時checkpointのみを使い、外部pickle・ダウンロード・実データ学習は使わなかった。

OpenAI Docs skillをエージェント運用の確認に使用し、[公式の外部アクセス安全指針](https://learn.chatgpt.com/docs/cloud/internet-access)に沿って、取得文書を実行命令として扱わない方針を確認した。世界モデル用の数学skillではない。相互監査は手続き上の確認であり、現在のfull-access環境に隔離を追加したものではない。

未完了：実データでの制約強度・表現力の診断、BT固有のz/u統計の長期監視、実計算予算比較、実モデル全体でのGPU長期resume評価、PushT/LIBERO-10の制御成功率。κ=.2とFrobenius構成の有用性はまだ判断しない。以下のIssue監査は以前の記録として保持する。

構成整理後の注記：以下は2026-09-07時点の監査履歴。評価・監査CLIは現在 `mylewm/tools/`、テストは `mylewm/tests/` に配置する。Sub-JEPA・RBG混合の専用trainerと旧実験runnerは削除済み。過去の件数・コマンドは当時の記録であり、現在の推奨手順は[README](../../README.md)を参照。

更新日：2026-09-07。対象はIssue #1〜#13。ユーザー依頼により#1と#3を検証結果付きでcloseした。残る11件はopen。closeはローカルでの完了確認に基づき、対応コード・記録が未コミット／未pushであることもGitHubコメントに明記した。
前提の学習・評価監査（#1〜#6）から対応し、研究候補（#7〜#13）の採用を先に決めない。
ユーザー指示により長時間学習は停止中。生成ログとチェックポイント136ファイル、約4.54 GiBをOSのごみ箱へ移した。データと公式配布重みは保持した。短い自動テストの一時出力は実験結果とは区別する。

## #1：学習率と保存・再開

実装：`training_state.py`と共通`train_rbg.py`。PushT、LIBERO、Sub-JEPA、Gaussian混合対照のCLIにも同じスケジュール引数を適用した。

- 更新番号を1始まりとして、最初に使う学習率は`max_lr / warmup_steps`。warmup最終更新がmax_lr、その後コサイン減衰、最終更新がmin_lr。既定は50,000/500/5e-5/0、batch 128。
- 全パラメータグループの使用学習率を毎更新保存する。schedulerの位置はoptimizer更新が完了してから進める。
- モデル、AdamW、正則化器、scheduler、Python/NumPy/PyTorch/全CUDA乱数、次のバッチ番号、初期化アーティファクトのSHA256を保存・復元する。
- DataLoaderは学習演算と別のGeneratorを使う。クリップ取得は元から更新番号に基づく決定論的な復元抽出で、画像のランダム拡張はない。worker側のランダム拡張を追加した場合、その乱数状態の扱いは別途必要。
- validationのPython/NumPy/PyTorch乱数は保存・復元する。再開時にcheckpointより先まで書かれたログは退避し、checkpoint直後のログ書き込み前に中断した場合は保存済み行を復元する。
- config互換性は出力パス等の運用引数以外の全項目で確認。学習率設定、損失係数、初期化、コード・アダプター、manifest、精度等の差分を拒否する。#2の実データSHA256も保存・再開互換性へ組み込んだ。
- 正則化は専用乱数列を使い、モデルのDropoutやデータ取得と分離する。正則化器の初期化と射影数の差によって次のモデル更新の乱数がずれない。validationによる正則化の乱数位置変更も復元する。

検証：`test_training_state.py`は実際の共通学習ループへ小型のBN/Dropout付きモデルを接続し、連続6更新と2更新後に中断・再開する場合を比較した。CPU float32、GPU bfloat16それぞれworker 0/2で、取得バッチ、LR列、パラメータ、AdamW状態、正則化器、全乱数状態、validationを含むlossが完全一致（atol=rtol=0）。GPUは`CUBLAS_WORKSPACE_CONFIG=:4096:8`と決定論アルゴリズムを有効化。この検証は実LeWM全体のGPU再現性や性能を保証しない。

## インストール済み公式経路のスケジューラ監査

stable-pretraining 0.1.8の`module.py`はmanual optimizationを使い、optimizer更新の直後に`scheduler.step()`を呼ぶ。`interval: epoch`という設定文字列から更新頻度を推定してはいけない。

`optim/lr_scheduler.py`の既定は推定最適化更新数の1%のwarmup、warmup_start_lr=0、eta_min=0。インストール済みのクラスを実際に50,000回進め、更新前LRを確認した。

| 更新 | 依存実装の使用LR（500/50,000を明示） | 今回の比較用LR |
|---|---:|---:|
| 1 | 0 | 1e-7 |
| 500 | 4.99e-5 | 5e-5 |
| 501 | 5e-5 | コサイン減衰開始 |
| 50,000 | 約5.035e-14 | 0 |

この1更新分の添字差は意図的に明示する。配布checkpointの過去の依存版・正確な学習履歴は未確認。今回の比較用スケジュールを公式履歴の再現と表記しない。

## #2：データ契約と実学習量

`audit_data_contract.py`で実ファイルのSHA256とインストール済みローダーの件数を測定した。レポートは`.cache/stable-wm/pusht/controlled_v2/data_audit.json`。

| 項目 | 実測値 |
|---|---:|
| HDF5サイズ | 46,300,921,856 bytes |
| 研究用訓練クリップ | 1,645,509 |
| 50,000更新、batch128で提示するクリップ | 6,400,000 |
| 提示数換算の周回数（均等周回ではない） | 3.889374 |
| 研究用集合をdrop_lastで100周する更新数 | 1,285,500 |
| インストール済みHDF5ローダー全クリップ | 1,981,721 |
| 同ローダーの90%訓練分割 | 1,783,549 |
| 実際の`len(train_loader)` | 13,933 |
| 同ローダーを100エポック学習する更新数 | 1,393,300 |

データSHA256は`b6ebd9ac94bbe9e383f6e7a9cd92d74e9aa665ea57b758ed3717b0ee7df8d4fb`。
SWM 0.0.6、SPT 0.1.8、torch 2.9.1+cu130。上流pinは`lewm/UPSTREAM.md`のcommitと一致。
ただし上流設定は`pusht_expert_train.lance`で、現環境のSWMはHDF5のみ。この設定は実際に解決エラーになった。`official_repro`のnative Lance経路は未解決であり、上記HDF5ローダー件数を「公式配布モデルの履歴」とは呼ばない。

trainerは`budget.json`を出力し、起動・再開ごとに実データの内容ハッシュを確認する。LIBEROではindexだけでなく全デモHDF5をハッシュする。size/mtimeを保った内容改変を検出するテストも合格。

今回GB10では大容量ハッシュ読み出し後のページキャッシュでCUDA初期化がOOMになった。対象HDF5のclean cacheだけを`POSIX_FADV_DONTNEED`で解放すると、free RAMは約7→50 GiBとなりCUDA割当が復帰した。大容量ハッシュ関数にも同じヒントを追加した。データの削除・変更や他GPUサービスの停止は行っていない。

## #3：公式重み＋実データのRaw経路

`audit_raw_path.py`は公式オブジェクト重みを複製し、訓練集合の実画像2クリップを使用。公式`lejepa_forward`と共通`one_step_objective`を直接実行する。stage JSONは`controlled_v2/raw_path_fp32.json`と`raw_path_bf16.json`。

- CPU float32：画像前処理は許容誤差内（最大差7.15e-7）。全データ統計とtrain-only統計による行動入力差は最大約0.00704。これは意図的なレシピ差として分離した。
- 同じtrain-only入力と使用LRへそろえると、eval/train両方でencode、行動埋め込み、predict、予測損失、Raw SIGReg、全勾配、1更新後パラメータが許容誤差内で一致。AdamWのパラメータグループ・設定も一致。許容誤差と各パラメータの結果をJSONに記録した。
- 未来教師専用の最後のフレームへの予測損失勾配は両経路で非零。AdaLN-zero初期化の初回ゼロ・後続非零は既存`test_rbg.py`の複数更新テストで分けて確認する。
- GPU bfloat16：encode／行動埋め込み／predictまでは一致し、最初の差はSIGReg。元コードのautocastと候補の明示float32統計計算による差で、train時SIGReg絶対差は約4.94e-4、loss差は約6.40e-5。FP32用の厳しい判定では勾配204項・更新後パラメータ152項に差が出た。GPU経路の完全一致とは報告しない。研究比較ではRaw/RBGとも統計float32を維持する。
- 追加切り分け：公式SIGRegもfloat32にしただけでは、trainモードでencoderの勾配81項に差が残った。さらに決定論的GPU演算を有効化すると、eval/train双方で全段階・全勾配・1更新後重みがFP32用の厳しい許容誤差内で一致した。`raw_path_bf16_deterministic_fp32_statistics.json`へ記録。GPUの丸め誤差を隠すために許容幅を広げず、演算条件を固定した比較にする。新planは`--deterministic`を指定する。速度への影響は本学習前に別測定が必要。
- 未来の4枚目だけを左右反転した監査：evalでは過去表現・予測の差は厳密に0。trainではprojector BNの時刻間統計依存により、過去表現の最大差約1.765、予測約1.739（CPU）。因果attentionだけでモデル全体の時刻間依存が消えると主張しない。

## #4：BN監査と再校正

再校正を関数化し、変更を許すBNバッファを名前で列挙して検証する。全学習パラメータと他のbufferはハッシュ一致を要求。train-onlyの取得index列・seed・順番・バッチ数・精度・列・ソースcheckpointを記録する。validation/testは校正へ渡さない。

`audit_rbg_checkpoint.py`は同一の公式重みを使い、保存BN／BNだけバッチ統計／train-only再校正の3経路を別コピーで比較。projectorとpred_projのrunning統計・実入力統計を個別記録する。`bn_diagnostics.py`は実際の`JEPA.get_cost`とrolloutで候補分割・順番・無関係候補追加を監査する。

公式モデル、校正2×4クリップ、検証2クリップの小規模接続確認では、予測MSEは保存BN 0.004846、バッチ統計0.203315、再校正0.086197となり、再校正が悪化した。主比較の32×128校正や制御性能の結果ではない。全経路の学習重みは不変。evalモードの候補不変性は保存BN・再校正とも設定した誤差内で合格。出力は`controlled_v2/bn_audit_official_smoke.json`。校正したcheckpointを新規保存せずメモリ上だけで監査した。

## #5：公式先行評価と対応付き比較

`evaluate_official_pusht.py`は独立した公式200ケース評価。既定はdry-run、`--execute`で50ケースずつ4回実行する。重み・実データ・manifest・前処理と評価ソース・依存版・計画設定・ケースが一致する場合だけ結果を再利用する。不完全結果は拒否する。

`lewm/eval.py`の監査オプションは初期状態/Goal画像等のハッシュと、実環境へ渡す物理行動・実行maskを保存する。提案checkpointの入力は保存された学習統計へ変換しつつ、CEMの物理探索分布を公式と共有できる`PlanningActionAdapter`を追加した。既存経路はオプション無しなら従来のまま。対応付き比較は初期/Goalと物理探索分布の不一致も拒否する。

公式先行評価は完了：200の異なるエピソードで176成功、88.0%。各50ケースは46/42/44/44成功。固定checkpointのケース試行に対する二項Clopper–Pearson 95%区間は約82.7–92.2%。独立学習seed間の変動や、タスク一般への保証ではない。集計は`.cache/stable-wm/pusht/controlled_v2_official_summary.json`、ケース単位JSONに全200件の初期/Goalハッシュと実行行動・maskを保存。配布モデルが当該軌道を見た可能性は未解決。

さらに最初の50ケースの記録物理行動を、`replay_pusht_evaluation.py`で別の環境インスタンスへ再実行し、初期状態・Goal・全実行mask・全成功/失敗が一致した。46/50の結果が再現された。これは同じ行動の環境再実行検証であり、独立CEM評価50回を追加したことにはしない。

SWMの既定`env_N.mp4`が次の50ケースで上書きされる問題を発見し、独立runnerにバッチ別動画退避を追加。初回50件の元動画は上書きされたため、記録行動から再生成した動画を`controlled_v2_official_confirm_0_replayed_videos`へ区別して保存した。残りの動画はバッチ別に退避済み。数値結果と行動・状態ハッシュは全200件を保持した。

## #6以降と未完了

`create_shared_initialization.py`と`plan_controlled_comparison.py`を追加した。新規未学習初期値をRaw/RBGで共有し、50,000/128/500/5e-5/0、保存5,000更新ごとのコマンドを出力する。総更新数を変えれば出力IDも変わる。TC/Sub/LIBERO/追加seedを自動起動しない。現在の停止指示の下ではplan出力までで、学習を実行しない。

`training_diagnostics.py`で係数込みの各項のencoder/projector勾配ノルム、validation先頭バッチの行動入れ替え・正規化ゼロ・物理ゼロ行動診断を追加。既定5,000更新ごとの疎な計測で、診断勾配を`.grad`へ加算せず学習目的も変更しない。毎更新の提示クリップ数・GPU peak allocated bytesも記録する。実LeWMの長時間学習での診断コストは未測定。

新予算の自動実行・診断の統合集計、同予算の学習比較、#7〜#13の研究候補の実データ比較は未完了。ユーザーとの議論により次回比較は各100,000更新の方針になったが、CLI既定は50,000であり明示指定が必要。既存200ケースは回帰試験であり、最終主張には別の未使用ケースが必要。native Lanceによる公式再現も未解決。#1/#3のみcloseし、残件のあるissueは維持した。

最終回帰確認：`CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=.:lewm:mylewm .venv/bin/python -m pytest mylewm -q`で83件合格（GPU統合を含む）。`git diff --check`も合格。GPU対応範囲やforkに関する依存ライブラリの警告は残るが、この実行ではテスト失敗はなかった。

## 付録：旧Frobenius試作


本節は旧Frobenius試作の履歴。現行コードは[BT v2](../research/BT_SIGREG.ja.md#6-現行実装cayley特異値制約v2)へ置換した。

`mylewm/bt_sigreg.py` の各ブロックを次で構成した。

```math
\bar W=\frac{W}{\sqrt{1+\|W\|_F^2}},\qquad
g(z)=\kappa(1-10^{-4})\bar W_2\tanh(\bar W_1 z),\qquad T_k(z)=z+g(z).
```

作用素ノルムはFrobeniusノルム以下なので、実数演算では常に `||W̄||₂ < 1`。tanhは1-Lipschitzで、`Lip(g) <= κ` が解析的に従う。バイアスなしのため `T(0)=0`。任意の有限重みの更新後も同じ構成を使い、power iterationや有限標本の距離罰則に依存しない。

これは保守的な上界で、スペクトルノルムを直接制約する方式より変形能力が弱くなる可能性がある。強度・幅・深さは未調整。数学上の上界は実数演算についてのもので、FP32の丸め誤差まで厳密保証する区間演算実装ではない。安全係数と数値テストを理論保証の代わりにしない。

W1を直交初期化、W2を0として、T全体を恒等写像から開始する。初回はW2に勾配が入り、その更新後にW1にも勾配が入る。両方の重みを0にする初期化ではない。depth=0は学習パラメータなしの恒等対照。

既定dim=hidden=192、depth=2、κ=.2ではTの学習パラメータ147,456、理論距離境界m=.64、M=1.44。推論用ネットワークの追加パラメータは0。TとSIGRegはautocast外のFP32（数学テストのTはFP64可）で計算する。

両trainerのbtモードは一段予測を元のzで計算し、全潜在に共通Tをかけた後、Rawと同じ1024射影・17周波数・時刻別バッチ集計のSIGRegを使う。cross項の係数は0。E/A/Fの構造・計画コードは変更しない。Tは学習専用optimizerグループとregularizer stateへ保存し、推論exportには含めない。Raw/TCの処理は維持する。

### 独立レビューで判明した表現力の制限

この実装は一般的なBTを代表する完成形ではなく、制限の強いFrobenius試作である。研究者役・数学者役の2体のAIが独立に検算し、相互確認した。

1. バイアスなしtanhにより `T(-z)=-T(z)`。可逆性から、Zが原点対称であることとT(Z)が原点対称であることは同値。非原点対称なZをこのTだけで厳密な標準Gaussianにはできず、encoderに対称化の負担が残る。原点固定だけから必要になる制約ではない。またtanhの有界性によりT(z)−zも大域的に有界となる。
2. Frobenius正規化は作用素ノルムに加えて `||Jg||_* <= κ`（核ノルム）まで制限する。行列積の展開より `||JT-I||_* <= ∏(1+κ_k)-1`。既定値では総予算は.44で、192次元の局所変形 `JT=1.1I` に必要な19.2を許さない。この変形は一般の距離境界[.64,1.44]なら許される。幅を増やすだけでは改善しないが、固定低ランク写像という意味ではない。

したがって、この試作が性能比較で負けてもBT仮説全体を棄却できない。性能学習前にこの制限を採用するか、原点固定と上下距離境界を保ちながら写像を再設計するかを研究判断する必要がある。今回のレビューではアルゴリズムを無断変更していない。

検証範囲と限界は[検証記録](../VALIDATION.ja.md)を参照。小型マルチタスク性能の改善は未検証である。
