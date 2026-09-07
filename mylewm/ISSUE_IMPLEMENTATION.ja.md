# Issue対応と検証記録

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
