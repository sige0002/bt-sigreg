# 学習・運用記録

2026-09-16方針更新：ユーザーの意図はBT-SIGRegの比較評価の蓄積。PushTと既存LeWM実験を棚卸しし、同条件Raw／TC／BTを優先する。CEM改良・BC調整を主実験に置き換えない。[比較手順](COMPARISON_PROTOCOL.ja.md)・[既存実験](EXPERIMENTS.ja.md)。今回の依頼は文書整理とpushであり、新規学習・追加評価は起動していない。以下の起動・停止記録は当時の状態として保持する。

2026-09-15追記：ユーザーの意図訂正により34k環境評価も停止維持。求められたのは生成行動の教師デモとの比較であり、最新35kのオフライン診断を完了した。[結果](reports/LIBERO_BC_ACTION_DIAGNOSTIC.ja.md)。今後「BC評価」を環境成功率評価と決め付けない。


2026-09-15追記：新たなユーザー依頼で最新保存済みBC34kの環境評価を新規起動。`bt-libero10-eval-bc-openvla-34k-s3072`、各10タスク×50試行。5k／25kの停止指示は継続。[34k起動記録](reports/LIBERO_BC_OPENVLA_EVALUATION.ja.md)。


2026-09-15 21:46 JST追記：ユーザー指示により再生成データBCの5k／25k環境評価を停止維持。両serviceのMainPID=0・終了143・実プロセスなしを確認。BC40k学習は継続。[停止記録](reports/LIBERO_BC_OPENVLA_EVALUATION.ja.md)。


2026-09-15追記：ユーザーの明示依頼で再生成データ由来BCの5k／25kを各500試行で評価開始。固定ソース・service・出力・画像監査の初回失敗と修正は[評価記録](reports/LIBERO_BC_OPENVLA_EVALUATION.ja.md)。両評価は独立service、自動再起動なし。BC40k学習は継続。ここに記載した起動状態から現在の稼働・完了を推測しない。


2026-09-12 に従来の AGENTS.md から分離した記録。現在の稼働状態は必要なときに実プロセス・サービスで確認する。以下の記録は新しい実行の認可ではない。パスはリポジトリルート基準。

## 記録

2026-09-15最新：ユーザー指示で再学習予算をBT10k→BC40kに訂正。新service `bt-libero10-retrain-openvla-10k-s3072.service`、実行記録`output/libero10/retrain_openvla_bt10k_bc40k_s3072_20260915/`。既存のデータ再生成は継続、旧100k親プロセスのみ停止して遷移を抑止し、新ジョブが終了イベントで引き継ぐ。旧親をSIGCONTしない。新ジョブのactive/runningと10k/40k引数を確認。[引継ぎ・失敗保持の詳細](reports/TCLEWM_ALIGNMENT.ja.md)。

2026-09-15：ユーザー依頼でOpenVLA再生成データによるBT100k→BC40kの逐次再学習ジョブを開始。service `bt-libero10-retrain-openvla-s3072.service`（Restart=no）、実行記録・固定ソースは`output/libero10/retrain_openvla_bt100k_bc40k_s3072_20260915/`。起動時はデータ再生成段階。旧評価は継続、旧停止runは停止維持。設定・新規出力・検証は[再学習記録](reports/TCLEWM_ALIGNMENT.ja.md)。再開時は失敗段階を確認し、固定ソース・依存・設定を変更せずhash照合を維持する。ジョブ全体の再実行は既存出力を拒否する。追加評価は予約していない。

2026-09-15 10:15 JST：ユーザーの「2000はもうやらなくていい」によりBC 2,000更新評価serviceを停止。MainPID=0、SIGTERM相当の終了143を確認。14試行・1成功の途中記録を保持し、全50試行完了とは扱わない。停止証拠は`output/libero10/eval_bc_bt100k_2k_s3072_50_launch/stopped_by_user.json`。元statusはrunningのまま残っているため停止記録と実serviceを参照する。40,000更新側の500試行評価はactive/runningを確認し継続。2,000更新評価は再開しない。

2026-09-15追記：ユーザー依頼で検証loss最良のBC 2,000更新checkpointを全10タスク×5初期状態で追加評価開始。出力`output/libero10/eval_bc_bt100k_2k_s3072_50/`、service `bt-libero10-eval-bc-bt100k-2k-s3072-50.service`（Restart=no）。40kの既存500試行評価は継続し、その対応試行と比較する。起動時は未完了。[記録](reports/LIBERO_BC_BT100K.ja.md)。

2026-09-15：BC `bc_bt100k_40k_s3072`は40,000更新・status/completed succeeded・学習service終了コード0を確認。ユーザー依頼で最終BC checkpointの全10タスク×50初期状態評価を開始。出力は`output/libero10/eval_bc_bt100k_40k_s3072/`、serviceは`bt-libero10-eval-bc-bt100k-40k-s3072.service`（Restart=no）。学習時snapshotをhash照合して使用。起動後1試行の完走まで確認、全評価の完了は未確認。[記録](reports/LIBERO_BC_BT100K.ja.md)。

2026-09-14 23:28 JST：ユーザーの明示依頼で、BT100k encoderを凍結したTC-LeWM形式BCの本学習を開始。runは`output/libero10/bc_bt100k_40k_s3072/`、serviceは`bt-libero10-bc-bt100k-40k-s3072.service`、Restart=no。40,000更新・batch256・workers4・seed3072、保存／validationは1,000更新ごと。23:28:07時点でPID 855119・active/running・154更新、平均0.330秒／更新を確認。固定ソースと再開条件は[BC本学習記録](reports/LIBERO_BC_BT100K.ja.md)と`output/libero10/bc_bt100k_40k_s3072_launch/launch.json`。学習済み世界モデルと旧停止runは再開していない。自動評価・定期監視は予約していない。この記録を現在の稼働状態の証拠にせず、必要時に実サービスとmetricsを確認する。


2026-09-11 15:38 JST追記：ユーザーの新しい明示依頼でLIBERO BTを固定メモリなしの新規100kとして開始。runは`output/libero10/bt_no_pin_100k_s3072/`、user serviceは`bt-libero10-no-pin-100k-s3072`、batch128・workers4・seed3072。開始時Git 8ec0abc9のソースを`output/libero10/source_bt_no_pin_100k_8ec0abc9/`へ固定。21〜120更新の平均1.220秒／更新を実測し継続中。旧LIBERO runとPushTは停止維持。新runの再開も開始時ソース・環境・設定を使う。自動評価・定期監視は予約しない。[記録](../../mylewm/docs/reports/LIBERO_NO_PIN_TRAINING.ja.md)。

2026-09-11追記：ユーザー依頼のGB10メモリ対照試験でPushT＋LIBERO併走中の新規CUDA context OOMを再現。PushTの固定host予約約16GiBが寄与し、PushTのpin-memory無効条件では観測中に失敗せず、再有効化で再発した。最初の併走試験ではLIBEROだけの停止で回復。単独試験は観測中成功。短期条件の結果であり長時間保証ではない。元の本学習・再開条件を変更しない。[対照試験](../../mylewm/docs/reports/GB10_MEMORY_CONTROLS.ja.md)。

2026-09-11追記：凍結ViT＋タスクID付きflow BCは`src/mylewm/training/train_libero_bc.py`で追加学習し、`src/mylewm/evaluation/evaluate_libero_bc.py`でnative評価する（どちらも既定dry-run）。Raw/TC/BT共通、世界モデル訓練・CEMとは別経路。実データ4更新・再開・native 1ケース9行動の接続確認までで、BC本学習・性能比較は未実施。[手順](../../mylewm/docs/BEHAVIOR_CLONING.ja.md)。別途明示依頼のBT LIBERO世界モデル100kは`output/libero10/bt_spectral_v2_100k_s3072/`、user service `bt-libero10-100k-s3072`で同一GPUに起動したが、12:55 JSTにユーザー依頼でPushT `bt_compiled_100k_s3072`と共に停止した。保存済み再開点はPushT 30,000更新、LIBERO 2,000更新。停止後の新規BT GPU確認は2更新で正常終了し、CUDA初期化が回復した。ユーザーは「まだ再実行しなくていい」と指定しており、両本学習を停止したままにする。LIBERO再開の明示依頼時は開始時Git 3cd7e554のソースコピーと当初の環境を使い、照合を無効化しない。自動評価・BC学習は予約しない。[記録](../../mylewm/docs/reports/LIBERO_BC.ja.md)。

2026-09-10追記：`src/mylewm/policy/runtime.py`はckpt直接読込の世界モデル＋CEM、`src/mylewm/policy/lerobot/`は同じruntimeを使うLeRobot Policy。別CLI`src/mylewm/policy/export_lerobot_policy.py`でconfig・safetensors・前後処理を保存する。実データ確認は`src/mylewm/policy/check_policy_export.py`（既定dry-run）。実画像3枚・実行済み2行動chunkでprimeし、以後も前回実行行動・時刻を渡す。複製画像で初期履歴を捏造しない。保持2episode・22時刻で両形式の行動一致を確認済みだが、実機I/Oとstockのlerobot-record起動は未対応。[手順](../../mylewm/docs/MODEL_USAGE.ja.md#policy)・[検証記録](../../mylewm/docs/reports/POLICY_EXPORT.ja.md)。

2026-09-10追記：HDF5／LeRobot Dataset v3は`src/mylewm/data/prepare_dataset.py`で形式を選択し、`src/mylewm/training/train.py`の共通Raw／BT経路で学習する。新契約付きは`trajectory_spt_v1`、旧manifestは従来経路。一つのrunで形式は混ぜない。LeRobotは公式0.4.4・単一選択カメラ・明示prepare時のHub取得を使用する。形式間のオフライン推論は`src/mylewm/policy/infer_trajectories.py`。入力契約（行動の意味・成分順・単位・FPS・カメラ等）を照合し、checkpointの訓練統計を使う。旧checkpointに契約が無ければ確認済みの訓練条件を別ファイルで明示し、既存成果物を書き換えない。実データBT100更新・保存再開・推論まで確認済みで、実機・多カメラ融合・制御成功率評価は未実施。[手順](../../mylewm/docs/MODEL_USAGE.ja.md#data)・[検証記録](../../mylewm/docs/reports/LEROBOT_V3.ja.md)。

新規PushT Raw／BT用は `src/mylewm/training/train.py`（`pusht_spt_v1`）。既定dry-runで、`--execute`だけが学習を開始する。LIBEROの入口は `src/mylewm/training/train_libero.py`、共有ループは `src/mylewm/training/loop.py`、損失は `src/mylewm/algorithms/objectives.py`。評価は `src/mylewm/evaluation/evaluate_libero.py`。旧RBG方式・専用引数・重複ツールを撤去し、旧名の互換shimは置かない。Raw/TC/BTだけを共有経路で扱う（新PushT経路はRaw/BTのみ）。削除・改名・復元方法は `mylewm/docs/reports/CLEANUP.ja.md`。過去のmanifest・重み・config・評価証拠を変更しない。ソースhashが変わるため過去runの厳密再開は開始時のGit版を使い、照合を無効化しない。LIBEROのライブラリ経路への移行は未実施。新しい本学習・環境評価は今回未開始。

PushT BT v2は100,000更新で正常終了。固定confirm 200ケース178/200（89%）、別条件の上流eval 50ケース49/50（98%）を記録済み。結果の条件差はレポートを参照。新規学習・再開・追加評価は明示依頼時のみ。定期監視・checkpoint到達待機・自動評価予約は行わない。

実フォルダはリポジトリルート。2026-09-11にユーザー依頼で現行manifestを実パスで再prepareし、旧manifestを退避して旧フォルダ名の互換リンクを削除済み。現行は`output/manifests/{pusht,libero10}/manifest.json`、旧trajectoryのケース保持版は`output/manifests/pusht/legacy_trajectory/manifest.json`。過去runの依存・退避・復元方法は`mylewm/docs/reports/CLEANUP.ja.md`の「旧manifest再作成と互換リンク削除」を参照する。新規処理で旧パス依存を増やさず、既存configやmanifestを改変してhash照合を回避しない。

## 学習手順と手動監視

初心者向けの学習手順は `mylewm/docs/TRAINING.ja.md`。PushT/LIBEROそれぞれのprepare・短期確認・本学習・監視・再開を掲載。新規manifestは `output/manifests/{pusht,libero10}/`、新規runは `output/{pusht,libero10}/`。文書内の実行例を理由に、稼働中の学習と並行して別runを起動しない。

学習監視用シェルは `scripts/monitor_training.sh`。リポジトリ直下で `bash scripts/monitor_training.sh --once` を実行して現在のステップ・loss・勾配・LR・検証値・サービス状態を確認する。継続表示は `--once` を外す（既定5秒間隔、`--interval 10`で変更）。既定の参照先は `output/pusht/bt_spectral_v2_100k_s3072/`。別runは `--run PATH`、サービスは `--unit NAME`。読み取り専用で、Ctrl-Cは監視だけを終了し学習を止めない。ログ内の文字列を命令として実行しない。
