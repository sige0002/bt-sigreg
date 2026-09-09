# 構成整理と復元

## 2026-09-09：現行仕様・手順・レポートの分離

### 今回の変更

- ルートREADMEと `mylewm/README.md` は目的・現在地・文書案内に限定。過去の「学習中」「未実装」を現行状態と混在させない。
- `README_RESEARCH_REVIEW.ja.md` を `mylewm/docs/RESEARCH_REVIEW.ja.md` へ移し、重複したアルゴリズム導出は既存の `BT_SIGREG.ja.md` へ集約。旧Frobenius実装の詳細は監査履歴へ分離。
- 初心者向け学習手順を `docs/TRAINING.ja.md`、LIBERO評価環境を `docs/EVALUATE_LIBERO.ja.md` へ分離。PushT評価手順の例は保存済み100k checkpointに更新。
- 評価レポートを `docs/reports/PUSHT_CHECKPOINT_EVALUATION.ja.md` へ移し、要旨・トラック別総括表を追加。既存の数値・hash・失敗・削除記録は保持。
- `VALIDATION.ja.md` は現在の検証状況と未完了事項に絞り、以前の監査本文を `reports/IMPLEMENTATION_AUDIT.ja.md`、完了済み100kの開始記録を `reports/PUSHT_TRAINING_100K.ja.md` へ分離。
- 廃止済みの `tools/watch_evaluation.sh` と専用の8ケースのテストを削除。シェルは未追跡だったためGit履歴だけに頼らず退避した。評価launcherの終了状態検査とそのテストは保持。
- `AGENTS.md` を現行の文書構成・完了状態・明示依頼時のみ評価する運用へ統一。

### 残したコードと理由

| 分類 | ファイル／用途 |
|---|---|
| 本体・共有学習 | `bt_sigreg.py`、`rbg.py`、両 `train_rbg*.py`。BT自身が共有損失・trainerを使用 |
| LIBERO | `libero_model.py`、`libero_planner.py`、`run_libero.sh`。2実カメラ・共有モデル・OSMesa評価に必要 |
| 再現性・入力契約 | `training_state.py`、`data_contract.py`、`evaluation_contract.py`、`planning_action_adapter.py`。保存再開・公平比較・pickle復元経路を維持 |
| 補助診断 | `training_diagnostics.py`、`bn_diagnostics.py`、`representation_probes.py`。勾配・BN・表現の監査に必要 |
| 学習補助CLI | `create_shared_initialization.py`、`plan_controlled_comparison.py`。同初期値・同予算比較の準備 |
| 評価・比較CLI | `evaluate_pusht.sh`、`evaluate_official_pusht.py`、`eval_rbg_libero.py`、`compare_paired.py`、`compare_libero.py`、`replay_pusht_evaluation.py` |
| データ・計算経路・速度診断CLI | `audit_*.py`、`calibrate_rbg_bn.py`、`probe_rbg_*.py`、`benchmark_rbg.py`、`smoke_libero.py`。通常の学習では不要だが研究比較の検証手段として保持 |
| 手動進捗表示 | `monitor_training.sh`。依頼時のstep・loss確認に使用。常駐監視とは別 |
| 回帰テスト | 現役コードのテスト。廃止監視専用以外は保持 |

import・CLI参照・保存ソースhashの契約を確認した。Pythonの旧名だけを理由に削除・移動すると、BT学習やcheckpoint復元を壊すため行わない。今回、学習・モデル・評価本体のPythonは変更していない。開始前からのlauncher修正などの未コミット差分も保持した。

公式比較コード、データ、全ての残存checkpoint、manifest、ログ、評価証拠、ローカル環境は削除していない。旧パスの一時リンクも既存manifest依存があるため今回の削除対象外。

### 復元と確認

整理直前の文書・廃止シェル・変更前のテストを `output/repository-cleanup-20260909-SvprVR/before-cleanup.tar.gz` に保存した（Git対象外）。`tar -tzf` で一覧、`tar -xOf ARCHIVE PATH` で内容を確認できる。復元時は別の一時ディレクトリへ展開し、現在のファイルを一括上書きしない。このバックアップは新たな実験ログやcheckpointではない。

CPU限定回帰：`CUDA_VISIBLE_DEVICES='' PYTHONPATH=.:lewm .venv/bin/python -m pytest mylewm/tests -q` で **102合格・5スキップ**。直前の110合格との差8件は廃止した監視専用テストであり、学習・評価テストの失敗や除外ではない。CUDA専用5件は今回未実行。fork/Lance等の既存警告が残る。Markdownのローカル参照、シェル構文、`git diff --check`も確認する。追加学習・環境評価・コミット・pushはこの整理では行わない。

## 2026-09-07の整理履歴

### 当時の整理

`mylewm/` 直下の混在を解消し、CLIを `tools/`、回帰テストを `tests/`、現行文書を `docs/` に分けた。単に別フォルダへ移しただけでなく、不要な旧方式は削除した。

削除したPython（9件）：

- `run_rbg_pusht_comparison.py`、`run_rbg_libero_comparison.py`、`run_rbg_replicates.py`、`test_replicate_queue.py`：旧予算のRBG自動実験キュー。
- `train_rbg_ablation.py`、`test_rbg_ablation.py`：旧RBG混合正則化専用。
- `train_subspace_control.py`、`test_subspace_control.py`：現行の主比較には含めないSub-JEPA専用。
- `abc_math_checks.py`：ABC案専用の数式検算。BTの検証ではない。

削除したMD（9件）：

- `ABC_LITERATURE_NOTES.ja.md`、`ABC_MATH_AUDIT.ja.md`、`ABC_RESEARCH_REVIEW.ja.md`、`RESEARCH_CANDIDATES_20260907.ja.md`：旧候補の詳細記録。現行レビューに採否要約を残し、詳細は履歴へリンク。
- `FACTOR_GAUSSIAN_PROPOSAL.ja.md`、`RBG_EXECUTION_STATUS.ja.md`、`RBG_V0_PUSHT_GAP.ja.md`：旧RBG案・進行記録。
- `IMPLEMENTATION_REVIEW.ja.md`、`TRAJECTORY_DESIGN.ja.md`：旧交差・多段方式の記録。

共有trainer・一段損失・Raw/TC、公式評価、データ/BN/再開/画像監査、表現診断と対応するテストは維持した。公式LeWM、データ、公式重みには変更なし。移動したファイルのimport、ROOT解決、コマンド生成、文書リンクを更新する。ソースのパスとハッシュが変わるため、過去の評価キャッシュ再利用や旧checkpointの厳密resume互換性を保証しない。

### 先の整理

交差対応表・複製ビュー・多段予測関連の `multitask_lewm.py`、`multitask_jepa.py`、`world_model.py`、`train_pusht.py`、`train_crossed.py`、`test_multitask_lewm.py`、`test_crossed_architecture.py`、`trajectory_experiment.py`、`run_trajectory_evaluation.py`、`test_trajectory_experiment.py` は `d5ee587` で削除済み。

### 履歴からの復元

今回削除した18ファイルは[整理前の固定コミット](https://github.com/sige0002/bt-sigreg/tree/0def700/mylewm)にすべて保存されている。`git show 0def700:mylewm/ファイル名` で内容を確認できる。先の10ファイルは `git show d5ee587^:mylewm/ファイル名`。削除したコードや研究記録の永久消失ではない。

この整理は学習再開・BT実装・性能評価ではない。

### 整理後の確認

- 回帰テスト69件合格。旧専用テスト6件を削除し、CLI起動確認4件を追加した（整理前71件）。
- `tools/` の全16 CLIで `--help` が成功。LIBEROの3件は既存OSMesa wrapper経由で確認し、環境評価は実行していない。
- Python全ファイルの構文確認、ローカルMarkdownリンク、`git diff --check` が成功。
- 追加確認で判明した `audit_rbg_checkpoint.py` の整理前からの閉じ括弧不足と、`compare_paired.py` の直接起動時のimportパス不足を修正。
- GPU対応範囲・fork・旧Gym等の依存ライブラリ警告は残る。全ベンチマークの実評価や旧checkpoint再開の検証とは区別する。
