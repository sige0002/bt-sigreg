# 構成整理と復元（2026-09-07）

## 現在の整理

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

## 先の整理

交差対応表・複製ビュー・多段予測関連の `multitask_lewm.py`、`multitask_jepa.py`、`world_model.py`、`train_pusht.py`、`train_crossed.py`、`test_multitask_lewm.py`、`test_crossed_architecture.py`、`trajectory_experiment.py`、`run_trajectory_evaluation.py`、`test_trajectory_experiment.py` は `d5ee587` で削除済み。

## 履歴からの復元

今回削除した18ファイルは[整理前の固定コミット](https://github.com/sige0002/bt-sigreg/tree/0def700/mylewm)にすべて保存されている。`git show 0def700:mylewm/ファイル名` で内容を確認できる。先の10ファイルは `git show d5ee587^:mylewm/ファイル名`。削除したコードや研究記録の永久消失ではない。

この整理は学習再開・BT実装・性能評価ではない。

## 整理後の確認

- 回帰テスト69件合格。旧専用テスト6件を削除し、CLI起動確認4件を追加した（整理前71件）。
- `tools/` の全16 CLIで `--help` が成功。LIBEROの3件は既存OSMesa wrapper経由で確認し、環境評価は実行していない。
- Python全ファイルの構文確認、ローカルMarkdownリンク、`git diff --check` が成功。
- 追加確認で判明した `audit_rbg_checkpoint.py` の整理前からの閉じ括弧不足と、`compare_paired.py` の直接起動時のimportパス不足を修正。
- GPU対応範囲・fork・旧Gym等の依存ライブラリ警告は残る。全ベンチマークの実評価や旧checkpoint再開の検証とは区別する。
