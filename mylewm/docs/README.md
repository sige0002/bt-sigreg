# 文書の読み方

このリポジトリでは、**仕組みを理解するガイド、操作手順、実験の記録**を分けています。初めて読む場合、過去のrun名や実行ログを覚える必要はありません。

## 基本の4章

| 順番 | 文書 | 読み終えたらできること |
|---|---|---|
| 1 | [アルゴリズム](ALGORITHM.ja.md) | 状態z、予測、正則化、BTの役割を説明できる |
| 2 | [実装](IMPLEMENTATION.ja.md) | 修正したい機能のファイルと、データの流れを探せる |
| 3 | [学習](TRAINING.ja.md) | データ準備から世界モデル・BC学習までの工程を選べる |
| 4 | [評価](EVALUATION.ja.md) | CEM／BCのどちらで何を測るか決め、結果を確認できる |

## 目的別の資料

| 目的 | 文書 |
|---|---|
| BCの入力・保存再開を詳しく知る | [BCガイド](BEHAVIOR_CLONING.ja.md) |
| HDF5／LeRobot、推論、Policy exportを使う | [モデル利用](MODEL_USAGE.ja.md) |
| 数学的な導出・Cayley v2の詳細を読む | [BT仕様](research/BT_SIGREG.ja.md) |
| 関連研究・研究仮説・比較条件を読む | [研究レビュー](research/RESEARCH_REVIEW.ja.md) |
| 実証されたことと残る課題を知る | [検証状況](VALIDATION.ja.md) |
| 個別の実験条件・失敗記録・証拠を探す | [実験レポート一覧](reports/README.md) |
| このPCの稼働run・停止指示・再開条件を確認する | [運用記録](AGENT_OPERATIONS.ja.md) |
| 評価や依存変更の作業ルールを確認する | [検証上の制約](AGENT_VALIDATION.ja.md) |
| 旧構成の改名・復元方法を調べる | [整理履歴](reports/CLEANUP.ja.md) |

## 詳細手順と履歴

`reference/`には従来の細かなCLI例・旧レシピ・トラブルシューティングを保持しています。現在のガイドから必要な節へ移動してください。

- [学習の詳細参照](reference/TRAINING.ja.md)：PushT、LeRobot、旧LIBEROデータ経路
- [評価の詳細参照](reference/EVALUATION.ja.md)：描画監査、CEM設定、途中checkpoint、viewer
- [BCの詳細参照](reference/BEHAVIOR_CLONING.ja.md)：保存・再開契約とCLI

`reports/`の「学習中」「未実装」はその記録時点の説明です。最新の操作条件にはガイドと運用記録を使い、過去の実験証拠は書き換えません。
