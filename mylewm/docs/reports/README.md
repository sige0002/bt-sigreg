# 実験レポート

更新日：2026-09-09。研究仕様と実行手順は[提案側README](../../README.md)を参照。

| レポート | 内容 |
|---|---|
| [PushT評価](PUSHT_CHECKPOINT_EVALUATION.ja.md) | 途中・最終checkpoint、固定200ケースと上流50ケース、比較条件・失敗・hash |
| [PushT 100k学習](PUSHT_TRAINING_100K.ja.md) | 完了済みrunの開始条件・出力・最終loss |
| [実装・運用監査履歴](IMPLEMENTATION_AUDIT.ja.md) | 数学・接続・再開テスト、LIBERO短期診断、Issue対応、旧試作・失敗記録 |

[現在の検証状況と未完了項目](../VALIDATION.ja.md)を入口とし、履歴の「未実装」「学習中」を現在の状態と混同しない。生データ・重み・ログはGit対象外の `output/` と既存 `.cache/` に保持している。新しい学習・評価の起動を指示する文書ではない。

記載された旧ファイル名・実行コマンドは当時の証拠です。現行の改名・削除・issue整理は[整理記録](../CLEANUP.ja.md)を参照し、過去コマンドを現在の実行手順として使用しないでください。
