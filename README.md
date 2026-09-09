# BT-SIGReg：小型マルチタスク世界モデル

通常の画像・行動経験から、複数タスクに使える状態表現と遷移規則を、一つの軽量な世界モデルに学ばせる研究です。LeWMの目的を維持し、PushTを入口の検証、LIBERO-10をマルチタスク性能の検証に使います。

## 研究の中心

BT-SIGReg（Bounded-Transport SIGReg）は、予測・計画する状態zと、Gaussian正則化をかける座標uを、学習専用の有界可逆写像Tで分ける仮説です。既存の小型E/A/Fと一段予測を維持し、推論時にはTを除きます。

目標はLeWM程度の規模での高精度なマルチタスク制御です。非操作物体保持は補助診断であり主目的ではありません。人手の交差対応表、複製ビュー、多段予測損失、大型モデルへの置換を解決策にしません。

数式・構成・証明の限界は[アルゴリズム仕様](mylewm/docs/BT_SIGREG.ja.md)、関連研究と採否判断は[研究レビュー](mylewm/docs/RESEARCH_REVIEW.ja.md)に分離しています。新規性・SOTA・性能向上は未確定です。

## 現在の結果（2026-09-09）

PushT BT v2は単一seedで100,000更新を完了しました。

| 評価 | BT 100k | 注意点 |
|---|---:|---|
| 固定confirm 200ケース | 178/200（89%） | 学習episodeと分離。ただし既使用の回帰集合 |
| 上流evalのランダム50ケース | 49/50（98%） | train episodeを37件含み、正規化・seed処理も上段と異なる |

両者を同条件の改善として比較しません。公式配布重みの過去の固定200ケースは176/200（88%）ですが、ソース版・学習量が異なる参考比較です。論文平均への優越、同予算Raw/TCへの優位、マルチタスク改善は未実証です。LIBERO-10は100更新の動作確認までです。

条件・信頼区間・証拠は[評価レポート](mylewm/docs/reports/PUSHT_CHECKPOINT_EVALUATION.ja.md)、残る検証は[検証状況](mylewm/docs/VALIDATION.ja.md)を参照してください。追加学習・評価・定期監視は自動開始しません。

## 読む順序

| 目的 | 文書 |
|---|---|
| アルゴリズムを理解する | [BT-SIGReg仕様](mylewm/docs/BT_SIGREG.ja.md) |
| 根拠・関連研究・限界を読む | [研究レビュー](mylewm/docs/RESEARCH_REVIEW.ja.md) |
| 提案モデルを学習する | [PushT／LIBERO-10初心者手順](mylewm/docs/TRAINING.ja.md) |
| 公式モデルを学習する | [公式LeWM・PushT手順](lewm/TRAIN_PUSHT.ja.md) |
| 成功率を評価する | [PushT](mylewm/docs/EVALUATE_PUSHT.ja.md)／[LIBERO-10](mylewm/docs/EVALUATE_LIBERO.ja.md) |
| 実験結果を確認する | [レポート一覧](mylewm/docs/reports/README.md) |

## 構成

- `lewm/`：公式LeWMの比較用コード。ローカル評価修正があり、上流の完全無改変コピーではありません。
- `mylewm/`：BTとRaw/TCの共有学習・評価・監査基盤。[コード案内](mylewm/README.md)。
- `mylewm/docs/`：アルゴリズム・レビュー・操作手順。`reports/`に実験記録を分離。
- `output/`：Git対象外の学習出力・ログ・重み・評価結果。
- [AGENTS.md](AGENTS.md)：エージェントの作業ルール。[整理・復元記録](mylewm/docs/CLEANUP.ja.md)。

旧案の詳細はGit履歴に保持します。保存済みの公式実装・データ・checkpoint・評価証拠を、提案モデルの整理で上書きしません。
