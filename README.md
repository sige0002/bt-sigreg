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

2026-09-10追記：ローカル学習のSIGReg（新経路Raw）とBT v2（旧経路）を70,000更新同士で固定50ケース評価し、45/50（90%）と47/50（94%）でした。学習手順差と実物理初期状態の微差が残るため、方式の優位性とは断定しません。Raw学習はユーザー指示で停止済みです。[評価経路の修正・追試記録](mylewm/docs/reports/PUSHT_ISSUE20.ja.md)。

新規PushT Raw／BTの学習入口は `mylewm/train.py` です。公式SWM/SPT/Lightningへの移行経路で、既存100kとは別レシピです。エピソード分離を維持し、重み・評価記録は保持しています。旧RBG専用コードを撤去し、共有処理を役割名へ整理しました。LIBEROは `mylewm/train_libero.py` から共有ループを使い、ライブラリ経路への移行は未実施です。

| 目的 | 文書 |
|---|---|
| アルゴリズムを理解する | [BT-SIGReg仕様](mylewm/docs/BT_SIGREG.ja.md) |
| 根拠・関連研究・限界を読む | [研究レビュー](mylewm/docs/RESEARCH_REVIEW.ja.md) |
| 提案モデルを学習する | [PushT／LIBERO-10初心者手順](mylewm/docs/TRAINING.ja.md) |
| 公式モデルを学習する | [公式LeWM・PushT手順](lewm/TRAIN_PUSHT.ja.md) |
| 成功率を評価する | [PushT](mylewm/docs/EVALUATE_PUSHT.ja.md)／[LIBERO-10](mylewm/docs/EVALUATE_LIBERO.ja.md) |
| 実験結果を確認する | [レポート一覧](mylewm/docs/reports/README.md) |

## 今後の対応：LeRobot Dataset

LeRobot Dataset形式の画像・行動軌道を、ローカル保存先またはHugging Face Hubから読み込み、既存のSIGReg／BTの世界モデルを学習できるようにする予定です。**現時点では未実装・未検証**で、新規PushT学習はHDF5入力です。

- モデルと損失を維持してデータ読み込みを接続し、カメラ選択、FPS、画像・行動の時刻対応、エピソード境界を検証する。
- 行動の次元・単位・絶対／相対座標を明示し、エピソード単位で学習／検証を分離して、学習側だけから正規化統計を計算する。
- 対象データと対応するLeRobotの版を決め、実データの読み込み、短期学習、checkpoint保存・復元を実行確認してから手順を掲載する。

データ入力の対応と、LeRobotのpolicyとして世界モデル＋CEMを統合する作業は別です。safetensorsによる重みの配布も別途扱います。また、データセットを読めることだけでは、環境での成功率評価まで対応したことにはしません。[LeRobot Dataset公式仕様](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)。

## 構成

- `lewm/`：公式LeWMの比較用コード。ローカル評価修正があり、上流の完全無改変コピーではありません。
- `mylewm/`：BTとRaw/TCの共有学習・評価・監査基盤。[コード案内](mylewm/README.md)。
- `mylewm/docs/`：アルゴリズム・レビュー・操作手順。`reports/`に実験記録を分離。
- `output/`：Git対象外の学習出力・ログ・重み・評価結果。
- [AGENTS.md](AGENTS.md)：エージェントの作業ルール。[整理・復元記録](mylewm/docs/CLEANUP.ja.md)。

旧案の詳細はGit履歴に保持します。保存済みの公式実装・データ・checkpoint・評価証拠を、提案モデルの整理で上書きしません。
