# 実験レポート

一覧整理：2026-09-15。研究仕様と現行手順は[文書一覧](../README.md)を参照。各レポートの実験日・検証範囲は本文に記載しています。

## PushTの学習・制御評価

| レポート | 内容 |
|---|---|
| [PushT 100k学習](PUSHT_TRAINING_100K.ja.md) | 完了済みrunの開始条件・出力・最終loss |
| [PushT評価](PUSHT_CHECKPOINT_EVALUATION.ja.md) | 途中・最終checkpoint、固定200ケースと上流50ケース、比較条件・失敗・hash |
| [PushT失敗監査](PUSHT_FAILURE_AUDIT.ja.md) | 固定ケースの一段・5遷移誤差と、遷移／cost／探索の切り分け範囲 |
| [PushT Issue #20](PUSHT_ISSUE20.ja.md) | 画像/Goal欠落・動画・CEM監査メモリ修正、複数ケースの実機確認と未確認範囲 |
| [CEM評価高速化・途中checkpoint](CACHED_CEM.ja.md) | 探索設定を維持した画像キャッシュ、10k／20k固定50ケース評価、同一GPUの時間・メモリ実測 |
| [Goal付き動画・viewer](PUSHT_VIEWER.ja.md) | 保存の重複整理、シーク・選択保持の修正、Playwright検証 |

## LIBEROの学習・評価

| レポート | 内容 |
|---|---|
| [TC-LeWM条件と再学習](TCLEWM_ALIGNMENT.ja.md) | 再生成データ、採用した構成、最新のBT10k→BC40kと固定ジョブ |
| [LIBERO BC実装](LIBERO_BC.ja.md) | 凍結ViT方策の保存再開・native接続確認、依頼されたBT世界モデル100kの起動記録 |
| [BT100kのTC-LeWM形式BC](LIBERO_BC_BT100K.ja.md) | 論文仕様の再照合、実データGPU保存再開一致、10タスク推論と短期native接続 |
| [GB10メモリ対照試験](GB10_MEMORY_CONTROLS.ja.md) | PushT／LIBERO単独・併走・pin切替と新規CUDA context OOMの再現 |
| [LIBERO固定メモリなしの本学習](LIBERO_NO_PIN_TRAINING.ja.md) | 新規BT100kの開始条件、pin切替の検証、1更新時間の実測 |
| [LIBERO BT100kの初回CEM評価](LIBERO_BT100K_EVALUATION.ja.md) | 10タスク各1試行・520行動で0/10、全描画監査・終了確認・画像一覧 |
| [LIBERO CEM切り分け](LIBERO_CEM_DIAGNOSTIC.ja.md) | 成功デモ対ランダム50窓と探索予算60件のオフライン診断、成功率は未再評価 |
| [旧・新モデルの予測比較](LIBERO_PREDICTION_COMPARISON.ja.md) | 再生成データBT10kの351条件、旧BT100kとの参考比較、旧BC21%の解釈 |
| [LIBEROモデル診断](LIBERO_MODEL_DIAGNOSTICS.ja.md) | 保持50デモの行動依存予測と、公式初期状態からの候補順位診断 |
| [実装・運用監査履歴](IMPLEMENTATION_AUDIT.ja.md) | 数学・接続・再開テスト、LIBERO短期診断、Issue対応、旧試作・失敗記録 |

## データ形式・Policy・学習基盤

| レポート | 内容 |
|---|---|
| [LeRobot v3対応](LEROBOT_V3.ja.md) | HDF5との共存、形式間推論、実データBT100更新・保存再開・回帰確認 |
| [ckpt／LeRobot Policy変換](POLICY_EXPORT.ja.md) | 共通CEM・config・コンバーター、実データ2episode／22時刻の行動一致 |
| [学習高速化（2026-09-10）](TRAINING_SPEED_20260910.ja.md) | GB10での短期速度比較、同期・転送・任意コンパイル、依存版による差と再開検証 |

## 研究調査・構成履歴

| レポート | 内容 |
|---|---|
| [LeWM引用研究のウォッチリスト](LEWM_CITATION_WATCH_20260909.ja.md) | TC-LeWM、LpWM、Fast-LeWM等の採否・比較順・再現性上の注意 |
| [構成整理と復元](CLEANUP.ja.md) | 文書統合、旧コードの改名・削除、復元方法 |

[現在の検証状況と未完了項目](../VALIDATION.ja.md)を入口とし、履歴の「未実装」「学習中」を現在の状態と混同しない。生データ・重み・ログはGit対象外の `output/` と既存 `.cache/` に保持している。新しい学習・評価の起動を指示する文書ではない。

記載された旧ファイル名・実行コマンドは当時の証拠です。現行の改名・削除・issue整理は[整理記録](CLEANUP.ja.md)を参照し、過去コマンドを現在の実行手順として使用しないでください。
