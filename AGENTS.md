# BT-SIGReg

LeWM 程度の小型共有世界モデルで、PushT と LIBERO-10 の画像・行動軌道からダイナミクスを学ぶ研究リポジトリ。原則として日本語で報告する。

## 作業範囲と運用

- 依頼された実装・文書・関連検証まで進める。新規・再開の長時間学習、追加評価、自動評価予約はユーザーの明示依頼があるときだけ行う。定期監視や checkpoint 到達待機の service / エージェントは起動しない。
- 旧 PushT `bt_compiled_100k_s3072` と旧 LIBERO `bt_spectral_v2_100k_s3072` は停止維持の指示がある。別途依頼された `bt_no_pin_100k_s3072` の記録と再開条件は [運用記録](mylewm/docs/AGENT_OPERATIONS.ja.md)。記録を現在の稼働状態の証拠にしない。
- 再開は開始時のソース・環境・設定を使い、hash 照合を解除しない。過去の manifest、checkpoint、評価証拠を書き換えない。
- 稼働中の学習用 `.venv` に依存を同期しない。依存変更の検証は `UV_PROJECT_ENVIRONMENT` で隔離する。旧 object checkpoint の復元も確認する（既存 Transformers 固定の理由は検証資料を参照）。
- 他のプロセス・GPU サービスを検証目的で停止しない。失敗ログを保持し、無限再起動しない。

## 研究の不変条件

- LeWMの目的に従い、通常の画像・行動軌道から、報酬・タスク仕様なしに環境のダイナミクスを学ぶ。学習データの作製そのものを研究目的にしない。
- LeWM程度の小型共有世界モデルで高精度なマルチタスク制御を目指す。対象はPushTとLIBERO-10。LIBERO-10では10タスクを一つの世界モデルで学ぶ。
- 非操作物体保持は補助診断であり主目的ではない。人手の交差対応表、複製ビューを実測と扱う学習、多段予測損失、大型事前学習モデルへの置換で主題をすり替えない。
- 実装の容易さは研究案の採択理由にしない。モデル規模、推論速度、学習計算量、安定性は評価対象とする。

- 現行 BT は Cayley 特異値制約 v2。旧 Frobenius checkpoint と互換性はない。旧 RBG の結果を BT と呼ばない。実施済み範囲と未実証の主張は [VALIDATION](mylewm/docs/VALIDATION.ja.md) で照合する。
- 予測・rollout・Goal 距離は状態 z で計算し、学習専用の同次元可逆写像 u=T(z) に SIGReg を適用する。T は全タスク・時刻で共有し、ID・Goal・行動・batch 統計で条件付けず、乱数で分散を作らない。
- 固定の大域的 bi-Lipschitz 上下界が設計条件。可逆性だけで尺度逃避を防げるとしない。近似スペクトルノルムを保証された上界と呼ばない。
- 推論から T を除く一方、再開 checkpoint には T と optimizer 状態を保存する。未来教師側 encoder への勾配を維持する。
- 全体分散の条件付き境界を、タスク内情報・収束・成功率の保証に拡張しない。研究案を変えるときは理由と比較条件を示す。

## 実装とデータ

- 実装は `src/mylewm/{algorithms,training,data,evaluation,environments,policy}/`、共通パスは `src/mylewm/paths.py`。`mylewm/` は設定・テスト・文書、`lewm/` は公式比較用。上流側にも既存のローカル評価修正があるため無改変コピーとは呼ばない。
- Raw / TC / BT の共有基盤を方式名だけで不要と判断しない。LIBERO 共有ループには `TrainingAdapter` を渡し、共有関数を上書きしない。保存済みクラス復元用の再公開は保持する。
- 現行 manifest は `output/manifests/{pusht,libero10}/`、run は `output/{pusht,libero10}/`。旧形式・削除・復元の依存確認には [CLEANUP](mylewm/docs/reports/CLEANUP.ja.md) を使う。
- 起動時のデータ検証はサイズ・mtime が既定。新規 prepare 時に SHA-256 を記録し、全量再検証は `--verify-data` 時だけ行う。prepare 時の hash と今回の実測を区別する。
- データ・公式重み・生成ログ・環境・認証情報を Git に入れない。保存済み HDF5 に評価用の補完列を書き足さない。

## 必要な資料と検証

| 作業 | 参照先 |
|---|---|
| 研究設計・主張 | [総合レビュー](mylewm/docs/research/RESEARCH_REVIEW.ja.md)、[BT-SIGReg](mylewm/docs/research/BT_SIGREG.ja.md) |
| prepare・学習・再開 | [TRAINING](mylewm/docs/TRAINING.ja.md)、[運用記録](mylewm/docs/AGENT_OPERATIONS.ja.md) |
| 評価・比較・依存更新 | [検証上の制約](mylewm/docs/AGENT_VALIDATION.ja.md)、[EVALUATION](mylewm/docs/EVALUATION.ja.md) |
| dataset / 推論 / policy export | [MODEL_USAGE](mylewm/docs/MODEL_USAGE.ja.md) |
| BC の追加学習・native 評価 | [BEHAVIOR_CLONING](mylewm/docs/BEHAVIOR_CLONING.ja.md) |

- コード変更は対応する回帰テストを実行する。全体の回帰入口は `.venv/bin/python -m pytest mylewm -q`。文書だけの修正ではリンクと契約の整合を確認する。
- CPU での合格・GPU skip・学習・制御評価を区別する。依頼された評価は終了コード・結果・`status.json` を照合し、SIGKILL 等では実プロセスも確認する。
- 比較では Raw / TC / BT の初期値・データ順・予算・前処理・Goal・成功関数を揃え、T の追加計算量を報告する。公式 checkpoint の再現と同予算の再学習、BC と CEM を別の比較として扱う。
- PushT のみでマルチタスク改善を主張しない。LIBERO の平均・各/下位タスク・seed 間変動、生の潜在 MSE 以外の制御成績・尺度対照・信頼区間を確認する。
- 主張は一次資料と照合し、既存技術・独自導出・実験仮説を区別する。古い資料との矛盾は最新のユーザー指示と総合レビューを優先して明記する。
- 委任はユーザーが求めた場合に行う。AI の役割レビューを実在専門家の査読と呼ばない。
- push 前は差分、`git diff --check`、追加ファイルと機密・生成物の混入を確認する。変更点・検証結果・未完了部分を報告する。
- セッションの model / provider / reasoning 等を聞かれた場合は、そのセッションに結び付いた runtime 記録で検証し、設定から推測しない。
