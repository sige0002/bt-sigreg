# 停止指示と再開条件

この文書は実行の許可・停止指示・再開条件の正本です。現在の稼働状態は記録から推測せず、依頼時に実プロセス・ログ・完了記録で確認します。実験の条件・結果は`reports/EXPERIMENTS.ja.md`に集約します。

## 停止を維持するもの

- 旧PushT `bt_compiled_100k_s3072`と旧LIBERO `bt_spectral_v2_100k_s3072`。別途明示依頼があるまで再開しません。
- BC 2,000更新の環境評価、および再生成データBCの5k／25k／34k環境評価。停止済みの未実施試行を失敗と数えません。
- 旧100k→BC40kの引継ぎ親プロセスは、10k→BC40kへ方針変更済みです。旧親をSIGCONTしません。

## 再開するとき

1. ユーザーの明示依頼と対象runを確認します。文書の実行例や過去の起動記録は新しい実行の認可ではありません。
2. 元のプロセスが終了していることを確認します。開始時の固定ソース・環境・manifest・設定・総更新数を復元し、hash照合を解除しません。
3. PushTは再開用Lightning checkpointと新規出力先、LIBERO世界モデルとBCは元の出力先・`resume.pt`を使います。
4. 失敗ログ、過去のmanifest、checkpoint、評価証拠は保持します。稼働中の`.venv`へ依存同期せず、他のGPUプロセスを検証目的で止めません。

自動評価予約、checkpoint到達待機、定期監視service、無限再起動は開始しません。「BC評価」の依頼では、生成行動の教師デモ比較か環境成功率かを区別します。

## run別の固定ソース・証拠

| 対象 | 条件・固定ソース・停止記録 |
|---|---|
| PushT Raw／BT・各140k | [起動条件](../reports/PUSHT_RAW_BT_140K.ja.md) |
| LIBERO `bt_no_pin_100k_s3072` | [学習記録](../reports/LIBERO_NO_PIN_TRAINING.ja.md)。旧停止runとは別 |
| 旧LIBERO BT・BC接続 | [停止と実装記録](../reports/LIBERO_BC.ja.md) |
| 再生成BT10k→BC40k | [条件変更・引継ぎ](../reports/TCLEWM_ALIGNMENT.ja.md) |
| 旧BT100k encoder＋BC | [学習・2k評価停止](../reports/LIBERO_BC_BT100K.ja.md) |
| 再生成BC環境評価 | [5k／25k／34k停止](../reports/LIBERO_BC_OPENVLA_EVALUATION.ja.md) |

マシン固有の設定は[環境別メモ](MACHINE_NOTES.ja.md)、検証の制約は[検証ルール](AGENT_VALIDATION.ja.md)です。日付順の重複記録と旧操作手順はGit履歴で参照します。
