# 検証状況と不足している比較

整理日：2026-09-16。現在の主方針は**CEMを固定してBT-SIGRegの比較評価を蓄積すること**。BCやCEM改良の診断は補助資料として扱う。[比較手順](../reference/COMPARISON_PROTOCOL.ja.md)・[既存実験一覧](EXPERIMENTS.ja.md)から進めてください。

## 現在言えること

| 対象 | 確認済み | まだ言えないこと |
|---|---|---|
| BT Cayley v2 | 学習専用T、共有世界モデル、保存再開・推論からT除去の実装と回帰検証 | 非崩壊や距離境界による制御成功の保証 |
| PushT | 旧BT100kの固定200ケース178成功。公式／Rawとの参考比較記録あり | 正則化以外を揃えたRaw／TCに対する優位性 |
| LIBERO世界モデル | 旧BT100k・新BT10kの学習、保持デモ予測とnative候補の診断 | 同条件Raw／TCに対するマルチタスク改善 |
| LIBERO制御診断 | 新BT10kの全110候補で予測・Goal距離・探索量を切り分け | 32行動の分岐成績を通しのタスク成功率に換算すること |
| BC | 旧モデルの環境21%、新5k／25k／35kの教師行動診断、新BC40kの学習完了記録 | 世界モデルの未来予測による計画能力、未評価の40k制御成績 |

## 比較を完成させるための不足

- PushTの主比較には、同じ学習経路・予算・物理初期状態のRaw／BTが必要。過去Raw70k／BT70kの条件差を残したまま主比較に昇格させない。
- LIBEROは旧128px・100kと再生成256px・10kを分け、選んだデータ版・予算でRaw／TC／BTを揃える。既存BTの再利用には共通初期値等の照合が必要。
- 同じ固定CEM・Goal・成功関数で制御成績、予測と尺度対照、計算コストを集める。BCの追加学習を必須工程にしない。
- 独立学習seed間の変動とLIBEROの各／下位タスクを確認する。単一runの成績だけで優位性・非劣性を確立しない。
- 他のLeWM環境は設定と実行実績を区別し、追加の前にデータ・重み・共通経路の成立を確認する。

## 検証記録の入口

| 内容 | 記録 |
|---|---|
| PushTの実測と比較条件差 | [PushT評価](PUSHT_CHECKPOINT_EVALUATION.ja.md)、[Raw／BT70kと初期状態差](PUSHT_ISSUE20.ja.md) |
| 世界モデルの行動依存性 | [旧モデル診断](LIBERO_MODEL_DIAGNOSTICS.ja.md)、[新旧の予測比較](LIBERO_PREDICTION_COMPARISON.ja.md) |
| 予測・Goal・CEMの限界 | [世界モデル制御診断](LIBERO_WORLD_MODEL_CONTROL_DIAGNOSTIC.ja.md) |
| BCの生成行動と停止済み環境評価 | [行動診断](LIBERO_BC_ACTION_DIAGNOSTIC.ja.md)、[環境評価記録](LIBERO_BC_OPENVLA_EVALUATION.ja.md) |
| 実装・データ・互換性 | [実装監査](IMPLEMENTATION_AUDIT.ja.md)、[検証履歴](../reference/VALIDATION_HISTORY_20260915.ja.md) |

2026-09-15の全回帰は240件合格・skipなし（61.03秒）。その後の世界モデル診断関連回帰は25件合格・skipなし（3.69秒）。これは別の変更時点・別範囲の検証であり、「現在の全回帰25件」と読まない。今回の2026-09-16整理は文書のみで、学習・制御評価・全回帰の再実行はしていない。

## 文書と実験の更新方法

新しい実験は個別レポートへ条件・結果・失敗・証拠を記録し、[実験一覧](EXPERIMENTS.ja.md)へ追加する。このページには結論と不足だけを反映する。操作手順はガイド、日付順の経緯はレポート、稼働確認と停止指示は[運用記録](../operations/AGENT_OPERATIONS.ja.md)に置く。

過去の詳細は[2026-09-15までの検証履歴](../reference/VALIDATION_HISTORY_20260915.ja.md)へ移した。履歴の「未実施」「稼働中」は当時の説明で、現在のプロセス・完了状態の証拠ではない。

<details>
<summary>以前の見出しへの互換リンク</summary>

<a id="実施済み"></a>

[実施済み](../reference/VALIDATION_HISTORY_20260915.ja.md#実施済み)

<a id="issue-20pusht評価画像動画診断メモリ2026-09-10"></a>

[issue-20pusht評価画像動画診断メモリ2026-09-10](../reference/VALIDATION_HISTORY_20260915.ja.md#issue-20pusht評価画像動画診断メモリ2026-09-10)

<a id="issue-19評価設定の列リスト変更2026-09-10"></a>

[issue-19評価設定の列リスト変更2026-09-10](../reference/VALIDATION_HISTORY_20260915.ja.md#issue-19評価設定の列リスト変更2026-09-10)

<a id="公式ライブラリへのpusht移行2026-09-09"></a>

[公式ライブラリへのpusht移行2026-09-09](../reference/VALIDATION_HISTORY_20260915.ja.md#公式ライブラリへのpusht移行2026-09-09)

<a id="旧方式撤去役割名への整理2026-09-09"></a>

[旧方式撤去役割名への整理2026-09-09](../reference/VALIDATION_HISTORY_20260915.ja.md#旧方式撤去役割名への整理2026-09-09)

<a id="既存経路での実績"></a>

[既存経路での実績](../reference/VALIDATION_HISTORY_20260915.ja.md#既存経路での実績)

<a id="未完了主張できないこと"></a>

[未完了主張できないこと](../reference/VALIDATION_HISTORY_20260915.ja.md#未完了主張できないこと)

<a id="issueと運用"></a>

[issueと運用](../reference/VALIDATION_HISTORY_20260915.ja.md#issueと運用)

<a id="学習起動時のデータ検証変更2026-09-10"></a>

[学習起動時のデータ検証変更2026-09-10](../reference/VALIDATION_HISTORY_20260915.ja.md#学習起動時のデータ検証変更2026-09-10)

<a id="既存checkpointの依存互換性修正2026-09-10issue-15"></a>

[既存checkpointの依存互換性修正2026-09-10issue-15](../reference/VALIDATION_HISTORY_20260915.ja.md#既存checkpointの依存互換性修正2026-09-10issue-15)

</details>
