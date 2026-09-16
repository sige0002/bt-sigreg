# 既存実験と比較に使える範囲

確認日：2026-09-16。[比較手順](../reference/COMPARISON_PROTOCOL.ja.md)の棚卸し用一覧。全ファイルを網羅した自動台帳ではない。下表の「記録」は文書の実施結果、「今回確認」はファイル・完了記録の読み取りであり、再評価・重みの全読込・データ全量hash検証ではない。稼働状態はこの表から推測しない。

## PushT：世界モデルの制御比較

| 対象 | 既存の証拠 | 比較に使える範囲／不足 |
|---|---|---|
| Raw／BT clip90・各140k | 20k時点の同条件50ケースでRaw32成功／BT42成功。[記録](PUSHT_RAW_BT_140K.ja.md) | 同初期値・同予算の途中比較。14万更新完了と最終制御成績は未確認 |
| 旧経路BT Cayley v2・100k | 固定200ケース178成功。[評価記録](PUSHT_CHECKPOINT_EVALUATION.ja.md) | 公式176/200はソース版と学習条件が違う参考値。BT優位性の証明ではない |
| 公式LeWM配布モデル | 既存の200／50ケース評価記録、ローカルの配布重み | 同一評価器での参考比較に利用可能。公式学習量と同予算のRaw再学習は別 |
| ローカルRaw70k／旧BT70k | 45/50対47/50。[対応付き記録](PUSHT_ISSUE20.ja.md) | 初期モデルhash等は同じだが、抽出法・LR添字と10ケースの物理初期状態が異なる。主比較として未完成 |
| 新経路BT compiled・予定100k | 10k／20k等の固定ケース評価。[記録](CACHED_CEM.ja.md) | 停止維持指示あり。旧100k経路とはbatch・実装が違う。同じ新経路の対照を確認する必要がある |
| 旧Raw／TC／RBG試作 | 古い`.cache/stable-wm/pusht/rbg_v0/`等 | 歴史資料。現行BT v2の対照に流用する前に全契約を確認。RBG結果をBTと呼ばない |

今回、旧BTの実体`.cache/stable-wm/pusht/bt_spectral_v2_100k_s3072/`で100k object checkpointとmetrics末尾step100000を確認した。`completed.json`はこの旧runには見当たらず、現行の完了形式を遡及して要求・作成しない。公式重み`.cache/stable-wm/pusht/lewm_object.ckpt`も既存参照先。新BTの`output/pusht/bt_compiled_100k_s3072/`には10k／20k／30kのobject checkpointがあり、予定100kというrun名を100k完了と読まない。

旧レポート中の`output/pusht/raw70000_confirm50/`等は、現在の同じパスに存在するとは限らない。今回の`output/pusht/`直下では確認できず、再集計時に実体・アーカイブ・[復元記録](CLEANUP.ja.md)を確認する。履歴のパスを一括置換したり、欠けた証拠を推測で埋めたりしない。

## LIBERO-10：世界モデルの比較

| 対象 | 既存の証拠 | 比較に使える範囲／不足 |
|---|---|---|
| 旧128pxデータBT100k | `bt_no_pin_100k_s3072`完了記録。CEM10試行0成功、保持50デモとnative候補の診断 | 単一BTの能力と失敗例。同データ・同予算Raw／TCの完了比較は未確認 |
| 再生成256pxデータBT10k | `bt_openvla_10k_s3072`の10k完了記録。保持39デモ・351条件、native110候補の診断 | 現行比較のBT候補。Raw／TCの同条件runが不足。旧BT100kとの比較はデータ・予算が異なる |
| 旧`bt_spectral_v2_100k_s3072` | 停止維持指示あり | `bt_no_pin_100k_s3072`とは別run。名前や同じ予定予算だけで同一視しない |

世界モデルrunは`output/libero10/`、manifestは`output/manifests/libero10/`。今回、旧BT100kと新BT10kの`completed.json`を実際に確認した。新データは再実行500デモ中388本を保持、分割はtrain310／validation39／test39。旧manifestと混ぜない。

結果の入口は[旧CEM評価](LIBERO_BT100K_EVALUATION.ja.md)、[旧モデル診断](LIBERO_MODEL_DIAGNOSTICS.ja.md)、[新旧予測診断](LIBERO_PREDICTION_COMPARISON.ja.md)、[新モデル制御診断](LIBERO_WORLD_MODEL_CONTROL_DIAGNOSTIC.ja.md)。最後の診断のCEM改善案は補助的な次案であり、現在の主方針は[方式間比較](../reference/COMPARISON_PROTOCOL.ja.md)を先に揃えること。

## BC：視覚表現の補助評価

| 対象 | 既存の証拠 | 位置付け |
|---|---|---|
| 旧BT100k encoder＋BC40k | 学習完了、環境105/500成功＝21%。[記録](LIBERO_PREDICTION_COMPARISON.ja.md) | 未来予測器を使わない補助評価 |
| 再生成BT10k encoder＋BC40k | 今回`completed.json`の40,000更新succeededと最終BC checkpointを確認 | 学習完了。40kの行動診断・環境成績はこの一覧では未確認 |
| 新BC5k／25k／35kの行動生成 | 全388デモ・3時刻・各3生成。[結果](LIBERO_BC_ACTION_DIAGNOSTIC.ja.md) | 教師行動への近さ。世界モデルによる計画能力ではない |
| 新BC5k／25k／34kの環境評価 | ユーザー指示・意図訂正により停止。[停止記録](LIBERO_BC_OPENVLA_EVALUATION.ja.md) | 未完了。残り試行を失敗扱いせず、再開しない |

BCの途中checkpointと各診断は保持するが、BT正則化の主比較の不足をBC成績で埋めない。

## 他のLeWM環境

| 設定 | リポジトリ内の入口 | 現時点の扱い |
|---|---|---|
| Reacher | `lewm/config/train/data/dmc.yaml`、`lewm/config/eval/reacher.yaml` | 設定は存在。今回、同条件Raw／TC／BTの完了実験は確認していない |
| TwoRoom | `lewm/config/train/data/tworoom.yaml`、`lewm/config/eval/tworoom.yaml` | 同上 |
| Cube | `lewm/config/train/data/ogb.yaml`、`lewm/config/eval/cube.yaml` | [LeWMに揃える手順](../reference/OGBENCH_WORKFLOW.ja.md)を追加。公開配布メタデータと設定展開を確認。実行・方式比較は未実施 |

設定が置かれていることを「この環境でもBTを実行済み」と説明しない。追加前の確認順は[比較手順](../reference/COMPARISON_PROTOCOL.ja.md)を参照する。
