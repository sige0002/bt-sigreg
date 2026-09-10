# mylewm：BT-SIGRegの学習・比較基盤

LeWMの一段予測と小型E/A/Fを維持し、学習専用の有界可逆写像Tを通した表現だけにSIGRegを適用します。推論時にはTを使いません。

## はじめて学習する方へ

[PushT／LIBERO-10の学習手順](docs/TRAINING.ja.md)に、環境・データ確認、分割作成、短期確認、本学習、進捗表示、再開をまとめています。このPCの既存環境を前提とし、別PCへの完全な環境構築手順ではありません。

| 読みたい内容 | 文書 |
|---|---|
| 研究目的と全体像 | [ルートREADME](../README.md) |
| アルゴリズム・数式・保証の限界 | [BT-SIGReg](docs/BT_SIGREG.ja.md) |
| 関連研究・採否判断 | [研究レビュー](docs/RESEARCH_REVIEW.ja.md) |
| PushTの成功率評価 | [評価手順](docs/EVALUATE_PUSHT.ja.md) |
| LIBERO-10の評価環境 | [評価手順](docs/EVALUATE_LIBERO.ja.md) |
| 公式LeWMの学習 | [公式PushT手順](../lewm/TRAIN_PUSHT.ja.md) |
| 実験結果・監査・未完了項目 | [レポート一覧](docs/reports/README.md) |
| ファイルを残した理由・復元 | [整理記録](docs/CLEANUP.ja.md) |

## 現在の状態（2026-09-09）

新規PushTのRaw／BT比較用に `train.py` を追加しました。公式SWM/SPT/Lightningを使い、エピソード分離を維持します。[新経路の手順と条件差](docs/TRAINING.ja.md#新しいpusht経路公式ライブラリへ委託2026-09-09)を参照。旧100kの継続ではなく新レシピで、本学習・成功率評価は未実施です。

PushT BT v2の100,000更新は正常終了。固定confirm 200ケースでは178/200（89%）、別条件の上流eval 50ケースでは49/50（98%）です。ケース・正規化・seed処理が異なるので混ぜません。[評価レポート](docs/reports/PUSHT_CHECKPOINT_EVALUATION.ja.md)に条件と留保を記録しています。

LIBERO-10は100更新の動作確認まで。同予算Raw/TC比較、複数学習seed、マルチタスク性能向上は未実証です。新規学習・評価・定期監視を自動開始しません。

## コードの役割

| 場所 | 用途 |
|---|---|
| `bt_sigreg.py` | 現行BTの有界変換・正則化 |
| `train.py` | 新規PushT Raw／BT用。SWMデータ読込・公式forward・SPT更新・Lightning管理への接続 |
| `training.py` | LIBERO用共有ループ、PushTの分割作成・比較検査用の参照経路 |
| `train_libero.py` | LIBERO-10のRaw／TC／BT学習・分割作成 |
| `objectives.py` | 全次元Gaussian正則化・Raw／TC／BTの一段損失 |
| `libero_model.py` / `libero_planner.py` | 2つの実カメラを使う共有モデル・計画器 |
| その他の直下Python | データ・再開・評価契約、行動正規化、学習中診断の共通部品 |
| `tools/` | PushT/LIBERO評価・結果比較・行動再実行・LIBERO画像監査 |
| `tests/` | 残した実装の回帰テスト |
| `docs/` / `docs/reports/` | 手順・研究仕様／実験結果・監査記録 |
| `run_libero.sh` | ローカルOSMesa環境の起動wrapper |

旧RBGのブロック分割・交差共分散損失・専用設定は撤去しました。Pythonは28個（直下12、tools5、tests11）です。共有部品は現在の役割に合わせて改名し、旧名の互換ファイルは増やしていません。LIBEROの共有ループ、Raw/TC比較、データ・再開・評価の安全確認は必要なので残しています。

過去の重み・manifest・評価結果は保持しています。ソースhashが変わるため、過去runの厳密再開は開始時のGit版を使ってください。削除・改名一覧と復元方法は[整理記録](docs/CLEANUP.ja.md)を参照。新規PushTは `train.py`、LIBEROは `train_libero.py` が入口です。

## シェルと回帰確認

進捗表示は手動・読み取り専用です。既定runは完了済みのPushTなので、新規実験では `--run` を指定します。

```bash
bash mylewm/tools/monitor_training.sh --once
bash mylewm/tools/evaluate_pusht.sh --help
CUDA_VISIBLE_DEVICES='' PYTHONPATH=.:lewm uv run python -m pytest mylewm/tests -q
```

CPU限定の回帰確認ではCUDA専用テストはスキップされます。環境での成功率評価や追加学習は行いません。全実験出力はGit対象外の `output/` 以下へ保存します。
