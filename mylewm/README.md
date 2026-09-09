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

PushT BT v2の100,000更新は正常終了。固定confirm 200ケースでは178/200（89%）、別条件の上流eval 50ケースでは49/50（98%）です。ケース・正規化・seed処理が異なるので混ぜません。[評価レポート](docs/reports/PUSHT_CHECKPOINT_EVALUATION.ja.md)に条件と留保を記録しています。

LIBERO-10は100更新の動作確認まで。同予算Raw/TC比較、複数学習seed、マルチタスク性能向上は未実証です。新規学習・評価・定期監視を自動開始しません。

## コードの役割

| 場所 | 用途 |
|---|---|
| `bt_sigreg.py` | 現行BTの有界変換・正則化 |
| `train_rbg.py` / `train_rbg_libero.py` / `rbg.py` | Raw/TC/RBG/BTの共有trainer・一段損失。旧名だが現役 |
| `libero_model.py` / `libero_planner.py` | 2つの実カメラを使う共有モデル・計画器 |
| その他の直下Python | データ・再開・評価契約、行動正規化、診断の共通部品 |
| `tools/` | 学習補助・評価・比較・監査CLI |
| `tests/` | 残した実装の回帰テスト |
| `docs/` / `docs/reports/` | 手順・研究仕様／実験結果・監査記録 |
| `run_libero.sh` | ローカルOSMesa環境の起動wrapper |

既存checkpointの復元・ソース照合を壊さないため、共有Pythonの名前や配置は維持します。補助診断を実行できることと、研究目的を達成したことは別です。

## シェルと回帰確認

進捗表示は手動・読み取り専用です。既定runは完了済みのPushTなので、新規実験では `--run` を指定します。

```bash
bash mylewm/tools/monitor_training.sh --once
bash mylewm/tools/evaluate_pusht.sh --help
CUDA_VISIBLE_DEVICES='' PYTHONPATH=.:lewm .venv/bin/python -m pytest mylewm/tests -q
```

CPU限定の回帰確認ではCUDA専用テストはスキップされます。環境での成功率評価や追加学習は行いません。全実験出力はGit対象外の `output/` 以下へ保存します。
