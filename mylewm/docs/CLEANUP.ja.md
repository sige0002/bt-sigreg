# 構成整理と復元

更新日：2026-09-09。現行はBT-SIGRegとRaw/TC比較のためのコードのみを残す。過去の研究案・失敗記録はGit履歴と実験レポートへ分離し、現在の操作手順と混ぜない。

## 今回の改名

| 旧名 | 現在 | 役割 |
|---|---|---|
| `train_rbg.py` | `training.py` | LIBERO共有ループ・PushT分割作成・比較検査用参照経路 |
| `train_rbg_libero.py` | `train_libero.py` | LIBERO-10の学習・分割 |
| `rbg.py` | `objectives.py` | 全次元SIGRegとRaw/TC/BT一段損失 |
| `tools/eval_rbg_libero.py` | `tools/evaluate_libero.py` | LIBERO制御評価 |
| `tests/test_rbg.py` | `tests/test_objectives.py` | 共有損失と公式一致のテスト |
| `tests/test_rbg_data.py` | `tests/test_dataset_clips.py` | データの時間・episode・実カメラ契約 |

新規PushTの入口は引き続き `train.py`。旧名の互換shimは増やさない。Pythonは **30→28個**（直下12、tools5、tests11）。名前変更だけでなく以下の旧機能を撤去した。

## 撤去した処理

- RBGのブロック分割、ブロック間交差共分散、`mode=rbg`、`--blocks`、`--cross-weight`。全次元用 `GaussianSIGReg` とスカラー損失へ整理し、無効なcross値をログへ出さない。
- RBG専用の交差共分散検算・周辺分布反例テスト。現行SIGRegの非崩壊の限界・公式一致・TC・BT数学のテストは残す。
- 廃止済み生成CLIの初期値を読む `--initialization`。同seedの初期モデルhash照合と、各runの `initialization.pt` 保存・再開時照合は残す。
- `tools/evaluate_official_pusht.py`：旧公式専用launcher。公式・BTとも現在の `tools/evaluate_pusht.sh` へ統一し、専用CLI/planテスト2件も撤去。結果・動画の履歴は保持。
- `tools/smoke_libero.py`：環境起動は画像監査、所定初期状態は評価で確認する。
- 画像監査の `--export-textures`：過去の描画不具合調査用。現在必須の実画像一致監査は残す。

CLI・import・テスト・現行Markdownを更新した。LIBERO学習と評価の新規manifest/監査出力の既定は `output/` 以下へ統一。既存PushT評価の既定manifestは固定済み回帰集合を維持するため変更しない。

## 残すものと理由

- BTの有界写像、Raw・TC比較、共有学習・保存・再開：現在の研究・LIBEROで使用。
- データと評価契約、行動座標adapter、LIBEROモデル・計画器：削除すると比較条件や推論が変わる。
- 結果比較、記録行動の再実行、画像監査：現在の成功率確認・評価条件の検証で使用。
- JSONL監視シェル：LIBERO共有ループと既存結果の手動確認に使用。新PushTはCSVを読む。
- `lewm/` と `external/` の公式・外部コード：比較・依存先であり提案側の整理で削除しない。
- 全ての重み・データ・manifest・config・ログ・評価結果：改名に合わせた内容の書換えも行わない。

`rbg_pusht_v1` / `rbg_libero10_v1` は保存データ形式の識別子として残す。旧 `rbg_v0` パスは既存評価証拠の参照であり、RBG学習の存続を意味しない。`agentview_rgb` などは実カメラのRGB画像キーなので改名しない。BTの残差 `blocks` も旧RBGの座標分割とは別物。

## 検証と互換性

CPU限定回帰は **102合格・5スキップ**（15.02秒）。公式Rawのloss・全モデル勾配一致、恒等BT、TCの適用座標、未来教師勾配、Tの境界・更新・推論分離、保存再開一致、改名CLIと旧方式拒否を確認した。CUDA専用5件は未実行。整理のための本学習・実制御評価は実行しない。

モデルの数式・E/A/F・データ抽出・optimizer・学習率・成功判定は今回変更しない。ただしソースhash・CLI・ログ項目は変わるため、**過去runの厳密resumeには開始時のGit版が必要**。既存configを編集したり、hash照合を無効化したりしない。推論用checkpointと学習再開用checkpointの互換性は別問題である。既存PushT 100k推論重みをCPUでロードし、SHA-256が `af541c74fbc03a4739bb4d5c82d28313c508a185b1cb77509cae8436afbbe45b` のまま、18,034,478パラメータ・行動統計buffer保持・Tなしを確認した。これは制御成功率の再評価ではない。

## Git履歴からの復元

今回の整理前は [3c68364](https://github.com/sige0002/bt-sigreg/tree/3c68364/mylewm)。例えば `git show 3c68364:mylewm/rbg.py` で旧内容を読める。過去実験を再開・追試する場合は、runに対応するGit版を別の作業ディレクトリへ展開し、現在のファイルを一括上書きしない。

前回撤去した独立診断・旧比較準備13ファイルは [5559095](https://github.com/sige0002/bt-sigreg/tree/5559095/mylewm) に保存済み。[以前の整理・削除一覧](https://github.com/sige0002/bt-sigreg/blob/3c68364/mylewm/docs/CLEANUP.ja.md)には、さらに前の整理履歴とローカルバックアップの場所も記録している。

実験レポート内の旧CLI名・ソースhash・コマンドは当時の証拠なので置換しない。[レポート一覧](reports/README.md)は履歴、[学習手順](TRAINING.ja.md)と[評価手順](EVALUATE_LIBERO.ja.md)は現行操作を記す。

## Issueの整理

ユーザー承認により、旧issue #2・#4〜#13 の11件を `not planned` で閉じた。openは0件。実装・学習・検証の完了認定ではなく旧計画の廃止であり、未完了の同予算比較・LIBERO本評価等は[検証状況](VALIDATION.ja.md)に残す。
