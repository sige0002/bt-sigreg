# 構成整理と復元

## src移行・共通処理整理（2026-09-11）

Python実装を`src/mylewm/`の`algorithms`・`training`・`data`・`evaluation`・`environments`・`policy`へ集約し、シェルはリポジトリ直下の`scripts/`へ移動しました。[現行構成と起動](../../README.md#コードの構成と起動)を参照してください。

| 変更 | 内容 |
|---|---|
| CLI | editable導入後の`python -m mylewm.training.train`等へ統一。旧ファイルパスのCLIは撤去 |
| 共通パス | `src/mylewm/paths.py`でリポジトリ・設定・ソース一覧を解決。LIBERO外部環境は環境変数を尊重 |
| 学習adapter | データ取得・前処理・モデル構築を`TrainingAdapter`で渡す。LIBEROによる共有モジュール関数の上書きを撤去 |
| hash計算 | 訓練・診断・上流評価の重複wrapperを共通`file_sha256`へ統合。大きなcheckpointを一括で`read_bytes`しない |
| ソース照合 | パッケージ内の全Pythonと該当する上流ソースを記録。新しい共通処理も変更検出の対象 |
| 固定実験シェル | 旧`mylewm/tools/run_pusht_repeated_evals.sh`を撤去。過去runの固定名と自動反復実行に特化し、現行手順からの参照が無いため。単発評価・比較CLIは保持 |
| 画像監査 | LIBERO画像監査に`--dataset`を追加し、指定したデータ場所を使う |
| 比較用コード | `lewm/eval.py`はimport先6か所のみ更新。環境成功判定・評価ロジックは変更しない |
| 保存物 | データ・重み・manifest・config・実験出力を移動・書換えしない |

保存済みLIBERO object checkpointの`mylewm.libero_model.TwoViewJEPA`参照だけは、`src/mylewm/libero_model.py`から現行クラスを再公開して維持しています。モデル実装の重複や旧RBGの復活ではありません。過去runの**厳密な学習再開には開始時のGit版・環境が必要**で、設定・ソース照合の解除は行いません。

今回の移行直前のコード・文書・未コミットのPolicy実装は`/tmp/bt_code_before_src/`へ退避しました（一時バックアップ）。Git管理済みの旧コードは移行前コミットと旧パスからも復元できます。過去レポート内のコマンドは当時の証拠として保持しました。

検証：固定済み訓練環境の依存を参照する隔離環境`.venv-refactor`へeditable導入し、不足するLeRobot依存だけをlock記載版で追加しました。既存`.venv`・`.venv-training`には同期していません。最初はCLIテストの旧パス、移動時の構文エラー、隔離環境のLeRobot依存不足を検出して修正しました。

最終コマンドは`CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv-refactor/bin/python -m pytest mylewm -q --disable-warnings`で、**182合格・スキップ0・失敗0（45.71秒）**。CUDA／コンパイル／連続更新と保存再開の一致、LeRobot形式間推論・Policy保存読込、リポジトリ外からのCLI起動、外部環境変数、旧クラス名のpickle読込を含みます。既存由来の警告629件は残っています。一時ログは`/tmp/bt_src_final.log`です。

実在する旧LIBERO100更新object checkpointもCPUでロードし、現行`TwoViewJEPA`・18,427,874パラメータを確認しました。読込前後のSHA-256は`327feee6bd7e90576c07abaea4769752a7e2ca52a5e9b7405123d9d3342adbae`で不変です。対象は`.cache/stable-wm/libero10/bt_spectral_v2_smoke_20260908_s3072/step_100_object.ckpt`。長時間学習・新しい制御成功率評価・別PCの描画検証は実行していません。文書リンク、シェル構文、Python構文、lock整合性、差分も確認しました。

## 文書の統合・配置整理（2026-09-11）

`docs/README.md`を入口とし、直下の文書を10個から5個へ整理しました。評価3文書を1冊、データ形式とPolicyの2文書を1冊へ統合しています。実験レポートの数値・コマンド・hashは保持し、参照リンクを更新しました。

| 旧配置（docs基準） | 現在の配置 |
|---|---|
| EVALUATE_PUSHT.ja.md / EVALUATE_LIBERO.ja.md / EVALUATE_INTERMEDIATE.ja.md | [EVALUATION.ja.md](../EVALUATION.ja.md) の各章 |
| DATA_FORMATS.ja.md / LEROBOT_POLICY.ja.md | [MODEL_USAGE.ja.md](../MODEL_USAGE.ja.md) の各章 |
| BT_SIGREG.ja.md / RESEARCH_REVIEW.ja.md | `research/` |
| CLEANUP.ja.md | 本文書（`reports/`） |

旧ファイルの本文は統合先へ保持しています。整理直前のローカル文書は `/tmp/bt_docs_before_organization/` に退避しました（一時バックアップ）。Git管理済みの過去版は、その時点のコミットから旧パスを指定して参照できます。未コミットだったPolicy文書も統合先へ収録しています。旧パスの案内専用ファイルは残しません。

以降は過去のコード整理記録です。

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

実験レポート内の旧CLI名・ソースhash・コマンドは当時の証拠なので置換しない。[レポート一覧](README.md)は履歴、[学習手順](../TRAINING.ja.md)と[評価手順](../EVALUATION.ja.md#libero)は現行操作を記す。

## Issueの整理

ユーザー承認により、旧issue #2・#4〜#13 の11件を `not planned` で閉じた。openは0件。実装・学習・検証の完了認定ではなく旧計画の廃止であり、未完了の同予算比較・LIBERO本評価等は[検証状況](../VALIDATION.ja.md)に残す。
