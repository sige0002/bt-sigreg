# ckpt／LeRobot Policy・コンバーター実データ確認（2026-09-10）

ユーザー依頼で、共通の世界モデル＋CEMをckpt直接読込とLeRobot Policyの両方から使えるようにし、独立したコンバーターCLIを追加した。[操作手順・config・API](../MODEL_USAGE.ja.md#policy)。新規学習とロボット接続は行っていない。

## 入出力と保存

`policy_runtime.py`が実履歴、訓練統計、物理座標CEM、行動queueを管理する。`lerobot_policy/`は公式`PreTrainedPolicy`／config／processor factoryに接続する。初期履歴は実画像3枚とその間で実行した2行動chunkを明示し、複製画像で埋めない。以後は実際の前回行動と時刻を渡す。

`export_lerobot_policy.py`は推論用object ckptまたは新経路Lightning訓練ckptから世界モデルだけを取り出す。LeRobot形式のconfig・safetensors・前後処理・hash証跡を保存する。モデル構成は既存単一カメラの`lewm_tiny_v1`を厳密に再構築し、未対応のキー・形状を拒否する。元モデルとのencoder／predictor probeと、保存後の全state・訓練統計の完全一致を確認する。Tとoptimizerは輸出しない。

最終変換元は、前回実データで学習した`output/lerobot_validation/bt_100_s3072/step_100_object.ckpt`。変換先は`output/lerobot_validation/bt_100_lerobot_policy/`。

| 成果物 | SHA-256 |
|---|---|
| `model.safetensors` | `52996c2fbb270ae2b1b9a7832bad67ca532519e5cbb9e65d8ed1a01dbb08507f` |
| `config.json` | `7c86da60d5e34b1102ae4957b72a0f87867ea97281a10ec04dbc37583f05c520` |

`policy_preprocessor.json`、`policy_postprocessor.json`、`export.json`も保存した。画像・行動の正規化は共通runtimeで行い、標準processorで重ねて正規化しない。ckptだけの読込は、LeRobot未インストールの既存隔離環境でも成功した。

## 実データによる確認

入力は取得済み公式`lerobot/pusht`、revision `7628202a2180972f291ba1bc6723834921e72c19`。manifestは`output/lerobot_validation/manifest.json`。固定保持test episodeの173と118を使用した。各episodeの実画像frame0/5/10、実行動0〜9、最後の実画像を目標としてprimeし、frame10〜20の11時刻を順に入力した。

CEMはhorizon4モデル遷移、samples32、elites8、iterations3、seed3072、物理行動範囲各成分0〜512。CPU、FP32。各11時刻でframe10・15・20の3回計画し、その間の5行動queueも比較した。

| 確認 | 結果・出力先 |
|---|---|
| 推論用object ckpt対LeRobot形式、episode173 | 11/11時刻で行動が完全一致。`policy_final_object_check/` |
| 訓練用`last.ckpt`対LeRobot形式、episode118 | 11/11時刻で行動が完全一致。`policy_final_training_check/` |
| LeRobot factory・前後処理 | 登録から保存済みprocessor再読込、batch/device処理、`select_action`まで実行 |
| 行動・cost | 全て有限、出力行動は指定した物理範囲内 |
| 完了 | 両方の終了コード0、`status.json`はsucceeded・ticks11、resultsの行動列22件を確認 |

上の出力先の共通基点は`output/lerobot_validation/`。入力・重み・Policyファイルのhashは各`results.json`に保存した。これは記録された画像と実行済み行動を使うオフライン比較である。提案行動を環境で実行した結果の画像ではないため、制御成功率・実機動作・CEMの有効性を実証したものではない。

## 実行で見つかった問題と修正

1. config再読込：登録名`type`を含むJSONを具象configクラスから読むと拒否された。公式の`PreTrainedConfig`から登録クラスへ復元するよう修正。
2. 実データを標準processorへ通した際、追加の`timestamp`が落ちた。`observation.timestamp`と`observation.executed_action`で渡すよう修正。
3. 標準batch processorは追加行動キーへbatch軸を付けなかった。単一ロボットの`(D,)`と`(1,D)`を明示的に受け付け、実データの2回目以降の呼出しも確認。

失敗ログは`policy_export_v2.log`、`policy_realdata_check.log`、`policy_realdata_check_v2.log`に保持。修正後は最終成果物を新ディレクトリへ書き出し、上記2つの実データ経路を再実行した。失敗した出力を上書きして成功扱いしていない。

## テストと限界

GPU・compile・従来HDF5／LeRobotデータ経路・既存checkpoint復元を含む全回帰は**171合格、スキップ0、27.21秒、警告629件**。`output/lerobot_validation/policy_all_tests.log`。追加11件は既知の線形ダイナミクスでのCEM、乱数分離、実行履歴・時刻・reset、不正な範囲、実JEPAの変換とfactory／processors、設定変更拒否、訓練checkpoint抽出、重み欠落拒否を検査する。

最後のexport証跡追加後もPolicy関連テストを再実行した。実データを入れない単体試験だけで完了とはしていない。

対象実機の初期履歴収集・目標入力・ロボットI/O・実行周期への接続は未実施。標準`lerobot-record`へパスを渡すだけの起動は対応していない。訓練済みモデルと一致する入力条件が必要で、任意のロボットや多カメラモデルへの自動変換は行わない。
