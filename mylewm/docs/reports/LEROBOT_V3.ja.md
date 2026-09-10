# LeRobot v3対応と実データ100更新（2026-09-10）

ユーザー依頼により、HDF5とLeRobot v3を選択する学習入力と、形式をまたぐオフライン推論を実装した。学習ループ・一段予測・Raw／BTの目的は共通。一つのrunで形式は混ぜない。LeRobotは公式0.4.4のreaderとPyAVで直接読み、HDF5へ変換しない。手順は[TRAINING](../TRAINING.ja.md#lerobot-v3で学習する)と[データ形式・推論](../DATA_FORMATS.ja.md)。

## 実データの取得と分割

- Hub: `lerobot/pusht`、固定revision `7628202a2180972f291ba1bc6723834921e72c19`。
- 取得先: `output/lerobot_validation/pusht/`。meta・Parquet・MP4の6ファイル、合計約7.68 MB。
- `meta/info.json`: v3.0、206episode、25,650frame、10 Hz、RGB96×96、行動2次元。選択cameraは`observation.image`。
- 訓練／validation／test: 166／20／20episode。訓練17,429clip、固定validation 20clip。訓練の非終端行動のみから平均・標準偏差を計算した。
- manifest: `output/lerobot_validation/manifest.json`、SHA-256 `02e99a009044703559bdca9bd07671b7a491c287e5e788a3497b025bc394e88d`。
- フレーム番号・episode境界・timestampとFPSの整合性をprepareで確認し、ファイルのSHA-256を記録。学習時はサイズ・mtimeのみを照合した。

Dataset cardの埋め込み仕様は旧v2だが、上記固定revisionの実際のinfo.jsonはv3.0。現在使っている大規模SWM HDF5と同じ収録集合・FPS・カメラ条件だとは確認していない。保存形式が違うという理由だけで同一条件の比較にしない。

## 実モデルBTの100更新

隔離環境`output/lerobot_validation/venv`を`uv sync --locked --group libero --group lerobot`で構築。既存`.venv`と既存runを変更していない。Torch2.9.1+cu130、Transformers4.57.6を維持。LeRobotの依存制約に合わせdatasets4.8.5・huggingface-hub0.35.3等をlockへ記録した。

runは`output/lerobot_validation/bt_100_s3072/`。GB10、BF16、batch32、workers2、seed3072、warmup10、lr5e-5、frameskip5、history3、公式224画像前処理、既存E/A/FとBT v2。画像encoderのcompileは無効。100更新のループは約26秒で、初期化・データ取得・最終保存を含む総時間とは異なる。

| 確認 | 結果 |
|---|---|
| 訓練終了 | 終了コード0、`completed.json`はstep100・completed |
| loss | 100行すべて有限、最初1.14474475、最後0.53672284 |
| 最終訓練pred loss / SIGReg | 0.19776876 / 3.76615620 |
| validation | 10,20,…,100更新の10回 |
| 最終validation loss / pred loss | 1.10619402 / 0.51735449 |
| 保存 | step50／100の訓練checkpointとTなしobject export、最終last.ckpt |

最終推論exportは`step_100_object.ckpt`、SHA-256 `8e394f4820b1b9577523d309717cbb7d1558f4f7570e783a15ce2620fcab7519`。入力契約と訓練行動統計を保持する。開始前には4更新の接続確認も実施したが、上の結果は独立した100更新runのもの。

step50の訓練checkpointから同じ100更新予算へ再開し、`output/lerobot_validation/bt_100_s3072_resume/`へ保存。終了コード0・step100を確認し、連続実行と全state_dict、optimizer、scheduler、乱数状態が完全一致した。ソースhashも現在の実装と一致する。長期・別GPU・別依存での一致保証ではない。

## 保存モデルの推論

100更新exportをCPUへロードし、保持test episode由来の32clipへオフライン予測を実行。`output/lerobot_validation/inference_100/`にpredictions.pt・results.json・status.jsonを保存し、終了コード0・succeeded・32clipを照合。潜在MSEは0.54055673。正規化はcheckpointの訓練統計を使用した。先頭32clipの診断で、制御成功率や独立32episodeの試験ではない。

入力条件が等しいHDF5／LeRobotの自己生成fixtureでは、前処理後の画像・行動、訓練分割・統計が一致した。HDF5で短期学習したモデルをLeRobotへ、逆方向も適用し、訓練統計を使うことを確認。評価manifestの統計を意図的に変えても出力は変わらない。FPS・カメラ・単位・成分順・行動規約・domainが違う場合は拒否する。旧checkpointは確認済みの訓練契約の明示を必須にした。

## 回帰確認

```bash
TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas \
CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=.:lewm \
output/lerobot_validation/venv/bin/python -m pytest mylewm -q
```

**160合格・スキップ0、29.06秒、警告629件**。GPU・compile・既存公式object checkpoint読込を含む。ログは`output/lerobot_validation/pytest.log`。追加20件は公式LeRobot writer/reader、画像と行動の対応、train-only統計、worker0/2、timestamp等の改変拒否、欠落／追加ファイル拒否、7次元行動・frameskip2の実モデル入力、両形式での短期保存／再開・推論を検査した。

文書のbash 35ブロックの構文、相対リンク、新入口CLIの11実行例の引数解析、`uv lock --check`、`git diff --check`も合格。引数解析ではprepare／学習／推論を呼ばず、例の本学習は開始していない。

初回の実データprepareで、公式ローダーがshapeをtupleに正規化することを見落として次元照合が失敗した。tupleとして比較する修正後に完了。初回のfixture推論比較では、同じ値でもメモリ配置の違いによる最大2.4e-7の差があった。新契約付き入力を同じcontiguous配置に揃えて解消した。旧契約なしHDF5の配置は変更していない。旧環境でもLeRobotをインストールせずにHDF5 dry-runが通ることを確認した。

PyAV経路のTorchvision非推奨警告、Lightningの保存／再開警告等は残る。元の失敗ログは`new_tests.log`、修正後は`new_tests_retry.log`と全回帰ログに保持。長期収束、実機、複数カメラ融合、CEM環境成功率は今回の達成事項に含めない。
