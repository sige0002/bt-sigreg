# HDF5／LeRobot v3の学習・推論

一つのrunは一つの形式だけを使います。`prepare_dataset.py`が形式・分割・入力条件・訓練統計・ファイル一覧・SHA-256をmanifestへ保存し、`train.py`が同じRaw／BTループへ接続します。従来のHDF5 manifestも従来の読み方で使えます。

LeRobotは`lerobot==0.4.4`の公式`LeRobotDataset`とPyAVで直接読みます。中間HDF5への変換は行いません。学習時の画像は選択した1カメラだけで、状態・報酬・タスクIDはモデルに入力しません。多カメラ融合と既存LIBEROの2カメラモデルへの接続は今回の範囲外です。正規化の対象は物理行動であり、評価データの統計へ置き換えません。

## 入力条件は保存形式から独立

`input_contract`には次を記録します。

| 項目 | 意味 |
|---|---|
| `domain` | 対象環境／ロボットとダイナミクスの識別名 |
| `camera` | 物理的なカメラの対応。ファイルの列名とは独立 |
| `action_names` / `action_units` | 行動の成分順・単位 |
| `action_convention` | 絶対／相対、座標系、制御コマンドの意味 |
| `fps` / `frameskip` / `history` | 元の時間間隔・行動の束ね方・履歴長（現行3） |
| `preprocessing` | 共通のImageNet正規化・224 resize（`lewm_imagenet_resize224_v1`） |

新規契約付きHDF5とLeRobotは連続20ステップ（frameskip=5の場合）から4画像・4行動chunkを作り、先頭3chunkで一段先を予測します。エピソード境界のpaddingを訓練サンプルにしません。prepareはLeRobotのフレーム番号・episode ID・timestampの整合性を確認します。HDF5は`ep_len`／`ep_offset`の連続性、RGB uint8画像、行動次元を確認します。HDF5に時間情報が無い場合、FPSは収録仕様から人が明示する必要があります。

保存形式をまたぐ推論では上の条件の一致を確認します。カメラの列名は`pixels`と`observation.image`のように違って構いません。契約の同じ名前だけで、実物のカメラ姿勢や行動座標まで自動的に保証できるわけではありません。対応は収録側の仕様から確認してください。FPS変更、行動の単位変換・並べ替え・リサンプリングを暗黙には行いません。

訓練正規化はエピソード分割後の訓練部分だけから、最終行動を除いて計算します。定数の行動成分は標準偏差1として記録します。無効な非終端行動は拒否します。新しい分割は約80/10/10で、最低1episodeずつをvalidation/testへ確保します。従来manifestの固定分割は変更しません。

## 契約付きHDF5で学習する例

次は**10 Hz・PushT絶対位置行動・同じ俯瞰カメラで収録したことを確認済みのHDF5**の例です。既存の`pusht_expert_train.h5`のFPSはHDF5属性だけでは確認できていないため、この契約をそのまま既存データへ付けないでください。任意のHDF5を自動解釈する入口ではなく、`pixels`、`action`、`ep_len`、`ep_offset`を持つ形式が対象です。

```bash
cd "/home/sadasue/bt-sigreg"
export UV_PROJECT_ENVIRONMENT="$PWD/.venv-lerobot"
uv sync --locked --group lerobot --group libero
uv run --no-sync python -m mylewm.prepare_dataset \
  --format hdf5 \
  --dataset "/mnt/data/pusht_10hz/trajectories.h5" \
  --contract mylewm/configs/lerobot_pusht_input.json \
  --manifest "$PWD/output/manifests/pusht/hdf_10hz/manifest.json"

CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run --no-sync python -m mylewm.train \
  --manifest "$PWD/output/manifests/pusht/hdf_10hz/manifest.json" \
  --output "$PWD/output/pusht/hdf_10hz_bt_100" \
  --mode bt --steps 100 --warmup-steps 10 --batch-size 32 --workers 2 \
  --save-every 50 --val-every 10 --seed 3072 \
  --accelerator gpu --precision bf16-mixed --execute
```

## HDF5学習モデルをLeRobotデータに適用する例

上で学習したモデルへ、ローカルLeRobotデータを渡します。`--purpose inference`は訓練統計を作らず、全有効episodeを推論対象にします。同じデータ由来の訓練・評価episodeが重なる場合は汎化評価とは呼べません。学習済みmanifestを使う場合は`--split test`でその保持episodeを選べます。

```bash
uv run --no-sync python -m mylewm.prepare_dataset \
  --format lerobot \
  --dataset "/mnt/data/pusht_10hz/lerobot_v3" \
  --camera-key observation.image \
  --contract mylewm/configs/lerobot_pusht_input.json \
  --purpose inference \
  --manifest "$PWD/output/manifests/pusht/lerobot_inference/manifest.json"

uv run --no-sync python -m mylewm.tools.infer_trajectories \
  --checkpoint "$PWD/output/pusht/hdf_10hz_bt_100/step_100_object.ckpt" \
  --manifest "$PWD/output/manifests/pusht/lerobot_inference/manifest.json" \
  --split test --limit 32 --batch-size 8 --device cpu \
  --output "$PWD/output/pusht/hdf_to_lerobot_predictions" --execute
```

出力は`predictions.pt`（予測・教師の潜在表現、episode/start）、`results.json`（潜在MSE、重み・manifestのhash、データ確認）、`status.json`です。`state: succeeded`と終了コード0を確認します。`--execute`を外すとcheckpointをロードしないdry-runです。これは保存軌道上のオフライン予測で、画像生成・実機制御・CEM成功率評価ではありません。潜在MSEだけで方式間の性能を順位付けしません。

## 入力契約のない古いcheckpoint

旧checkpoint・manifestは変更しません。訓練時のカメラ・FPS・行動仕様を確認し、同じJSON形式の別ファイルへ記録してから明示します。checkpoint内の訓練行動統計は必須です。既に契約を持つcheckpointの契約を上書きする用途には使えません。

```bash
uv run --no-sync python -m mylewm.tools.infer_trajectories \
  --checkpoint "/mnt/models/verified_hdf_training/step_100_object.ckpt" \
  --training-contract "/mnt/models/verified_hdf_training/input_contract.json" \
  --manifest "$PWD/output/manifests/pusht/lerobot_inference/manifest.json" \
  --split test --limit 32 --batch-size 8 --device cpu \
  --output "$PWD/output/pusht/legacy_hdf_to_lerobot_predictions" --execute
```

訓練条件が不明なら対応可能とは判定しません。また、上の推論利用と厳密な学習再開は別です。再開は開始時のコード・依存・manifest・設定を維持し、今回変更前のrunを新コードへ移しません。

## 参照した仕様

- [LeRobot Dataset v3](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)
- [使用する公式ローダーv0.4.4](https://github.com/huggingface/lerobot/blob/v0.4.4/src/lerobot/datasets/lerobot_dataset.py)
- [取得版のPushTメタデータ](https://huggingface.co/datasets/lerobot/pusht/blob/7628202a2180972f291ba1bc6723834921e72c19/meta/info.json)
- [PushT環境の行動仕様](https://github.com/huggingface/gym-pusht/blob/main/gym_pusht/envs/pusht.py)

2026-09-10確認。Dataset cardの埋め込み例は旧v2を記載していますが、取得した実際の`meta/info.json`はv3.0です。Hubの全体統計・旧cardのパスを新manifestの根拠にしていません。
