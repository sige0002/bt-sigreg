# モデル利用：データ形式・推論・LeRobot Policy

LIBEROのタスクID付き行動模倣方策は[凍結ViT＋BC](BEHAVIOR_CLONING.ja.md)を参照してください。この文書のLeRobot Policyは世界モデル＋CEMを包む形式で、BC方策とは別です。

コードは`src/mylewm/`へ移動しました。環境準備でeditableパッケージを導入し、`python -m mylewm.…`で起動します。[構成・環境準備](../README.md#コードの構成と起動)。

学習の実行順は[学習手順](TRAINING.ja.md)、環境での成功率計測は[評価手順](EVALUATION.ja.md)を参照してください。この文書は入力条件の照合と学習済みモデルの利用を扱います。

| 目的 | 読む章 |
|---|---|
| HDF5／LeRobotの入力契約、形式間のオフライン推論 | [第1章 データ形式と推論](#data) |
| ckptを直接使う、LeRobot Policyへ変換・検証する | [第2章 Policy利用](#policy) |

[文書一覧へ戻る](README.md)。

<a id="data"></a>

## 第1章 データ形式・入力契約・オフライン推論

一つのrunは一つの形式だけを使います。`prepare_dataset.py`が形式・分割・入力条件・訓練統計・ファイル一覧・SHA-256をmanifestへ保存し、`train.py`が同じRaw／BTループへ接続します。従来のHDF5 manifestも従来の読み方で使えます。

学習後のckptを世界モデル＋CEMのLeRobot Policyへ変換する手順は[Policyとコンバーター](#policy)を参照してください。ckpt直接読込も可能です。

LeRobotは`lerobot==0.4.4`の公式`LeRobotDataset`とPyAVで直接読みます。中間HDF5への変換は行いません。学習時の画像は選択した1カメラだけで、状態・報酬・タスクIDはモデルに入力しません。多カメラ融合と既存LIBEROの2カメラモデルへの接続は今回の範囲外です。正規化の対象は物理行動であり、評価データの統計へ置き換えません。

<a id="data-入力条件は保存形式から独立"></a>

### 入力条件は保存形式から独立

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

<a id="data-契約付きhdf5で学習する例"></a>

### 契約付きHDF5で学習する例

次は**10 Hz・PushT絶対位置行動・同じ俯瞰カメラで収録したことを確認済みのHDF5**の例です。既存の`pusht_expert_train.h5`のFPSはHDF5属性だけでは確認できていないため、この契約をそのまま既存データへ付けないでください。任意のHDF5を自動解釈する入口ではなく、`pixels`、`action`、`ep_len`、`ep_offset`を持つ形式が対象です。

```bash
cd "$(git rev-parse --show-toplevel)"
export UV_PROJECT_ENVIRONMENT="$PWD/.venv-lerobot"
uv sync --locked --group lerobot --group libero
uv run --no-sync python -m mylewm.data.prepare_dataset \
  --format hdf5 \
  --dataset "/mnt/data/pusht_10hz/trajectories.h5" \
  --contract mylewm/configs/lerobot_pusht_input.json \
  --manifest "$PWD/output/manifests/pusht/hdf_10hz/manifest.json"

CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run --no-sync python -m mylewm.training.train \
  --manifest "$PWD/output/manifests/pusht/hdf_10hz/manifest.json" \
  --output "$PWD/output/pusht/hdf_10hz_bt_100" \
  --mode bt --steps 100 --warmup-steps 10 --batch-size 32 --workers 2 \
  --save-every 50 --val-every 10 --seed 3072 \
  --accelerator gpu --precision bf16-mixed --execute
```

<a id="data-hdf5学習モデルをlerobotデータに適用する例"></a>

### HDF5学習モデルをLeRobotデータに適用する例

上で学習したモデルへ、ローカルLeRobotデータを渡します。`--purpose inference`は訓練統計を作らず、全有効episodeを推論対象にします。同じデータ由来の訓練・評価episodeが重なる場合は汎化評価とは呼べません。学習済みmanifestを使う場合は`--split test`でその保持episodeを選べます。

```bash
uv run --no-sync python -m mylewm.data.prepare_dataset \
  --format lerobot \
  --dataset "/mnt/data/pusht_10hz/lerobot_v3" \
  --camera-key observation.image \
  --contract mylewm/configs/lerobot_pusht_input.json \
  --purpose inference \
  --manifest "$PWD/output/manifests/pusht/lerobot_inference/manifest.json"

uv run --no-sync python -m mylewm.policy.infer_trajectories \
  --checkpoint "$PWD/output/pusht/hdf_10hz_bt_100/step_100_object.ckpt" \
  --manifest "$PWD/output/manifests/pusht/lerobot_inference/manifest.json" \
  --split test --limit 32 --batch-size 8 --device cpu \
  --output "$PWD/output/pusht/hdf_to_lerobot_predictions" --execute
```

出力は`predictions.pt`（予測・教師の潜在表現、episode/start）、`results.json`（潜在MSE、重み・manifestのhash、データ確認）、`status.json`です。`state: succeeded`と終了コード0を確認します。`--execute`を外すとcheckpointをロードしないdry-runです。これは保存軌道上のオフライン予測で、画像生成・実機制御・CEM成功率評価ではありません。潜在MSEだけで方式間の性能を順位付けしません。

<a id="data-入力契約のない古いcheckpoint"></a>

### 入力契約のない古いcheckpoint

旧checkpoint・manifestは変更しません。訓練時のカメラ・FPS・行動仕様を確認し、同じJSON形式の別ファイルへ記録してから明示します。checkpoint内の訓練行動統計は必須です。既に契約を持つcheckpointの契約を上書きする用途には使えません。

```bash
uv run --no-sync python -m mylewm.policy.infer_trajectories \
  --checkpoint "/mnt/models/verified_hdf_training/step_100_object.ckpt" \
  --training-contract "/mnt/models/verified_hdf_training/input_contract.json" \
  --manifest "$PWD/output/manifests/pusht/lerobot_inference/manifest.json" \
  --split test --limit 32 --batch-size 8 --device cpu \
  --output "$PWD/output/pusht/legacy_hdf_to_lerobot_predictions" --execute
```

訓練条件が不明なら対応可能とは判定しません。また、上の推論利用と厳密な学習再開は別です。再開は開始時のコード・依存・manifest・設定を維持し、今回変更前のrunを新コードへ移しません。

<a id="data-参照した仕様"></a>

### 参照した仕様

- [LeRobot Dataset v3](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)
- [使用する公式ローダーv0.4.4](https://github.com/huggingface/lerobot/blob/v0.4.4/src/lerobot/datasets/lerobot_dataset.py)
- [取得版のPushTメタデータ](https://huggingface.co/datasets/lerobot/pusht/blob/7628202a2180972f291ba1bc6723834921e72c19/meta/info.json)
- [PushT環境の行動仕様](https://github.com/huggingface/gym-pusht/blob/main/gym_pusht/envs/pusht.py)

2026-09-10確認。Dataset cardの埋め込み例は旧v2を記載していますが、取得した実際の`meta/info.json`はv3.0です。Hubの全体統計・旧cardのパスを新manifestの根拠にしていません。

<a id="policy"></a>

## 第2章 ckpt読込・LeRobot Policy変換

世界モデル＋CEMの共通コントローラーを、既存ckptから直接、またはLeRobot形式の保存ディレクトリから使えます。コンバーターは別CLIです。実データの画像・行動・目標を使った比較まで確認しています。[実行記録](reports/POLICY_EXPORT.ja.md)。

現行対応は単一カメラの`lewm_tiny_v1`（192次元の既存LeWM E/A/F）、Raw／BTの推論用object ckpt、および新経路Lightningの訓練ckptです。BTのT・optimizerはPolicyへ含めません。LIBEROの2カメラモデルや任意のPyTorchモデルを自動変換するものではありません。

<a id="policy-変換する"></a>

### 変換する

以下は今回実際に学習したBT100更新checkpointから、LeRobotの保存形式へ変換する例です。専用環境を使い、使用中の学習環境には同期しません。

```bash
cd "$(git rev-parse --show-toplevel)"
export UV_PROJECT_ENVIRONMENT="$PWD/.venv-lerobot"
uv sync --locked --group lerobot --group libero
uv run --no-sync python -m mylewm.policy.export_lerobot_policy \
  --checkpoint "$PWD/output/lerobot_validation/bt_100_s3072/step_100_object.ckpt" \
  --planner-config mylewm/configs/pusht_cem.json \
  --camera-key observation.image \
  --output "$PWD/output/pusht/bt_100_lerobot_policy" --execute
```

出力先は新しいディレクトリが必要です。`--execute`を外すとcheckpointをロードしないdry-runです。入力は自分が管理する信頼済みcheckpointを使います。訓練用checkpointを直接入力する場合の完全な例：

```bash
uv run --no-sync python -m mylewm.policy.export_lerobot_policy \
  --checkpoint "$PWD/output/lerobot_validation/bt_100_s3072/last.ckpt" \
  --planner-config mylewm/configs/pusht_cem.json \
  --camera-key observation.image \
  --output "$PWD/output/pusht/bt_100_from_training_ckpt" --execute
```

保存されるもの：

| ファイル | 内容 |
|---|---|
| `model.safetensors` | 世界モデルと訓練時の行動平均・標準偏差 |
| `config.json` | Policy種別、モデル構成識別子、入力契約、カメラキー、入出力次元、CEM条件 |
| `policy_preprocessor.json` | LeRobotのbatch追加・device転送 |
| `policy_postprocessor.json` | 物理座標の出力をCPUへ戻す処理 |
| `export.json` | 元ckpt・出力ファイル・関連ソースのhash、依存版、照合結果、訓練統計の参照値 |

画像のImageNet正規化・224 resizeと、行動の訓練統計による正規化は共通コントローラー内で実施します。前後処理で重ねて正規化しません。コンバーターは全重みの厳密再読込だけでなく、元モデルと再構築モデルのencoder・predictor出力も照合します。変換は追加学習を行いません。

`pusht_cem.json`はPushTの絶対x/y座標0〜512を使う例です。horizon4はモデルの4遷移（frameskip5では物理20step）、samples32・elites8・iterations3・seed3072です。最終候補のうち実際に採点した最良候補の先頭5行動を順番に返します。この設定が全環境に適するとは検証していません。別ロボットの範囲・単位は対象に合わせて設定します。

入力契約を持たない旧ckptの場合は、確認済みの訓練条件を別ファイルで渡します。元ckptを書き換えず、既に保存された契約の変更にも使えません。

```bash
uv run --no-sync python -m mylewm.policy.export_lerobot_policy \
  --checkpoint "/mnt/models/verified_pusht/step_100_object.ckpt" \
  --training-contract "/mnt/models/verified_pusht/input_contract.json" \
  --planner-config mylewm/configs/pusht_cem.json \
  --camera-key observation.image \
  --output "$PWD/output/pusht/legacy_lerobot_policy" --execute
```

<a id="policy-実データで変換前後を確認する"></a>

### 実データで変換前後を確認する

ダウンロード・prepare済みのLeRobot PushTを使い、保持episode 173の実画像・過去行動・最終目標画像で確認します。11時刻の行動出力を比較し、途中の履歴更新と3回のCEM計画を含みます。

```bash
uv run --no-sync python -m mylewm.policy.check_policy_export \
  --checkpoint "$PWD/output/lerobot_validation/bt_100_s3072/step_100_object.ckpt" \
  --policy "$PWD/output/pusht/bt_100_lerobot_policy" \
  --manifest "$PWD/output/lerobot_validation/manifest.json" \
  --episode 173 --ticks 11 \
  --output "$PWD/output/pusht/policy_realdata_check" --execute
```

`results.json`の行動列と`status.json`の`succeeded`、終了コード0を確認します。比較にはLeRobot標準factoryから読み直した前後処理も含みます。失敗時は`failed`とエラーを保存します。履歴には記録された実行済み行動を使い、CEMが提案した行動を実機へ送ることはありません。従って、この確認は保存形式の互換性・実データ入出力の検証であり、提案行動による制御成功率ではありません。

<a id="policy-読込apiと制御への接続"></a>

### 読込APIと制御への接続

ckpt直接読込はLeRobotのインストールを要求しません。

```python
import json
from pathlib import Path
from mylewm.policy.runtime import load_checkpoint_controller

planner = json.loads(Path("mylewm/configs/pusht_cem.json").read_text())
controller = load_checkpoint_controller(
    "output/lerobot_validation/bt_100_s3072/step_100_object.ckpt",
    planner, device="cpu",
)
```

LeRobot形式はPolicyを登録してから読みます。保存ファイルに加えて、このリポジトリのPolicy実装が必要です。LeRobot 0.4.4のfactory経路にも登録できます。

```python
from mylewm.policy.lerobot import BTSIGRegPolicy
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

policy = BTSIGRegPolicy.from_pretrained("output/pusht/bt_100_lerobot_policy")
assert get_policy_class("bt_sigreg") is BTSIGRegPolicy
pre, post = make_pre_post_processors(
    policy.config, pretrained_path="output/pusht/bt_100_lerobot_policy"
)
```

制御開始時は`prime(images, past_actions, goal, timestamps)`で実際の初期履歴を渡します。

| 引数 | 形状・意味 |
|---|---|
| `images` | `(3,3,H,W)`、frameskip間隔で取得した実画像3枚 |
| `past_actions` | `(2,frameskip,action_dim)`、画像間で実行した物理行動 |
| `goal` | `(1,3,H,W)`、目標RGB画像 |
| `timestamps` | `(3,)`、画像に対応する秒単位の時刻 |

RGBはuint8または未正規化float[0,1]。初回`select_action`の画像・時刻はprimeの最後と一致させます。その後は契約のFPSで呼び、前回実際に実行した行動を毎回報告します。5行動を返し終えると、新画像と実行済み5行動で履歴を更新して再計画します。未初期化・時刻間隔の不一致・実行済み行動の欠落は拒否します。時刻照合の許容差は1e-4秒で、今回の記録データの周期を対象としています。実機の不規則なカメラ時刻を自動補間する機能はありません。

直接コントローラーは`select_action(image[None], timestamp, executed_action)`、LeRobot Policyは前処理後の辞書を受け取ります。辞書にはカメラキー、`observation.timestamp`（秒）、2回目以降は`observation.executed_action`（物理行動ベクトル）を入れます。LeRobotの標準前処理を通すため、追加情報も`observation.`名前空間を使います。

環境reset時は`reset()`し、再び実履歴をprimeします。目標変更もreset・primeで行います。Policyが返した指令と実際に実行された行動が違う場合も、後者を報告します。学習専用Tや評価データの再正規化は使いません。

`predict_action_chunk`は履歴管理を呼出元が行う場合のAPIです。`history`、`past_actions`、`goal`、`timestamps`を渡すと`(1,frameskip,action_dim)`を返します。`select_action`の履歴や行動queueは変更しません。

現時点ではLeRobot Policyとしての保存・factory読込・行動選択APIとオフライン実データ確認までです。stockの`lerobot-record --policy.path=...`だけで実機を動かす入口は追加していません。初期履歴の収集、目標指定、実行結果・時刻の受け渡し、ロボットI/Oは対象機体のアプリケーションから接続する必要があります。
