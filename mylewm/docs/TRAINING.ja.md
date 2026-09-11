# LeRobot／HDF5：初心者向け学習・評価手順

学習済みLIBERO世界モデルにタスクID付き行動模倣方策を追加する場合は[凍結ViT＋BCの学習・評価](BEHAVIOR_CLONING.ja.md)を参照してください。世界モデル学習とは別工程で、既定dry-runです。

コードは`src/mylewm/`へ移動しました。環境準備でeditableパッケージを導入し、`python -m mylewm.…`で起動します。[構成・環境準備](../README.md#コードの構成と起動)。

使用するデータ形式の章を選び、その中で「準備 → 学習 → ログ・再開 → 評価」の順に進んでください。HDF5はPushTとLIBERO-10で入口が異なるため、章を分けています。

| データ | 学習手順 | 評価手順 |
|---|---|---|
| LeRobot Dataset v3（単一カメラ） | [第1章](#lerobot-training) | [保存軌道での予測確認・Policy確認](#lerobot-evaluation) |
| HDF5（PushT） | [第2章](#hdf5-pusht-training) | [PushT環境での成功率評価](#hdf5-pusht-evaluation) |
| HDF5（LIBERO-10） | [第3章](#hdf5-libero-training) | [LIBERO環境での成功率評価](#hdf5-libero-evaluation) |

学習中のvalidationは予測損失などの確認です。環境を動かして測る制御成功率とは別です。LeRobot入力からの環境成功率評価は未対応です。

PushT／LIBEROでは、保存済みの途中checkpointも評価できます。[途中checkpoint評価手順](EVALUATION.ja.md#intermediate)のGPU指定・実行・結果確認に従ってください。PushTは共有RAMに余裕があるGB10で同一GPUの学習と併走できます。途中評価に`completed.json`は不要です。

一つのrunにHDF5とLeRobotを混ぜません。以下のコマンド例は必要なものだけを選び、学習を重複起動しないでください。`train.py`は既定dry-runで、`--execute`で学習を開始します。LIBEROの`train_libero.py train`はそのまま学習を開始します。

### このPCのデータ場所を確認する（2026-09-11確認）

現在のリポジトリは`/home/sadasue/bt-sigreg`、準備済みのPython環境はその直下の`.venv`です。取得済みデータとmanifestは以下にあります。パスはリポジトリ直下を基準にしています。

| データ | 実データの場所 | 学習へ渡すmanifest |
|---|---|---|
| PushT HDF5（約46.3 GB） | `.cache/stable-wm/datasets/pusht_expert_train.h5` | `output/manifests/pusht/manifest.json` |
| LIBERO-10（HDF5 10個） | `.cache/libero-datasets/libero_10/` | `output/manifests/libero10/manifest.json` |

**場所の確認だけなら、次のブロックをそのまま実行してください。** 前の端末の環境変数に依存せず、manifestが指すファイルの存在・サイズ・更新時刻を確認します。データの全量hash計算、再取得、prepare、依存同期、学習は行いません。

```bash
cd /home/sadasue/bt-sigreg
.venv/bin/python - <<'PYCODE'
import json
from pathlib import Path
from mylewm.data.verification import verify_training_data

for name in ("pusht", "libero10"):
    path = Path("output/manifests") / name / "manifest.json"
    manifest = json.loads(path.read_text())
    verify_training_data(manifest, full=False)
    print(f"{name}: OK — manifest={path.resolve()}")
    if "files" in manifest:
        print("  ファイル一覧:", manifest["dataset"])
        print("  HDF5数:", len(manifest["files"]))
        for entry in manifest["files"]:
            print(" ", entry["path"])
    else:
        print("  HDF5:", manifest["dataset"])
PYCODE
```

以前のLIBERO確認例にあった`/mnt/data/bt-sigreg-data/libero_10`は、このPCにはありません。`/mnt/data/robot datasets/...`も外部ストレージの書き方の例で、取得済みデータの場所ではありません。新しい端末で`PUSHT_MANIFEST`等が未設定だと、旧確認コマンドは空のパスを読み、`IsADirectoryError: '.'`等になります。まず上の確認を使い、学習コマンドを使う場合は該当章の変数設定から実行してください。

<a id="lerobot-training"></a>
<a id="lerobot-v3で学習する"></a>

## 第1章 LeRobot v3の学習・評価

学習後のckptは直接CEMコントローラーへ読み込むか、別CLIでLeRobot Policyへ変換できます。[変換・config・実データ確認の手順](MODEL_USAGE.ja.md#policy)。

HDF5とLeRobot v3は同じ`src/mylewm/training/train.py`を使い、manifestの`data_format`でローダーを選びます。一つのrunに両形式を混ぜません。以下は公式`lerobot/pusht`を取得し、BTを100更新する完全な例です。HDF5の手順は[第2章](#hdf5-pusht-training)を参照してください。入力契約付きの新HDF5 prepareと、HDF5で学習したcheckpointをLeRobotデータへ適用する例は[データ形式と推論](MODEL_USAGE.ja.md#data)を参照してください。

### 1.1 環境を用意する

このPCでは準備済みの標準環境`.venv`を選び、依存版を確認します。

```bash
cd "$(git rev-parse --show-toplevel)"
export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
uv run --no-sync python -c 'import importlib.metadata as m; print({n: m.version(n) for n in ["lerobot", "torch", "transformers", "av"]})'
```

環境を初めて構築する場合だけ、リポジトリ直下で`uv sync --locked --group lerobot --group libero`を実行します。学習・評価で使用中の`.venv`では実行しません。

### 1.2 データを取得・指定し、manifestを作る

Hubのコミットを固定して取得し、新規manifestを作成します。`--download-root`はまだ存在しないディレクトリを指定します。保存先に空白がある場合も引用符で囲みます。取得後の学習・推論はローカルファイルだけを使います。

```bash
export LEROBOT_DATASET="$PWD/output/datasets/lerobot/pusht_v3"
export LEROBOT_MANIFEST="$PWD/output/manifests/pusht/lerobot_v3/manifest.json"
uv run --no-sync python -m mylewm.data.prepare_dataset \
  --format lerobot \
  --repo-id lerobot/pusht \
  --revision 7628202a2180972f291ba1bc6723834921e72c19 \
  --download-root "$LEROBOT_DATASET" \
  --camera-key observation.image \
  --contract mylewm/configs/lerobot_pusht_input.json \
  --manifest "$LEROBOT_MANIFEST"
```

この契約JSONはPushTの絶対位置行動x/y・10 Hz・俯瞰RGB用です。別ロボットには流用せず、その行動の意味・単位・カメラ・FPSを確認して専用の契約を用意します。メタデータのFPS・行動次元と契約が異なる場合はprepareを拒否します。Hubの全体統計は使わず、分割後の訓練エピソードから平均・標準偏差を計算します。

既にローカルへ保存してあるLeRobot v3を使う場合は、上の取得コマンドの代わりに次の例を使います。以後の学習もこのmanifestを参照するよう、変数を設定します。

```bash
export LEROBOT_MANIFEST="$PWD/output/manifests/pusht/lerobot_local/manifest.json"
uv run --no-sync python -m mylewm.data.prepare_dataset \
  --format lerobot \
  --dataset "/mnt/data/robot datasets/pusht_v3" \
  --camera-key observation.image \
  --contract mylewm/configs/lerobot_pusht_input.json \
  --manifest "$LEROBOT_MANIFEST"
```

### 1.3 dry-runで学習設定を確認する

Hub取得のmanifestで、学習設定だけを確認するdry-run：

```bash
uv run --no-sync python -m mylewm.training.train \
  --manifest "$LEROBOT_MANIFEST" \
  --output "$PWD/output/pusht/lerobot_bt_100_s3072" \
  --mode bt --steps 100 --warmup-steps 10 --batch-size 32 --workers 2 \
  --save-every 50 --val-every 10 --seed 3072 \
  --accelerator gpu --precision bf16-mixed
```

### 1.4 100更新の短期学習を実行する

BTを100更新する実行例。画像encoder・行動encoder・予測器・BTを実際に学習します。検証は10更新ごと、checkpointは50更新ごとです。100更新は動作確認の予算で、長期収束・制御性能の評価ではありません。

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run --no-sync python -m mylewm.training.train \
  --manifest "$LEROBOT_MANIFEST" \
  --output "$PWD/output/pusht/lerobot_bt_100_s3072" \
  --mode bt --steps 100 --warmup-steps 10 --batch-size 32 --workers 2 \
  --save-every 50 --val-every 10 --seed 3072 \
  --accelerator gpu --precision bf16-mixed --execute
```

### 1.5 100,000更新の本学習を実行する

100,000更新の新規本学習を選ぶ場合の例。上の100更新から予算だけを変えてresumeすることはできません。必要なrunだけを開始し、他の学習と重複させないでください。

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run --no-sync python -m mylewm.training.train \
  --manifest "$LEROBOT_MANIFEST" \
  --output "$PWD/output/pusht/lerobot_bt_100k_s3072" \
  --mode bt --steps 100000 --warmup-steps 500 --batch-size 32 --workers 2 \
  --save-every 5000 --val-every 500 --seed 3072 \
  --accelerator gpu --precision bf16-mixed --execute
```

### 1.6 中断した学習を再開する

途中で停止した100更新runを50更新の訓練checkpointから再開する例。開始時と同じコード・依存・manifest・条件を使い、出力先だけを新しくします。`*_object.ckpt`は推論用なのでresumeには使いません。

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run --no-sync python -m mylewm.training.train \
  --manifest "$LEROBOT_MANIFEST" \
  --output "$PWD/output/pusht/lerobot_bt_100_s3072_resume1" \
  --mode bt --steps 100 --warmup-steps 10 --batch-size 32 --workers 2 \
  --save-every 50 --val-every 10 --seed 3072 \
  --accelerator gpu --precision bf16-mixed \
  --resume "$PWD/output/pusht/lerobot_bt_100_s3072/step_50.ckpt" --execute
```

### 1.7 ログと完了を確認する

実行後は`completed.json`の`step: 100`と終了コードを確認します。ログは`metrics/version_0/metrics.csv`、推論用モデルは`step_100_object.ckpt`、最終訓練状態は`last.ckpt`です。学習時の入力契約と行動統計も保存されます。今回実行済みの保存先・検証結果は[LeRobot対応レポート](reports/LEROBOT_V3.ja.md)に記録しています。

<a id="lerobot-evaluation"></a>

### 1.8 評価：保持データでオフライン予測を確認する

1.4の100更新モデルを、同じmanifestの学習に使わなかったtest episodeで確認する例です。1.5の本学習モデルならcheckpointと出力先を対応するものへ変更します。

```bash
uv run --no-sync python -m mylewm.policy.infer_trajectories \
  --checkpoint "$PWD/output/pusht/lerobot_bt_100_s3072/step_100_object.ckpt" \
  --manifest "$LEROBOT_MANIFEST" \
  --split test --limit 32 --batch-size 8 --device cpu \
  --output "$PWD/output/pusht/lerobot_bt_100_test_predictions"
```

まず上のdry-runを確認し、実行するときは同じコマンドの末尾に`--execute`を追加します。終了コード0、`status.json`の`state: succeeded`、`results.json`、`predictions.pt`を確認してください。これは保存軌道の潜在予測を調べるもので、潜在MSEだけでは制御性能を判断できません。

### 1.9 Policyの動作確認と環境評価の対応範囲

ckpt直接読込とLeRobot Policy形式のCEM行動一致は、[Policy変換・実データ確認手順](MODEL_USAGE.ja.md#policy)で確認できます。実機I/O・stockの`lerobot-record`起動・環境成功率評価への接続は未対応です。第2章のHDF5用評価コマンドへLeRobot manifestを渡して評価することはできません。

保存形式をまたいでモデルを使う場合は、[入力契約と形式間の推論](MODEL_USAGE.ja.md#data)でカメラ・行動・FPSなどの一致を確認してください。

<a id="hdf5-pusht-training"></a>

## 第2章 HDF5（PushT）の学習・評価

この章は既存PushT HDF5と従来manifestを使う `pusht_spt_v1` の手順です。入力契約付きHDF5を新規prepareする場合は[契約付きHDF5の学習例](MODEL_USAGE.ja.md#data-契約付きhdf5で学習する例)を使います。既存データへ未確認のFPSなどを付与しないでください。

<a id="新しいpusht経路公式ライブラリへ委託2026-09-09"></a>

### 0. 実行フォルダと固定依存の環境を用意する

以下はこのPC（GB10／CUDA 13）の具体例です。すべてリポジトリ直下で実行します。標準環境はリポジトリ直下の`.venv`です。固定依存（Transformers 4.57.6）と現在のsrcパッケージを導入します。旧`.venv-training`・`.venv-refactor`は使用しません。学習・評価で使用中の環境には同期しないでください。

```bash
cd "$(git rev-parse --show-toplevel)"
export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
uv run --no-sync python -c 'import torch, transformers; print("torch:", torch.__version__); print("transformers:", transformers.__version__)'
ls -l "/usr/local/cuda-13.0/bin/ptxas"
nvidia-smi
```

以後の例も同じ端末で実行します。新しい端末やtmuxへ移った場合は、同じ`cd`と`export UV_PROJECT_ENVIRONMENT`、次節のデータ／manifestの`export`を設定し直してください。同期済み環境を使う各コマンドは`uv run --no-sync`としており、起動のたびに依存を変更しません。ここにあるコード例をまとめて実行する必要はありません。短期確認・本学習・再開は、それぞれ目的のrunだけを実行します。

上は準備済み環境の確認です。初回構築が必要な場合だけ、学習・評価が動いていない状態で`uv sync --locked --group libero --group lerobot`を実行します。

#### PushT HDF5が未取得の場合

圧縮済みで約13.1GB、展開後は約46.3GBです。保存先と空き容量を確認してください。取得済みなら次節へ進みます。

```bash
export BT_SIGREG_DATA_ROOT="$PWD/.cache/stable-wm"
export PUSHT_HDF5="$BT_SIGREG_DATA_ROOT/datasets/pusht_expert_train.h5"
mkdir -p "$(dirname "$PUSHT_HDF5")"
uv run --no-sync hf download quentinll/lewm-pusht --repo-type dataset \
  --local-dir "$BT_SIGREG_DATA_ROOT/pusht-source"
zstd -d --keep "$BT_SIGREG_DATA_ROOT/pusht-source/pusht_expert_train.h5.zst" \
  -o "$PUSHT_HDF5"
```

`zstd: command not found`ならOS側のZstandardコマンドを導入してから展開行を再実行します。次節では取得した`PUSHT_HDF5`を使い、新規manifestの保存先を指定してください。

### 1. データセットのパスを指定する

**PushTでは、prepareコマンドの`--dataset`にHDF5ファイルのパスを渡します。学習コマンドは`--manifest`を受け取り、そのJSON内の`dataset`に保存された絶対パスから読み込みます。`src/mylewm/training/train.py`に`--dataset`を渡す形式ではありません。**

このPCにある既存PushTデータと既存manifestを使う例：

```bash
cd /home/sadasue/bt-sigreg
export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
export PUSHT_HDF5="$PWD/.cache/stable-wm/datasets/pusht_expert_train.h5"
export PUSHT_MANIFEST="$PWD/output/manifests/pusht/manifest.json"
ls -lh -- "$PUSHT_HDF5" "$PUSHT_MANIFEST"
```

別ストレージで新規実験を始める場合だけ、`PUSHT_HDF5`を実在するHDF5のパスへ、`PUSHT_MANIFEST`を新しいmanifestの保存先へ変更します。空白を含むパスは引用符で囲み、ファイル名まで指定してください。例えば`/mnt/data/robot datasets/pusht_expert_train.h5`は書き方の例であり、このPCにはありません。相対パスの`./`はリポジトリ直下が基準で、prepareが絶対パスに変換して記録します。データ未取得の場合は、前節の「PushT HDF5が未取得の場合」を先に実施します。

### 2. manifestを作成し、記録されたデータの場所を確認する

新規manifestを作る完全な例です。既存manifestがある場合は作り直さず、その下の確認コマンドへ進みます。prepareは分割・学習側の正規化統計・絶対パス・サイズ・更新時刻を記録し、このときだけ全データのSHA-256を一度計算します。データ本体を複製・生成する処理ではありません。

```bash
uv run --no-sync python -m mylewm.training.loop prepare \
  --dataset "$PUSHT_HDF5" \
  --manifest "$PUSHT_MANIFEST"
```

既存・新規のどちらも、学習前に記録された場所を確認します。下のコマンドはJSONとパスを確認するだけで、全量データhashは走査しません。

```bash
uv run --no-sync python - "${PUSHT_MANIFEST:?先に節1のPUSHT_MANIFESTを設定してください}" "${PUSHT_HDF5:?先に節1のPUSHT_HDF5を設定してください}" <<'PYCODE'
import json
import sys
from pathlib import Path
manifest_path = Path(sys.argv[1])
manifest = json.loads(manifest_path.read_text())
dataset = Path(manifest["dataset"])
print("manifest:", manifest_path)
print("dataset:", dataset)
assert dataset.is_file(), "manifestに記録されたデータが見つかりません"
assert dataset.resolve() == Path(sys.argv[2]).resolve(), "指定したデータとmanifestのパスが異なります"
PYCODE
```

`PUSHT_HDF5`の変数だけを変更しても、既存manifestの読込先は変わりません。場所が違う場合、移転後の新規実験には別名の新しいmanifestを作ります。過去runを再開する目的で、元のmanifestや保存済みconfigを書き換えないでください。

### 3. 学習を開始せず、設定を確認する

BT本学習の設定確認例です。validationは500更新ごと、保存は5,000更新ごと、画像エンコーダのコンパイルを有効にしています。`--execute`がないため、学習・GPU初期化・runフォルダの作成は行いません。

```bash
TRITON_PTXAS_PATH="/usr/local/cuda-13.0/bin/ptxas" \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
uv run --no-sync python -m mylewm.training.train \
  --mode bt \
  --manifest "$PUSHT_MANIFEST" \
  --output "output/pusht/bt_compiled_100k_s3072" \
  --steps 100000 --warmup-steps 500 \
  --batch-size 128 --workers 4 --seed 3072 \
  --lr 5e-5 --accelerator gpu --precision bf16-mixed \
  --val-every 500 --save-every 5000 \
  --pin-memory --compile-encoder
```

出力の`val_every`が500、`save_every`が5000、`compile_encoder`がtrueであることを確認します。dry-runは設定・ソース識別の確認までで、学習動作や全データの内容まで検証するものではありません。

### 4. 100更新の短期確認を実行する

本学習とは別の新規runとして、batch16・warmup10・validation10更新ごと・保存50更新ごとに100更新だけ実行する例です。以下のBTとRawは別々のrunです。同時に起動しません。出力先は未使用の名前にし、runフォルダ自体を先にmkdirしないでください。

BTの短期確認：

```bash
TRITON_PTXAS_PATH="/usr/local/cuda-13.0/bin/ptxas" \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
uv run --no-sync python -m mylewm.training.train \
  --mode bt \
  --manifest "$PUSHT_MANIFEST" \
  --output "output/pusht/bt_compiled_smoke_s3072" \
  --steps 100 --warmup-steps 10 \
  --batch-size 16 --workers 0 --seed 3072 \
  --lr 5e-5 --accelerator gpu --precision bf16-mixed \
  --val-every 10 --save-every 50 \
  --pin-memory --compile-encoder --execute
```

Rawの短期確認：

```bash
TRITON_PTXAS_PATH="/usr/local/cuda-13.0/bin/ptxas" \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
uv run --no-sync python -m mylewm.training.train \
  --mode raw \
  --manifest "$PUSHT_MANIFEST" \
  --output "output/pusht/raw_compiled_smoke_s3072" \
  --steps 100 --warmup-steps 10 \
  --batch-size 16 --workers 0 --seed 3072 \
  --lr 5e-5 --accelerator gpu --precision bf16-mixed \
  --val-every 10 --save-every 50 \
  --pin-memory --compile-encoder --execute
```

正常終了後、選んだrunの`completed.json`と`step_100_object.ckpt`を確認します。短期確認のcheckpointを10万更新へ延長する用途のresumeはできません。次の本学習は新規初期値から開始します。

### 5. 100,000更新の本学習を実行する

短期確認が終わり、他の学習が動いていない状態で、目的の方式の例を実行します。端末切断に備える場合はtmuxを使用し、新しい端末内でも節0・1のフォルダと環境変数を設定してください。両方式ともbatch128・warmup500・seed3072・validation500更新ごと・保存5,000更新ごとです。

BTの本学習：

```bash
TRITON_PTXAS_PATH="/usr/local/cuda-13.0/bin/ptxas" \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
uv run --no-sync python -m mylewm.training.train \
  --mode bt \
  --manifest "$PUSHT_MANIFEST" \
  --output "output/pusht/bt_compiled_100k_s3072" \
  --steps 100000 --warmup-steps 500 \
  --batch-size 128 --workers 4 --seed 3072 \
  --lr 5e-5 --accelerator gpu --precision bf16-mixed \
  --val-every 500 --save-every 5000 \
  --pin-memory --compile-encoder --execute
```

Rawの本学習：

```bash
TRITON_PTXAS_PATH="/usr/local/cuda-13.0/bin/ptxas" \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
uv run --no-sync python -m mylewm.training.train \
  --mode raw \
  --manifest "$PUSHT_MANIFEST" \
  --output "output/pusht/raw_compiled_100k_s3072" \
  --steps 100000 --warmup-steps 500 \
  --batch-size 128 --workers 4 --seed 3072 \
  --lr 5e-5 --accelerator gpu --precision bf16-mixed \
  --val-every 500 --save-every 5000 \
  --pin-memory --compile-encoder --execute
```

`3072`はこの例で指定する学習seedです。Raw／BTの比較では、manifest・seed・batch・更新数・コンパイル・validation間隔等を揃え、出力configの初期モデルhashも照合します。

このPCのコンパイルには`/usr/local/cuda-13.0/bin/ptxas`を使います。同梱TritonのCUDA 12.8コンパイラはGB10に未対応です。他のPCへこのパスや速度の結果をそのまま一般化しないでください。固定依存のTransformers 4.57.6とGB10では通常stepが約0.79〜0.80秒から約0.53〜0.54秒へ短縮しましたが、初回の学習・validationにはコンパイル待ちがあります。推論exportにはコンパイルを含めません。演算融合による丸め差があるため、コンパイルなしとの学習軌跡はビット一致しません。[計測・検証記録](reports/TRAINING_SPEED_20260910.ja.md)。

<a id="ログ保存再開"></a>

### 6. ログ・保存・再開

上の本学習例はvalidation500更新ごと、checkpoint保存5,000更新ごとです。`--val-every`と`--save-every`はそれぞれ正の整数で指定します。CLIで`--val-every`を省略した場合だけ、従来どおり保存間隔と同じになります。validationを頻繁にすると、その計算時間は増えます。

- `metrics/version_N/metrics.csv`：`update`は1始まりの更新番号、`lr_used`はその更新に使った学習率、`fit/loss`・`fit/pred_loss`・`fit/sigreg_loss`が学習損失。CSVの標準`step`は0始まりです。
- `step_N.ckpt`／`last.ckpt`：Lightning形式のモデル・T・optimizer・scheduler・乱数・再開条件。上の本学習例では5,000更新ごとと最終時点に保存します。
- `step_N_object.ckpt`：Tなし・非コンパイルの推論専用モデル。再開用ではありません。
- `completed.json`：訓練ループの正常完了後に生成します。

本学習のCSVを別端末から確認する例：

```bash
cd "$(git rev-parse --show-toplevel)"
tail -f "output/pusht/bt_compiled_100k_s3072/metrics/version_0/metrics.csv"
```

```bash
cd "$(git rev-parse --show-toplevel)"
tail -f "output/pusht/raw_compiled_100k_s3072/metrics/version_0/metrics.csv"
```

この`tail`のCtrl-Cは表示だけを終了します。学習画面のCtrl-Cは学習そのものを中断します。旧JSONL用`monitor_training.sh`は新PushTのCSV表示には使いません。

以下は**この文書の本学習例で新規開始し、その後中断したrun**を再開する完全な例です。まず元の学習プロセスが終了していることと、指定した`last.ckpt`が保存済みであることを確認します。開始時と同じソース・固定依存環境・manifest・総更新数・保存間隔・validation間隔・コンパイラを使います。新しい端末でも節0・1の環境変数を同じ値で設定してください。出力は元runを上書きせず、新規フォルダへ分離します。

BT本学習の再開：

```bash
TRITON_PTXAS_PATH="/usr/local/cuda-13.0/bin/ptxas" \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
uv run --no-sync python -m mylewm.training.train \
  --mode bt \
  --manifest "$PUSHT_MANIFEST" \
  --output "output/pusht/bt_compiled_100k_s3072_resume1" \
  --steps 100000 --warmup-steps 500 \
  --batch-size 128 --workers 4 --seed 3072 \
  --lr 5e-5 --accelerator gpu --precision bf16-mixed \
  --val-every 500 --save-every 5000 \
  --pin-memory --compile-encoder \
  --resume "output/pusht/bt_compiled_100k_s3072/last.ckpt" --execute
```

Raw本学習の再開：

```bash
TRITON_PTXAS_PATH="/usr/local/cuda-13.0/bin/ptxas" \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
uv run --no-sync python -m mylewm.training.train \
  --mode raw \
  --manifest "$PUSHT_MANIFEST" \
  --output "output/pusht/raw_compiled_100k_s3072_resume1" \
  --steps 100000 --warmup-steps 500 \
  --batch-size 128 --workers 4 --seed 3072 \
  --lr 5e-5 --accelerator gpu --precision bf16-mixed \
  --val-every 500 --save-every 5000 \
  --pin-memory --compile-encoder \
  --resume "output/pusht/raw_compiled_100k_s3072/last.ckpt" --execute
```

`--steps 100000`は残り更新数ではなく開始時に決めた総更新数です。保存以前の未確定分はやり直します。既存の過去100k BTや停止済みRaw 70kなど、**変更前コードで始めたrunへこの再開例を適用しません**。開始時のGit版・環境を使い、照合を解除しないでください。旧`resume.pt`と新Lightning checkpointも互換ではありません。

完走時は`completed.json`の`state=completed`と`step=100000`、`step_100000_object.ckpt`を確認します。成功率評価は別工程です。途中checkpointの評価では`completed.json`を必須とせず、[途中評価手順](EVALUATION.ja.md#intermediate)に従います。[PushT評価手順](EVALUATION.ja.md#pusht)も参照してください。

<a id="hdf5-pusht-evaluation"></a>

### 7. 評価：PushT環境で成功率を測る

本章のBT本学習が正常終了し、`step_100000_object.ckpt`が保存済みの場合のdry-runです。準備した環境とデータ変数を同じ端末で使います。

```bash
UV_NO_SYNC=1 bash scripts/evaluate_pusht.sh \
  --checkpoint output/pusht/bt_compiled_100k_s3072/step_100000_object.ckpt \
  --manifest "$PUSHT_MANIFEST" --dataset "$PUSHT_HDF5" \
  --output output/pusht/eval_bt_compiled_100k_confirm50 \
  --num-eval 50 --seed 42
```

内容を確認した後、同じコマンドの末尾に`--execute`を追加します。Rawならcheckpointと出力先をRawのrunに替え、manifest・ケース数・seed・CEM条件を揃えます。GB10固有のcache対処、結果ファイル、終了コードと`status.json`の照合は[PushT評価手順](EVALUATION.ja.md#pusht)を参照してください。途中checkpointは[途中評価手順](EVALUATION.ja.md#intermediate-pusht)を使います。

入力契約付きHDF5の保存軌道上の予測やLeRobotへの適用は、[データ形式とオフライン推論](MODEL_USAGE.ja.md#data)を参照してください。

### 8. 学習レシピ・過去runとの違い

`Raw`は**BTの写像TなしでSIGRegを直接適用するモデル**であり、正則化なしという意味ではありません。公式配布重みと、自分で学習したRaw/SIGRegの重みも区別します。

2026-09-10の実測：新経路Rawは74,504更新でユーザー指示により停止し、最後の保存済みcheckpointは70,000更新です。同じ固定50ケースでRaw 70kは45/50、旧経路BT v2の70kは47/50。学習レシピ差と実物理初期状態の微差が残るため、方式の優位性とは断定しません。[実測・保存先・留保](reports/PUSHT_ISSUE20.ja.md)を参照してください。

新規のRaw／BT比較には `src/mylewm/training/train.py` を使います。データ読込はstable-worldmodel、画像前処理と一段損失は公式LeWM、逆伝播・optimizer・schedulerはstable-pretraining、訓練ループ・CSVログ・checkpointはLightningへ委託します。BT固有の処理は学習専用Tと正則化分岐です。

これは新レシピ `pusht_spt_v1` です。既存10万更新の再現経路と互換ではありません。重み・manifest・評価結果は保持しています。旧RBG専用処理は撤去し、共有ループを `training.py`、LIBERO入口を `train_libero.py` へ改名しました。過去runの厳密再開は開始時のGit版が必要です。LIBEROは後半の共有経路を使います。本学習・環境評価はユーザーの明示依頼時だけ実行します。

#### 変更する条件・維持する条件

旧経路と新経路でBT v2の基本方式・推論モデルが別物になったわけではありません。保存済み推論重みは同じ評価経路で扱えます。ただしデータ抽出などのレシピが違うため、旧経路のcheckpointから新経路への厳密な学習再開はできません。

| 項目 | 新しいRaw／BTで共通の条件 |
|---|---|
| 分割・行動統計 | 既存manifestのエピソード分離80/10/10とtrain-only統計を維持 |
| 画像・行動の取得 | SWMの4フレーム・frameskip5。20ステップ分を取得し、予測には先頭3行動chunkを使う。旧ローダーの16ステップ分とは末尾の有効クリップ数が異なる |
| 抽出 | PyTorchのepoch単位シャッフル・drop_last。旧経路の更新ごとの復元抽出から変更 |
| 予測損失 | 公式 `lejepa_forward` をそのまま使用。未来教師への勾配を維持 |
| 正則化 | 公式SIGRegを両方式ともFP32で計算。BTだけTを追加。射影乱数はモデルから分離 |
| 更新予算 | 両方式100,000、batch128、seed3072、同一の新規E/A/F初期値 |
| 学習率 | SPTのwarmup500＋cosine、最大5e-5、目標最小0。初回LRは0、最終更新に使うLRはごく小さい正値。旧添字規約とは異なる |
| optimizer | AdamW、全重みweight decay .001。モデルとTの勾配を別々にノルム1でclip |
| validation | manifestにある固定validationケースを使用（既存PushTは256件）。無ければvalidationエピソードの全有効クリップ |
| 推論 | 公式E/A/Fのみ。Tなし。訓練統計bufferを保持し、既存評価シェルで使用可能 |

各runにmanifest・初期モデル・ソース・主要依存ソースのhash、依存版、レシピを記録します。データは通常起動時に存在・サイズ・更新時刻を確認し、新規prepareで保存したSHA-256を参照情報として残します。全量を再読込して検証する場合だけ学習コマンドに`--verify-data`を追加してください。旧manifestにSHA-256が無くても通常起動では走査しません。明示検証時はhashを計算して記録しますが、比較対象が無ければ過去との同一性確認にはなりません。新Rawと新BTを比較し、旧10万BTとの違いをTだけの効果とは解釈しません。

2026-09-10の高速化：新PushTのGPU用DataLoaderは既定で`pin_memory`を使います。無効化する場合は`--no-pin-memory`を指定し、Raw／BTで設定を揃えてください。射影乱数のカウンタはCPUの整数としてcheckpointに保存し、毎更新のGPU同期を減らします。BTはCUDA上で同じ次元のCayley行列を一括計算します。バッチサイズ・射影数・損失・決定論設定・ログ頻度は変更しません。

固定CPUメモリと先読み画像にもRAMを使います。このPCではPushTのbatch256の画像588MiBをpinすると1GiBの確保になり、workers8・既定prefetch2の16枠が全てpin済みなら画像だけで約16GiBになります（1バッチ実測からの計算値で、総使用量や常時使用量ではありません）。上の本学習例のbatch128／workers4とは別条件です。GB10での併走時はGPUの使用量だけでなく、固定CPUメモリの保持量も確認してください。停止済みrunの再開条件は変更せず、調整は別の診断条件として扱います。[実測・原因切り分け](reports/LIBERO_BC.ja.md#memory-attribution)。

2026-09-11の対照試験では、このbatch256／workers8のPushTで固定host予約量約16GiBを実測しました。LIBERO（batch128／workers4、GPU予約約25.4GiB）との併走で新規CUDA初期化OOMが再現し、PushTだけ`--no-pin-memory`にした条件では観測中の失敗がなく、有効に戻すと再発しました。利用可能RAMが約40GiBあっても連続領域を確保できない場合があります。短期試験の結果であり、長時間の再発防止保証ではありません。[対照条件・結果](reports/GB10_MEMORY_CONTROLS.ja.md)。

ソースと乱数カウンタの保存形式が変わるため、**変更前runの厳密再開は変更前のコードで行います**。保存済みconfigや照合条件を変更して再開しないでください。[高速化の計測・検証記録](reports/TRAINING_SPEED_20260910.ja.md)を参照してください。

<a id="hdf5-libero-training"></a>

## 第3章 HDF5（LIBERO-10）の学習・評価

新規PushTの学習は冒頭の `train.py` に統一します。PushTのパス指定・prepare・学習・再開は前半の完全な例を使います。LIBEROは `train_libero.py` から共有ループを使用します。旧RBG、`--blocks`、`--cross-weight`、廃止済み初期値ファイルの読込引数 `--initialization` はありません。Raw/TC/BTは同じseedから初期化し、記録された初期モデルhashで照合します。過去の完全な手順・再開は[整理記録](reports/CLEANUP.ja.md)のGit履歴を参照してください。

公式LeWM側を学習したい場合は[公式PushT学習の説明書](../../lewm/TRAIN_PUSHT.ja.md)を参照してください。公式trainerの経路と、公平なBT比較向けRaw経路を分けています。

以下はLIBERO-10のHDF5取得・学習手順です。PushTの学習は前半の例を使い、ここでは重ねて起動しません。LIBEROには新PushT用の`--val-every`と`--compile-encoder`はありません。PushTとLIBEROの学習は一方ずつ実行してください。

旧経路PushT BT v2の100,000更新は完了済みです。新経路Rawの100k完了という意味ではありません。以下は新規実験の手順で、文書整理を理由に学習を自動起動しません。既存の重み・出力を上書きしないでください。

### 0. 固定依存の環境を用意する

以下は環境準備の工程です。学習・評価が稼働中の共有`.venv`では`uv sync`や自動同期を行わず、準備済み環境を`UV_NO_SYNC=1`で使うか、分離環境を用意します。[途中評価時の環境注意](EVALUATION.ja.md#intermediate-実行前の確認)を参照してください。

前半で固定依存の環境を準備済みなら、同期を繰り返す必要はありません。別端末では`cd`と環境変数を同じ値で設定します。以下は準備済みの環境を確認する例です。`pyproject.toml`と`uv.lock`はGB10/CUDA 13向けです。

```bash
cd "$(git rev-parse --show-toplevel)"
export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
uv run --no-sync python -c 'import hydra, h5py, lightning, stable_pretraining, stable_worldmodel, torch; print("hydra", hydra.__version__, "torch", torch.__version__)'
uv run --no-sync python -m mylewm.training.train --help
```

初回構築が必要な場合だけ、学習・評価が動いていない状態で`uv sync --locked --group libero --group lerobot`を実行します。

`ModuleNotFoundError: No module named 'hydra'` の配布名は`hydra-core`であり、`pyproject.toml`の依存`stable-worldmodel[env,train]`経由で導入されます。最後の2コマンドが通れば、少なくともPushT trainerのimportとCLIまで確認できています。CUDAを使わないPCはこのlockfileをそのまま使わず、対応するPyTorch backendで別途lockを作成してください。

初回構築用の`--group libero`はLIBEROのPython依存も導入します。実環境評価にはさらにLIBERO本体（この作業木では`external/libero`）、OSMesa、`LIBERO_CONFIG_PATH`、10個のHDF5データが必要です。公式LIBEROのPython 3.8 / CUDA 11.3手順をこのPython 3.12 / CUDA 13環境へ一般化した完全な構築手順は未検証です。したがって、新しいPCでLIBERO評価まで行う場合は、ここにない依存を推測で導入せず、対応環境を先に検証してください。

### 1. LIBERO-10のHDF5を取得する

既に10個のHDF5がある場合は次節へ進みます。未取得の場合は、実際の保存先に合わせて以下を実行します。容量は約13.7GBです。

```bash
export BT_SIGREG_DATA_ROOT="$PWD/.cache/libero-datasets"
uv run --no-sync hf download yifengzhu-hf/LIBERO-datasets --repo-type dataset \
  --include 'libero_10/*' --local-dir "$BT_SIGREG_DATA_ROOT"
```

### 2. 作業フォルダ・環境・データを確認する

全コマンドはリポジトリ直下で実行します。別の場所に置いた場合は、最初の`cd`だけ実際の場所に変更してください。仮想環境のactivateは不要です。

```bash
cd "$(git rev-parse --show-toplevel)"
export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
export LIBERO10_DATASET="$PWD/.cache/libero-datasets/libero_10"
uv run --no-sync python --version
uv run --no-sync python -m mylewm.training.train_libero --help
nvidia-smi
ls -lh "$LIBERO10_DATASET"/*.hdf5
```

LIBERO-10はタスクごとのHDF5が10個必要です。`LIBERO10_DATASET`には10個の`.hdf5`を直接含むディレクトリを指定します。PushTのように単一ファイルを指定する形式ではありません。例えば別の保存先なら`export LIBERO10_DATASET="/mnt/data/robot datasets/libero_10"`と引用符で囲みます。対象benchmarkのデータが見つからなければ先へ進まないでください。`uv run --no-sync python`やimportが見つからない場合も環境準備が必要です。任意の最新版をまとめてインストールして既存環境を上書きしないでください。

学習では実画像・実行行動から一段先の潜在状態を予測します。LIBEROは2つの実カメラを使い、10タスクで1つのモデルを共有します。HDF5からの学習にはMuJoCo/OSMesaの起動は不要です。シミュレータが必要なのは後述の制御評価です。

### 3. 学習・検証の分割ファイルを作る（最初の1回だけ）

新規prepareはPushT・LIBEROとも全量SHA-256を一度計算してmanifestの`data_fingerprints`へ保存するため、データ量に比例して時間がかかります。既存manifestの作り直しは不要です。以降の通常学習起動ではこの全量走査を繰り返しません。

`manifest.json`はデータの場所、学習/検証/テストの分割、行動の正規化統計を記録するファイルです。データ本体を複製・生成する処理ではありません。新しいmanifestを**現在の実フォルダで**作り、旧フォルダ名の一時リンクに依存しないようにします。

LIBERO-10用：

```bash
uv run --no-sync python -m mylewm.training.train_libero prepare \
  --dataset "$LIBERO10_DATASET" \
  --manifest output/manifests/libero10/manifest.json
```

完了すると分割件数が表示されます。LIBEROのmanifestの`dataset`は、隣に作る`files.json`の絶対パスです。`files.json`には10個のHDF5それぞれの絶対パス・サイズ・更新時刻を保存し、学習はこの一覧から読み込みます。prepare時のデータhashも記録します。既に作成済みならこの工程を飛ばしてください。`FileExistsError`は上書き防止です。学習開始後はmanifest・データ・フォルダ名を変更しないでください。移転後の**新規実験**は新しいmanifestを作れますが、途中再開のために元のmanifestを書き換えると互換性チェックで拒否されます。

### 4. まず100ステップだけ動作確認する

`--steps`はoptimizerの呼出し数、`--batch-size`は1回に使うクリップ数です。出力先は未使用の名前にします。**runフォルダ自体を先にmkdirしないでください**。trainerが作成します。

LIBERO-10：

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run --no-sync python -m mylewm.training.train_libero train \
  --mode bt --manifest output/manifests/libero10/manifest.json \
  --output output/libero10/bt_smoke \
  --steps 100 --batch-size 16 --workers 0 --seed 3072 \
  --warmup-steps 10 --lr 5e-5 --min-lr 0 \
  --save-every 50 --diagnostics-every 25 --deterministic
```

通常起動は全量ハッシュを走査しません。LIBEROでは`verifying_training_data`（`size_mtime`）の後に`start`、`step`を確認します。`--verify-data`指定時だけ全量走査で待ち時間が生じます。サイズ・更新時刻を保存したままの改変は軽量確認では検出できません。最後にstep100と`resume.pt`を確認してください。loss低下だけでは制御性能やマルチタスク能力を評価できません。

### 5. 本学習を開始する

まず端末を閉じても実行を保持するため、既存のtmuxで作業用セッションを作ります。このPCではtmuxを確認済みです。

```bash
tmux new -s bt-training
cd "$(git rev-parse --show-toplevel)"
export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
```

短期学習のcheckpointは使わず、新規初期値から始めます。

LIBERO-10・10万ステップ（本学習のこの設定はまだ完走検証していません）：

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run --no-sync python -m mylewm.training.train_libero train \
  --mode bt --manifest output/manifests/libero10/manifest.json \
  --output output/libero10/bt_train \
  --steps 100000 --batch-size 128 --workers 4 --seed 3072 \
  --warmup-steps 500 --lr 5e-5 --min-lr 0 \
  --bt-depth 2 --bt-kappa .2 --bt-hidden 192 \
  --save-every 5000 --diagnostics-every 1000 --deterministic
```

`Ctrl-b`を押して離し、次に`d`を押すとtmuxから離れます。戻るには`tmux attach -t bt-training`。学習画面でのCtrl-Cは**学習そのものを中断**します。tmuxは端末切断対策で、OS再起動や停電後に学習を自動再開するものではありません。

これらは単独学習の手順です。Raw/TCとの厳密な比較では同一の未学習初期重み・データ順・batch・予算等をそろえます。旧初期値生成CLIは削除しました。新規PushTの比較は冒頭の新経路で共通seedから初期化し、configの初期モデルhashを照合します。完了済みPushTの開始コマンドは[実行記録](reports/PUSHT_TRAINING_100K.ja.md)に残しています。

### 6. 別の端末から進捗を見る

PushTの新経路は冒頭のCSVを確認します。LIBEROのJSONLログを見る場合は次を使います。

```bash
bash scripts/monitor_training.sh --run output/libero10/bt_train
```

100ステップ確認を見る場合は`bt_train`を`bt_smoke`に変更。`--once`を付けると1回表示、`--interval 10`で10秒間隔です。`--run`を省略した既定画面は**完了済みPushT run**なので、新規実験では必ず指定してください。tmuxで起動した学習にはsystemd serviceがないため、`--unit`は付けません。

| 表示・ファイル | 意味 |
|---|---|
| Step / loss / Loss mean | 進捗、最新バッチの損失、直近最大100件の平均 |
| prediction / SIGReg | 予測誤差とGaussian正則化。小さいだけで性能が良いとは限らない |
| Grad / T grad / LR | 勾配ノルムと学習率。warmup後にLRが減少し、最終ステップは0 |
| Validation / T diagnostics | 最新の疎な診断値。どのstepで測った値かを併記 |
| `metrics.jsonl` | 全ステップの数値ログ。学習中に編集・削除しない |
| `config.json` / `budget.json` | 設定・データ/コードの識別情報と学習量 |
| `initialization.pt` | 初期状態の記録。再開時の照合にも必要 |
| `resume.pt` | 最新保存時点のE/A/F・T・optimizer・乱数等。途中再開用 |
| `step_5000_object.ckpt`など | その時点の推論モデル。Tを含まず、途中再開には使わない |

出力はすべて`output/`以下でGit対象外です。通常の数値ログは自動保存されますが、このtmux手順は端末の全stdout/stderrを別のconsole.logへ自動保存するものではありません。エラー発生時は端末のtracebackも残してください。

### 7. 中断した学習を再開する

学習プロセスが終了していることと、対象runに`resume.pt`があることを確認します。以下は上のLIBERO本学習例で開始し、中断したrunを同じ条件で再開する例です。LIBEROでは`--resume`はファイルパスを取らず、元の`--output`内の`resume.pt`を使います。PushT新経路は前半のLightning再開例を使ってください。開始時のソース・環境を維持し、改名前のrunは開始時のGit版を使います。

```bash
cd "$(git rev-parse --show-toplevel)"
export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run --no-sync python -m mylewm.training.train_libero train \
  --mode bt --manifest "output/manifests/libero10/manifest.json" \
  --output "output/libero10/bt_train" \
  --steps 100000 --batch-size 128 --workers 4 --seed 3072 \
  --warmup-steps 500 --lr 5e-5 --min-lr 0 \
  --bt-depth 2 --bt-kappa .2 --bt-hidden 192 \
  --save-every 5000 --diagnostics-every 1000 --deterministic --resume
```

- `--output`は元のrunのまま。`--steps`も元の総数のままです（残りステップ数ではありません）。
- ほかの引数・manifest・データ・学習ソース・環境を変えないでください。`resume mismatch`を無理に解除してはいけません。
- 保存は本学習なら5,000ステップごと。保存前の未確定分はやり直しになります。初回保存前に停止した場合は、未使用の出力名で新規学習し直してください。
- 100ステップの確認runを`--steps 100000 --resume`で延長することはできません。短期診断と本学習ではスケジュールが違います。
- `torch.load`を使うため、自分で作成した信頼済みcheckpointだけを使ってください。不明な配布重みを再開ファイルにしないでください。

<a id="hdf5-libero-evaluation"></a>

### 8. 評価：LIBERO環境で成功率を測る

学習後の`output/libero10/bt_train/step_100000_object.ckpt`と、学習に使った`output/manifests/libero10/manifest.json`を使います。環境評価にはLIBERO本体・OSMesaなどの準備とrender監査が必要です。[LIBERO評価手順](EVALUATION.ja.md#libero-2-osmesa画像監査)の監査を済ませ、同文書の「実環境CEM評価」へ進んでください。BTではcheckpointと出力先をBTのrunに合わせます。

dry-run後に`--execute`を付けて実行します。終了コード0と`status.json`の`succeeded`、`summary.json`、`episodes.jsonl`、各初期状態の記録を照合します。平均成功率に加え、10タスクそれぞれの結果も確認してください。途中checkpointは[途中評価手順](EVALUATION.ja.md#intermediate-libero-10)を使います。LIBERO-10の本学習・制御成功率は未検証です。

### 9. よくある問題

| 症状 | 対処 |
|---|---|
| `FileExistsError` | 新規学習なら未使用の出力名にする。再開なら保存済みrunを確認して`--resume`。既存成果を消して回避しない |
| データが見つからない | `--manifest`と内部のパスを確認。稼働中のmanifestを書き換えない。新規実験なら現フォルダでprepareする |
| `warmup_steps`のエラー | warmupは総steps未満が必要。短期確認は100/10、本学習は100000/500 |
| CUDA out of memory | 他の学習を重複起動していないか確認。新規runとしてbatchを下げる。途中resumeのbatchを勝手に変更しない。特にLIBERO batch128はこのPCで長期未検証 |
| NaN/Inf・10分以上進まない | 端末の例外、プロセス、データ読込を確認し報告。自動再起動や設定変更で隠さない |
| lossが短期確認より大きい | SIGRegの値はbatch数にも依存する。batch16と128の生lossを直接比較しない |

学習が終わっても、PushT/LIBEROの**成功率評価は別工程**です。[PushT評価手順](EVALUATION.ja.md#pusht)・[LIBERO評価手順](EVALUATION.ja.md#libero)と[検証記録](VALIDATION.ja.md)を参照してください。
