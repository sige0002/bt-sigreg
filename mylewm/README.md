# mylewm：BT-SIGRegの学習・比較基盤

LeWMの一段予測と小型E/A/Fを維持し、学習専用の有界可逆写像Tを通した表現だけにSIGRegを適用します。推論時にはTを使いません。

## はじめて学習する方へ

PushT／LIBEROの通常学習起動はデータのサイズ・更新時刻を確認します。新規prepare時に全量SHA-256を一度記録し、起動時の全量再検証は`--verify-data`指定時だけです。再開では開始時と同じ検証設定・コード・環境を使います。

[PushT／LIBERO-10の学習手順](docs/TRAINING.ja.md)に、PushT用の`uv`環境構築、環境・データ確認、分割作成、短期確認、本学習、進捗表示、再開をまとめています。LIBERO実環境評価までを含む別PC用の完全な環境構築手順は未検証です。

| 読みたい内容 | 文書 |
|---|---|
| 研究目的と全体像 | [ルートREADME](../README.md) |
| アルゴリズム・数式・保証の限界 | [BT-SIGReg](docs/research/BT_SIGREG.ja.md) |
| 関連研究・採否判断 | [研究レビュー](docs/research/RESEARCH_REVIEW.ja.md) |
| PushTの成功率評価 | [評価手順](docs/EVALUATION.ja.md#pusht) |
| LIBERO-10の評価環境 | [評価手順](docs/EVALUATION.ja.md#libero) |
| 公式LeWMの学習 | [公式PushT手順](../lewm/TRAIN_PUSHT.ja.md) |
| 実験結果・監査・未完了項目 | [レポート一覧](docs/reports/README.md) |
| ファイルを残した理由・復元 | [整理記録](docs/reports/CLEANUP.ja.md) |

## 現在の状態（2026-09-09）

新規PushTのRaw／BT比較用に `train.py` を追加しました。公式SWM/SPT/Lightningを使い、エピソード分離を維持します。[新経路の手順と条件差](docs/TRAINING.ja.md#新しいpusht経路公式ライブラリへ委託2026-09-09)を参照。旧100kの継続ではなく新レシピです。2026-09-10、新Raw（SIGReg）は74,504更新でユーザー指示により停止し、保存済み70kを評価済み。同じ固定50ケースでSIGReg 70kは45/50、旧BT v2の70kは47/50。学習手順差と物理初期状態の微差があるため、方式の優位性とは断定しません。[詳細](docs/reports/PUSHT_ISSUE20.ja.md)。

PushT BT v2の100,000更新は正常終了。固定confirm 200ケースでは178/200（89%）、別条件の上流eval 50ケースでは49/50（98%）です。ケース・正規化・seed処理が異なるので混ぜません。[評価レポート](docs/reports/PUSHT_CHECKPOINT_EVALUATION.ja.md)に条件と留保を記録しています。

LIBERO-10は100更新の動作確認まで。同予算Raw/TC比較、複数学習seed、マルチタスク性能向上は未実証です。新規学習・評価・定期監視を自動開始しません。

## コードの構成と起動

実装はリポジトリ直下の`src/mylewm/`へ集約しています。`mylewm/`には設定例・文書・テストを置き、シェルは`scripts/`へ分離しました。

| 場所（リポジトリ基準） | 用途 |
|---|---|
| `src/mylewm/algorithms/` | BTの写像、Raw／TC／BTの損失、2カメラモデル |
| `src/mylewm/training/` | Raw／BTの学習CLI、LIBERO共有ループ、再開状態・診断 |
| `src/mylewm/data/` | HDF5／LeRobot読込・prepare・入力契約・データ検証 |
| `src/mylewm/evaluation/` | 環境評価・監査・結果比較・可視化 |
| `src/mylewm/environments/` | LIBERO計画器、環境へ渡す行動の座標変換 |
| `src/mylewm/policy/` | 共通CEM runtime、オフライン推論、LeRobot Policy・変換CLI |
| `src/mylewm/paths.py` | リポジトリ・設定・外部環境のパス、ソース一覧 |
| `scripts/` | 評価・監視・OSMesa起動シェル |
| `mylewm/configs/` / `mylewm/tests/` / `mylewm/docs/` | 設定例／回帰テスト／文書 |

`training.loop.TrainingAdapter`がデータ取得・前処理・モデル構築を共有ループへ渡します。LIBERO起動時に共有モジュールの関数を上書きしません。ファイルのSHA-256は`data.verification.file_sha256`へ集約し、ソースの移動や共通処理の変更も再開時の照合対象に含めます。

このパッケージは、比較用`lewm/`と設定を同じcheckoutに持つ**editableインストール**で使用します。新しい専用環境を作る例です。使用中の環境には同期しません。

```bash
cd "$(git rev-parse --show-toplevel)"
export UV_PROJECT_ENVIRONMENT="$PWD/.venv-training-new"
uv sync --locked --group libero --group lerobot
uv run --no-sync python -m mylewm.training.train --help
uv run --no-sync python -m mylewm.training.train_libero --help
uv run --no-sync python -m mylewm.data.prepare_dataset --help
```

以後は`python -m mylewm.…`で起動します。旧`python mylewm/train.py`等のパスは廃止しました。既に依存が揃った停止中の環境へパッケージだけ導入する場合は、対象環境を明示して`uv pip install --python /path/to/venv/bin/python --no-deps --editable .`を使えます。`--no-sync`はパッケージ導入後に使ってください。

`src/mylewm/libero_model.py`だけは、保存済みLIBERO object checkpointのクラス参照を解決するために残します。実装は`algorithms/libero_model.py`に一つだけです。旧RBGや旧CLIの互換ファイルは追加しません。既存checkpointの推論読込と、厳密な学習再開は別です。**変更前runの再開には開始時のGit版・環境が必要です。** 元のmanifest・config・照合値を変更して再開しません。

## PCごとの外部環境設定

リポジトリ内の場所は`paths.py`から解決します。入力データ・manifest・出力先は各CLIの引数で指定します。LIBEROの外部環境は次の環境変数で変更できます。

| 環境変数 | 省略時 | 指定するもの |
|---|---|---|
| `LIBERO_ROOT` | `external/libero` | LIBEROソースの場所 |
| `LIBERO_MUJOCO_PATH` | `.cache/libero-runtime` | ローカルのPython runtime依存 |
| `LIBERO_CONFIG_PATH` | `.cache/libero-config` | LIBERO設定ディレクトリ |
| `LIBERO_OSMESA_DIR` | `.cache/libero-osmesa/usr/lib/<実行CPUアーキテクチャ>-linux-gnu` | `libOSMesa.so.8`を含むディレクトリ |

```bash
export LIBERO_OSMESA_DIR="/opt/mesa/lib"
export LIBERO_ROOT="/opt/LIBERO"
bash scripts/run_libero.sh -m mylewm.evaluation.evaluate_libero --help
```

明示した設定を起動シェルが上書きしません。画像監査のデータ場所は`python -m mylewm.evaluation.audit_libero_images --dataset /path/to/libero_10`で指定できます。パス指定への対応は、別PCでの描画・制御検証を済ませたという意味ではありません。OSMesaの画像監査は評価前に必要です。

削除・移動の詳細と復元方法は[整理履歴](docs/reports/CLEANUP.ja.md)を参照してください。

## シェルと回帰確認

進捗表示は手動・読み取り専用です。既定runは完了済みのPushTなので、新規実験では `--run` を指定します。

```bash
bash scripts/monitor_training.sh --once
bash scripts/evaluate_pusht.sh --help
CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run --no-sync python -m pytest mylewm -q
```

CPU限定の回帰確認ではCUDA専用の5ケースがスキップされます：共有ループの連続学習と再開の一致（worker 0/2の2ケース）、BTの再開・推論出力（1/2カメラの2ケース）、CPU履歴とGPU候補を混ぜた入力の処理（1ケース）。GPUで検証する場合は学習終了後に`CUDA_VISIBLE_DEVICES=''`を外し、`CUBLAS_WORKSPACE_CONFIG=:4096:8`を指定します。公式checkpointが無い環境では、その読込テストも別途スキップされます。全実験出力はGit対象外の `output/` 以下へ保存します。
