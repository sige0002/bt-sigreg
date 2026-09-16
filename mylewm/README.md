# 実装・環境への入口

BT-SIGRegの実装は`src/mylewm/`、この`mylewm/`ディレクトリは設定・テスト・文書です。初めて読む方は[ルートREADME](../README.md)から、[アルゴリズム](docs/ALGORITHM.ja.md)→[実装](docs/IMPLEMENTATION.ja.md)→[学習](docs/TRAINING.ja.md)→[評価](docs/EVALUATION.ja.md)へ進んでください。

## コードの構成と起動

パッケージは比較用`lewm/`と同じcheckoutにeditableインストールして使います。

```bash
cd "$(git rev-parse --show-toplevel)"
export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
uv run --no-sync python -m mylewm.training.train --help
uv run --no-sync python -m mylewm.training.train_libero --help
uv run --no-sync python -m mylewm.training.train_libero_bc --help
```

初回の環境作成時は`uv sync --locked --group libero --group lerobot`を使います。既に学習・評価に使っている環境へ同期しないでください。[学習ガイドの環境準備](docs/TRAINING.ja.md#setup)に実行順序があります。

| 場所 | 内容 |
|---|---|
| `src/mylewm/algorithms/` | 写像・損失・2カメラモデル |
| `src/mylewm/training/` | 世界モデル・BCの学習と再開 |
| `src/mylewm/data/` | データ取得・再生成・入力契約・検証 |
| `src/mylewm/evaluation/` | 環境評価・診断・結果比較・viewer |
| `src/mylewm/environments/` | 計画器と行動座標変換 |
| `src/mylewm/policy/` | BC、CEM runtime、LeRobot Policy export |
| `mylewm/configs/`・`mylewm/tests/`・`mylewm/docs/` | 設定例・テスト・文書 |

ファイル単位の案内とバッチの形は[実装ガイド](docs/IMPLEMENTATION.ja.md)へ。保存済みクラス復元用の互換ファイルと比較用LeWMを、未使用と判断して削除しないでください。

<a id="pcごとの外部環境設定"></a>

## 外部環境

LIBERO環境は`scripts/run_libero.sh`で起動します。Pythonパッケージのインストールだけで、LIBEROソース・描画runtime・データまで揃うわけではありません。

| 環境変数 | 既定の場所・意味 |
|---|---|
| `LIBERO_ROOT` | `external/libero`：LIBEROソース |
| `LIBERO_MUJOCO_PATH` | `.cache/libero-runtime`：専用Python runtime |
| `LIBERO_CONFIG_PATH` | `.cache/libero-config`：LIBERO設定 |
| `LIBERO_OSMESA_DIR` | `.cache/libero-osmesa/usr/lib/<アーキテクチャ>-linux-gnu`：OSMesaライブラリ |

```bash
bash scripts/run_libero.sh -m mylewm.evaluation.evaluate_libero --help
```

別の場所へ導入している場合は対応する環境変数を指定します。[描画監査](docs/reference/CEM_AND_RENDERING.ja.md#libero-2-osmesa画像監査)を通し、学習と評価のカメラ・向き・解像度を照合してください。

## 検証と運用

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python -m pytest mylewm -q
git diff --check
```

CPU実行や外部ファイルがない環境ではskipがあり得ます。実測した件数を記録します。特定PCのrun・サービス状態・再開用の固定ソースは[運用記録](docs/operations/AGENT_OPERATIONS.ja.md)、旧構成の復元は[整理履歴](docs/reports/CLEANUP.ja.md)に分けています。
