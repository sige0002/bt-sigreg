# 実装と文書の配置

実装は`src/mylewm/`、この`mylewm/`ディレクトリは設定・テスト・文書です。

**最初の環境構築から学習開始までは[ルートREADME](../README.md)にコマンドを載せています。** 再開まで続けて読むなら[PushT学習手順](docs/TRAINING.ja.md)だけで完結します。

| 作業 | 正本 |
|---|---|
| PushTの取得・prepare・学習・再開 | [PushT学習](docs/TRAINING.ja.md) |
| LIBEROの取得・prepare・学習・再開 | [LIBERO学習](docs/TRAINING_LIBERO.ja.md) |
| 世界モデル＋CEMの評価 | [評価](docs/EVALUATION.ja.md) |
| BCの追加学習・再開・環境評価 | [BC](docs/BEHAVIOR_CLONING.ja.md) |

<a id="コードの構成と起動"></a>

## コードの構成

| 場所 | 内容 |
|---|---|
| `src/mylewm/algorithms/` | 写像・損失・2カメラモデル |
| `src/mylewm/training/` | 世界モデル・BCの学習と再開 |
| `src/mylewm/data/` | データ取得・再生成・入力契約・検証 |
| `src/mylewm/evaluation/` | 環境評価・診断・結果比較・viewer |
| `src/mylewm/environments/` | 計画器と行動座標変換 |
| `src/mylewm/policy/` | BC、CEM runtime、LeRobot Policy export |
| `mylewm/configs/`・`mylewm/tests/`・`mylewm/docs/` | 設定例・テスト・文書 |

バッチの形とコードの流れは[実装ガイド](docs/IMPLEMENTATION.ja.md)にあります。保存済みクラスの復元用互換ファイルと比較用LeWMは保持します。

<a id="pcごとの外部環境設定"></a>

## 作業者向け

マシン固有設定は[環境別メモ](docs/operations/MACHINE_NOTES.ja.md)、停止指示と再開条件は[運用](docs/operations/AGENT_OPERATIONS.ja.md)に分けています。LIBERO描画の初回準備は評価・BCそれぞれの正本内にあります。

コード変更時の回帰入口は`.venv/bin/python -m pytest mylewm -q`です。文書だけの変更ではリンクとコマンドの整合を確認します。
