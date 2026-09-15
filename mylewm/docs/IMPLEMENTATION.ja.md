# 2. 実装：ファイルとデータの流れ

[アルゴリズム](ALGORITHM.ja.md) → 実装 → [学習](TRAINING.ja.md) → [評価](EVALUATION.ja.md)

## 最初に見るファイル

Python実装は`src/mylewm/`です。`mylewm/`は設定・テスト・文書の置き場です。以下のリンクはリポジトリ内の実ファイルを指します。

| 変更・確認したいもの | 入口 |
|---|---|
| BTの可逆写像・特異値制約 | [algorithms/bt_sigreg.py](../../src/mylewm/algorithms/bt_sigreg.py) |
| Raw／TC／BTの損失 | [algorithms/objectives.py](../../src/mylewm/algorithms/objectives.py) |
| LIBEROの2カメラ融合 | [algorithms/libero_model.py](../../src/mylewm/algorithms/libero_model.py) |
| LIBEROのprepare・世界モデル学習 | [training/train_libero.py](../../src/mylewm/training/train_libero.py) |
| LIBEROの更新ループ・保存・検証 | [training/loop.py](../../src/mylewm/training/loop.py) |
| PushT／LeRobotの新規学習 | [training/train.py](../../src/mylewm/training/train.py) |
| OpenVLA方式のデモ再生成 | [data/regenerate_libero_openvla.py](../../src/mylewm/data/regenerate_libero_openvla.py) |
| BCのデータ・画像前処理 | [data/libero_bc_data.py](../../src/mylewm/data/libero_bc_data.py) |
| BCの構成・行動生成 | [policy/libero_bc.py](../../src/mylewm/policy/libero_bc.py) |
| BCの学習・再開 | [training/train_libero_bc.py](../../src/mylewm/training/train_libero_bc.py) |
| BCの環境評価 | [evaluation/evaluate_libero_bc.py](../../src/mylewm/evaluation/evaluate_libero_bc.py) |
| LIBEROのCEM環境評価 | [evaluation/evaluate_libero.py](../../src/mylewm/evaluation/evaluate_libero.py) |
| データのサイズ・mtime・hash照合 | [data/verification.py](../../src/mylewm/data/verification.py) |

## LIBERO世界モデルの1バッチ

1. `LiberoClips`が1本のデモから画像4時刻と行動12時刻を取り出す。
2. 画像は4行動おき、行動は4個ずつまとめた3区間になる。2カメラの画像を224×224へresizeし、正規化する。
3. 共有ViTで各カメラの特徴を取り、CLSを連結して192次元のzへ変換する。
4. 過去3状態と対応する行動から次のzを予測する。
5. 一段予測lossと方式ごとのSIGRegを計算し、optimizerを更新する。

画像バッチの概形は`[batch, time=4, cameras=2, channels=3, height, width]`、行動は`[batch, transitions=3, native_actions=4, dimensions=7]`です。世界モデルの4行動区切りと、BCの8連続行動は別の契約です。

LIBEROは`TrainingAdapter`でデータ取得・前処理・モデル構築を共有ループに渡します。Raw／TC／BTで基盤を共有し、共有関数を起動時に書き換えません。

## BCの1バッチと推論

BCは、現在の実測画像1組と、その直後の8連続行動を教師にします。世界モデルのような4行動ごとの間引きは行いません。1つのchunkがデモ境界を越えることもありません。

| 部分 | 現行構成 |
|---|---|
| 画像encoder | 世界モデルのViTを凍結・evalモードで使用 |
| 画像特徴 | 各カメラCLS 1個＋4×4 pool patch 16個、計34token |
| 行動head | 幅256、4ブロック、8 heads、タスクIDとflow時刻で条件付け |
| 出力 | 8行動×7次元 |
| 学習 | ノイズからデモ行動へ向かう速度をflow matchingで学習 |
| 推論 | Gaussian noiseから10 Euler stepで生成し、逆正規化・範囲制限 |

既存構成のパラメータはencoder約550万（凍結）、head約599万（学習）、合計約1,149万です。TC-LeWMの未公開の幅・層数を再現した値ではありません。

データ内のタスク順とLIBERO環境のtask ID順は異なるため、名前を経由して対応付けます。画像・行動の順序、正規化統計、分割はmanifestとcheckpointで照合します。

## checkpointを取り違えない

| 保存物 | 用途 |
|---|---|
| `manifest.json` | データの場所、分割、前処理、行動統計、データ由来 |
| `config.json` | 学習・評価の設定、依存版、ソースhash |
| `step_N_object.ckpt` | 世界モデル。CEM評価やBC用encoder取得に使用 |
| `step_N_bc.pt` | encoder＋BC head。BC評価に使用 |
| `resume.pt` | LIBERO共有ループ／BCのoptimizer・乱数を含む再開状態 |
| `metrics.jsonl` | 各更新のloss等。成功率ではない |
| `completed.json` | 学習予算の完了記録 |

PushTのライブラリ経路では`resume.ckpt`など保存形式が異なります。[詳細手順](reference/TRAINING.ja.md#hdf5-pusht-training)を参照してください。object checkpointの読込には信頼済みファイルを使います。

過去checkpointのクラス名を復元する`src/mylewm/libero_model.py`と、比較用`lewm/`は必要な互換部分です。見た目が重複していても削除しません。ソースや環境を変えた新runと、開始時条件が一致した厳密再開は区別します。

<a id="tests"></a>
## コードを変更したら

関連する回帰を実行します。全体の入口は次です。CuBLASの設定はGPU決定論テストのために、Python起動前に指定します。

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python -m pytest mylewm -q
git diff --check
```

2026-09-15の確認は240件合格、skipなしでした。CPUだけの実行ではGPUテストがskipされ得るため、実際の件数を報告します。docsのみの変更では、リンクとCLI・設定契約の整合を確認します。

稼働中runは開始時の固定ソース・環境を使います。リポジトリの修正を反映させるために、進行中runのhashを上書きしません。運用上の制約は[AGENT_VALIDATION](AGENT_VALIDATION.ja.md)を参照してください。
