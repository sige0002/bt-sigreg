# 3. 学習：データから世界モデル、BC方策まで

[アルゴリズム](ALGORITHM.ja.md) → [実装](IMPLEMENTATION.ja.md) → 学習 → [評価](EVALUATION.ja.md)

## 何を学習するか選ぶ

| 目的 | 経路 |
|---|---|
| 現在のLIBERO設定で学習する | このページの手順。再生成データ→BT10k→BC40k |
| PushTで基本動作を確認する | [PushT HDF5手順](reference/TRAINING.ja.md#hdf5-pusht-training) |
| LeRobot Dataset v3を使う | [LeRobot手順](reference/TRAINING.ja.md#lerobot-training)。単一カメラ経路 |
| 既存のLIBERO世界モデルにBCを追加する | [BCガイド](BEHAVIOR_CLONING.ja.md) |
| 旧データ・100kレシピを確認する | [従来LIBERO手順](reference/TRAINING.ja.md#hdf5-libero-training)。今回の10k設定とは別 |

**世界モデルの更新回数とBCの更新回数は別です。** 最新のLIBERO再学習設定は世界モデル10,000回、BC40,000回。1更新は1バッチに対するoptimizer更新であり、環境の1行動ではありません。

このページは新しい実験を立ち上げる手順です。このPCで既に動いている逐次ジョブを確認する場合は[運用記録](AGENT_OPERATIONS.ja.md)を使い、同じ処理を重複起動しないでください。

<a id="setup"></a>
## 1. 環境を確認する

コマンドはリポジトリ直下で実行します。実装と比較用`lewm/`を同じcheckoutに置くeditableインストールを使います。

```bash
cd "$(git rev-parse --show-toplevel)"
export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
uv run --no-sync python -m mylewm.training.train_libero --help
uv run --no-sync python -m mylewm.training.train_libero_bc --help
```

**初回の環境作成時だけ** `uv sync --locked --group libero --group lerobot` を実行します。稼働中の学習・評価で使っている`.venv`には同期しません。依存変更の検証には別の`UV_PROJECT_ENVIRONMENT`を使います。

LIBEROのデモ再生成・環境評価にはLIBERO本体、MuJoCo、描画環境も必要です。[LIBERO環境と画像監査](reference/EVALUATION.ja.md#libero-2-osmesa画像監査)・[外部パス設定](../README.md#外部環境)を参照してください。別PCでの全工程の動作は保証済みではありません。

このPCで不足依存を隔離したruntimeを使う場合は、追加で以下を設定します。このパスはGitには含まれません。

```bash
export PYTHONPATH="$PWD/output/libero10/bc_implementation_check/runtime${PYTHONPATH:+:$PYTHONPATH}"
```

## 2. データを用意する

LIBERO-10の元データは、各タスク1つの`*_demo.hdf5`、計10ファイルです。取得方法は[データ取得手順](reference/TRAINING.ja.md#1-libero-10のhdf5を取得する)を参照してください。

以下は**新規run用の名前の例**です。実行前に未使用であることを確認します。出力ディレクトリを先に作らないでください。

```bash
export LIBERO_RAW="$PWD/.cache/libero-datasets/libero_10"
export LIBERO_REGENERATED="$PWD/output/libero10/openvla_example"
export LIBERO_MANIFEST="$PWD/output/manifests/libero10/openvla_example/manifest.json"
export LIBERO_WM_RUN="$PWD/output/libero10/bt_openvla_example_10k"
export LIBERO_BC_RUN="$PWD/output/libero10/bc_openvla_example_40k"
```

まず再生成の設定を確認します。以下はdry-runで、環境は動きません。

```bash
bash scripts/run_libero.sh -m mylewm.data.regenerate_libero_openvla \
  --raw-data "$LIBERO_RAW" --output "$LIBERO_REGENERATED"
```

実行するときは同じコマンドに`--execute`を追加します。OpenVLAの固定コードで元デモを再実行し、256×256の画像を保存、無動作行動と失敗デモを除きます。元のHDF5は変更しません。全デモの再実行は数時間かかり得ます。

終了コード0と`$LIBERO_REGENERATED/status.json`の`succeeded`を確認してから次へ進みます。生成データは`data/`、元コード・由来・hash・成功判定記録も同じ出力先に保存されます。描画環境の違いにより保持デモ数は変わり得ます。

## 3. デモを分割してmanifestを作る

**manifest**は、データの場所・学習／検証／テストの分割・行動の正規化統計を固定したJSONです。1本のデモを丸ごと1つの用途へ割り当てます。

```bash
uv run --no-sync python -m mylewm.training.train_libero prepare \
  --dataset "$LIBERO_REGENERATED/data" --manifest "$LIBERO_MANIFEST" \
  --split-mode ratio_80_10_10
```

各タスクの残ったデモを約80%学習・10%検証・10%テストに分けます。50本なら40・5・5です。各タスク10本以上が必要で、各デモは13行動以上を必要とします。統計は学習用デモだけから計算します。この分割はローカル仮定で、TC-LeWM著者の分割と一致確認したものではありません。

prepare時はデータ全量のSHA-256を記録します。通常の学習起動ではサイズ・mtimeを確認し、全量の再検証は`--verify-data`指定時だけです。再生成後のデータを既存manifestへ差し替えないでください。

## 4. BT世界モデルを10,000更新する

以下は**そのまま学習を開始するコマンド**です。`train_libero train`にはdry-run用の`--execute`はありません。

```bash
uv run --no-sync python -m mylewm.training.train_libero train \
  --dataset "$LIBERO_REGENERATED/data" --manifest "$LIBERO_MANIFEST" \
  --output "$LIBERO_WM_RUN" --mode bt \
  --steps 10000 --batch-size 128 --workers 4 --no-pin-memory \
  --lr 5e-5 --warmup-steps 500 --min-lr 0 \
  --bt-depth 2 --bt-kappa 0.2 --bt-hidden 192 \
  --save-every 1000 --seed 3072
```

10タスクで1モデルを共有します。学習率はwarmup後cosineで低下します。学習時のBT変換は既定2ブロックで、後段BCの4層とは別です。

短期の接続確認を行う場合は、別出力で更新数・warmup・batchを小さくします。短期runの予算を後から変更して本学習へ再開することはできません。Raw／TC比較は、モード以外の初期値・データ・予算を揃えた独立runで行います。

## 5. 凍結encoder上のBCを40,000更新する

世界モデルの`completed.json`が10,000更新の完了を記録し、`step_10000_object.ckpt`が保存されたことを確認します。次は設定確認だけのdry-runです。

```bash
uv run --no-sync python -m mylewm.training.train_libero_bc \
  --checkpoint "$LIBERO_WM_RUN/step_10000_object.ckpt" \
  --manifest "$LIBERO_MANIFEST" --output "$LIBERO_BC_RUN" \
  --steps 40000 --batch-size 256 --workers 4 --seed 3072 \
  --width 256 --depth 4 --heads 8 --horizon 8 --euler-steps 10
```

実行するときは同じコマンドに`--execute`を追加します。BCは世界モデルと同じmanifestを使用します。既存の旧データencoderへ新データを渡すためにhash照合を解除しません。

画像encoderは固定し、約599万パラメータの方策headを学習します。AdamW・LR2e-4・weight decay0.01・一定LR・clip norm1、FP32、augmentationなしです。幅・層数等は採用したローカル構成です。[BCの入力と保存](BEHAVIOR_CLONING.ja.md)

## 6. 進捗・完了・再開

| 確認対象 | 見るもの |
|---|---|
| 世界モデルの進捗 | `metrics.jsonl`の最後のstep、実プロセス、実行ログ |
| 世界モデルの完了 | 終了コード0、`completed.json`のstepとstate、最終object checkpoint |
| BCの進捗 | `metrics.jsonl`、`status.json`、実プロセス |
| BCの完了 | 終了コード0、`status.json`と`completed.json`がsucceeded、最終BC checkpoint |
| 失敗・中断 | 実行ログ、終了コード、実プロセス。古いrunning表示だけで判断しない |

validation lossは保持デモ上の誤差であり、制御成功率ではありません。学習が正常終了しても、環境で動かして成功するとは限りません。

再開は開始時と同じソース・依存・manifest・構成・予算・出力で`--resume`を指定します。BCでは`--execute`も必要です。LIBEROの再開状態は`resume.pt`です。未使用出力へ新規起動する場合と混同しないでください。固定ソースの場所と、このPCの逐次ジョブの引継ぎは[運用記録](AGENT_OPERATIONS.ja.md)にあります。

このガイドのコマンドは各工程を手動で実行するものです。今回の承認済みジョブは固定ソースから逐次実行しています。ジョブの起動記録を再実行せず、学習後の制御確認は[評価ガイド](EVALUATION.ja.md)へ進んでください。

<details>
<summary>従来の詳細手順へのリンク（旧URL互換）</summary>

<a id="lerobot-training"></a>

[lerobot-training](reference/TRAINING.ja.md#lerobot-training)

<a id="lerobot-v3で学習する"></a>

[lerobot-v3で学習する](reference/TRAINING.ja.md#lerobot-v3で学習する)

<a id="lerobot-evaluation"></a>

[lerobot-evaluation](reference/TRAINING.ja.md#lerobot-evaluation)

<a id="hdf5-pusht-training"></a>

[hdf5-pusht-training](reference/TRAINING.ja.md#hdf5-pusht-training)

<a id="新しいpusht経路公式ライブラリへ委託2026-09-09"></a>

[新しいpusht経路公式ライブラリへ委託2026-09-09](reference/TRAINING.ja.md#新しいpusht経路公式ライブラリへ委託2026-09-09)

<a id="ログ保存再開"></a>

[ログ保存再開](reference/TRAINING.ja.md#ログ保存再開)

<a id="hdf5-pusht-evaluation"></a>

[hdf5-pusht-evaluation](reference/TRAINING.ja.md#hdf5-pusht-evaluation)

<a id="pin-memory-tradeoffs"></a>

[pin-memory-tradeoffs](reference/TRAINING.ja.md#pin-memory-tradeoffs)

<a id="hdf5-libero-training"></a>

[hdf5-libero-training](reference/TRAINING.ja.md#hdf5-libero-training)

<a id="hdf5-libero-evaluation"></a>

[hdf5-libero-evaluation](reference/TRAINING.ja.md#hdf5-libero-evaluation)

</details>
