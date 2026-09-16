# 3. 学習する

[前：実装](IMPLEMENTATION.ja.md) · [次：評価](EVALUATION.ja.md)

学習では、記録済みの画像と行動から「この行動で状態がどう変わるか」を覚えさせます。まずPushTを例に、データ準備からモデル保存までを説明します。

## 用語

- **ステップ（更新）**：1バッチのデータでモデルの重みを1回更新すること。
- **エポック**：学習データを一巡すること。ステップ数はデータ量とバッチサイズで変わります。
- **manifest**：使うデータ、学習・検証の分割、行動の正規化統計を固定するファイル。
- **checkpoint**：保存したモデル。評価用と学習再開用があります。

<a id="setup"></a>
## 1. 環境とデータを用意する

コマンドはリポジトリ直下で実行します。初回の環境構築は[インストール手順](../README.md#コードの構成と起動)を参照してください。準備済みの環境では`.venv/bin/python`を使います。

PushTは、T字の物体を押して目標の位置・向きへ動かす課題です。入力には公開LeWM用のPushTデータ（画像・行動を保存したHDF5）を使います。[取得方法](reference/DATASET_RECIPES.ja.md#hdf5-pusht-training)

## 2. 学習・検証用に分ける

次の例はクリップ（連続する短い時系列）を90%／10%に分け、共通のmanifestを作ります。初回はデータ全体を読み、識別用のhashも計算します。

```bash
.venv/bin/python -m mylewm.data.prepare_pusht_comparison \
  --dataset .cache/stable-wm/datasets/pusht_expert_train.h5 \
  --manifest output/manifests/pusht/example/manifest.json --seed 3072
```

隣接クリップでは画像が重なるため、この検証集合は「初めて見る軌道」での評価ではありません。制御能力は次章の環境評価で別に確認します。

## 3. RawとBTを学習する

RawはLeWMのモデル・損失を使う比較対象、BTは正則化に学習専用の変換Tを加えた方式です。同じmanifestと設定を使い、方式と出力先だけを変えます。

以下はRawの**設定確認だけ**を行う例です。

```bash
.venv/bin/python -m mylewm.training.train \
  --manifest output/manifests/pusht/example/manifest.json \
  --output output/pusht/raw_example --mode raw \
  --steps 140000 --batch-size 128 --seed 3072 \
  --lr 5e-5 --warmup-steps 500 --workers 4 --no-pin-memory \
  --save-every 20000 --val-every 20000
```

設定を確認したら同じコマンドに`--execute`を付けて学習を開始します。BTは`--mode bt`、出力先は`output/pusht/bt_example`に変えます。出力先は未使用の名前を指定してください。

この例では各14万更新で比較します。公開データをこの設定で分割した10エポックは139,330更新なので、厳密には同じではありません。両方式の初期値・データ順・学習率の変化も揃えます。[比較条件の補足](reference/COMPARISON_PROTOCOL.ja.md)

## 4. 保存結果を確認する

| ファイル | 用途 |
|---|---|
| `config.json` | 実際に使った設定 |
| `metrics/version_0/metrics.csv` | 更新数と学習・検証の誤差。`update`が実更新数 |
| `step_N_object.ckpt` | 評価に使う世界モデル |
| `step_N.ckpt`／`last.ckpt` | optimizerなども含む学習再開用 |
| `completed.json` | 指定した更新数まで到達した記録 |

終了コード・完了記録・最終checkpointを確認します。再開には、開始時と同じソース・環境・設定、再開用checkpoint、未使用の出力先が必要です。評価用objectだけでは学習を再開できません。

学習誤差が下がっても、操作に成功するとは限りません。[次の章で環境評価を行います](EVALUATION.ja.md)。LIBERO・OGBenchなどは、必要になった時点で[補足資料](reference/README.md)を参照してください。
