# LIBEROのBC：凍結画像encoderから行動を学ぶ

[文書一覧](README.md) · [学習ガイド](TRAINING.ja.md) · [評価ガイド](EVALUATION.ja.md)

BC（Behavior Cloning、行動模倣）は、成功デモの画像に対応する行動を学ぶ方策です。このリポジトリでは、世界モデルで学習したViTを凍結し、タスクID付きのflow matching方策を追加学習します。世界モデルの正則化方式はRaw／TC／BTのままです。

## 今回の構成

| 項目 | 設定 |
|---|---|
| 画像 | 現在の外部・手首カメラの実測画像1組 |
| encoder | 世界モデルのViT。重み固定、evalモード |
| 画像特徴 | 各カメラCLS＋4×4 pool patch、計34token |
| 行動head | 幅256・4層・8 heads、画像へのcross-attention |
| 条件 | タスクIDとflow時刻 |
| 教師 | 8個の連続native行動、各7次元 |
| 学習 | 40,000更新、batch256、AdamW、LR2e-4、weight decay0.01 |
| 推論 | 10 Euler step、学習統計で逆正規化、[-1,1]に制限 |

幅・層数等は採用したローカル設定です。TC-LeWM論文の公開条件を参考にしていますが、公式コードの移植ではありません。[条件の照合と採用理由](reports/TCLEWM_ALIGNMENT.ja.md)

## 学習・評価の入口

新データから世界モデルも学ぶ場合は[学習ガイド](TRAINING.ja.md)の順序で実行します。BCだけを追加する場合も、encoderを学んだときと同じmanifestが必要です。

```bash
uv run --no-sync python -m mylewm.training.train_libero_bc --help
bash scripts/run_libero.sh -m mylewm.evaluation.evaluate_libero_bc --help
```

両CLIとも通常の実行はdry-runで、`--execute`を付けると学習・環境評価を開始します。入力には、学習時は`*_object.ckpt`、BC評価時は`*_bc.pt`を使います。具体的な引数は[学習例](TRAINING.ja.md#5-凍結encoder上のbcを40000更新する)・[評価例](EVALUATION.ja.md#libero)に集約しています。

## 入力と再開の契約

- 世界モデルとBCでtrain／validation／testのデモ分割と行動統計を共有する。
- BCは毎時刻の8行動を使う。世界モデルの4行動ごとの間引きと組み合わせて32行動にしない。
- Goal・未来画像・報酬・内部状態はBCへ入力しない。projector・予測器・BT写像もBC推論では使わない。
- native task IDとデータの順序は名前で対応付ける。
- `step_N_bc.pt`はencoderと方策、`resume.pt`はoptimizer・乱数等も含む。再開時は構成・ソース・環境・データhashを照合する。

validationでは保持デモごとの固定開始位置を使います。検証lossが低いcheckpointが、最も高い環境成功率を持つとは限りません。現行の学習コードは予算終了時のcheckpointも保存します。

保存移転、厳密再開、旧環境でのCLI例は[詳細参照](reference/BEHAVIOR_CLONING.ja.md)へ。旧BT100kからのBC学習と途中評価の記録は[実験レポート](reports/LIBERO_BC_BT100K.ja.md)に残しています。
