# BCの保存再開契約・native接続の詳細

BCは補助評価です。現在の操作入口は[BCガイド](BC_GUIDE.ja.md)、実施状況は[実験一覧](../reports/EXPERIMENTS.ja.md)です。

> 詳細参照資料：2026-09-15の整理時点の手順を保持しています。新規作業は[学習ガイド](../TRAINING.ja.md)・[評価ガイド](../EVALUATION.ja.md)から選んでください。過去runの出力名を新規実行に流用しないでください。

学習済みRaw／TC／BTのLIBERO世界モデルからViTだけを取り出して固定し、現在の2カメラ画像とタスクIDから行動列を出す方策を追加学習します。CEMとは別の評価経路です。タスクIDは方策にだけ入力し、世界モデルの学習目的・SIGReg・BTを変更しません。

2026-09-11：学習、保存・再開、推論、native LIBERO評価、動画・参考画像付きレポートを実装。[検証範囲とBT本学習の起動記録](../reports/LIBERO_BC.ja.md)。BCの本学習・成功率比較は未実施です。

2026-09-14：既存実装をTC-LeWM v3と再照合し、現在のBT100k encoderでGPU短期学習・厳密再開・全10タスクの行動生成を確認。評価のsettling既定を公式LIBEROの5回に統一しました。[今回の確認](../reports/LIBERO_BC_BT100K.ja.md)。BTで学習したencoderを使う下流BC評価であり、世界モデルの正則化をTCへ変更する処理ではありません。

2026-09-14 23:28 JST追記：ユーザー依頼で上記BT100k由来のBC本学習（40,000更新）を開始しました。runは`output/libero10/bc_bt100k_40k_s3072/`。以降、同じ出力先の新規起動は行わず、稼働状況と固定ソースを[運用記録](../reports/LIBERO_BC_BT100K.ja.md)で確認してください。完了・成功率はまだ未確認です。

2026-09-15追記：上記BCの40,000更新完了・終了コード0とencoder凍結を確認し、ユーザー依頼で全10タスク×50初期状態のnative評価を開始しました。[完了確認と評価記録](../reports/LIBERO_BC_BT100K.ja.md)。評価結果はまだ未確定です。

2026-09-15条件整備：[TC-LeWM条件の照合とOpenVLA再生成入口](../reports/TCLEWM_ALIGNMENT.ja.md)を追加。その後、ユーザー承認のローカル設定で再生成→BT1万更新→BC4万更新のジョブを開始しました。実行段階は運用記録を参照。未公開のBC詳細や分割は公式一致と扱いません。

## 論文との関係

[TC-LeWM v3、付録A.1](https://arxiv.org/html/2607.26924v3#A1.SS1)を参照したローカル実装です。2026-09-14に再確認した[著者ページ](https://ryuuchou17.github.io/tclewm/)は「Code Coming soon」であり、公式コード移植や成功率再現とは呼びません。

| 項目 | 実装 |
|---|---|
| 画像特徴 | 凍結ViT、各カメラのCLS＋4×4平均プールpatch、合計34 token |
| 方策 | 行動のself-attention、画像へのcross-attention、時刻とタスクIDによるLayerNormの変調 |
| 学習目標 | Gaussian noiseから正規化デモ行動列へのflow matching |
| 行動・推論 | 既定8個の連続native 7次元行動、10 Euler step、訓練統計で逆変換して[-1,1]に制限 |
| 論文に記載の学習設定 | batch 256、AdamW、LR 2e-4、10タスク共同方策40,000更新 |
| ローカル設定 | 幅256、4ブロック、8 heads、MLP比4、dropoutなし、weight decay .01、一定LR、gradient clip 1、FP32 |
| その他のローカル条件 | augmentationなし、不完全な末尾chunkは除外、既存manifestのデモ分割と固定validation開始位置 |

ネットワーク幅・深さ・最適化の詳細・評価時の実行chunk長など、論文だけでは確定しない条件はローカル設定です。評価既定は8行動を実行してから次の観測で生成し、最大520行動です。

世界モデルのLIBERO学習は4行動ごとの画像遷移を使いますが、BCの教師は**毎制御時刻の連続した8行動**です。8×4行動ではありません。未来画像・Goal・報酬・内部状態は方策に入力しません。ViTはevalモードで固定し、projector・潜在予測器・BT変換はBC推論に使いません。

## 入力契約

- 信頼済みの2カメラLIBERO `*_object.ckpt`と、同じディレクトリの`config.json`が必要です。PushT checkpointは非対応です。
- 世界モデルと同じtrain／validation／testデモ分割・カメラ順・行動統計を使います。chunkはデモ境界を越えません。
- 既定のデータ確認はサイズ・mtimeのみ。全量再hashは`--verify-data`時だけです。
- 移転前のcheckpointには`--world-model-manifest 保存してある元のmanifest.json`を指定します。元のhash、現行の分割・前処理・記録済みデータhashを照合します。過去manifestを編集して照合を回避しません。
- 方策にタスク名の順序を保存し、native task IDから名前を経由してembedding番号を引きます。HDF5の並び順とnative task IDを同一と仮定しません。

## 学習と再開

リポジトリ直下で、保存完了済みのcheckpointを指定します。下は今回確認済みのBT100,000更新checkpointです。このCLI単体はBCだけを扱います。今回承認された逐次ジョブは世界モデル完了後にBCを起動します（[実行記録](../reports/TCLEWM_ALIGNMENT.ja.md)）。

```bash
export LIBERO_WM=output/libero10/bt_no_pin_100k_s3072/step_100000_object.ckpt
test -f "$LIBERO_WM"
.venv/bin/python -m mylewm.training.train_libero_bc \
  --checkpoint "$LIBERO_WM" \
  --manifest output/manifests/libero10/manifest.json \
  --output output/libero10/bc_bt100k_40k_s3072 \
  --steps 40000 --batch-size 256 --workers 4 --seed 3072
```

既定はdry-runで、重み読込・学習・output作成はしません。明示的に学習する場合だけ`--execute`を追加します。Raw／TCもcheckpointとoutput以外の方策条件を揃えます。

短期確認は別outputに`--steps 40000 --stop-after 10 --batch-size 2 --workers 0 --device cpu --execute`を指定できます。10更新で`status.json=paused`となり、全更新予算を維持します。同じ設定に`--resume --execute`を付け、`--stop-after`を外せば再開します。batch等を変えて同じrunを再開することはできません。

| 保存物 | 内容 |
|---|---|
| `config.json` | 元モデル・manifest・ソースhash・依存版・学習条件 |
| `step_N_bc.pt` | ViT＋方策、タスク名、行動統計、由来を含む推論用state dict |
| `resume.pt` | 方策・optimizer・乱数・完了更新数を含む信頼済み再開状態 |
| `metrics.jsonl` | flow loss・勾配・固定validationのflow loss |
| `status.json` | running／paused／succeeded／failed |
| `completed.json` | 全更新予算を終了した場合だけ作成 |

ViTのstate hashを保存ごとに照合します。再開はソース・依存・元モデル・データ・条件が一致する場合だけです。validation lossは制御成功率ではありません。

## 環境評価

[LIBERO環境とOSMesa画像監査](../EVALUATION.ja.md#libero)が必要です。`libero`依存グループにはnative環境に必要な`bddl`・`easydict`・`future`・`gym`も固定しています。稼働中の学習環境へ`uv sync`をかけないでください。

今回のPCでは不足分だけを`output/libero10/bc_implementation_check/runtime/`へ隔離して導入しました。この環境を使う場合は下記のPYTHONPATHを付けます。別PCでは対応する隔離環境を用意してください。

```bash
export PYTHONPATH="$PWD/output/libero10/bc_implementation_check/runtime${PYTHONPATH:+:$PYTHONPATH}"
bash scripts/run_libero.sh -m mylewm.evaluation.evaluate_libero_bc \
  --checkpoint output/libero10/bc_bt100k_40k_s3072/step_40000_bc.pt \
  --manifest output/manifests/libero10/manifest.json \
  --output output/libero10/eval_bc_bt100k_40k_s3072 \
  --task-ids 0 1 2 3 4 5 6 7 8 9 --episodes 50 \
  --budget 520 --execute-actions 8 --settling-steps 5 --device cuda:0 \
  --render-audit-dir output/libero10/eval_bt100k_20260914/render_audit
```

ここも既定dry-runです。`--execute`だけがモデル・環境を起動します。別outputで`--task-ids 0 --episodes 1 --budget 9`とすれば、8行動実行後の再生成を含む短い接続確認ができます。短期checkpointでの成否を性能値とは扱いません。

公式LIBEROと同じく、通常reset→固定init state復元→ゼロ行動5回で初期化します。旧ローカル条件の10回は`--settling-steps 10`で明示できます。回数を評価configとinitial_historyへ記録し、異なる条件の評価を同条件比較しません。BCは現在画像1組だけを使うため、世界モデル診断の追加8行動や複製履歴は不要です。

環境の`env.check_success()`で判定します。`episodes.jsonl`にnative task ID・タスク名・初期状態番号・1からの試行番号・成否・実行step・方策生成時間、`results.txt`に一覧、`summary.json`に各タスクと平均の成功率を保存します。各試行のMP4、初期／最終画像、参考の成功デモ画像を`viewer/index.html`に表示します。参考画像は**方策には渡しません**。物体配置も異なり得るため、参考画像との完全一致は成功条件ではありません。

映像はnative制御20Hzに合わせた20fpsで、生成の計算待ち時間を含みません。終了コード0、`status.json=succeeded`、結果とviewerを確認してください。

MP4は`viewer/videos/trial_001.mp4`から試行順に一部だけ保存し、再生に同じファイルを使います。viewerを別の場所へ再生成する場合も、評価元の動画を参照し、コピーを増やしません。

## 比較

Raw／TC／BTでは同じ世界モデル訓練予算・データ分割・BC初期seed・構成・更新予算・評価初期状態を揃えます。`compare_libero.compare()`はBC同士の条件・checkpoint更新数・初期状態を照合し、BCとCEMの混在を拒否します。複数学習seedと各タスクの下位成績は別途検証が必要です。学習済みタスクID集合の共有方策であり、自然言語理解・未知タスク名への汎化・実機I/Oは未対応です。
