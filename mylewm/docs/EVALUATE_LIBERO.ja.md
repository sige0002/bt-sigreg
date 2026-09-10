# LIBERO-10：学習から環境評価・可視UIまで

更新日：2026-09-10。新規学習・環境評価は明示依頼時だけ実行し、自動実行・定期監視はしない。

## 比較の位置付け

上流の[公式LeWM](https://github.com/lucas-maes/le-wm)とローカル`lewm/`には、LIBERO用の学習config・評価config・配布checkpointがない。従ってLIBEROの`--mode raw`は公式LeWMのE/A/F・一段予測・全次元SIGRegを使う**ローカルの公式コア比較ベースライン**であり、公式提供のLIBERO実装／結果とは呼ばない。`--mode bt`は同じ基盤に学習専用BTを追加する提案側である。

Raw/BTは同じ新規manifest、seed、batch、workers、更新数、optimizer、固定初期状態、CEM条件で比較する。LIBERO-10の10タスクを一つの共有モデルで学習する。短期smokeのlossや1初期状態の成否は性能比較ではない。

## 1. 分割を作る

データを複製せず、分割とtrain-only行動統計をmanifestへ記録する。既にmanifestがあれば変更・再作成しない。

```bash
cd /home/USER/bt-sigreg
ls .cache/libero-datasets/libero_10/*.hdf5
uv run python mylewm/train_libero.py prepare \
  --dataset .cache/libero-datasets/libero_10 \
  --manifest output/manifests/libero10/manifest.json
```

正常ならtrain 400 demonstrations、validation 50、test 50と表示され、隣に`files.json`が作られる。

## 2. OSMesa画像監査

このGB10では隔離したOSMesa wrapperを使う。評価前に公式LIBERO task ID 0〜9の全てを監査する。`--task-id`はLIBERO task IDでありHDF5のソート番号ではない。出力は必ず対応する`task_N`に保存する。

```bash
for task_id in {0..9}; do
  bash mylewm/run_libero.sh mylewm/tools/audit_libero_images.py \
    --task-id "$task_id" \
    --output "output/libero10/render_audit/task_${task_id}"
done
```

各`report.json`で`passed: true`、`render_backend: osmesa`、native MAEが縦反転MAEより小さいことを確認する。評価器はtask名、MuJoCo版、backendが一致しない監査を拒否する。

## 3. Raw／BTを新規学習する

先に短期接続確認をする。下はRaw例で、BTは`--mode bt`だけを変え、その他をそろえる。短期checkpointを本学習へ延長しない。

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run python mylewm/train_libero.py train \
  --mode raw --manifest output/manifests/libero10/manifest.json \
  --output output/libero10/raw_smoke_s3072 \
  --steps 100 --batch-size 16 --workers 0 --seed 3072 \
  --warmup-steps 10 --lr 5e-5 --min-lr 0 \
  --save-every 50 --diagnostics-every 25 --deterministic
```

本比較は新規outputで逐次実行する。RawとBTを同時起動しない。

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run python mylewm/train_libero.py train \
  --mode raw --manifest output/manifests/libero10/manifest.json \
  --output output/libero10/raw_train_s3072 \
  --steps 100000 --batch-size 128 --workers 4 --seed 3072 \
  --warmup-steps 500 --lr 5e-5 --min-lr 0 \
  --save-every 5000 --diagnostics-every 1000 --deterministic
```

BTは`--mode bt --output output/libero10/bt_train_s3072`に変えるだけにする。各runの`config.json`の`initial_model_sha256`を照合する。最終評価へ進めるのは`completed.json`が`{"step": 100000, "state": "completed", "recipe": "controlled_comparison"}`であり、`step_100000_object.ckpt`がある場合だけである。`resume.pt`は再開専用で評価に渡さない。

## 4. 実環境CEM評価

既定はdry-runで、checkpointをロードせず環境も動かさずoutputも作らない。下はRaw最終checkpointの固定10タスク・各50初期状態評価である。BTもcheckpointとoutputだけを変え、同じmanifest・監査・CEM条件を使う。

```bash
# 設定・監査・入力だけを確認する。
bash mylewm/run_libero.sh mylewm/tools/evaluate_libero.py \
  --checkpoint output/libero10/raw_train_s3072/step_100000_object.ckpt \
  --manifest output/manifests/libero10/manifest.json \
  --render-audit-dir output/libero10/render_audit \
  --output output/libero10/eval_raw_s3072_confirm50

# dry-runを確認した後だけ環境を動かす。
bash mylewm/run_libero.sh mylewm/tools/evaluate_libero.py \
  --checkpoint output/libero10/raw_train_s3072/step_100000_object.ckpt \
  --manifest output/manifests/libero10/manifest.json \
  --render-audit-dir output/libero10/render_audit \
  --output output/libero10/eval_raw_s3072_confirm50 --execute
```

評価中に学習を併走しない。成功は`env.check_success()`だけで決め、デモ状態距離やBCスコアと混ぜない。終了時は`status.json=succeeded`、`summary.json`、10×50行の`episodes.jsonl`、全`taskN_initM.npz`を確認する。途中終了は未測定であり0%ではない。

## 5. ブラウザの可視UI

保存済み評価から、初期・goal・最終の2カメラ（agent view | wrist view）を静的HTMLにまとめる。UIは記録を読むだけで、CEM・環境・checkpointを再実行しない。

```bash
uv run python mylewm/tools/build_libero_ui.py \
  --evaluation output/libero10/eval_raw_s3072_confirm50 \
  --output output/libero10/ui_raw_s3072_confirm50

# ローカルブラウザで http://127.0.0.1:8000 を開く。Ctrl-Cは表示だけを停止する。
uv run python -m http.server 8000 --directory output/libero10/ui_raw_s3072_confirm50
```

`index.html`を直接開くこともできる。画面のsuccess/failureは固定実行結果であり、goalへの見た目の近さによる判定ではない。

## 実行確認した範囲

2026-09-10に、新manifest作成、Raw CPU 2更新（`completed.json`と`step_2_object.ckpt`）、task ID 0のOSMesa画像監査、RawとBT短期checkpointの各1初期状態・budget 1 CEM評価、dry-run、`status.json=succeeded`、HTML UI生成を実行した。これは接続確認のみであり、100,000更新・全10監査・各50初期状態・Raw対BT性能比較は未実施である。
