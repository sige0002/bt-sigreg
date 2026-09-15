# LIBERO BT100kの初回CEM評価

実施日：2026-09-14。ユーザーの評価依頼に基づく、学習済み世界モデルの環境評価です。追加学習やBC方策の学習ではありません。

## 対象と条件

| 項目 | 条件 |
|---|---|
| 世界モデル | `output/libero10/bt_no_pin_100k_s3072/step_100000_object.ckpt` |
| 学習 | BT v2、LIBERO-10共有モデル、100,000更新、seed3072 |
| ケース | native task ID 0〜9、各init ID 0の1試行、計10試行 |
| 制御 | 世界モデル＋CEM、最終予測zとGoalの潜在MSE |
| CEM | horizon 8、候補128、elite16、反復5、seed42 |
| 実行 | 4行動chunkごとに再計画、1試行最大520行動 |
| 成功条件 | native `env.check_success()` |
| Goal | 各タスクのheld-out testから最初の成功デモの最終2カメラ画像 |
| 初期履歴 | 既存評価器どおり、10回のゼロ行動後の観測を3回繰り返し、過去行動をゼロに設定 |
| 実行環境 | GB10、PyTorch 2.9.1+cu130、Transformers 4.57.6、MuJoCo 3.3.7、robosuite 1.4.0、OSMesa |
| データ確認 | 存在・サイズ・mtimeと全タスクの成功Goalの存在を確認。全量SHA-256の再走査は未実施 |

各タスク1試行は初回確認であり、各50初期状態の本比較ではありません。単一学習seed・単一初期状態なので、タスクごとの成功確率や方式の優位性を確定できません。Raw／TCとの同予算比較、尺度・アフィン対照は含みません。初期履歴の複製を実測された3時刻の履歴とは扱いません。

## 描画と起動確認

最初の起動は共有`.venv`に`bddl`がなく、環境操作前に終了コード1で停止しました。ログを`output/libero10/eval_bt100k_20260914/audit_task_0.log`に保持しています。

以前の検証用`output/libero10/bc_implementation_check/runtime/`にあるbddl 1.0.1、easydict 1.9、future 1.0.0、gym 0.25.2を`PYTHONPATH`で参照して解消しました。描画には既存`.cache/libero-runtime/`のMuJoCo 3.3.7とOSMesa wrapperを使用し、共有環境の依存同期・他プロセスの停止はしていません。

全10タスクの新規描画監査が終了コード0で完了しました。2カメラ×3時刻×10タスクの60比較すべてでnative MAEが閾値10以下かつ縦反転MAEより小さく、native MAEの範囲は1.337〜7.896でした。監査は`output/libero10/eval_bt100k_20260914/render_audit/task_0`〜`task_9`へ保存しました。既存監査は上書きしていません。

評価器のdry-runは終了コード0。全タスクの監査名・MuJoCo版・backend、checkpoint・manifestの読込条件を確認しました。

## 実行コマンド

```bash
UV_PROJECT_ENVIRONMENT=/home/sadasue/bt-sigreg/.venv \
PYTHONPATH=/home/sadasue/bt-sigreg/output/libero10/bc_implementation_check/runtime \
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
bash scripts/run_libero.sh -m mylewm.evaluation.evaluate_libero \
  --checkpoint output/libero10/bt_no_pin_100k_s3072/step_100000_object.ckpt \
  --manifest output/manifests/libero10/manifest.json \
  --render-audit-dir output/libero10/eval_bt100k_20260914/render_audit \
  --output output/libero10/eval_bt100k_20260914/cem_10 \
  --episodes 1 --offset 0 --budget 520 \
  --horizon 8 --samples 128 --iterations 5 --seed 42 --device cuda --execute
```

標準出力・標準エラーは同じ親ディレクトリの`console.log`へ保存しています。

## 結果

成功は **0/10、タスクmacro成功率0%** でした。全試行が520行動の上限まで完走し、実行行動は合計5,200件、試行ループの合計時間は708.38秒（約11分48秒）です。描画監査・モデル読込・環境の初期化時間はこの合計に含みません。

| native task ID | タスク概要 | 成功数／試行数 |
|---|---|---|
| 0 | スープとトマトソースをかごに入れる | 0/1 |
| 1 | クリームチーズとバターをかごに入れる | 0/1 |
| 2 | コンロをつけてモカポットを置く | 0/1 |
| 3 | 黒いボウルを下の引き出しに入れて閉める | 0/1 |
| 4 | 白いマグを左、黄色いマグを右の皿に置く | 0/1 |
| 5 | 本をキャディーの奥の区画に入れる | 0/1 |
| 6 | 白いマグを皿に置き、プリンをその右に置く | 0/1 |
| 7 | スープとクリームチーズをかごに入れる | 0/1 |
| 8 | 2個のモカポットをコンロに置く | 0/1 |
| 9 | マグを電子レンジに入れて閉める | 0/1 |

評価プロセスは終了コード0、`status.json`は`succeeded`で終了しました。ここで`succeeded`は評価処理の完了を表し、タスク成功率は別フィールドの0%です。終了後に対象PIDが存在せず、GPU compute processも残っていないことを確認しました。

`verify_results.py`を一度実行して次を照合し、終了コード0、`verification.json`の`verified: true`を確認しました。

- 重複・欠落のないnative task 0〜9、init 0の10ケース。
- `summary.json`、`status.json`とケースごとの成功判定・件数。
- 全10個のNPZ内に520×7の有限行動があり、全成分が[-1, 1]内。
- 各NPZの開始・Goal・終了画像が2×128×128×3のuint8。
- 開始状態・開始画像・Goal画像のSHA-256が各試行記録と一致。
- 評価終了後のcheckpoint・manifestのSHA-256が開始時の記録と一致。

開始と終了の画像は全試行で変化していました。ただし画像変化や有限の行動だけで、遷移予測やGoal距離が正しいと保証するものではありません。

今回のモデル＋CEM設定では、10万更新後もタスク成功を確認できませんでした。表現に必要な情報があるか、長い予測が正しいか、Goal距離が適切か、探索が十分かは未切り分けです。Tの倍率効果が失敗原因だと結論付ける証拠でもありません。

出力は`output/libero10/eval_bt100k_20260914/`にあります。`cem_10/`が評価記録、`verification.json`が照合結果、`ui/index.html`が10試行の開始・Goal・終了画像一覧です。UI生成も終了コード0でした。動画は生成していません。

## 識別情報

評価ソースはGit `22c5c51a17958a81ae354a8de988d415b30a87b5`と一致します。開始時の作業ツリーには前ターンのvalidation末尾バッチ修正がありましたが、評価器・CEM・世界モデルの推論コードは変更していません。

`algorithms/libero_model.py`、`lewm/jepa.py`、`lewm/module.py`、`lewm/config/train/model/lewm.yaml`の実測SHA-256が、対象runの学習configに保存された値と一致することも確認しました。

| 対象 | SHA-256 |
|---|---|
| checkpoint | `f7581ab6b98f2396959ce077413db23e1af378cc839dc28f7f2c2a74a67b52cc` |
| manifest | `f7a3e92718512189dba86cee40a26d3b1bacd13da8b11329a1d92811ce4e4f35` |
| `evaluation/evaluate_libero.py` | `57f99761cbe10120bc2eb95b3999d467d0cbc817f3b409d77cb154e3d7bda18d` |
| `environments/libero_planner.py` | `1219acd7cf5e93ed797566f32f6e3943bd0363d1665989af18c52e01dd7b3fbc` |
| `scripts/run_libero.sh` | `de3b29db11fb01b224fed37aec5e7e9fd02f40dfc16190e9fb03821c0a885ecf` |
