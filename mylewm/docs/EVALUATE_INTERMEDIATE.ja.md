# 学習中の途中checkpointを評価する（PushT／LIBERO）

学習済みの`step_N_object.ckpt`は、本学習の終了前でも評価できる。途中評価には`completed.json`は不要。`last.ckpt`や`resume.pt`は再開用なので渡さない。以下は自分で作成した信頼済みcheckpointを前提とする。

## 実行前の確認

`nvidia-smi -L`と`nvidia-smi`で学習GPUと空いている評価GPUを確認する。以下の`CUDA_VISIBLE_DEVICES=1`は物理GPU 1で評価する例で、実際の空きGPU番号またはUUIDに変更する。この指定で評価プロセス内では選んだGPUが`cuda:0`になる。別端末で実行し、学習側の設定は変えない。同じGPUでの同時実行は速度低下・OOMのおそれがあるため、この手順では別GPUを使う。GPUが1台なら学習終了後に評価するか、再開可能なcheckpointを確保して学習を正常に中断してから評価する。

checkpointの保存完了と、学習が次の更新へ進んだことをログで確認する。ファイルが存在するだけでは書込み終了の証拠にならない。特にPushTの保存は実行環境によって直接書込みになる。以下の50,000は例で、保存完了した更新数へ変更する。入力checkpointは評価終了まで移動・削除・上書きしない。

評価環境はcheckpointと互換な依存版を事前に準備する。学習中の共有`.venv`に`uv sync`を実行しない。下記の`UV_NO_SYNC=1`は起動時の自動同期も抑えるため、準備済み環境が必要。別環境を使う場合は`UV_PROJECT_ENVIRONMENT`も設定する。別GPUでもCPU・ディスクは共有するため、学習速度への影響はあり得る。

## PushT

HDF5はmanifest内の任意パスを使う。移転先を指定する場合は以下の両コマンドへ`--dataset /absolute/path/to/pusht_expert_train.h5`を追加する。元manifestを書き換えず、サイズを確認し、`--verify-data`指定時は保存済みhash（存在する場合）とも照合する。

リポジトリ直下で、実際のrun名・保存済みmanifestに置き換える。各出力先は未使用名にする。

```bash
export PUSHT_EVAL_CKPT=output/pusht/spt_raw/step_50000_object.ckpt
export PUSHT_EVAL_OUT=output/pusht/eval_raw_step50000_seed42
CUDA_VISIBLE_DEVICES=1 UV_NO_SYNC=1 bash mylewm/tools/evaluate_pusht.sh \
  --checkpoint "$PUSHT_EVAL_CKPT" \
  --manifest output/manifests/pusht/manifest.json \
  --output "$PUSHT_EVAL_OUT" --num-eval 50 --seed 42

# 上のdry-runを確認してから実行する。
CUDA_VISIBLE_DEVICES=1 UV_NO_SYNC=1 bash mylewm/tools/evaluate_pusht.sh \
  --checkpoint "$PUSHT_EVAL_CKPT" \
  --manifest output/manifests/pusht/manifest.json \
  --output "$PUSHT_EVAL_OUT" --num-eval 50 --seed 42 --execute
```

BTはcheckpointと出力名をBT runに変更する。GB10のcache解放フラグは学習データの再読込を増やし得るため、この同時評価例では付けていない。評価も通常は全量hashを走査せず、`--verify-data`指定時だけ走査する。

`spt_raw`はSIGRegを直接適用するモデルで、公式配布checkpointとは別のローカル学習runである。70k同士を評価する場合は両方の入力を`step_70000_object.ckpt`にし、それぞれ未使用の出力名を指定する。学習ログの最新更新数でなく、実際に保存完了したcheckpointを選ぶ。

終了コード0、出力先の`status.json`が`succeeded`、`results.txt.json`を確認する。詳細は[PushT評価手順](EVALUATE_PUSHT.ja.md)の終了確認を参照する。

## LIBERO-10

[LIBERO評価手順](EVALUATE_LIBERO.ja.md)のOSMesa環境と対象タスクの画像監査を事前に準備する。`run_libero.sh`は現在GB10/aarch64用OSMesaパスを持つため、RTX/x86環境でそのまま動くとは限らない。そこでのOSMesa構築・監査を通してから使う。

```bash
export LIBERO_EVAL_CKPT=output/libero10/raw_train/step_50000_object.ckpt
export LIBERO_EVAL_OUT=output/libero10/eval_raw_step50000
CUDA_VISIBLE_DEVICES=1 UV_NO_SYNC=1 bash mylewm/run_libero.sh mylewm/tools/evaluate_libero.py \
  --checkpoint "$LIBERO_EVAL_CKPT" \
  --manifest output/manifests/libero10/manifest.json \
  --render-audit-dir output/libero10/render_audit \
  --output "$LIBERO_EVAL_OUT" --device cuda:0

# 上のdry-runを確認してから実行する。
CUDA_VISIBLE_DEVICES=1 UV_NO_SYNC=1 bash mylewm/run_libero.sh mylewm/tools/evaluate_libero.py \
  --checkpoint "$LIBERO_EVAL_CKPT" \
  --manifest output/manifests/libero10/manifest.json \
  --render-audit-dir output/libero10/render_audit \
  --output "$LIBERO_EVAL_OUT" --device cuda:0 --execute
```

終了コード0、`status.json=succeeded`、`summary.json`と`episodes.jsonl`を確認する。UI生成はLIBERO評価手順の`--evaluation`に今回の出力先を指定する。

## 結果の扱いと確認範囲

結果にはrun名と更新数を付け、「50,000更新時点の途中評価」等と記録する。比較は同じmanifest・ケース・seed・CEM予算で行う。繰り返し見たconfirm集合はモデル選択に影響するため、未使用の最終テスト成績とは呼ばない。

2026-09-10、ユーザー依頼で学習停止後にGB10上で新Raw 10k・70k、旧BT 15k・70kなどの途中checkpointを実評価し、終了コード・status・結果・動画を確認した。70k同士の固定50ケースはSIGReg 45/50、BT 47/50。[実測と比較上の留保](reports/PUSHT_ISSUE20.ja.md)を参照。別GPUでの同時学習・環境評価の完走、RTX側の該当BT 10k、LIBEROのこの途中評価手順の実環境完走は未検証であり、PushTの実績で代用しない。
