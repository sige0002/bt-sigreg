# PushT Raw LeWM／BTの同条件140k比較

2026-09-16、ユーザー依頼でRawとBTを各140,000更新、同一GPUで同時学習する。旧checkpointの継ぎ足しではなく、共通初期値からの新規比較。環境評価は今回の学習起動に含めず、自動予約しない。

## 旧100kを継続しない理由

`.cache/stable-wm/pusht/bt_spectral_v2_100k_s3072/`の100k object checkpoint・resume状態とconfigは残っている。`controlled_training_v2`の旧経路で、LRの終了点は100,000更新。新経路・140,000更新終了のLRで学習するRawとの主比較には使えない。40k継続すれば別の継続学習実験になる。既存重み・記録は参考評価用に保持する。

停止中の`bt_compiled_100k_s3072`も今回再開しない。公式配布checkpointとの参考比較と、今回再学習したRawとの方式比較を分ける。

## 固定する条件

| 項目 | 今回の設定 |
|---|---|
| データ | 公開PushT HDF5、既存ファイルを読み取り専用で使用 |
| クリップ | SWMの4フレーム・frameskip5、全1,981,721件 |
| 分割 | SPT random_splitの90%／10%、seed3072、train1,783,549・validation198,172 |
| バッチ・順序 | batch128、drop_last、単一GPU、勾配蓄積なし、同じepoch shuffle |
| 予算 | 両方式140,000更新。10 epochs＝139,330より670更新多い |
| LR | 5e-5、warmup500、140,000終了のcosine、最小0。step単位で同じ進行 |
| モデル | 共通LeWM E/P/A/F、初期値hash一致を確認。BTだけCayley v2 Tを追加 |
| 正則化 | 係数0.09、射影1024。BTはdepth2・hidden192・kappa0.2 |
| 実行 | bf16-mixed、workers4、pin memory無効、encoder compile無効、seed3072 |
| 保存・validation | 20,000更新ごと。validationは共通の全198,172クリップ |

**クリップ分割なので、隣接窓の画像がtrainとvalidationで重なる。** validation lossを未知episodeへの汎化成績とは扱わない。環境成功率の正式比較では、ケースの由来・Goal・物理初期状態・CEM条件を別途固定する。

正規化統計は訓練クリップが参照する物理行動行の集合から計算し、同一行を重複カウントしない。非有限行を除外してsample stdを使う。公開trainerの「分割前の全体統計」とは区別し、双方に同じtrain由来統計を使用する。公式のモデル・損失を使った同条件再学習であり、公式trainerの完全再現とは呼ばない。

## 実装と再現

`mylewm.data.prepare_pusht_comparison`が専用manifestを作る。旧episode分割manifestは変更せず、`split_kind=random_clip_90_10`だけ新しい経路へ分岐する。train／validationの順序付きindex hash、nativeクリップ順hashを保存し、起動時に一致を検証する。初回prepareでデータ全量SHA-256、通常起動はサイズ・mtime照合。

```bash
.venv/bin/python -m mylewm.data.prepare_pusht_comparison \
  --dataset .cache/stable-wm/datasets/pusht_expert_train.h5 \
  --manifest output/manifests/pusht/clip90_140k_s3072/manifest.json --seed 3072
```

両方式とも次の共通引数を使い、`--mode raw`／`--mode bt`と未使用出力だけを変える。実起動はソースを固定したコピーから行う。以下は再起動指示ではなく条件の記録。

```bash
.venv/bin/python -m mylewm.training.train \
  --manifest output/manifests/pusht/clip90_140k_s3072/manifest.json \
  --output output/pusht/raw_clip90_140k_s3072 --mode raw \
  --steps 140000 --batch-size 128 --workers 4 --no-pin-memory \
  --lr 5e-5 --warmup-steps 500 --save-every 20000 --val-every 20000 \
  --seed 3072 --execute
```

ソース・環境・manifest・予算の照合を外さない。再開時は同じ固定ソースと環境、同一recipe、Lightning checkpointと未使用出力を使う。旧`resume.pt`は新経路へ渡さない。

## 検証と実行記録

分割・train統計・hash不一致拒否・既存学習／保存再開の関連回帰は21件合格（15.73秒）。実データprepareで上表の件数と13,933バッチ／epochを確認した。起動・GPU接続確認は以下に追記する。ここに書かれた予定は完了した学習成績ではない。

追加のデータ契約・検証回帰は11件合格（0.59秒）。実データGPU併走はRaw／BTとも4更新・終了コード0・completed.json一致。共通初期値hashは`1c1962d31c04bb65617d468686d5f23efe777058988c032408429d64dc02bc7d`。同時実行中のGPU表示は各約13.2 GiB、利用可能RAM約68 GiB。これは短期観測であり長時間完走の証明ではない。証拠は`output/pusht/raw_bt140k_preflight_20260916/`。
