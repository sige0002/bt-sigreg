# PushT Raw LeWM／BTの同条件140k比較

2026-09-19確認。**Raw LeWM／BTとも共通初期値から140,000更新の学習と最終制御評価を完了した。最終成功率はRaw 89.5%、BT 92.0%（固定50ケース・CEM 4 seed平均）。** 終了コード0、学習完了記録、評価statusと成否配列、比較条件の一致を確認済み。

## 結果の要約

| 更新数 | Raw成功率 | BT成功率 | 評価したCEM seed |
|---|---:|---:|---|
| 20,000 | 64% | 84% | 42のみ |
| 40,000 | 76% | 84% | 42のみ |
| 60,000 | 88.5% | 85.5% | 42・7・43・123の平均 |
| 80,000 | 84.0% | 92.0% | 同上 |
| 100,000 | 92.5% | 93.0% | 同上 |
| 120,000 | 90.5% | 90.5% | 同上 |
| 140,000 | 89.5% | 92.0% | 同上 |

今回のPushTでは、BTへの変更による一貫した制御性能低下は観測されず、基本制御性能がRawと同程度であることを確認した。Rawの8万での低下は10万で回復しており、継続的な劣化とは言えない。低下の原因は未確定で、過学習や潜在距離の変化によるものと断定しない。

**PushTは基本的な予測・制御性能を確認する位置付けであり、BTのマルチタスク優位性を主張する実験ではない。** 学習seedは3072の1つだけ。同じ50ケースを4探索seedで繰り返しており、独立な200ケースや4学習seedではない。一般的な非劣化・統計的非劣性も確立していない。LIBERO等の共有世界モデルでのマルチタスク効果、単独タスク学習と共同学習の差は今後の検証課題。

以下に条件と証拠をまとめる。途中評価の節は各実施時点の記録であり、その中の「最終結果未確認」は当時の状態を示す。

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

## 本学習の開始

2026-09-16 09:19 JST、固定ソースGit `41693e1`から2本を同時起動。両serviceのactive/running、実GPUプロセス、CSVの更新進行を確認した。本番configの差はmodeのみで、共通初期値hash・manifest・データ順・LR等は一致。開始時の数更新を確認した段階で、140k完了ではない。

| 方式 | 出力 | user service |
|---|---|---|
| Raw | `output/pusht/raw_clip90_140k_s3072/` | `bt-pusht-raw-clip90-140k-s3072` |
| BT | `output/pusht/bt_clip90_140k_s3072/` | `bt-pusht-bt-clip90-140k-s3072` |

起動引数・ログ・方式別status・初回進捗証拠は`output/pusht/raw_bt140k_launch_20260916/`、固定ソースはその`source/`。guardは単一の学習子プロセスを待ち、終了コードとcompleted.jsonの140kを照合して状態を保存する。自動再起動なし（Restart=no）、追加評価・定期監視なし。再開時はこの固定ソースを使い、新しいGit HEADで照合を回避しない。

## 20,000更新checkpointの評価（2026-09-16）

ユーザーの明示依頼により、両方式の保存済み20,000更新object checkpointを固定confirm先頭50ケースで評価する。学習は継続し、同じGPUで両評価を実行。候補300・反復30・elite30・seed42・計画5×5行動・環境予算50を固定した。clip90学習なので、未知episodeだけの保持評価とは呼ばない。

評価用固定ソースはGit `d6cd395`、`output/pusht/eval_clip90_20k_20260916_launch/source/`。両学習configに記録されたソースhashと一致することを確認した。manifestは`output/manifests/pusht/manifest.json`で、clip90学習manifestとは役割が異なる。両dry-runのデータ・ケース・seed・launcher hash一致を確認。依存同期・学習再起動・全量データhash再走査は行っていない。

| 方式 | checkpoint SHA-256 | 評価出力 |
|---|---|---|
| Raw | `3c2626efb580b7046d6351a33c414c94def914dd4277c4182c54443e43f75ee1` | `output/pusht/eval_raw_clip90_step20000_seed42_50/` |
| BT | `0ed07eb82dac3854bc40cb0204097b6f91aa261b4765752f7cc81221402210b0` | `output/pusht/eval_bt_clip90_step20000_seed42_50/` |

起動条件・dry-run・serviceログは`output/pusht/eval_clip90_20k_20260916_launch/`。user serviceは`bt-pusht-eval-{raw,bt}-clip90-20k-20260916`、Restart=no。これは指定された2checkpointだけの評価で、将来checkpointの自動評価や監視serviceは予約していない。

### 20k評価結果

両service終了コード0・`status.json=succeeded`・50件の成否・checkpoint hash・viewer生成を照合した。

| 方式 | 成功数 | 成功率 | 評価本体の時間 |
|---|---:|---:|---:|
| RAW | 32/50 | 64% | 493.56秒 |
| BT | 42/50 | 84% | 445.02秒 |

BT−Rawは+20ポイント。同じケースでBTだけ成功12件、Rawだけ成功2件、両方成功30件、両方失敗6件。`compare_paired`でケース・物理初期状態／Goal hash・探索設定・行動統計・ソース・依存の一致を検証した。集計は`output/pusht/eval_clip90_20k_20260916_launch/paired_comparison.json`。

このcheckpoint・固定50ケースではBTの観測成功率が高い。ただし単一学習seed・途中更新の結果であり、14万更新時や別seedでの優位性は未確認。既存の保守的な片側95%差下限は−0.65ポイントで、同集計器の厳密非劣性条件は未達。評価時間は2本の学習・2本の評価が同一GPUで併走した壁時計時間であり、方式間の推論速度比較には使わない。起動から保存確認までは約10分。

## 40,000更新checkpointの評価（2026-09-17）

ユーザーの明示依頼でRaw／BTの4万更新object checkpointを評価。2万更新時と同じ固定ソース・manifest・confirm先頭50ケース・seed42・CEM候補300／反復30／elite30・計画5×5行動・環境予算50を使用した。両学習configの記録済みソースhashを固定ソースと照合した。学習を継続したまま同一GPUで2評価を併走し、依存・評価条件は変更していない。

| 方式 | checkpoint SHA-256 | 評価出力 |
|---|---|---|
| RAW | `b9f49b6bb36764a3c46a9a5b78d5ce1657afb06ed0a23e14c105e27eb14b311d` | `output/pusht/eval_raw_clip90_step40000_seed42_50/` |
| BT | `9e0d89ac45371dd3fd3e64211ee2def89facbeab25e9016458f6694dc5c43f00` | `output/pusht/eval_bt_clip90_step40000_seed42_50/` |

起動条件・dry-run・serviceログは`output/pusht/eval_clip90_40k_20260917_launch/`。評価用固定ソースは2万評価時の`output/pusht/eval_clip90_20k_20260916_launch/source/`を共用。user serviceは`bt-pusht-eval-{raw,bt}-clip90-40k-20260917`、Restart=no。

### 40k評価結果

両service終了コード0・`status.json=succeeded`・50件の成否・checkpoint hash・viewer保存を照合した。

| 方式 | 成功数 | 成功率 | 評価本体の時間 |
|---|---:|---:|---:|
| RAW | 38/50 | 76% | 444.95秒 |
| BT | 42/50 | 84% | 405.29秒 |

4万更新でBT−Rawは+8ポイント。BTだけ成功9件、Rawだけ成功5件、両方成功33件、両方失敗3件。対応付き比較で、ケース・実際の初期状態／Goal・探索設定・行動統計・評価ソース・依存の一致を検証した。

| 方式 | 20k → 40k | 新たに成功 | 失敗へ変化 |
|---|---|---:|---:|
| Raw | 32/50 → 38/50（64% → 76%） | 10 | 4 |
| BT | 42/50 → 42/50（84% → 84%） | 5 | 5 |

Rawは改善し、BTとの差は20ポイントから8ポイントへ縮まった。BTは成功率が同じでも成功ケースが入れ替わっているため、「何も学習していない」「性能が収束した」とは言えない。固定50ケース・単一学習seedの途中評価で、最終予算や複数seedでの優位性は未確認。4万Raw対BTの保守的な片側95%差下限は−13.24ポイントで、既存集計器の厳密非劣性条件は未達。

集計は起動記録内の`paired_comparison.json`、`raw_20k_vs_40k.json`、`bt_20k_vs_40k.json`。全3比較で対応関係の検証を通過した。評価時間は学習・評価のGPU併走下の値であり、方式間の推論速度比較には使わない。14万更新への両学習は停止していない。

## 60,000更新checkpointの評価（2026-09-17）

ユーザーの明示依頼でRaw／BTの6万更新object checkpointを評価。2万・4万更新時と同じ固定ソース・manifest・confirm先頭50ケース・seed42・CEM候補300／反復30／elite30・計画5×5行動・環境予算50を使用した。学習configの記録済みソースhashを照合し、dry-run成功後に同一GPUで2評価を併走。学習・依存・探索条件は変更していない。

| 方式 | checkpoint SHA-256 | 評価出力 |
|---|---|---|
| RAW | `927282b2af12ae43b5f6f41086e0ce880338e92478188508fdfc6ea690339b60` | `output/pusht/eval_raw_clip90_step60000_seed42_50/` |
| BT | `e73bafbea23ded49b23d0f612f6ef5ca331b4aeb6a056af023c8fe32753c8bef` | `output/pusht/eval_bt_clip90_step60000_seed42_50/` |

起動条件・dry-run・serviceログは`output/pusht/eval_clip90_60k_20260917_launch/`。固定ソースは2万評価時の`output/pusht/eval_clip90_20k_20260916_launch/source/`を共用。user serviceは`bt-pusht-eval-{raw,bt}-clip90-60k-20260917`、Restart=no。

### 60k評価結果

両service終了コード0・`status.json=succeeded`・50件の成否・checkpoint hash・viewer保存を照合した。

| 方式 | 成功数 | 成功率 | 評価本体の時間 |
|---|---:|---:|---:|
| RAW | 46/50 | 92% | 235.61秒 |
| BT | 45/50 | 90% | 233.44秒 |

6万更新でBT−Rawは−2ポイント（1ケース差）。BTだけ成功2件、Rawだけ成功3件、両方成功43件、両方失敗2件。対応付き比較でケース・実際の初期状態／Goal・探索設定・行動統計・評価ソース・依存の一致を検証した。

| 方式 | 40k → 60k | 新たに成功 | 失敗へ変化 |
|---|---|---:|---:|
| Raw | 38/50 → 46/50（76% → 92%） | 9 | 1 |
| BT | 42/50 → 45/50（84% → 90%） | 4 | 1 |

両モデルとも4万から改善した。2万・4万で見られたBTの成功率の上回りは、6万の同じ50ケースでは続かなかった。1ケース差・単一学習seedの途中結果からRawの一般的優位性やBTの無効性は結論できない。6万Raw対BTの保守的な片側95%差下限は−16.06ポイントで、既存集計器の厳密非劣性条件も未達。

集計は起動記録内の`paired_comparison.json`、`raw_40k_vs_60k.json`、`bt_40k_vs_60k.json`。全3比較で対応関係の検証を通過した。評価時間は学習とGPUを共有した壁時計時間であり、過去の評価との速度比較には使わない。14万更新への両学習は停止していない。

## 60k：CEMの乱数だけを変える評価（2026-09-17）

ユーザー依頼で環境seed42・固定50ケース・6万checkpoint・探索量を維持し、CEM seed7／43／123を追加した。既存のseed42を合わせて4種類の探索乱数を比較する。起動スクリプトに`--cem-seed`を追加し、既存の`solver.seed`へ渡す。CEM本体・モデル・学習コード・依存は変更していない。省略時の挙動は従来と同じ。

関連CPU回帰は25件合格、GPU対象2件をskip。実solverの同seed候補再現、別seed候補変化、global RNG不変を確認した。評価用ソースはGit `9084a7d`で固定し、学習configの記録済みソースhashとも照合した。`output/pusht/eval_clip90_60k_cem_seeds_20260917_launch/`に起動条件・dry-run・各serviceログ・固定ソース・集計スクリプトを保存した。6評価を同じGPUで併走、Restart=no、学習は停止していない。

集計では各評価の終了コード・status・50成否・checkpoint hashを検証し、全8評価の環境seed・ケース・初期状態／Goal hash・前処理・探索量・依存・評価ソースhashを照合する。過去の固定ソースとの比較は保存ディレクトリの接頭辞だけを正規化し、各ソースの相対パスとhashは一致を要求する。同じ50ケースの繰り返しを独立な200ケースとは扱わず、学習seed間の変動とも区別する。

### 探索seed比較の結果

追加6評価はすべて終了コード0・status succeeded・50件・checkpoint hash・viewer保存を確認した。既存seed42の2評価を含む全8評価で、初期状態／Goal hashとCEM seed以外の比較条件が一致した。実評価でも環境seed42／solver seed指定値で動作したことを保存configから確認した。

| CEM seed | Raw | BT | BT−Raw |
|---|---:|---:|---:|
| 42 | 46/50（92%） | 45/50（90%） | -2ポイント |
| 7 | 42/50（84%） | 41/50（82%） | -2ポイント |
| 43 | 43/50（86%） | 44/50（88%） | +2ポイント |
| 123 | 46/50（92%） | 41/50（82%） | -10ポイント |

4 seed平均はRaw88.5%、BT85.5%（BT−Raw −3ポイント）。範囲はRaw84〜92%、BT82〜90%、探索seed間の標本標準偏差は両方4.12ポイント。成功と失敗がseed間で入れ替わるケースはRaw12/50、BT11/50。

CEMの乱数だけで成績が動き、seed43ではBTが1件上回る一方、ほかの3 seedではRawが上回った。従来のseed42における1件差を安定した優位性と解釈できない。今回の平均はRawが高いが、4探索seed・固定50ケース・単一学習seedの6万checkpointに限る。学習seed間の再現性や最終14万更新の優劣は未確認。

集計と全条件照合は起動記録内の`aggregate.py`、`summary.json`、`paired_cem{42,7,43,123}.json`。同一ケースのseed反復を独立試行として信頼区間を狭める集計は行っていない。起動から完了確認は約14分で、学習と6評価が同じGPUで併走した。コード変更は起動引数の分離と回帰テストのみで、CEMの探索式・候補数・計画長・モデル重み・学習中の固定ソースは変更していない。

## 学習完了と80k〜140kの評価（2026-09-19集計）

2026-09-16 09:19 JST開始、Rawは9月19日04:07、BTは04:17に正常終了。両方の`completed.json`はstep140000・state completed、guardのstatusはsucceeded・exit_code0、user serviceはinactive・ExecMainStatus0。最終のobject checkpointと再開用checkpointを保存済み。

| 項目 | Raw | BT |
|---|---:|---:|
| 起動から正常終了までの壁時計時間 | 66.81時間 | 66.97時間 |
| 最終validation潜在予測MSE | 0.003442 | 0.002702 |

壁時計時間には全量validation・保存・途中評価とのGPU競合を含む。約9分の差をT単体の追加計算量とは解釈しない。Tの追加計算コストを分離した測定は未実施。潜在MSEはモデル間で尺度が異なり得るため、BTの予測・制御の優位性の根拠にしない。

### 共通評価条件と全seedの結果

80k以降は6万の追加探索seed評価と同じ固定ソースGit `9084a7d`を使用。環境seed42、評価manifest `output/manifests/pusht/manifest.json`のconfirm先頭50ケース、CEM seed42・7・43・123を固定した。CEM候補300・反復30・elite30・計画5×5行動・Goal間隔25・環境予算50、前処理と物理行動探索を共通化。方式ごとに探索量を増やしていない。clip90分割なので未知episodeだけの評価とは呼ばない。

| 更新数 | CEM seed | Raw成功数／50 | BT成功数／50 |
|---|---:|---:|---:|
| 80k | 42 | 42 | 45 |
| 80k | 7 | 44 | 46 |
| 80k | 43 | 41 | 48 |
| 80k | 123 | 41 | 45 |
| 100k | 42 | 48 | 47 |
| 100k | 7 | 47 | 47 |
| 100k | 43 | 45 | 46 |
| 100k | 123 | 45 | 46 |
| 120k | 42 | 45 | 46 |
| 120k | 7 | 45 | 45 |
| 120k | 43 | 44 | 45 |
| 120k | 123 | 47 | 45 |
| 140k | 42 | 46 | 47 |
| 140k | 7 | 45 | 46 |
| 140k | 43 | 44 | 47 |
| 140k | 123 | 44 | 44 |

最終14万の探索seed間の標本標準偏差はRaw 1.91ポイント、BT 2.83ポイント。最終平均差はBT−Raw +2.5ポイントだが、これは上記条件での観測差。学習seed間の変動を測ったものではない。各seedの対応付き比較は`strict_noninferiority_established=false`であり、平均値から非劣性を主張しない。繰り返し50ケースを独立試行として扱う信頼区間は作成していない。

### 証拠と照合

80k〜140kの計32評価について各集計の`aggregate.py`で、service終了コード0・inactive、status succeeded、50件の成否と成功率、checkpoint SHA-256、viewer保存を確認。各更新数の8評価でケース・初期状態／Goal hash・環境seed・前処理・探索量・行動統計・依存・ソースhashが一致することを確認した。異なるCEM seedだけを比較条件から除外する。

| 更新数 | 起動・集計記録のディレクトリ（リポジトリルート基準） |
|---|---|
| 80k | `output/pusht/eval_clip90_80k_cem_seeds_20260918_launch/` |
| 100k | `output/pusht/eval_clip90_100k_cem_seeds_20260918_launch/` |
| 120k | `output/pusht/eval_clip90_120k_cem_seeds_20260918_launch/` |
| 140k | `output/pusht/eval_clip90_140k_cem_seeds_20260919_launch/` |

各ディレクトリの`launch.json`にcheckpoint hash・個別出力先・コマンド・service名、`summary.json`に集計、`paired_cem{seed}.json`に対応付き比較、`aggregate.py`に照合手順がある。個別出力の`status.json`・`results.txt.json`・`console.log`・`viewer/`を保持する。学習完了証拠は`output/pusht/raw_bt140k_launch_20260916/{raw,bt}_status.json`と各学習runの`completed.json`。これらの生成物・checkpoint・データはGitに含めない。

80k・100kは8評価を学習と併走。120kは保存完了の時刻に合わせRawの4評価、BTの4評価を別に起動。140kは学習終了後に8評価を併走し、9月19日06:46 JSTまでに全件完了。実行時間は競合条件が異なるため方式間の推論速度比較に流用しない。将来checkpointの自動評価・定期監視は起動していない。
