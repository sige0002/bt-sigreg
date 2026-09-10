# 学習高速化の短期計測（2026-09-10）

ユーザー依頼により、PushTの学習時間を計測し、同期とデータ転送を改善した。本学習・成功率評価は開始していない。以下は単一GB10での短期計測であり、100,000更新の所要時間や制御性能を保証するものではない。

## 採用した変更

- `mylewm/train.py`：GPU用DataLoaderの`pin_memory`を既定で有効化。Lightning既存の非同期転送を利用する。`--no-pin-memory`で無効化でき、設定はrecipeに記録する。CPUでは利用しない。
- `GaussianBranch`：射影乱数の呼出し番号をGPU TensorからPython整数へ変更し、毎forwardの整数読出しによる同期をなくす。`get_extra_state`／`set_extra_state`でcheckpointに保存する。射影seedの列、学習時だけ番号を進める規則、モデルの乱数との分離は維持する。
- `BoundedTransport`：CUDA上で同じ次元のCayley行列を一括計算する。既定192次元・2ブロックでは8回の`torch.linalg.solve`呼出しを1回へまとめる。矩形の場合は次元別にまとめる。CPUは従来の逐次計算を使う。
- `--compile-encoder`：画像エンコーダだけをInductorでコンパイルする任意オプション。既定は無効。予測器のDropoutと射影乱数は従来の経路に残す。compilerの版・実行ファイルhashと有効化設定をrecipeに記録し、再開時の照合を維持する。

Cayleyのパラメータ、特異値制約、損失、射影数1024、周波数点17、未来教師の勾配、モデルとTの別々の勾配clip、決定論設定を維持した。`solve_ex`でエラー検査を省略する方式は採用せず、`solve`の検査を残した。GPUの`solve`にCPU同期があることは[PyTorch公式仕様](https://docs.pytorch.org/docs/stable/generated/torch.linalg.solve.html)と手元の実装・docstringで確認した。

行列のバッチ計算は浮動小数点の計算経路を変えるため、変更前BTとのビット単位の学習軌跡一致は主張しない。ソースhashと射影カウンタの保存形式も変わる。**過去runの厳密再開は開始時のコード・環境で行い、configや照合を変更しない。** 推論exportは引き続きTなしである。

## 計測条件と結果

主計測は`uv.lock`から作成した隔離環境：Python 3.12.12、PyTorch 2.9.1+cu130、Transformers 4.57.6、Lightning 2.6.5、stable-pretraining 0.1.8、stable-worldmodel 0.0.6。元の`.venv`は変更していない。

既存PushT HDF5／manifest、224×224画像4枚、batch128、worker4、CPU thread4、BF16混合精度、seed3072を使用した。変更前後それぞれRaw／BTを新規初期値から32更新し、warmup4、保存・validation間隔16で比較した。学習は実際のSWM/SPT/Lightning経路を使い、既存runを再開しなかった。初回5更新と保存・validation境界の前後を除いた24区間の中央値を通常step時間とする。タイマはCUDAを同期して測る。総学習時間には保存とvalidationを含むが、データセット構築・モデル初期化・fit終了後の最終保存は含めない。

| 方式 | 変更前の通常step | 変更後の通常step | 時間短縮 | 保存・validation込み32更新（前→後） |
|---|---:|---:|---:|---:|
| Raw | 0.7908秒 | 0.7502秒 | 5.13% | 32.49→31.85秒（1.99%短縮） |
| BT | 0.7986秒 | 0.7578秒 | 5.12% | 32.83→31.32秒（4.62%短縮） |

両方式・変更前後の4runで終了コード0と`completed.json`の32更新完了を確認した。Rawの32個の学習lossは前後で完全一致した。BTの最大loss差は約0.01487で、変更前との学習軌跡はビット一致しない。速度比較は同じ更新数・データ順・初期値によるものであり、lossの差を性能改善とは解釈しない。一回ずつの短期比較で、反復実験の信頼区間や他GPUへの一般化は未確認。

上の表はコンパイル無効時。さらに同じ固定依存で`--compile-encoder`を有効にして実学習ループを32更新ずつ計測した。

| 方式 | コンパイル有効時の通常step | 変更前からの時間短縮 |
|---|---:|---:|
| Raw | 0.5301秒 | 32.96% |
| BT | 0.5395秒 | 32.45% |

この2runも終了コード0・32更新完了を確認した。総学習時間はRaw 30.03秒、BT 24.10秒だったが、コンパイルcacheの状態が異なる。Rawの初回validationにはコンパイルを含む5.59秒がかかり、同一プロセス内で続けたBTの初回validationは1.79秒だった。通常stepはコンパイル待ちを除く値であり、初回コンパイルを含む総所要時間が約33%減るとは主張しない。コンパイル有効時はRawも丸め差により変更前とlossがビット一致しない。

開始時の`.venv`はTransformers 5.17.0で、プロジェクトの固定版と異なっていた。そこで得た予備計測はRaw 0.7780→0.7441秒、BT 0.7922→0.7544秒／通常stepだった。これは上記固定版の主計測と混ぜない。同環境の計算部分の分割計測では、Rawのencoder約0.203秒、逆伝播全体約0.478秒が支配的だった。SIGReg forwardはRaw約0.0022秒、BT約0.0116秒だった。データ転送単独の中央値はpinなし約0.0227秒、pinあり約0.0079秒だったが、非同期読込と重なるため、単純に全体の短縮率へ換算できない。

## コンパイルの検討

画像エンコーダだけの`torch.compile`を試した。同梱Tritonの`ptxas`はCUDA 12.8で、GB10の`sm_121a`を認識せず失敗した。既存の`/usr/local/cuda-13.0/bin/ptxas`を`TRITON_PTXAS_PATH`で当該プロセスだけに指定すると動作した。パッケージやコンパイラは更新していない。

Transformers 5.17.0の予備計測では、コンパイル後の計算部分がRaw約0.727秒、BT約0.738秒／stepとなり、コンパイルなしの約0.697秒、約0.705秒より遅かった。この時点では不採用と判断したが、固定版4.57.6で条件を揃えて再計測するとRawの計算部分が約0.702→0.490秒／stepへ短縮した。上記の実学習ループでも改善したため、最終的には固定版環境向けの任意オプションとして採用した。ライブラリ版を混ぜた速度比較や、他の版でも同じ改善になるという主張はしない。

## 検証と証拠

固定依存環境で`TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=.:lewm <隔離環境>/bin/python -m pytest mylewm -q`を実行し、**139合格・失敗0・スキップ0（30.74秒）**。警告592件は残る。CUDA専用・コンパイルのテストも実行した。コンパイル追加前の固定版環境でも136件合格を確認している。

- CPU／CUDA、Raw／BTで、射影乱数カウンタ変更前の式との損失・入力勾配の完全一致、モデル用乱数の不変、保存・復元後の射影列を確認。
- CUDAのFP32／FP64、192×192と矩形5×7で、非恒等BTの逐次計算と一括計算の出力・入力勾配・全パラメータ勾配が許容誤差内で一致。
- 新PushT経路の小型BN／Dropoutモデルで、CPUとGPU（BF16）、worker0／2の連続6更新と3更新＋再開が一致。バッチ列、loss、全state、optimizer、scheduler、Tなしexportを検査した。実LeWMの長期GPU再開の証明ではない。
- コンパイル有効のGPU小型モデルでもworker0／2の保存・再開が完全一致。実224×224のViTエンコーダでも、BF16のコンパイル前後で出力の相対L2差2%未満、入力勾配と全パラメータ勾配の相対L2差5%未満を確認した。ビット単位の同一性や制御性能の保証ではない。
- コンパイル有効の実Raw／BTの32更新object exportをCPUで読み込み、通常の非コンパイルモデルでTが無く、学習checkpointのモデル重みと完全一致することを確認した。
- 公式object checkpointの読込・rollout一致を含む既存回帰も合格。

初回の元`.venv`では135合格・1失敗（19.65秒）。失敗はTransformers 5.17.0で既存checkpointの`ViTEncoder`を復元できない既知の依存不一致で、固定版の隔離環境では解消した。速度計測の初回退避コード読込でも相対ルートの補正不足で一度失敗し、計測スクリプトのROOTを修正した。いずれも失敗ログを保持した。

証拠はGit対象外の`output/benchmarks/training_speed_20260910/`に保存した。

- `train_before.py`／`bt_sigreg_before.py`：変更前ソース。基準Gitは`691bf19ebbd03fea135269dce7516aed3d25945b`。
- `end_to_end.py`、`locked_before.json`／`locked_after.json`と同名ログ：固定依存の実学習ループ計測。各Raw／BT出力にconfig・CSV・checkpoint・`completed.json`を保存。
- `locked_eager.json`／`locked_compile.json`／`locked_compiled_e2e.json`と同名ログ：固定依存のコンパイル比較。`compiled_export_check.json`に実exportの読込・重み照合、`provenance.json`にソース識別を保存。
- `measure.py`、`before.json`、`sync_batched.json`、`compile_cuda13.json`：予備計測。実データの12バッチ読込・転送と、最後のバッチを再利用した計算部分を分離。計算はwarmup4回、通常12回、区間別3回。環境制御の性能評価ではない。
- `pytest.log`／`pytest_locked.log`／`pytest_complete.log`、`environment.log`、`compile_probe.log`、`e2e_before.log`：回帰・隔離環境・失敗記録。

変更前計測では退避した2つのPythonファイルを読み込み、BTも退避実装へ明示的に差し替えた。run config内の共通ソース一覧だけではこの差替えを表せないため、変更前コードの識別には退避ソースと計測スクリプトを併用する。これらの短期checkpointを本学習の再開元には使わない。
