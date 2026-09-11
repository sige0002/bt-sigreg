# LIBERO BT：固定メモリなしの本学習と1更新の時間

2026-09-11、ユーザーの「リベロを学習初めて。固定はしない。1ステップの時間をレポート」に従って開始した。ここで1ステップは**batch128クリップに対するoptimizer更新1回**であり、環境の1行動や実機制御周期ではない。

15:38:44 JSTに新規BT LIBERO-10の100,000更新を起動した。前回の固定メモリ有効runは2,000更新の保存状態で保持し、PushTも停止を維持した。新規runは未学習初期値から始めている。

## 起動条件

| 項目 | 今回の値 |
|---|---|
| run | `output/libero10/bt_no_pin_100k_s3072/` |
| user service | `bt-libero10-no-pin-100k-s3072.service`、Restart=no |
| 開始時Git | `8ec0abc907de4304c77c3ef40cbd398671c54aa3` |
| 固定ソース | `output/libero10/source_bt_no_pin_100k_8ec0abc9/` |
| 環境 | 既存`.venv`、PyTorch 2.9.1+cu130、依存同期なし |
| データ | 現行`output/manifests/libero10/manifest.json`、10タスク・2カメラ |
| device / precision | 同じGB10のGPU0、bf16_autocast、他の学習は併走なし |
| 更新 / batch / workers | 100,000 / 128 / 4 |
| pin-memory | `--no-pin-memory`。config=false、起動ログでもtrain／validationともfalse |
| seed / LR / warmup | 3072 / 5e-5 / 500更新 |
| BT | Cayley v2、depth2、kappa .2、hidden192 |
| 保存・validation | 1,000更新ごと |
| 疎な診断 | 1更新目と5,000更新ごと |

前回のLIBERO本学習と同じ通常の数値設定を使い、`CUBLAS_WORKSPACE_CONFIG`は未指定、`--deterministic`は付けていない。初期モデルhashは`03d730556d3bc7e1d7725d2cbcccd4ca66774478c78fad1276a758f040872e04`、manifest hashは`f7a3e92718512189dba86cee40a26d3b1bacd13da8b11329a1d92811ce4e4f35`で、前回および対照試験と一致した。学習実行中のコードを今後の編集から隔離するため、Gitのソースコピーを`PYTHONPATH`と`BT_SIGREG_ROOT`へ指定した。

## 1更新あたりの時間

15:41:26 JSTに集計し、serviceがactive/running、PID 4055599のCUDA利用と125更新への進行を確認した。100,000更新に向けてバックグラウンドで継続中であり、完了扱いではない。

| 指標 | 実測 |
|---|---:|
| 平均 | **1.2199秒／更新** |
| 中央値 | 1.2243秒／更新 |
| 95パーセンタイル（線形補間） | 1.2275秒／更新 |
| 最小〜最大 | 1.2024〜1.2287秒／更新 |
| 更新速度 | 0.8198更新／秒 |
| クリップ処理量 | 約104.93クリップ／秒 |
| 初回更新（下記の起動・診断を含む） | 5.3279秒 |

120更新時のlossは1.59161、GPU peak allocatedは26,934,989,312 bytes（約25.09GiB）。メモリ固定を無効にしたのはCPU側のDataLoaderで、GPU上のモデル・学習用メモリは使用している。短期lossや起動成功を、制御性能・長時間安定性の検証とはしない。

既存`metrics.jsonl`の`elapsed_session`について、更新NとN−1の差を集計した。初期化後の20更新を除き、21〜120更新の100件を使用する。データ待ち、GPU転送、forward/backward、optimizer更新と更新間のログ処理を含む経過時間である。CUDA kernelだけの時間ではない。既存のGPU scalar読出しによる同期を含み、計測用の追加GPU処理や毎更新の同期を挿入していない。

計測区間にvalidation・checkpoint保存・疎な診断はない。これらの周期的な追加時間や、長時間稼働後の変化を含む10万更新全体の平均ではない。また固定メモリON/OFFの厳密な速度比較を行った結果ではない。1更新目の値はDataLoader起動と初期診断を含むが、Python importやモデル構築を含む起動全体の時間ではない。

## 実装と検証

これまでLIBERO共有ループの訓練DataLoaderはCUDA使用時にpinを常時有効にしていた。`train_libero.py`と共有CLIへ`--pin-memory`／`--no-pin-memory`を追加し、共有ループで指定値を適用・configへ保存するよう変更した。既定は従来と同じ有効、validationは従来どおり無効。再開時の設定照合を維持し、pin設定が違う再開は拒否する。

起動前に`test_training_state.py`・`test_bt_sigreg.py`・`test_source_layout.py`の関連テストが**66合格、スキップ0、28.57秒**。CPU／GPU、workers0／2、pinのON／OFFで実際のbatchの`is_pinned()`と保存・再開一致を確認し、設定変更による再開拒否も確認した。警告213件はテストログに保持している。全回帰pytestを追加実行した記録ではない。

## 証拠と操作

`output/libero10/bt_no_pin_100k_s3072_launch/`へ`launch.json`（実コマンド・環境）、`console.log`、`pytest.log`、`timing_report.json`、`timing_steps_1_120.jsonl`、集計用`summarize_timing.py`を保存した。計測用ログsnapshotのSHA-256は`7d836b54143f68bef81b747bf6dfc4f8d4e7b9bd9161626628bee34680cb271e`。開始時configに記録されたソースhashと固定コピーの一致も確認した。config・metrics・重みはrun内に保存する。生成物はGitに追加しない。

現在の進捗を1回確認するコマンド：

```bash
bash scripts/monitor_training.sh \
  --run output/libero10/bt_no_pin_100k_s3072 \
  --unit bt-libero10-no-pin-100k-s3072 --once
```

ユーザーが停止する場合は`systemctl --user stop bt-libero10-no-pin-100k-s3072.service`。再開はこの新runの開始時ソース・環境・引数を使い、元出力へ`--resume`を付ける。初回保存前には再開checkpointがない。自動再起動・自動評価・定期監視serviceは予約していない。
