# BT100k encoderを使うTC-LeWM形式BCの確認

2026-09-14、TC-LeWM形式の実装依頼に対応。既存の凍結ViT＋タスクID条件付きflow BCを論文と照合し、現在のBT100k checkpointで学習・保存再開・推論・native接続を確認した。BC本学習・成功率比較は未実施。

## 実装と論文の対応

[TC-LeWM v3 Appendix A.1](https://arxiv.org/html/2607.26924v3#A1.SS1)と[著者ページ](https://ryuuchou17.github.io/tclewm/)を2026-09-14に再確認。コード公開は引き続きComing soon。公式コードの移植ではなく、論文に基づく既存ローカル実装を利用した。

- BT100k世界モデルのViT encoderを凍結。projector・予測器・BT変換はBC推論には含めない。
- 2カメラそれぞれのCLS＋4×4平均pool patch、計34 tokenを使用。
- DiT型headへ画像tokenをcross-attentionで渡し、task IDとflow時刻で条件付ける。10タスクで1つの共有方策。
- Gaussian noiseから連続8 native行動×7次元へのflow matching。推論10 Euler step、訓練統計による逆正規化。
- 本設定はbatch256、AdamW、LR2e-4、40,000更新。幅256・depth4・heads8など論文から確定できない詳細はローカル設定として保持。
- ViTは5,501,376パラメータで凍結、学習対象headは5,985,287パラメータ。合計11,486,663。

今回の重みはBTで100,000更新したもの。TC-LeWM論文の10タスク世界モデル10k更新とは異なり、論文の成功率再現を主張しない。これはBTのencoderを使う下流方策評価形式で、世界モデルの正則化をTCへ置き換える処理ではない。

## 今回の変更

BC評価の初期化を関数化し、公式LIBEROのreset→固定init state→ゼロ行動5回を既定にした。`--settling-steps 10`で旧ローカル回数も選択できる。回数をconfigとinitial_historyへ記録するため、異なる初期化条件は比較時に拒否される。既存評価記録は変更していない。

通常resetでrobot／gripperを初期化してからstateを復元する。BCは現在の実測2カメラ画像だけを入力し、世界モデル診断で必要だった追加の履歴収集はしない。

[利用手順](../reference/BC_GUIDE.ja.md)の例を、`bt_no_pin_100k_s3072/step_100000_object.ckpt`と本学習出力予定`bc_bt100k_40k_s3072/`へ更新した。本学習コマンドのdry-runを確認し、出力予定ディレクトリは作成されていない。

## 実測した検証

| 項目 | 結果 |
|---|---|
| 関連回帰 | 23合格、skipなし、3.39秒。CUDA forward/backwardを含む |
| BT100k由来の実データGPU短期学習 | BC head 4更新、batch2、workers0、FP32、決定論モード、5.07秒 |
| 保存再開 | 2更新でpaused→再開して4更新。各終了コード0。再開後status succeeded |
| 連続学習との一致 | policy全state・optimizer・全保存乱数状態が完全一致 |
| encoder凍結 | 学習前BT100k encoderとBC保存後encoderのhashが一致 |
| 全タスク推論 | 保持画像10タスク、特徴10×34×192、生成行動10×8×7、有限かつ[-1,1]内 |
| native接続確認 | task0/init0、9行動、2chunk生成、終了コード0、status succeeded、動画とviewer生成 |

短期学習は4更新の独立した検証用runであり、4万更新の本学習へ延長するrunではない。native接続確認の成否はFalseだが、短期重み・9行動のため性能評価とは扱わない。nativeのloopは約1.29秒、うち方策生成約0.32秒。環境生成・読込などを含む全所要時間ではない。

共有環境への依存同期、世界モデルの追加学習、既存学習の再開、他サービスの停止は行っていない。データはサイズ・mtimeで確認し、全量hashは再走査していない。世界モデルcheckpoint・manifestのhash不変を照合。GB10 capabilityに関するPyTorch警告はログに保持した。

## 証拠

`output/libero10/bc_bt100k_implementation_20260914/`に、`commands.json`、各起動ログ、`continuous/`、`resumed/`、全タスク生成行動、`verification.json`、検証コードを保存。`native_smoke/`に実環境の結果・動画・viewerを保存。

encoder SHA-256：`a3ba5a5fdbfff435ec1128dc8bb2246c264d278eaa1b906a5c09bc40bdf0d06d`。

本学習・全10タスクの通しの成功率評価・複数seedの比較は未実施。CEM成績とBC成績は別の評価経路として報告する。


## ユーザー依頼によるBC本学習開始（2026-09-14 23:27 JST頃）

上の短期検証後、ユーザーの「学習やっておいて」に基づきBC方策の40,000更新を開始した。これは世界モデルの追加学習ではなく、BT100kのViTを凍結した共有BC headの本学習である。

| 項目 | 設定／起動時確認 |
|---|---|
| run | `output/libero10/bc_bt100k_40k_s3072/` |
| 元encoder | `bt_no_pin_100k_s3072/step_100000_object.ckpt` |
| 予算 | 40,000更新、batch256、workers4、seed3072 |
| 方策 | 幅256、depth4、heads8、8 native行動、10 Euler step |
| optimizer | AdamW、LR2e-4、weight decay .01、FP32、clip norm 1 |
| 保存・validation | 1,000更新ごとと最後 |
| サービス | `bt-libero10-bc-bt100k-40k-s3072.service`、Restart=no |
| 起動時PID | 855119 |
| 固定ソース | `output/libero10/source_bc_bt100k_40k_20260914_232644/` |
| 開始時Git | `22c5c51a17958a81ae354a8de988d415b30a87b5`＋保存済み作業差分・未追跡ソースを含むsnapshot |
| 起動確認 | 23:28:07 JST、active/running、154更新、flow loss 0.7218 |
| 速度 | 21〜154更新、平均0.330秒／更新 |
| GPUプロセス使用量 | 2,607 MiB（起動時観測） |

起動前にGPU compute processなし、使用可能RAM約102 GiBを確認。固定ソースのdry-run終了コード0を確認してから実行した。固定ソース81ファイルのhashとrun config内のソースhashを照合済み。実行時bootstrapでBC trainer・policy・loop・世界モデルクラス・jepa・moduleがsnapshotから読み込まれることも検証する。出力・データは元のリポジトリに保持する。

再開にはこのsnapshot内の`launch_bc.py`と同じ`.venv`・引数・依存を使い、`--resume --execute`を指定する。ソース・設定・encoder・データのhash照合を外さない。snapshotと実コマンドは`output/libero10/bc_bt100k_40k_s3072_launch/launch.json`へ保存。標準出力／エラーは同ディレクトリの`console.log`、起動照合は`startup_verification.json`にある。

観測速度では残り約3.7時間（保存・validationなどの追加時間を除く）。長時間の速度や完了時刻の保証ではない。現時点は起動・更新進行の確認であり、40,000更新完了や制御成功率ではない。自動再起動・自動評価・定期監視サービスは設定していない。

手動確認：

```bash
systemctl --user status bt-libero10-bc-bt100k-40k-s3072.service --no-pager
tail -n 5 output/libero10/bc_bt100k_40k_s3072_launch/console.log
cat output/libero10/bc_bt100k_40k_s3072/status.json
```


## 本学習完了と評価開始（2026-09-15）

ユーザーの評価依頼を受け、40,000更新の`status.json`と`completed.json`が`succeeded`、学習サービスが停止済み・終了コード0であることを確認した。最終checkpointのstep、10タスク、encoder state hashを実際に読み込んで照合し、encoderは上記の凍結前hashと一致した。学習metricsは全て有限、最後のflow lossは0.132505、validation flow lossは0.898379。これらのlossは制御成功率ではない。

最終checkpointのSHA-256は`ba6b1b18e060595cf0149f06e8fa0afd97f496fd2d6f01656538cef1f776e5ea`。照合結果は`output/libero10/eval_bc_bt100k_40k_s3072_launch/training_verification.json`。

全10タスク・各50初期状態（0〜49）、最大520行動、8行動chunk、settling 5回、評価seed42でnative BC評価を開始した。学習時snapshotの全記録hashを照合し、その中の評価・方策コードを利用している。環境は既存OSMesa・MuJoCo 3.3.7と全10タスクのrender auditを使用。新規依存同期は行っていない。

- 評価出力：`output/libero10/eval_bc_bt100k_40k_s3072/`
- 実行記録・ログ：`output/libero10/eval_bc_bt100k_40k_s3072_launch/`
- 評価service：`bt-libero10-eval-bc-bt100k-40k-s3072.service`、Restart=no

起動後のtask0/init0は520行動を54.15秒で完走し未成功。これは途中の1試行であり、全体成功率ではない。500試行は最大行動数近くまで進む場合に約8時間の見込み。評価は開始確認までで、終了コード・全結果・viewerの最終照合は未完了。定期監視serviceや自動再起動は設定していない。


## 検証loss最良checkpointの少数評価開始（2026-09-15）

40k評価の低い途中成功率を受けて学習ログを確認したところ、検証flow lossは2,000更新の0.392526が最良で、40,000更新では0.898379へ悪化していた。訓練バッチlossは同じ更新で0.408768→0.132505。過学習を示唆するが、検証flow lossだけでは制御成功率低下の原因は確定できない。

ユーザーの「じゃあそれで」に基づき、保存済み`step_2000_bc.pt`を全10タスク・各5初期状態（0〜4）の計50試行で評価開始。40k評価と同じseed42・最大520行動・8行動chunk・settling 5回・固定ソースと環境を使用した。checkpointを読み込み、encoderが40k側と同じSHA-256であることを照合した。

- 出力：`output/libero10/eval_bc_bt100k_2k_s3072_50/`
- launch・dry-run・console：`output/libero10/eval_bc_bt100k_2k_s3072_50_launch/`
- service：`bt-libero10-eval-bc-bt100k-2k-s3072-50.service`、Restart=no
- 比較対象：進行中の40k・500試行から同じtask/initの50件。40kの追加重複評価は起動していない。

起動とtask0/init0開始まで確認。全50試行の完了・結果照合はまだ未完了。比較時はtask/init番号だけでなく実際の初期sim state・画像hashとプロトコルを照合し、不一致は対応試行として集計しない。各5試行・学習seed1個の探索的診断であり、BT方式そのものの優劣や確定した成功率差は主張しない。


2026-09-15 10:15 JST追記：ユーザーの停止指示により2,000更新側の評価を停止した。14試行・1成功で打ち切り、結果・動画・ログを保持。service MainPID=0、終了143を確認。元status.jsonはrunningのままなので、launchディレクトリの`stopped_by_user.json`を停止証拠とする。全50試行の完了や過学習仮説の確定とは扱わない。40k側の既存評価は継続。
