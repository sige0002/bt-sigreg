# TC-LeWMのデータ・BC・評価条件を揃えるための確認

2026-09-15、ユーザーの「学習データ以下は揃えたい」に対応。BT方式と世界モデル更新予算はこの作業の変更対象にしない。現在の実験結果を公式再現へ読み替えない。

## 確定できる部分と公開情報の限界

[TC-LeWM v3 §5.1・Appendix A.1](https://arxiv.org/html/2607.26924v3)はOpenVLAに従ったデータ再構成を記載しているが、著者が使ったデータのrevision、分割したdemo ID、BCの幅・層数・heads、augmentation、末尾chunk、checkpoint選択、評価時のchunk実行数、seed具体値は確定できない。[著者ページ](https://ryuuchou17.github.io/tclewm/)は本日もCode Coming soon。OpenVLAの設定をTC-LeWMの未公開設定と同一視しない。

設定仕様は[libero_tclewm_alignment.json](../../configs/libero_tclewm_alignment.json)。未確定値はnullであり、既定値を埋めて公式一致と呼ばない。これは学習を起動する設定ではない。

| 項目 | 確認した条件／対応 |
|---|---|
| OpenVLA HDF5再生成 | 256×256、再実行、no-op除外、再実行成功デモのみ保存 |
| 再生成の初期化 | env.seed(0)、通常reset→元states[0]→[0,0,0,0,0,0,-1]を10回 |
| no-op条件 | 先頭6成分のnormが1e-4未満、かつgripperが直前の保持行動と同じ。最初はnormのみ |
| 画像と行動の対応 | 保持行動を実行する前の画像を保存。除外行動は環境でも実行しない |
| 画像の向き | OpenVLA HDF5再生成ではnative。後段RLDS変換・評価では180度回転を使用するが、TC側採用有無は未確認 |
| デモ分割 | 著者の分割未公開。現行40/5/5を公式分割と呼ばない |
| BC主要条件 | 凍結encoder、34token、horizon8、Euler10、batch256、AdamW LR2e-4、40kは既に一致 |
| BC詳細 | 現行幅256・depth4・heads8等はローカル選択のまま。公式値の入手が必要 |
| seed数 | 論文は3 full-pipeline学習seed×3評価seed×10タスク×50試行＝4,500試行 |

再生成は公式LIBERO評価の5回settlingとは別工程。OpenVLAの再生成条件を理由に、既存BC評価の初期化を遡って変更しない。

## 実装した再生成入口

[再生成ツール](../../../src/mylewm/data/regenerate_libero_openvla.py)はOpenVLAのcommit `c8f03f48af692657d3060c19588038c7220e9af9`を固定する。[上流replay](https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/experiments/robot/libero/regenerate_libero_dataset.py)・[utils](https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/experiments/robot/libero/libero_utils.py)・LICENSEをSHA-256照合して新規出力へ保存し、replay本体は無変更で実行する。TensorFlow/VLAを読み込まないため、使用する2つの環境helperのみ同じ引数・seed・dummy actionでローカル供給する。環境の版差は残るため、著者生成物とのバイト一致は主張しない。

```bash
export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
export PYTHONPATH="$PWD/output/libero10/bc_implementation_check/runtime${PYTHONPATH:+:$PYTHONPATH}"
bash scripts/run_libero.sh -m mylewm.data.regenerate_libero_openvla \
  --raw-data .cache/libero-datasets/libero_10 \
  --output output/libero10/openvla_regenerated_v1
```

既定dry-run。実際に再生成するときだけ`--execute`を追加する。共有環境へ依存同期しない。新規出力以外は拒否し、元HDF5は読み取りのみ。`upstream/`、`config.json`、`status.json`、`data/`、上流metainfo、生成データSHA-256を保存する。成功判定による除外はデータ再生成の処理であり、世界モデルへ報酬を入力する変更ではない。

**旧prepareの既定分割をそのまま新データに適用しない。** 現行prepareは各タスクの先頭40本をtrainへ割り当てるため、失敗デモ除外で本数が減るとvalidation/testが不足する。現在は下記のユーザー承認済み`--split-mode ratio_80_10_10`を指定してmanifestを作成する。元demo番号は除外後も保持される。

新データでBCだけ学習すると、encoderを学習したデータ・分割と異なる。現在のcheckpointのmanifest hash照合を緩めず、新しい確定分割を世界モデルとBCで共有する。再学習は独立した新runとし、旧checkpoint/manifest/evidenceを変更しない。

## 検証と未完了

4件のCPUテストでdry-runの無書込、既存出力拒否、上流hash不一致時の失敗記録、元データ保持、上流相対パスの隔離、module/cwd復元を確認。実データパスでdry-run成功。上流本体のnative再実行・全量データ再生成・新分割作成・追加学習・追加評価は未実施。進行中の40k評価は固定ソースのまま継続し、停止指示済みの2k評価は再開しない。

完全一致のため残る情報は、著者データrevisionとdemo分割、画像前処理、BC詳細設定、chunk実行数、checkpoint選択、seed値とreset条件。これらが公開されるまでは「公開条件に沿ったローカル実装」の範囲に留まる。ユーザーは後述のローカル仮定で進めることを承認済みであり、未公開項目は作業開始の阻害条件としない。


## ユーザー承認の構成と再学習開始（2026-09-15）

ユーザーがデモ分割は仮定でよいとし、構成を任せて再学習を進めるよう明示依頼したため、次の独立runを開始した。未公開設定の入手を開始条件にしない。

- データ：固定OpenVLA再生成、256px、no-op除外、再実行成功のみ。
- 分割：各タスクで約80/10/10。保持数Nに対しvalidation/test各max(1,round(N×.1))、残りtrain。50本なら40/5/5。最低10本・各demo13行動以上を要求。split seed20260906、統計はtrainのみ。
- 世界モデル：BT Cayley v2、100,000更新、batch128、workers4、pin-memoryなし、seed3072、LR5e-5、warmup500＋cosine、既存小型ViT共有構成。
- BC：凍結encoder、幅256・4層・8heads、34token、8行動、Euler10、40,000更新、batch256、workers4、seed3072、LR2e-4。augmentationなし等のローカル条件を維持。
- 順序：データ再生成→出力と成功デモ数・画像shape検証→新manifest→新規BT学習→最終checkpoint確認→BC学習。追加評価は起動しない。

出力はデータ`output/libero10/openvla_regenerated_v1/`、manifest`output/manifests/libero10/openvla_v1_s3072/manifest.json`、BT`output/libero10/bt_openvla_100k_s3072/`、BC`output/libero10/bc_bt_openvla100k_40k_s3072/`。初期値からの新規学習であり、旧run再開ではない。

実コマンド・固定ソース・依存版・各段階ログは`output/libero10/retrain_openvla_bt100k_bc40k_s3072_20260915/`。serviceは`bt-libero10-retrain-openvla-s3072.service`、Restart=no。各段階の開始前にソースと依存版を照合し、前段階の異常終了で後段を起動しない。逐次処理のジョブであり、checkpointを定期監視するserviceではない。既存評価は停止せず、旧停止runも再開しない。

可変成功デモ数に対応するprepareの明示オプション`--split-mode ratio_80_10_10`を追加。従来prepareの既定を維持。関連19テスト合格（skipなし）、実パスdry-runとBC引数受理を確認。起動時点は再生成中で、データ完成・BT/BCの学習開始・完了は未確認。現在の段階はジョブのstatus.jsonと実serviceで確認する。

起動後追記：native再生成が2デモまで進行、1成功・1失敗を確認。失敗デモの除外と成功デモの保存まで動作した。学習はまだ未開始。`startup_verification.json`に観測とservice状態を記録。


## 世界モデル予算を1万更新へ訂正（2026-09-15、最新指示）

ユーザーの明示指示により、上の10万更新計画を**BT10,000更新→BC40,000更新**へ置き換えた。変更時はデータ再生成中で、10万更新の学習runは未作成。100k計画の過去launch.json・固定ソース・ログを保持し、未開始の学習段階を新ジョブへ引き継ぐ。

新しい実行記録は`output/libero10/retrain_openvla_bt10k_bc40k_s3072_20260915/`、serviceは`bt-libero10-retrain-openvla-10k-s3072.service`（Restart=no）。出力BTは`output/libero10/bt_openvla_10k_s3072/`、BCは`output/libero10/bc_bt_openvla10k_40k_s3072/`。BCは新BTの`step_10000_object.ckpt`を使う。warmup500・cosineの総予算も10,000となる。BCの幅256・4層・8headsなどは維持。

稼働中の再生成子プロセスは継続し、旧親オーケストレータPID1467302だけをSIGSTOPして旧100k学習への遷移を防止。新ジョブは再生成launcher PID1467306の終了イベントをkernel pidfdで待ち、旧親を終了させ、再生成status succeededとデータ検証後にprepare→BT10k→BC40kを実行する。周期的なポーリングや監視agentは使わない。旧serviceのactive表示だけで旧学習計画が継続していると判断しない。引継ぎ情報は旧launchディレクトリの`superseded_by_user.json`と新launch.jsonに保存。

最初の新ジョブ起動はPython環境にos.pidfd_openがなく停止した。失敗コード・statusを`pipeline_attempt1.py`、`status_attempt1.json`とconsole.logに保持し、同じLinux APIをlibc経由で呼ぶ実装に修正して一度再起動。新ジョブactive/running・既存再生成の継続、WM引数10,000・BC引数40,000と入力checkpoint10,000を確認済み。新bootstrapのCLI読込成功、git diff --check合格。学習開始・完了はまだ未確認。
