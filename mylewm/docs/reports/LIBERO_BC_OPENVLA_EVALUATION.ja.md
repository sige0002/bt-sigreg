# 再生成データBC：5kと25kの環境評価

2026-09-15、ユーザーの明示依頼で、BT10kから学習したBCの5,000更新と25,000更新checkpointを比較する評価を開始した。起動記録であり、最終成功率は未確定。

## 比較条件

両方とも `bc_bt_openvla10k_40k_s3072/step_{5000,25000}_bc.pt` を使う。manifestは `output/manifests/libero10/openvla_v1_s3072/manifest.json`。256pxの実測2カメラ画像を学習と同じ224pxへ前処理する。全10タスク×50初期状態、seed42、最大520行動、8行動ずつ生成・実行、公式LIBEROのゼロ行動5回settling、20Hz、OSMesa／MuJoCo3.3.7。CEMは使わない。

同じ条件・固定ソースで2プロセスを並行実行する。BC学習の停止・再開や予算変更は行わない。画像は各試行の初期・最終・参考デモ画像をNPZへ保存し、全フレームは圧縮MP4へ保存する。参考デモ画像を方策に入力しない。

## 256px対応と画像監査

従来の評価器は128px固定だった。保存データの全デモ・両カメラの形状から128／256pxを判定し、解像度混在を拒否するよう修正した。評価時には画像監査の解像度と対象データのパス・サイズ・mtimeを照合する。旧128pxの歴史的監査との互換性は保持する。

OpenVLA再生成データにはmodel XMLがなく、先頭stateは元デモの初期状態で、保存画像はsettling後である。再生成データの監査は実測stateが記録されたt=1,4,8を使う。元デモ番号まで、棄却されたデモも含めてseed0からresetした順序を再現する。固定物配置にはflattened simulator stateに含まれない成分があるため、この順序が必要となる。

初回監査ではtask3の最大MAE12.746が既定閾値10を超え、終了コード1で停止した。reset順序の不一致を修正すると同タスクの最大MAEは0.297へ低下。閾値を緩めず、全10タスクを新出力へ再監査し、全60画像で合格した（最大MAE1.142、すべて上下反転画像より誤差が小さい）。初回失敗ログとreportは保持している。監査用のデモ場面復元と、評価時の公式初期状態resetは別の手順であり、評価の初期状態をデモへ差し替えていない。

関連回帰はGPUを含む15件合格、skipなし（6.52秒）。旧auditの256px誤用、データidentity不一致、解像度混在の拒否を検証した。全タスクの実機画像監査は終了コード0。これは操作成功率の検証ではない。

## 実行記録

起動ディレクトリは `output/libero10/eval_bc_openvla_5k_25k_20260915_launch/`。`source_v2/`へ実装を固定し、`evaluate_v2.py`はPythonソースhashを起動時に確認する。`launch.json`に両checkpoint hash・コマンド・環境を保存した。環境は既存`.venv`を同期せず使用する。

| checkpoint | user service | 出力 |
|---|---|---|
| 5k | `bt-libero10-eval-bc-openvla-5k-s3072` | `output/libero10/eval_bc_openvla_5k_s3072/` |
| 25k | `bt-libero10-eval-bc-openvla-25k-s3072` | `output/libero10/eval_bc_openvla_25k_s3072/` |

両serviceは起動直後activeを確認した。自動再起動なし。ログは起動ディレクトリ内のservice名`.log`、監査の最終結果は`render_audit_v2/`。完了判定には各serviceの終了コード0、`status.json`のsucceeded、`summary.json`と`episodes.jsonl`の500試行を照合する。比較では対応するtask/initの初期state・画像hashも確認する。完了前のタスク順途中集計を10タスクの最終平均としない。

## ユーザー指示による停止（2026-09-15 21:46 JST）

5k／25kの環境評価はユーザー指示で両方停止した。各25試行完了時点で5kは0成功、25kは1成功。最初のタスクの途中結果であり、全10タスク平均や最終比較結果ではない。両serviceはMainPID=0、終了コード143、実評価プロセスなしを確認。SIGTERMによる停止のため既存status.jsonのrunning表示は上書きせず、起動ディレクトリの`stopped_by_user.json`に停止理由と実測状態を記録した。再開・再評価は別途ユーザー指示がある場合のみ。BC本学習は停止せず、確認時33,011／40,000更新で稼働していた。

## 最新保存checkpointの評価依頼（2026-09-15）

続くユーザーの明示依頼で、確認時点の最新保存済み34,000更新を固定して評価を開始した。学習は34,810更新時点で継続中だった。5k／25kは停止維持し、34kのみ新規評価する。条件は同じ256px・全10タスク×50試行・520行動上限・seed42・8行動実行・ゼロ行動5回settling。

serviceは`bt-libero10-eval-bc-openvla-34k-s3072`、出力は`output/libero10/eval_bc_openvla_34k_s3072/`、起動記録・ログは`output/libero10/eval_bc_openvla_34k_s3072_launch/`。直前に検証した固定`source_v2`・256px画像監査を使用し、checkpoint hashと引数を新規launch.jsonへ記録した。自動再起動なし。学習が先へ進んでも評価対象を自動で差し替えない。これは起動記録であり、500試行完了や最終成功率を示さない。
