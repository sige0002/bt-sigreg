# LIBERO BT100k：行動依存予測とnative候補順位の診断

実施日2026-09-14。ユーザー依頼の①未見デモの予測・行動依存性、②モデルの候補採点と実環境の対応を調べる補助診断。追加学習や、探索予算を増やした通しの成功率評価ではない。

**保持デモ上では行動依存の未来予測が機能する一方、native候補の予測順位と実測順位の対応は弱かった。** 32行動先ではCEMがランダムより良い割合が予測97.5%に対して実測45.0%、タスク内順位相関の平均は−0.108。成功率の改善や、探索不足だけが原因であることを示す結果ではない。

## 公式LIBEROとの対応

一次資料は[公式評価ループ](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/8f1084e3132a39270c3a13ebe37270a43ece2a01/libero/lifelong/metric.py)、[公式環境wrapper](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/8f1084e3132a39270c3a13ebe37270a43ece2a01/libero/libero/envs/env_wrapper.py)、[公式環境のstep](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/8f1084e3132a39270c3a13ebe37270a43ece2a01/libero/libero/envs/bddl_base_domain.py)。参照したローカルLIBEROのHEADは同commit。既存のローカル環境・描画対応もあるため、無改変コピーや公式結果の完全再現とは呼ばない。

公式の手順は固定init state、reset→set_init_state、ゼロ行動5回、nativeのstep・doneによる成功判定。本診断は公式LIBERO-10の全10タスク・init 0を使い、Panda／OSC_POSE／7次元行動／20 Hz／2カメラ128画像を維持する。doneとenv.check_success()の一致も各行動で検証する。

世界モデルには4行動間隔の実測3観測が必要なため、5回のsettling後にゼロ行動8回を追加し、settling後・4行動後・8行動後を履歴にする。この追加履歴と32行動の分岐実験は独自の補助診断条件。公式の言語条件付き方策評価や、各タスク50初期状態の通しの成功率評価ではない。初回CEM評価のゼロ10回・複製履歴とは異なる。

候補ごとに通常hard resetを行い、初回resetで得た環境XMLを公式wrapperのreset_from_xml_stringで復元してから同じ手順を再実行し、コントローラやgripperの内部状態を前候補から持ち越さない。開始sim stateの絶対差1e-10以下と、3時刻2カメラの入力画像の完全一致を必須にする。sim stateだけを巻き戻して履歴一致としない。

## ① 保持デモの予測精度・行動依存性

- 既存manifestのtest 50デモ（10タスク×5）、trainデモと交差なし。50本すべて末尾rewardが成功。既に一部を診断に用いたtest集合であり、モデル選択から完全に隔離された最終テストとは呼ばない。
- 各デモの序盤・中盤・終盤で、同じ開始時刻から4／16／32行動先を予測。計450条件。終盤の開始時刻は末尾32行動前に固定し、4／16行動先はデモ末尾ではない。
- 履歴は実画像t−8,t−4,t、過去行動t−8:t。予測対象は観測t+4,t+16,t+32。訓練と同じ行動正規化・前処理。報酬・状態・タスクIDはモデルへ入力しない。
- 正しい行動、ゼロ行動、同タスクの次の保持デモの同じ進行区分の行動、16通りの時間シャッフルを比較。シャッフルは7成分一組の行動を並べ替え、各成分を別々には崩さない。
- 実測未来を同じencoderで得たzに対するMSEを測定。「現在zのまま」の予測誤差で尺度を対照する。これは画像復元誤差や物理座標誤差ではない。

| 予測先 | 正しい行動の誤差／現在zを保つ対照 | 正しい行動が別デモ行動より正確な条件 | 時間シャッフルによる平均誤差倍率 |
|---|---:|---:|---:|
| 4行動 | 0.307（95%区間0.236–0.387） | 90.0% | 1.031 |
| 16行動 | 0.178（0.149–0.208） | 85.3% | 1.183 |
| 32行動 | 0.162（0.136–0.189） | 87.3% | 1.257 |

比率は誤差平均の比。区間は10タスクを固定し、各タスク内の5デモを2000回bootstrap、同一デモの3窓をまとめたもの。単一学習seedの条件付き区間であり、学習seed間の不確実性ではない。予測先が長いほど比率が小さくても、絶対MSEは0.00708→0.04225→0.10003と増える。

デモ分布上では、モデルは行動に依存して未来zを予測し、無変化予測を上回る。4行動内の時間順シャッフルの影響は小さい。これだけで細かい接触ダイナミクスや、探索が出す行動の予測精度は保証しない。

## ② native候補実行の条件

現CEMの128候補×5反復・elite16・8chunk（32行動）を固定。各タスクで得た最終meanの行動列1本、標準偏差0.6のGaussianを[-1,1]へclipしたランダム行動列8本、ゼロ行動列1本の計10本を実行する。ランダム候補はモデルによる事前選別なし。全10タスクで100分岐、3,200行動（reset後のsettling・履歴収集を除く）。

4行動ごとに実測画像・sim state・予測z・実測z・公式Goal述語の成否を保存する。成功は各stepのnative判定のORと最終判定を別に保存。物体座標・手先・gripperの観測は補助診断として記録し、候補生成には入力しない。

実環境の実測画像をencoderへ入れたGoal距離と、予測Goal距離の順位相関を**各タスク内**で計算する。タスクをまたぐ生MSEの大小を混ぜて順位相関を作らない。実測潜在距離もencoderの表現に依存するため、公式Goal条件と物理変化を併記する。

32行動列全体は予測の監査のためopen-loopで実行する。通常のMPCは最初の4行動後に再計画するため、4行動先と16／32行動先を区別して報告する。

## 実行・証拠

実装は `src/mylewm/evaluation/diagnose_libero_model.py`。共有CEMや学習コードは変更していない。最終候補meanを記録するサブクラスは継承元のplanをそのまま使用し、返されるchunkと完全一致を検証する。

```bash
.venv/bin/python -m mylewm.evaluation.diagnose_libero_model \
  --stage offline \
  --checkpoint output/libero10/bt_no_pin_100k_s3072/step_100000_object.ckpt \
  --output output/libero10/model_diagnostics_20260914/offline

UV_PROJECT_ENVIRONMENT=/home/sadasue/bt-sigreg/.venv \
PYTHONPATH=/home/sadasue/bt-sigreg/output/libero10/bc_implementation_check/runtime \
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
bash scripts/run_libero.sh -m mylewm.evaluation.diagnose_libero_model \
  --stage native \
  --checkpoint output/libero10/bt_no_pin_100k_s3072/step_100000_object.ckpt \
  --output output/libero10/model_diagnostics_20260914/native_canonical \
  --task-ids 3 0 1 2 4 5 6 7 8 9
```

outputは新規ディレクトリが必須。既存HDF5・重み・manifest・過去評価は更新しない。データはサイズ・mtimeを検証し、全量SHA-256の再走査なし。出力configへcheckpoint・manifest・評価ソースと参照LIBEROソースのhash、候補ごとの記録へinit stateファイルと描画監査のhashを保存する。全10タスクの既存OSMesa監査がtask名・MuJoCo版・backendと一致することを検証する。

新規診断6件＋既存planner3件の回帰テストが合格。未来行動の時刻対応、記録によるCEM行動と乱数系列の不変、シャッフル時の行動成分保持、境界と順位相関を確認した。①の実行時間は13.04秒。実測環境と依存は初回BT100k評価と同じで、同期や追加インストールなし。PyTorchのGB10 capability警告などはconsole logに保持する。

## 途中失敗と修正

最初のnative診断はtask 0〜2の30分岐を実行後、task 3の最初の候補前に開始画像の完全一致検証で停止した。終了コード1、status failed、画像の21.8%の要素が不一致。sim stateの一致検証は通っていた。LIBEROのresetは固定物の位置・姿勢をmodel.body_pos／body_quatへ設定するため、それらはflattened sim stateだけでは保持されない。

一致検証を緩めず、初回resetで得た環境XMLを保存し、候補ごとに公式wrapperのreset_from_xml_string（robosuiteのdeterministic reset）と固定init stateの復元を行うよう修正した。過去ログ・途中結果・実行時ソースは `native/` と `native_console.log` に保持。修正後は問題のtask 3から全10タスクを新規 `native_canonical/` へ実行し、失敗runの候補と混ぜない。この環境復元の修正は本診断だけに適用し、既存の通し評価器や過去評価を変更していない。


XML復元のみの2回目はtask 3のCEM行動列を実行後、次候補の開始状態差0.003717で停止した（終了コード1、status failed）。robosuiteのPanda gripperはcurrent_actionを保持し、XMLによるdeterministic resetだけではgripperオブジェクトが再作成されない。通常のhard resetでrobot／gripperを初期化してから、固定XML・init stateを復元する順に修正した。2回目のログ・状態・ソースも `native_fixed_xml/` に保持。3回目の `native_canonical/` が最終診断で、開始状態と画像の一致閾値は変更していない。

## ② native候補実行の結果

以下の「CEMがランダムより良い割合」は、各タスクでCEMのGoal距離がランダム8本より小さい割合を求め、10タスクで平均したもの。**タスク成功率ではない。** 順位相関は1が完全一致、0付近は単調な順位対応が弱く、負なら逆方向。

| 予測・実行先 | 予測上でCEMがrandomより良い割合 | 実測でCEMがrandomより良い割合 | 予測・実測の順位相関：タスク平均 | CEMの実測距離が開始より縮んだタスク |
|---|---:|---:|---:|---:|
| 4行動 | 87.5% | 68.8% | 0.069 | 7/10 |
| 16行動 | 90.0% | 50.0% | -0.082 | 4/10 |
| 32行動 | 97.5% | 45.0% | -0.108 | 5/10 |

| 予測先 | native CEM行動の予測誤差／無変化予測の誤差 | nativeランダム行動の同比率 |
|---|---:|---:|
| 4行動 | 0.686 | 1.251 |
| 16行動 | 0.600 | 1.380 |
| 32行動 | 0.694 | 1.195 |

### 全タスク：32行動先

| task ID | 予測・実測の順位相関 | CEMが実測でrandomより良い割合 | CEMの予測距離短縮 | CEMの実測距離短縮 |
|---|---:|---:|---:|---:|
| 0 | -0.552 | 50.0% | +0.3403 | +0.2311 |
| 1 | -0.055 | 12.5% | +0.2933 | -0.2374 |
| 2 | -0.091 | 50.0% | +0.7704 | +0.2762 |
| 3 | -0.527 | 37.5% | +0.2811 | +0.0768 |
| 4 | 0.127 | 0.0% | +0.1416 | -0.6578 |
| 5 | 0.248 | 62.5% | +0.2636 | -0.0409 |
| 6 | -0.248 | 75.0% | +0.2175 | -0.0064 |
| 7 | 0.115 | 37.5% | +0.4177 | +0.1128 |
| 8 | 0.188 | 87.5% | +0.3553 | +0.0017 |
| 9 | -0.285 | 37.5% | +0.2247 | -0.0059 |

距離短縮は開始距離−末尾距離。正が短縮で、生の距離値によるタスク間の性能順位ではない。

32行動内に公式成功判定が立った分岐は、CEM 0/10、ゼロ 0/10、ランダム 0/80。これは長い複合タスクの開始直後32行動に限る値で、通常のepisode成功率には使わない。

CEMの手先移動は1.93〜6.96 cm。一方、保存された非robot・非相対の物体位置の開始終点変位は最大7.69e-09 mで、新たな公式Goal述語が成立したタスクはなかった。物体位置は観測に含まれるものの補助診断であり、姿勢や関節・接触を含む完全な物理誤差ではない。


## 解釈と範囲

①は、保持デモ分布上で行動を利用した予測機能があることを支持する。②は、CEMが作る行動を同じ精度で予測できるか、Goal距離による候補順位が実行後も保たれるかを別に測っている。探索数は増やしておらず、今回の結果を16倍の探索量による改善とは解釈しない。

予測MSEが小さいこと、候補順位が正しいこと、公式タスクを成功させることは別の検証である。nativeの実測Goal距離も学習済みencoderに依存し、潜在距離短縮を物体操作の進捗と同一視できない。4行動で再計画する通常MPCと、32行動のopen-loop監査も区別する。モデル分布外の行動、接触予測、Goal距離の意味のどれが主要因かの因果分離は未完了。

今回の対象はBTで学習した単一モデルの能力であり、BT正則化の追加による改善を識別するRaw／TC比較ではない。nativeは各タスク1初期状態・ランダム8本であり、他初期状態・独立学習seedへの一般化や通しの成功率改善は未検証。

## 最終照合・成果物

修正後のnative診断は828.21秒（13.80分）。①・最終②とも終了コード0、status succeeded、450／100条件の件数・重複なし、checkpointとmanifestのhash不変、両PIDの終了を確認した。保存NPZから予測誤差・予測／実測Goal距離・成功判定を再計算し、全候補の開始state・画像・固定XMLのhashと公式Goal述語の整合を検証した。途中失敗2回のログとstatus failedはそのまま保持している。

出力は `output/libero10/model_diagnostics_20260914/`。`offline/` が①、`native_canonical/` が最終②、`verification.json` が独立照合、`offline_analysis.json` がデモ単位の区間とタスク別予測指標、`native_analysis.json` が順位・予測誤差・物理変化。`analyze.py`、`plot.py`と実行時ソースも保持する。

`viewer/index.html` に全候補の数値と、CEM・ゼロ・実測距離で最良のランダム候補の開始／4／16／32行動後／Goal画像を保存。最良ランダムの画像選択は事後であり、モデルが選んだ候補ではない。表示画像だけ上下反転し、モデル入力・保存画像はnativeのまま。`diagnostics.png`・`diagnostics.svg` は共有可能な図。`runtime_versions.json`に実際に読み込んだMuJoCo 3.3.7のパスと依存版を記録した。
