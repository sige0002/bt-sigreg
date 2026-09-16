> 2026-09-15までの検証記録を保存した履歴です。「学習中」「未実装」は当時の記載で、現在の状況は[検証状況](../reports/VALIDATION.ja.md)、比較の主方針は[比較手順](COMPARISON_PROTOCOL.ja.md)を参照してください。内容の事実関係は遡及修正せず、移動に伴う相対リンクのみ補正しています。

# 検証状況と未完了事項

2026-09-15追記：新BT10k世界モデルの予測・Goal距離・CEM探索の診断を完了。全10タスク・110候補、256pxの初期状態と画像を揃え、通常CEMと探索16倍を比較。32行動先の予測コスト改善は9/10、実測改善は4/10。実測履歴からのCEM行動の一段予測誤差は無変化対照の2.83倍。通しの成功率やBT正則化の因果評価とは区別する。終了コード0・status succeeded・保存NPZ再検算、関連回帰25合格。[詳細](../reports/LIBERO_WORLD_MODEL_CONTROL_DIAGNOSTIC.ja.md)。


2026-09-15 push前検証：`CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python -m pytest mylewm -q`で240件合格、skipなし（GPUテスト含む）、61.03秒。最初の環境変数未指定の全回帰はCuBLASの決定論設定不足で15件失敗・225件合格。失敗ログを保持し、起動時に設定した新規テストプロセスで全回帰を再確認した。稼働中ジョブの環境や固定ソースは変更していない。ログは`output/libero10/prepush_20260915/`。再生成・再学習の進捗とは別の検証である。

2026-09-14 23:28 JST追記（BC本学習）：ユーザーの明示依頼でBT100k由来の凍結encoder＋flow BCを40,000更新・batch256・workers4・seed3072で開始。固定ソースのdry-runとhash照合を経てuser serviceで起動し、active/running・154更新・有限loss/勾配・GPU利用を確認。21〜154更新は平均0.330秒／更新。これは本学習開始の記録で、完了や制御成功率は未確認。[起動・再開条件](../reports/LIBERO_BC_BT100K.ja.md)。

2026-09-14追記（BT100k＋BC）：TC-LeWM形式の既存凍結ViT＋flow BCを論文と再照合。現在のBT100k encoderでGPU4更新・2＋2更新の保存再開を行い、policy・optimizer・全乱数の完全一致とencoder不変を確認。10タスクの行動生成、native task0の9行動／2chunk、終了コード0・status succeeded・動画を確認。関連23テスト合格、GPU含むskipなし。評価settling既定を公式LIBEROの5回に統一し、10回も明示指定可能。BC本学習・成功率比較は未実施。[検証記録](../reports/LIBERO_BC_BT100K.ja.md)。

2026-09-14追記（モデル診断）：保持50デモ・450条件の行動依存予測と、公式LIBERO固定初期状態の10タスク・100候補のnative分岐診断を完了。4行動先のデモ予測誤差は無変化予測の0.307倍。一方、現CEM128×5の32行動先はランダムより良い割合が予測97.5%／実測45.0%、順位相関のタスク平均−0.108。候補順位の指標であり、通しの成功率ではない。固定物配置とgripper状態の復元不一致を検出した2回の失敗ログを保持し、通常reset＋固定XML復元後に全100候補の開始画像・state一致を確認。最終両診断の終了コード0・status succeeded・保存配列からの再検算が合格、回帰9件合格。追加学習なし。[詳細](../reports/LIBERO_MODEL_DIAGNOSTICS.ja.md)。

2026-09-14追記（CEM切り分け）：成功デモの50窓をオフライン採点。終盤32行動のデモはランダム候補の平均99.9%より低コストだが、序盤は50.9%。同じ履歴で候補128×反復5から512×20へ増やすと、終盤のデモより低コストな計画は3/10から10/10へ増加。予測コスト上の結果であり、実環境の0/10改善や探索不足だけが原因との証明ではない。両診断の終了コード0・status succeeded・保存結果を照合、planner回帰3件合格。[診断記録](../reports/LIBERO_CEM_DIAGNOSTIC.ja.md)。

2026-09-14追記：`bt_no_pin_100k_s3072`の100,000更新完了記録・最終checkpointを確認し、ユーザー依頼で世界モデル＋CEMを評価しました。native 10タスク各init 0の1試行、最大520行動で **0/10（macro成功率0%）**。全タスクのOSMesa画像監査が合格し、評価の終了コード0・status succeeded・10試行の結果・行動・画像hashを照合しました。各タスク50初期状態の評価やRaw／TCとの比較ではありません。学習成功率やTの機構についての保証はなく、失敗原因も未切り分けです。依存不足による最初の起動失敗と、既存の専用ライブラリを参照して解消した経緯も[実測レポート](../reports/LIBERO_BT100K_EVALUATION.ja.md)に記録しました。追加学習は実行していません。

2026-09-11 15:38 JST追記：LIBERO共有ループへpin-memory切替を追加。関連66テスト合格・スキップ0で、GPU／CPU・workers0／2・pin ON／OFFの実batchと保存再開一致、設定変更時の拒否を確認しました。ユーザー依頼で固定メモリなしのBT100kを別runで開始し、21〜120更新の平均1.220秒／更新を実測しました。旧LIBEROとPushTは停止維持、新runは継続中です。10万更新完了・制御性能の検証ではありません。[起動と計測](../reports/LIBERO_NO_PIN_TRAINING.ja.md)。

2026-09-11追記：LIBEROの凍結ViT＋タスクID付きflow BCを実装。実データ4更新・保存再開・native task 0の9行動／2chunk生成とviewerまで確認しました。CPU全回帰185合格・CUDA専用18スキップ、追加BC GPUテスト1合格。BC本学習・成功率比較は未実施です。別途ユーザー依頼でBT LIBERO世界モデルの100,000更新を同一GPU上に起動し、開始時ソースを固定してバックグラウンド実行中です。[手順](BC_GUIDE.ja.md)・[検証と起動記録](../reports/LIBERO_BC.ja.md)。

2026-09-11：srcへの構成整理後、隔離環境の全回帰182合格・スキップ0。GPU・保存再開・LeRobot Policy・CLI・旧LIBERO object checkpointの読込を確認しました。長時間学習・制御成功率評価ではありません。[移行内容・検証記録](../reports/CLEANUP.ja.md#src移行共通処理整理2026-09-11)。

2026-09-10追記（Policy変換）：ckpt直接読込／LeRobot Policyの共通CEMと別CLIのコンバーターを追加。既存の実データBT100更新重みをconfig・safetensors・前後処理込みで変換。公式LeRobot PushTの保持2episode・計22時刻で、推論用／訓練用ckptとLeRobot形式の行動が完全一致し、実履歴更新と再計画まで確認した。標準processorで時刻が落ちる等の実行時問題も修正。全回帰171合格・スキップ0。オフライン記録データでの接続確認であり、実機I/O・環境での提案行動実行・制御成功率は未実施。[実データの結果と失敗記録](../reports/POLICY_EXPORT.ja.md)・[利用手順](MODEL_IO.ja.md#policy)。

2026-09-10追記（LeRobot v3）：HDF5／LeRobotを選択するRaw／BT入力と、訓練統計・入力条件を維持する形式間のオフライン推論を追加。公式`lerobot/pusht`の実データで実モデルBTの100更新、10更新ごとのvalidation、50／100更新保存、test由来32clip推論を完了。50更新から再開した100更新時点の全state・optimizer・scheduler・乱数状態も連続実行と完全一致。固定依存の隔離環境でGPU・compile・既存重み読込を含む160件合格・スキップ0。単一カメラ入力とオフライン予測の対応であり、実機・多カメラ融合・長期収束・制御成功率の検証ではない。[条件・証拠](../reports/LEROBOT_V3.ja.md)・[実行手順](DATASET_RECIPES.ja.md#lerobot-v3で学習する)。

2026-09-10追記（validation間隔）：新PushTに`--val-every`を追加し、`--save-every`から分離した。省略時は従来と同じ間隔。CPU小型モデルでvalidationを2／4／6更新、checkpointを3／6更新に実行し、3更新checkpointからの再開でloss・全state・optimizer・schedulerが連続実行と一致した。間隔変更での再開拒否と不正値拒否も確認。固定依存の隔離環境でGPU・コンパイルを含む全回帰140件合格・スキップ0件（25.73秒、警告601件）。ログは`output/benchmarks/validation_intervals_20260910/pytest.log`。本学習は開始していない。[指定方法](DATASET_RECIPES.ja.md#ログ保存再開)。

2026-09-10追記：学習高速化のユーザー依頼により、新PushTのGPU転送・射影乱数カウンタ・BTのCayley一括計算を改善し、任意の`--compile-encoder`を追加。固定依存の隔離環境とGB10による32更新比較で、通常stepはRaw／BTとも既定設定で約5%、コンパイル有効時は約33%短縮（初回コンパイル待ちを除く）。GPU・コンパイルを含む全回帰139件合格・スキップ0件。元の`.venv`はTransformers 5.17.0で公式checkpoint読込が1件失敗したため、変更せず隔離環境の固定4.57.6で検証した。BT一括計算・コンパイルの前後で学習軌跡はビット一致せず、変更後コード内の短期保存・再開一致を確認した。本学習・成功率評価ではない。[計測条件と詳細](../reports/TRAINING_SPEED_20260910.ja.md)。

更新日：2026-09-09。不要な独立診断・旧比較準備CLIと専用テスト13ファイルはユーザー承認で削除した。[削除一覧・復元方法](../reports/CLEANUP.ja.md)を参照。以下の旧診断実績は当時のコードでの結果である。

実験の現在地をまとめる文書です。個別の数値・ハッシュ・失敗記録は[レポート一覧](../reports/README.md)から参照してください。

## 実施済み

### Issue #20：PushT評価画像・動画・診断メモリ（2026-09-10）

明示HDF5読込で画像列を落としていた問題、動画が再利用バッファを参照する問題、CEM監査が反復ごとの展開画像を保持する問題を修正。公式・旧BT 15k・新Raw 10k・ランダムの固定10ケースを実行し、それぞれ10/10・4/10・5/10・0/10。全ケースでagent移動を確認し、終了コード・status・結果を照合した。性能比較の結論ではない。修正途中の50ケースはメモリ急増で中断、RTX側BT 10kと修正後50ケース完走は未確認。詳細・失敗記録は[実測レポート](../reports/PUSHT_ISSUE20.ja.md)。ユーザー指示で学習は停止済み。

最終コードはGPU有効の全回帰126件合格・スキップ0件（17.86秒）。CUDA専用5ケースも含む。警告236件は残る。

続くユーザー依頼の50件追試では、同じ公式重みで修正前`6272e47`は0/50、現行`5f19742`は45/50。開始/Goal画像のデータ一致も0/50→50/50で、評価バグを旧コミットで再現した。現行BT 100kは44/50。全実行の終了・結果・動画を確認済み。ただし現行の公式/BT間で13件の物理初期object状態に微差が見つかり、その原因と厳密一致は未確認。詳細は同レポート。

さらに70,000更新同士を同じ固定50ケースで評価し、SIGReg（新経路Raw）は45/50（90%）、BT（旧経路v2）は47/50（94%）。終了コード・status・結果と動画各50本を確認した。初期学習重みは一致するが学習手順に差があり、実物理初期object状態にも10件の微差が残るため、Tの効果や方式の優位性とは断定しない。重みhash・条件・出力先は同レポートの70k節。

### Issue #19：評価設定の列リスト変更（2026-09-10）

Issue #17の修正がHDF5Datasetへ渡した列リストを共有し、episode/step列を追加した結果、統計計算へ識別子が混入してAxisErrorになった。アダプターでリストをコピーし、統計計算は識別子を除外、1D/2Dの有限行を扱う。元データの行は削除しない。回帰5件合格。実PushT HDF5と評価yamlで統計計算まで通過し、ローカルRaw 50,000更新checkpointのCPU読込・行動統計次元を確認。RTX側BT 10,000更新checkpointによる環境評価完走は未確認。以前の列読込テストだけでは設定の副作用を検出できなかった。

### 公式ライブラリへのPushT移行（2026-09-09）

`src/mylewm/training/train.py` の新レシピ `pusht_spt_v1` を実装。SWMのHDF5Dataset、公式画像前処理・`lejepa_forward`・SIGReg、SPTの逆伝播／optimizer／scheduler、Lightningのループ・CSV・checkpointを使用する。エピソード分離・train-only統計を維持するが、クリップ末尾条件・抽出・LR添字は旧経路と異なる。[条件差と手順](../TRAINING.ja.md)を参照。

追加の9テストで次を確認した。性能実験ではない。

- 小型モデルのRaw／恒等BTで損失・全モデル勾配が一致し、旧Rawの同一入力での損失も一致。未来教師勾配とTへの予測損失勾配不在を確認。
- 本物のPushT E/A/FでもRaw／恒等BTの損失・勾配が一致（CPU、合成28×28画像）。実画像制御や224×224での長期学習ではない。
- native HDF5ローダーを自己生成fixtureへ適用し、episode分離・形状・正規化・固定validationケースを確認。
- CPU小型BN/Dropoutモデル、worker0/2で連続6更新と3更新＋再開のバッチ・loss・全state・optimizer・schedulerが完全一致。モデルとTの更新、Tなしexport、設定変更時の拒否を確認。
- 新entrypointのRaw／BT両方でnative fixture・小型モデルの3更新が完了し、CSV・最終checkpoint・完了記録と出力上書き拒否を確認。

最初の接続試験では画像前処理の引数不足、テスト内の非leaf Tensorのdeepcopy、再開時のvirtual epoch長の扱いを修正した。SPTが登録するHardwareMonitorは公開設定のキー一覧に無く、生成後・setup前に除外した。環境情報の背景収集・外部tracker・追加モデルexportも無効化。独自最適化ループは追加していない。

新経路追加時点では旧コード・manifest・既存100k重み・評価結果は変更しなかった。その後の今回のコード整理は下記のとおりで、重み・manifest・評価結果は引き続き保持している。新経路の本学習・PushT成功率・GPU長期再開・LIBERO移行は未実施で、旧checkpointからの互換resumeも許可しない。

全回帰は `CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:.:lewm .venv/bin/python -m pytest mylewm/tests -q` で **111合格・5スキップ**（14.35秒）。CUDA専用5件は未実行。fork/LanceとLightningのログ・再開に関する警告は残るが、上記CPU再開の実測一致を別途確認した。実manifestでの新CLI dry-run、Markdownリンク・見出し参照、Python構文、`git diff --check`も確認した。

実PushTデータでも学習を起動せずnative loaderを確認し、train 1,585,717クリップ、固定validation 256件、取得画像4×3×224×224・行動4×10を確認した。旧train 1,645,509クリップとは末尾条件が異なる。実データの確認は1クリップの読込までで、全クリップの内容監査・実データ学習・成功率試験ではない。

### 旧方式撤去・役割名への整理（2026-09-09）

旧RBGのブロック分割・交差共分散・専用引数とテストを撤去し、共有処理・CLI・テスト6ファイルを改名した。重複するLIBERO環境smoke、旧公式専用PushT評価launcherと専用テスト2件、旧テクスチャ書出しオプションも撤去。全次元SIGReg、Raw/TC/BTの一段予測、学習条件、公式評価の成功判定は維持した。

CPU回帰は **102合格・5スキップ**（15.02秒）。公式Rawとのloss・勾配一致、恒等BT、TCの適用座標、再開一致、旧方式の拒否を確認。CUDA専用5件は未実行で、長期学習・制御評価を行ったという意味ではない。詳細と互換性の境界は[整理記録](../reports/CLEANUP.ja.md)を参照。

### 既存経路での実績

| 項目 | 確認した範囲 |
|---|---|
| BT v2 | Cayley特異値制約、非奇関数の原点固定写像、距離境界、勾配・推論分離・保存再開のテスト |
| Raw経路 | 公式forwardとの一段損失・勾配・1更新比較。精度・決定論設定による差も記録 |
| PushT学習 | seed3072、新規初期値から100,000更新で正常終了 |
| PushT固定confirm評価 | 80k 179/200、90k 178/200、100k 178/200。同じrunのcheckpoint比較 |
| PushT上流eval | 100kで49/50。固定confirmとはケース・正規化・seed処理が異なる |
| LIBERO-10 | 10タスク・2実カメラの共有モデルで100更新の短期動作確認 |
| データ・再開・評価契約 | 内容hash、乱数・optimizer復元、初期状態/Goal・行動探索条件の監査 |
| 終了確認 | launcherが結果検証後にsucceededを記録。SIGKILL等ではstatusが残るため実プロセスも確認 |

[学習記録](../reports/PUSHT_TRAINING_100K.ja.md)・[評価レポート](../reports/PUSHT_CHECKPOINT_EVALUATION.ja.md)・[実装監査履歴](../reports/IMPLEMENTATION_AUDIT.ja.md)に証拠を分離しています。過去のテスト件数は実行時点の範囲を示し、現在のテスト件数や制御試行数と混同しません。

## 未完了・主張できないこと

- 同じ新規E/A/F初期値・データ順・100,000更新のRaw/TC/BT比較。
- LIBERO-10の同予算Raw／TC本学習と、複数初期状態・学習seedによる共有モデルの平均・各タスク・下位タスクの制御比較（BT100kと各タスク1試行は2026-09-14追記を参照）。
- 複数学習seedの変動、非線形Tの効果と単なる尺度・アフィン効果の切り分け。
- 追加Tの学習計算量を含めた同実測計算予算での比較。
- 公式配布重みの正確な過去学習履歴とnative Lance経路の再現。
- 上流eval今回50ケースでの公式配布重みの直接比較。

PushTだけでマルチタスク改善を証明しません。98%を論文の3学習seed平均96%への優越とはしません。固定confirm 200件も既使用の回帰集合であり、未使用最終テストではありません。有限試行で任意の条件に対する非劣化を保証しません。

## Issueと運用

2026-09-09、ユーザーの明示依頼で残っていた旧issue #2・#4〜#13 の11件をすべて `not planned` としてクローズした。#1・#3は以前にclose済み。現在openは0件。これは旧計画の整理であり、上記の未完了実験を達成済みに変更するものではない。本文・コメントはGitHubに保持し、現行の未完了事項は本書へ集約する。

追加学習・評価は明示依頼時のみ。定期監視・自動評価予約は行いません。手動の進捗確認は `bash scripts/monitor_training.sh --once`、評価手順は[PushT](../EVALUATION.ja.md#pusht)／[LIBERO](../EVALUATION.ja.md#libero)です。完了済み学習のログが増えないことを障害とは扱いません。

今回の構成整理と回帰確認は[整理記録](../reports/CLEANUP.ja.md)へ記録します。新しい性能試験は行いません。
## 学習起動時のデータ検証変更（2026-09-10）

Issue #14対応：通常起動は存在・サイズ・mtimeの確認、新規prepareで全量SHA-256を保存、`--verify-data`で明示再走査。旧manifestは書き換えない。prepare時のhashを保持し、軽量検証を全量検証と呼ばない。ソース・設定・依存照合は継続し、過去runの厳密再開は開始時のGit版・環境を使う。

CPU全回帰は108合格・5スキップ・1失敗。失敗は既存checkpoint読込時のTransformers 5系の`ViTEncoder`欠落であり、環境更新に伴う別件。今回の変更に関する軽量検証・改変検出・Raw/BT小型モデル実更新・再開テストは合格。RTX 6000 Adaでの起動時間は未測定で、HDF5読込・モデル初期化等の待ち時間まで無くなるとは主張しない。

## 既存checkpointの依存互換性修正（2026-09-10、Issue #15）

`pyproject.toml`でTransformers 4.57.6を固定し、`uv.lock`を再生成。既存checkpointが参照する`ViTEncoder`を復元できる版へ戻した。修正版lockから`UV_PROJECT_ENVIRONMENT`指定の新規隔離環境へ`uv sync --locked --group libero`を実行し、CPU全回帰は **109合格・5スキップ・失敗0**（51.86秒）。以前失敗した公式checkpointの読込・rollout比較も合格。稼働中の学習環境は同期しなかった。

5スキップはCUDA限定：BTの再開・推論出力2件（カメラ数1/2）、共有ループの再開一致2件（worker数0/2）、CPU/GPU混在入力1件。CPU限定のためでありGPU検証合格とは扱わない。Hugging Face CLI 0.36.2のdownload引数（`--repo-type`、`--include`、`--local-dir`）も確認。以前のCLI 1系の`--dry-run`はこの版では利用できない。
