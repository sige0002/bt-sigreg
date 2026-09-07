# myLeWM：共有世界モデルの研究実装領域

更新日：2026-09-07

現在の第一候補は[BT-SIGReg](BOUNDED_TRANSPORT_PROPOSAL.ja.md)。最新の採否と数学的条件は[総合レビュー](../README_RESEARCH_REVIEW.ja.md)を参照する。BTは未実装・未学習であり、以下のRBG実行例はBTの起動手順ではない。

整理により交差対応表・複製ビュー・多段予測の旧Pythonコード10ファイルを削除した。残した学習・評価基盤と復元方法は[整理記録](CLEANUP_20260907.ja.md)を参照。過去の実験記述は履歴であり、削除したコマンドを実行しない。

2026-09-07の運用状態：ユーザー指示で比較学習を停止し、生成ログ・学習チェックポイントをOSのごみ箱へ移動した。データセット、公式配布重み、研究レポートは保持している。現在は[Issue対応の検証記録](ISSUE_IMPLEMENTATION.ja.md)に沿って学習基盤を修正中で、長時間学習は実行していない。以下の旧RBG v0実行例・進行記録は履歴として扱い、削除したチェックポイントを前提に再開しない。

## 修正後の学習スケジュール（Issue #1）

2026-09-07：#1（スケジュール・再開）と#3（公式Raw計算経路の監査）は検証結果付きでclose。#2/#4/#5/#6には残件、研究#7〜#13には未実装・未検証事項がある。全issue完了ではない。整理前の回帰テストは83件合格。コード・文書の修正を今回の整理とともにバージョン管理する。

次回はRawと選定した提案を同一初期値・データ順で各100,000更新比較する方針。下記50,000はCLI既定の説明であり次回予算ではない。研究案の優先順位・先行研究・未実装の数式は[研究候補レポート](RESEARCH_CANDIDATES_20260907.ja.md)を参照。新しい候補のmodeはまだ存在しないため、既存RBGコマンドを新案の実装と解釈しない。

共有trainerの既定は50,000更新、batch 128、warmup 500更新、最大学習率5e-5、最小学習率0。最初の更新に1e-7、500回目に5e-5、50,000回目に0を使用する。各更新の全パラメータグループの学習率を`metrics.jsonl`へ記録する。これは研究比較用の短縮設定であり、公式100エポック再現ではない。

既存Raw/RBGの50,000更新CLI例。学習開始済みではない。native Lance履歴再現は研究比較の必須前提ではなく、候補学習後のBN・制御比較は後工程である。

```bash
.venv/bin/python mylewm/train_rbg.py train --mode raw --total-steps 50000 --warmup-steps 500 --max-lr 5e-5 --min-lr 0 --batch-size 128 --output .cache/stable-wm/pusht/controlled_v2/raw_s3072
.venv/bin/python mylewm/train_rbg.py train --mode rbg --total-steps 50000 --warmup-steps 500 --max-lr 5e-5 --min-lr 0 --batch-size 128 --output .cache/stable-wm/pusht/controlled_v2/rbg_s3072
```

同一実験の再開には同じ引数へ`--resume`を付ける。初期化は新規`initialization.pt`に保存し、旧10,000更新重みは読み込まない。総更新数、損失係数、データmanifest、ソース、アダプター、精度などの変更は拒否する。100,000更新へ変更する場合も別の出力先で新規実験にする。短い動作確認では`--steps 3 --warmup-steps 0`のようにwarmupを明示する。旧runnerの出力先・結果再利用規則は#5/#6で別途修正する。

共有初期値を使う正式な比較コマンド一覧は次で生成する。これはdry-runであり、学習プロセスは起動しない。`--methods raw`などの手法選択も可能。

```bash
.venv/bin/python mylewm/plan_controlled_comparison.py --steps 100000 --methods raw rbg
```

公式モデルだけの200ケース回帰評価は`.venv/bin/python mylewm/evaluate_official_pusht.py --execute`。`--execute`無しなら4つの評価コマンドを表示する。学習キューと独立しており、既存manifestの同じ200ケース・共通の物理CEM探索分布を使う。結果の重み・データ・環境版・設定・ケースの一致を確認して再利用する。

新trainerでは`budget.json`に実学習量を保存する。今回のPushT 50,000更新×128は提示数換算で約3.89周であり、100エポックではない。`--diagnostics-every 5000`で係数込みの各項のencoder/projector勾配ノルムを記録し、保存時にはvalidation先頭バッチで行動入れ替え・ゼロ行動診断を行う。これらは追加の学習損失ではない。

比較planは`--deterministic`を指定する。CUDAでは既定`CUBLAS_WORKSPACE_CONFIG=:4096:8`と決定論的演算を使い、Raw/RBGともSIGReg統計をfloat32に固定する。実LeWMのGPU一更新監査では、この条件で両経路の勾配・更新後重みが許容誤差内で一致した。公式配布重みの過去の学習設定とは区別する。

**目標は、通常の観測・行動経験から、複数タスクで再利用できる状態表現と遷移を学ぶこと。** 人手の交差対応表は必須にしない。現在の操作だけでなく後続の別操作に必要な情報を保持し、撮影条件が変わっても必要な物理状態差を認識できることを目指す。

**LeWM程度の小ささでPushTとLIBERO-10の高精度な制御を目指す。研究の中心はSIGRegの単一等方ガウス制約と情報配分の見直しであり、多段予測損失の追加ではない。**

実装済みの初期候補[RBG v0](FACTOR_GAUSSIAN_PROPOSAL.ja.md)は各ブロックのガウス性と弱い線形重複抑制を使う。要因間の有用な関係獲得やマルチタスク能力は未実証で、研究issueの完成版ではない。現在は学習停止中。最新公式PushT評価は176/200成功（88%）だが、提案側の新予算比較は未実施。

## RBG v0の旧実験手順（履歴・自動実行しない）

以下の10,000更新コマンドとrunnerは旧予算の参照。新しい100,000更新研究案の起動手順ではない。既存の旧学習済み重みはごみ箱へ移しており、待機・自動継続も現在は行っていない。

公式比較コードの依存環境とPushT HDF5が必要。新しい出力ディレクトリを使い、既存実験は上書きしない。

```bash
.venv/bin/python -m pytest mylewm/test_rbg.py mylewm/test_rbg_data.py mylewm/test_libero_planner.py -q
# 分割manifestがまだない場合のみ実行
.venv/bin/python mylewm/train_rbg.py prepare
.venv/bin/python mylewm/train_rbg.py train --mode rbg --steps 10000 --batch-size 128 --output .cache/stable-wm/pusht/rbg_v0/rbg_s3072
```

同じentryで`--mode raw`または`--mode tc`と別の出力先を指定すると対照条件になる。10,000更新は旧初期比較予算で、公式の100epoch再現済みという意味ではない。

LIBERO-10は公式配布の10 HDF5ファイルを`.cache/libero-datasets/libero_10`へ配置し、次を使う。2視点は実際の異なるカメラ画像であり、複製ビューではない。

```bash
.venv/bin/python mylewm/train_rbg_libero.py prepare
.venv/bin/python mylewm/train_rbg_libero.py train --mode rbg --steps 10000 --batch-size 128 --output .cache/stable-wm/libero10/rbg_v0/rbg_s3072
```

LIBEROシミュレータは`external/libero`とrobosuite1.4.0/bddl1.0.1を使用する。手元のMuJoCo3.12.0とは非互換だったため、MuJoCo3.3.7を`.cache/libero-runtime`に隔離。GB10上のEGL描画にはテクスチャ・カメラ画像の異常を確認したため、**評価には検証済みのOSMesa経路を使う**。MuJoCo3.3.7＋OSMesaで全10タスク×3時点×2カメラの保存状態・画像監査を通過（MAE平均3.45／255、完全一致ではない）。学習データやPushTの実行環境は変更しない。ローカル設定は`.cache/libero-config/config.yaml`。

```bash
bash mylewm/run_libero.sh mylewm/smoke_libero.py
bash mylewm/run_libero.sh mylewm/audit_libero_images.py --task-index 0 --output FRESH_AUDIT_DIRECTORY
bash mylewm/run_libero.sh mylewm/eval_rbg_libero.py --checkpoint CHECKPOINT --output FRESH_OUTPUT_DIRECTORY
```

このUbuntu 24.04 ARM64環境のOSMesaは、システムにインストールせず、公式Ubuntuパッケージ`libosmesa6=24.0.5-1ubuntu1`、`libglapi-mesa=24.0.5-1ubuntu1`、`libllvm17t64=1:17.0.6-9ubuntu1`を`apt download`で取得し、`dpkg-deb -x PACKAGE .cache/libero-osmesa`で展開した。wrapperがこのライブラリディレクトリと`MUJOCO_GL=osmesa`を子プロセスにだけ設定する。他OSでは対応するOSMesaを用意し、必ず画像監査を再実行する。

評価器の既定の監査パスは`.cache/stable-wm/libero10/rbg_v0/render_audit/task_N/report.json`（Nはファイル名ソート順）。全10タスクで`audit_libero_images.py --task-index N --output .../task_N`を実行してから評価する。既存の監査出力を上書きせず、再監査時は新規ディレクトリを用い、評価の`--render-audit-dir`で指定する。

評価器の単体テストは通過したが、学習済みLIBEROモデルによる制御評価は未実施。TC論文のBC成績ではなく、同条件の世界モデル＋CEMを比較する。GPU学習と制御評価はメモリ競合を避けて順次実行する。

初期seedの順次実験には`run_rbg_pusht_comparison.py --wait-pid PUSHT_TRAIN_PID`と、続く`run_rbg_libero_comparison.py --wait-pid PUSHT_COMPARISON_PID`を使う。前者はRBG/Raw/TC/Sub-JEPA対照と公式checkpoint、後者は共有2視点のRBG/Raw/TC/Sub-JEPA対照を比較する。各10,000更新の初期予算であり、公式の最終性能再現や複数seedでの優越性を既に達成したという意味ではない。LIBEROは各タスク50初期状態、520行動、CEM horizon8・128候補・5反復。未再計算BNとtrain-only再計算BNを両方評価する。不完全な学習・評価出力は自動で上書きせず、監査を求めて停止する。

独立学習3seedの継続実行は`run_rbg_replicates.py --wait-pid FIRST_LIBERO_COMPARISON_PID`。既定ではseed3072の両ベンチマーク完了後、3073・3074を順番に実行する。全比較が出力されても、表現診断・アブレーションと最終の研究結論は別途監査する。

PushTの凍結表現診断は`.venv/bin/python mylewm/probe_rbg_pusht.py --checkpoint CHECKPOINT --output FRESH_REPORT.json`。既定はtrain256／validation128／test128の異なる軌道。世界モデルの重みやBN統計を更新せず、線形状態読み出しと線形／非線形なブロック間予測可能性を測る。後者が高いだけでは有用な関係獲得を証明できない。制御成功率とは別の補助診断として報告する。

LIBEROは`.venv/bin/python mylewm/probe_rbg_libero.py --checkpoint CHECKPOINT --output FRESH_REPORT.json`。既定は各タスクtrain40／validation5／test5デモ。共通ロボット状態・各タスク内分散・タスク識別の補助診断を行う。保存画像を読むだけなのでシミュレータ起動は不要。物体属性の同定やマルチタスク制御の成功を、このprobeだけから主張しない。

小型性・推論microbenchmarkは`.venv/bin/python mylewm/benchmark_rbg.py --checkpoint CHECKPOINT --output FRESH_REPORT.json --benchmark pusht --device cpu`。LIBEROは`--benchmark libero10`。GPU測定は学習と重ねず`--device cuda`で実行する。パラメータ数・重みbytes・同期付き推論時間・CUDA peak allocatedを記録するが、CEM全体や環境描画の時間とは区別する。

機構の切り分けには`train_rbg_ablation.py --output FRESH_DIRECTORY`を使う。`--global-mix 0 --cross-weight 0`がブロックGaussian化だけの対照、`--global-mix .5 --cross-weight .01`が全体Gaussian化を戻す対照。`--benchmark libero10`にも対応する。混合条件はブロック512＋全体512射影、損失は`L_pred + .09*(.5*L_block + .5*L_global) + .01*L_cross`。係数と射影予算を保つが勾配強度まで整合したことにはならず、高次依存の因果効果を断定するには強度感度の追加確認が必要。本比較とは別名で保存し、初期のRBG学習を置き換えない。

## 実装に求めること

最新の候補レビューは[BT-SIGRegを第一候補とする総合レビュー](../README_RESEARCH_REVIEW.ja.md)。[ABC監査](ABC_RESEARCH_REVIEW.ja.md)は別仮説の検討記録として残す。目的は小型共有世界モデルのマルチタスク性能向上であり、非操作物体保持は補助診断に限る。

- 共有E・A・Fを出発点とし、異なる初期状態や行動による結果の違いを保持する。
- ある操作の予測状態を、実測未来の再入力なしに別操作の未来予測へ利用する。
- 撮影条件への依存を減らす際に、物体の位置や識別など後続操作に必要な情報まで消さない。
- 通常軌道から学び、人手の交差表や複製ビューを学習の必須条件にしない。
- 一段予測とE/A/Fを固定し、SIGReg、形状非指定の分散・共分散正則化、時間残差への適用を比較する。これは設計候補であり、実装済みという意味ではない。

## 過去のコード・実験との区別

| 項目 | 状態と制限 |
|---|---|
| 旧`multitask_lewm.py`、`multitask_jepa.py` | 交差予測・識別損失の旧実装。関連trainer・テストとともに削除しGit履歴に保存 |
| 旧`train_pusht.py`による50,000更新 | 同時刻教師と複製ビューに問題があり、研究比較には無効。旧チェックポイントは削除済み |
| ローカルの交差対応表方式の改訂 | 主要計算のテスト・レビューは行ったが、通常経験のみで学ぶ現在の要件を満たす完成方式ではない |
| ローカルの通常軌道・多段予測追加実験 | 公式重みから1,000更新の追加学習を実施。公式のネットワークとSIGRegに多段予測損失を追加した検討実験であり、目的に対応する提案方式として確定していない |

多段予測追加実験の同一50試行による予備評価は、公式45/50、追加学習モデル39/50。性能維持条件は未達であり、マルチタスク能力も未実証。この結果を研究アイデア全体の評価と混同しない。評価条件と結果の限界はルートREADMEに記載している。

旧コード・実験記録と、新たなRBG実装は区別する。旧多段予測案は不採用とし、その学習コマンドを現行方式の再現手順にはしない。RBGの手順は上記の専用entryを使う。

## 検証方針

実装の検証では、教師の未来時刻、行動と画像の時間幅、予測への未来画像の漏洩、勾配経路、学習・推論の前処理を確認する。損失単体や合成入力のテスト合格を、世界表現の形成結果として扱わない。

PushTでは公式と同じ開始状態・Goal・成功判定・計画予算で性能維持を検証する。LIBERO-10では10タスク共有モデルの平均・各タスクの性能向上を検証する。同条件で再学習するRawも比較し、規模・計算量と成功率差の不確実性を報告する。絶対的な性能保証とは呼ばない。世界モデルの計画成績と、別途学習したBC方策の成績は区別する。

## 過去の検討を参照する

- [研究目的・要件・現在の状態](../README.md)
- [SIGRegの数学的課題・先行研究・比較方針](../README_RESEARCH_REVIEW.ja.md)（現行レビュー）
- [旧交差対応表方式の仕様](https://github.com/sige0002/task-centered-sigreg-lewm/blob/04bf75f73fb0a785e193fac799bf45b8943c8da1/README.md)
- [過去のmyLeWM実装説明](https://github.com/sige0002/task-centered-sigreg-lewm/blob/04bf75f73fb0a785e193fac799bf45b8943c8da1/mylewm/README.md)

過去の記録は経緯の参照用であり、現在の研究目的や完成したアルゴリズムの仕様を置き換えるものではない。
