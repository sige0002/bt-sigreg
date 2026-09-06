# myLeWM：共有世界モデルの研究実装領域

更新日：2026-09-06

**目標は、通常の観測・行動経験から、複数タスクで再利用できる状態表現と遷移を学ぶこと。** 人手の交差対応表は必須にしない。現在の操作だけでなく後続の別操作に必要な情報を保持し、撮影条件が変わっても必要な物理状態差を認識できることを目指す。

**LeWM程度の小ささでPushTとLIBERO-10の高精度な制御を目指す。研究の中心はSIGRegの単一等方ガウス制約と情報配分の見直しであり、多段予測損失の追加ではない。**

初期候補[関係保持型ブロックSIGReg（RBG v0）](FACTOR_GAUSSIAN_PROPOSAL.ja.md)を実装し、PushTの新規学習を開始した。各ブロックのガウス性と弱い線形重複抑制を使い、単一E/A/Fで要因間の関係を予測する。性能・マルチタスク能力はまだ未実証。[進行記録](RBG_EXECUTION_STATUS.ja.md)、[ルートREADME](../README.md)、[背景レビュー](../README_RESEARCH_REVIEW.ja.md)を参照。

## RBG v0の実行方法

公式比較コードの依存環境とPushT HDF5が必要。新しい出力ディレクトリを使い、既存実験は上書きしない。

```bash
.venv/bin/python -m pytest mylewm/test_rbg.py mylewm/test_rbg_data.py mylewm/test_libero_planner.py -q
# 分割manifestがまだない場合のみ実行
.venv/bin/python mylewm/train_rbg.py prepare
.venv/bin/python mylewm/train_rbg.py train --mode rbg --steps 10000 --batch-size 128 --output .cache/stable-wm/pusht/rbg_v0/rbg_s3072
```

同じentryで`--mode raw`または`--mode tc`と別の出力先を指定すると対照条件になる。現在の学習は別プロセスで進行中なので、上記を重複起動しない。10,000更新は初期比較予算で、公式の100epoch再現済みという意味ではない。

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

- 共有E・A・Fを出発点とし、異なる初期状態や行動による結果の違いを保持する。
- ある操作の予測状態を、実測未来の再入力なしに別操作の未来予測へ利用する。
- 撮影条件への依存を減らす際に、物体の位置や識別など後続操作に必要な情報まで消さない。
- 通常軌道から学び、人手の交差表や複製ビューを学習の必須条件にしない。
- 一段予測とE/A/Fを固定し、SIGReg、形状非指定の分散・共分散正則化、時間残差への適用を比較する。これは設計候補であり、実装済みという意味ではない。

## 過去のコード・実験との区別

| 項目 | 状態と制限 |
|---|---|
| `multitask_lewm.py`、`multitask_jepa.py` | 過去の交差予測・識別損失の実装領域。名前だけでマルチタスク能力を実証したとはみなさない |
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
