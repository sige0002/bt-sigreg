# mylewm：学習・比較基盤

現在の研究案は[BT-SIGReg](docs/BT_SIGREG.ja.md)。**BT本体は未実装・未学習**で、既存のRBGをBT実装と呼ばない。研究目的・数学的限界は[総合レビュー](../README_RESEARCH_REVIEW.ja.md)を参照。長時間学習はユーザー指示で停止中。

## 構成

| 場所 | 残す理由 |
|---|---|
| 直下のPython | PushT/LIBERO共有trainer、モデル、損失、データ・再開・評価契約、診断の共通部品 |
| `tools/` | 公式評価、対応付き比較、BN校正、データ・画像・計算経路の監査、表現診断 |
| `tests/` | 残した実装の回帰テスト |
| `docs/BT_SIGREG.ja.md` | 現行アルゴリズム案 |
| `docs/VALIDATION.ja.md` | 学習・評価基盤の監査結果と未解決事項 |
| `docs/CLEANUP.ja.md` | 削除・移動の一覧と復元方法 |
| `run_libero.sh` | LIBERO専用OSMesa環境の起動wrapper |

`train_rbg.py` / `train_rbg_libero.py` / `rbg.py` はRaw/TCでも使う共有基盤なので残す。学習レシピを変更せず構成を整理したもので、RBGの比較モードは残っているが本命ではない。旧RBG実験キュー、混合アブレーション、Sub-JEPA専用trainer、ABC数値検算と重複した旧研究MDは作業ツリーから削除し、Git履歴に保存した。

## 検証と評価ツール

以下はリポジトリルートで実行する。

```bash
.venv/bin/python -m pytest mylewm/tests -q
# dry-runのみ。学習や環境評価は起動しない
.venv/bin/python mylewm/tools/plan_controlled_comparison.py --steps 100000 --methods raw
.venv/bin/python mylewm/tools/evaluate_official_pusht.py
```

公式PushT評価には `evaluate_official_pusht.py --execute`、記録行動の再実行には `replay_pusht_evaluation.py` を使う。過去の200ケース176成功はデータセット由来Goalの回帰評価であり、未使用最終テストや固定T被覆率とは区別する。比較は `compare_paired.py` / `compare_libero.py`、学習状態とBNの監査は `audit_rbg_checkpoint.py` / `calibrate_rbg_bn.py`。各CLIの引数は `--help` を参照。

## 学習基盤（自動実行しない）

PushTの共有trainerは `train_rbg.py`、LIBERO-10は `train_rbg_libero.py`。どちらも `prepare` と `train` を持ち、既存の `raw` / `tc` / `rbg` モードを扱う。BTモードはまだない。

主比較の計画はRaw/TC/BTを同じ新規初期値・データ順で100,000更新。CLI既定は50,000更新なので明示指定が必要。`plan_controlled_comparison.py` は旧Raw/RBG用のdry-run補助で、TC/BT/LIBEROを含む全比較パイプラインではない。

共有trainerの既定はbatch128、warmup500、max-lr5e-5、min-lr0。全更新の使用LRと提示数を記録する。`--resume` は同じ実験設定のみ許可し、予算やデータを変える場合は別出力先に新規実験を作る。旧checkpointは読み戻さない。初期値生成は `tools/create_shared_initialization.py`、検証済みの保存・復元処理は `training_state.py` にある。

100,000更新は公式配布checkpointの過去の学習履歴再現ではない。Raw/TC/BTの公平性、追加訓練計算量、seed差と実制御成績を評価する。残る課題は[検証記録](docs/VALIDATION.ja.md)を参照。

## LIBERO環境

データは `.cache/libero-datasets/libero_10` の10 HDF5。2視点は異なる実カメラ画像を使う。ローカルシミュレータは `external/libero`、robosuite1.4.0 / bddl1.0.1、隔離したMuJoCo3.3.7を使用する。

このGB10環境ではEGL画像に異常があったため、評価には検証済みのOSMesa wrapperを使う。

```bash
bash mylewm/run_libero.sh mylewm/tools/smoke_libero.py
bash mylewm/run_libero.sh mylewm/tools/audit_libero_images.py --task-index 0 --output FRESH_AUDIT_DIRECTORY
bash mylewm/run_libero.sh mylewm/tools/eval_rbg_libero.py --checkpoint CHECKPOINT --output FRESH_OUTPUT_DIRECTORY
```

OSMesaはUbuntu24.04 ARM64用の `libosmesa6=24.0.5-1ubuntu1`、`libglapi-mesa=24.0.5-1ubuntu1`、`libllvm17t64=1:17.0.6-9ubuntu1` を `apt download` し、`dpkg-deb -x PACKAGE .cache/libero-osmesa` で展開したもの。システム全体を変更しない。他環境では画像監査を再実行する。

評価には全10タスクの画像監査が必要。既定の監査先は `.cache/stable-wm/libero10/rbg_v0/render_audit/task_N/report.json`。新規監査は別出力先に保存し、評価の `--render-audit-dir` で指定する。学習済み共有LIBEROモデルの制御性能は未検証であり、BC成績とCEM成績を混ぜない。
