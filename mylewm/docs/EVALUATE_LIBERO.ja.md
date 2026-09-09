# LIBERO-10の評価環境

データは `.cache/libero-datasets/libero_10` の10 HDF5。2視点は異なる実カメラ画像を使う。ローカルシミュレータは `external/libero`、robosuite1.4.0 / bddl1.0.1、隔離したMuJoCo3.3.7を使用する。

このGB10環境ではEGL画像に異常があったため、評価には検証済みのOSMesa wrapperを使う。

```bash
bash mylewm/run_libero.sh mylewm/tools/smoke_libero.py
bash mylewm/run_libero.sh mylewm/tools/audit_libero_images.py --task-index 0 --output FRESH_AUDIT_DIRECTORY
bash mylewm/run_libero.sh mylewm/tools/eval_rbg_libero.py --checkpoint CHECKPOINT --output FRESH_OUTPUT_DIRECTORY
```

OSMesaはUbuntu24.04 ARM64用の `libosmesa6=24.0.5-1ubuntu1`、`libglapi-mesa=24.0.5-1ubuntu1`、`libllvm17t64=1:17.0.6-9ubuntu1` を `apt download` し、`dpkg-deb -x PACKAGE .cache/libero-osmesa` で展開したもの。システム全体を変更しない。他環境では画像監査を再実行する。

評価には全10タスクの画像監査が必要。既定の監査先は `.cache/stable-wm/libero10/rbg_v0/render_audit/task_N/report.json`。新規監査は別出力先に保存し、評価の `--render-audit-dir` で指定する。学習済み共有LIBEROモデルの制御性能は未検証であり、BC成績とCEM成績を混ぜない。


新規の画像監査・評価出力は `output/libero10/` 以下の未使用ディレクトリへ保存してください。環境の構築・画像監査・制御評価は別工程です。各CLIの `--help` で必須引数を確認し、PushTの重みをLIBEROへ渡さないでください。
