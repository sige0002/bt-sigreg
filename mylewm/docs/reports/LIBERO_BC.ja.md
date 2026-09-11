# LIBERO BCの実装・検証とBT学習起動（2026-09-11）

ユーザー依頼で、凍結ViT＋タスクID条件付きflow matching行動模倣方策を追加しました。[利用手順・論文との条件差](../BEHAVIOR_CLONING.ja.md)。BCの本学習・成功率比較は行っていません。

## 実装

- `data/libero_bc_data.py`：既存デモ分割の検証、2カメラ同時画像、境界を越えないnative行動chunk。移転した元モデルは元manifestと記録済みhashを照合。
- `policy/libero_bc.py`：凍結ViTのCLS＋4×4 patch平均、34 token、タスクID付きDiT型flow head、10 Euler step。学習済みタスク名からembeddingを引き、環境IDとファイル順を混同しない。
- `training/train_libero_bc.py`：既定dry-run、方策だけのAdamW更新、固定validation、ViT不変性、推論保存と厳密再開。
- `evaluation/evaluate_libero_bc.py`：既定dry-run、OSMesa監査、native固定初期状態、実観測からの再生成、native成功判定、動画・試行一覧・viewer自動生成。
- `compare_libero.py`：同じBC訓練条件・checkpoint更新数・評価条件を照合。BCとCEMの混在を拒否。

TC-LeWM v3の[付録A.1](https://arxiv.org/html/2607.26924v3#A1.SS1)を確認。[著者ページ](https://ryuuchou17.github.io/tclewm/)のコードは確認時点でComing soonです。幅256・depth4等はローカル選択であり、公式コードや論文成功率の再現ではありません。CEMによる世界モデルの計画評価は別経路として維持します。

## 検証

| 確認 | 結果 |
|---|---|
| CPU全回帰 | 185合格、CUDA専用18スキップ、68 warnings、91.96秒 |
| 追加BCのGPU確認 | 1合格、16.88秒。凍結ViT、方策の逆伝播、有限かつ有界の生成行動 |
| 保存配置変更後の関連CPU回帰 | 34合格、CUDA専用BC 1スキップ、8.35秒 |
| 最終変更の関連CPU回帰 | BC・両viewer・比較の24合格、CUDA専用BC 1スキップ、8.03秒 |
| 合成データでの再開 | 4連続更新と2＋2更新で方策・optimizer・Torch乱数が完全一致 |
| 実データBC | 旧BT LIBERO 100更新checkpointから、CPU／batch2でBC 2更新保存→再開して4更新 |
| 実環境接続 | native task 0、init 0、予算9行動、2回の行動列生成、正常終了・status succeeded・viewer生成 |

18スキップのうち17件は既存CUDA専用テスト、1件は今回追加したBCのGPUテストです。後者だけは別途GPUで実行しました。全GPU回帰合格とは呼びません。

最終確認で`--deterministic`用のcuBLAS設定も共有学習ループと揃えました。ただし決定論モードの追加GPU試験は、モデルをCUDAへ転送する段階でメモリ不足となり未検証です（`/tmp/bt-libero-bc-gpu-deterministic-test.log`）。先の通常GPU試験1合格と区別します。2本の本学習は稼働継続を確認し、停止や全体cache解放はしていません。

実データ確認は`output/libero10/bc_implementation_check/train/`に保存。凍結ViTは5,501,376パラメータ、方策は5,985,287パラメータです。固定validation flow lossは2更新時2.60468、4更新時2.43697でした。短期動作確認であり、学習性能の結論ではありません。

実環境の最終記録は`output/libero10/bc_implementation_check/eval_native_task0/`。9行動では未成功（0/1）で、環境ループ約3.20秒、CPU方策生成2回の合計約1.59秒です。短期checkpoint・短い予算の接続確認であり、成功率や実機制御Hzの評価ではありません。方策への入力は現在画像・タスクIDのみで、成功デモ画像は表示だけに使います。

### 接続確認で検出した環境問題

不足していた`bddl==1.0.1`、`easydict==1.9`、`future==1.0.0`、`gym==0.25.2`を検証専用`output/libero10/bc_implementation_check/runtime/`へ導入し、libero依存グループにも記録しました。共有`.venv`はsyncしていません。lockfileの既存パッケージ版は維持し、追加依存のみ解決しました。

`.cache/libero-config/config.yaml`の旧フォルダ参照を現行ルートへ修正し、元ファイルは`bc_implementation_check/libero_config_before_path_fix.yaml`へ退避しました。移行前の画像監査`task_0`はファイル順のtaskとnative taskが異なり、評価器が拒否しました。旧コピーを`mismatched_legacy_audit_task0/`へ退避してnative ID 0で再監査しました。6比較のnative MAEは3.38〜7.90で全て閾値10未満、縦反転より小さく、MuJoCo 3.3.7／OSMesaで合格しました。他9タスクの新規監査は今回未実施です。

途中の依存不足・監査不一致の失敗は各`eval*`ディレクトリのfailed statusと`/tmp/bt-libero-bc-real-eval*.log`に保持しています。失敗した起動を制御試行として数えていません。

## 依頼されたBT LIBERO-10の100,000更新

別途の明示依頼に従い、**世界モデル**のBT学習をバックグラウンド起動しました。BCの長時間学習ではありません。

- run：`output/libero10/bt_spectral_v2_100k_s3072/`
- service：`bt-libero10-100k-s3072.service`（user systemd、Restart=no）
- 100,000更新、batch128、workers4、seed3072、LR5e-5、warmup500、保存1,000更新ごと。
- BT Cayley v2、depth2、kappa .2、hidden192、2カメラ・共有世界モデル。
- 現行`output/manifests/libero10/manifest.json`を使用。起動時はサイズ・mtime確認。
- PushT学習と同じGPU0。起動後のGPU観測約26,267MiB、Torch peak約26,934,989,312 bytes、available RAM約33GiB。
- 開始時Gitは`3cd7e554168c1ce77ed038621ba944993c24eeb5`。`output/libero10/source_bt_100k_3cd7e554/`へソースを固定し、`PYTHONPATH`と`BT_SIGREG_ROOT`をそのコピーへ指定。今回の実装変更を稼働中runへ混ぜない。
- 実コマンドと場所は`output/libero10/bt_spectral_v2_100k_s3072_launch.json`。再開も固定ソース・当初の環境を使用し、hash照合を無効化しない。

記録時点では起動・更新進行を確認した段階です。10万更新の完了や制御性能は未報告です。自動評価・BCの自動起動は予約していません。

```bash
systemctl --user status bt-libero10-100k-s3072 --no-pager
journalctl --user -u bt-libero10-100k-s3072 -n 5 --no-pager
tail -n 1 output/libero10/bt_spectral_v2_100k_s3072/metrics.jsonl
```
