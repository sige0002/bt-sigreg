# 検証状況と未完了事項

更新日：2026-09-09。不要な独立診断・旧比較準備CLIと専用テスト13ファイルはユーザー承認で削除した。[削除一覧・復元方法](CLEANUP.ja.md)を参照。以下の旧診断実績は当時のコードでの結果である。

実験の現在地をまとめる文書です。個別の数値・ハッシュ・失敗記録は[レポート一覧](reports/README.md)から参照してください。

## 実施済み

### 公式ライブラリへのPushT移行（2026-09-09）

`mylewm/train.py` の新レシピ `pusht_spt_v1` を実装。SWMのHDF5Dataset、公式画像前処理・`lejepa_forward`・SIGReg、SPTの逆伝播／optimizer／scheduler、Lightningのループ・CSV・checkpointを使用する。エピソード分離・train-only統計を維持するが、クリップ末尾条件・抽出・LR添字は旧経路と異なる。[条件差と手順](TRAINING.ja.md)を参照。

追加の9テストで次を確認した。性能実験ではない。

- 小型モデルのRaw／恒等BTで損失・全モデル勾配が一致し、旧Rawの同一入力での損失も一致。未来教師勾配とTへの予測損失勾配不在を確認。
- 本物のPushT E/A/FでもRaw／恒等BTの損失・勾配が一致（CPU、合成28×28画像）。実画像制御や224×224での長期学習ではない。
- native HDF5ローダーを自己生成fixtureへ適用し、episode分離・形状・正規化・固定validationケースを確認。
- CPU小型BN/Dropoutモデル、worker0/2で連続6更新と3更新＋再開のバッチ・loss・全state・optimizer・schedulerが完全一致。モデルとTの更新、Tなしexport、設定変更時の拒否を確認。
- 新entrypointのRaw／BT両方でnative fixture・小型モデルの3更新が完了し、CSV・最終checkpoint・完了記録と出力上書き拒否を確認。

最初の接続試験では画像前処理の引数不足、テスト内の非leaf Tensorのdeepcopy、再開時のvirtual epoch長の扱いを修正した。SPTが登録するHardwareMonitorは公開設定のキー一覧に無く、生成後・setup前に除外した。環境情報の背景収集・外部tracker・追加モデルexportも無効化。独自最適化ループは追加していない。

旧コード・manifest・既存100k重み・評価結果は変更していない。新経路の本学習・PushT成功率・GPU長期再開・LIBERO移行は未実施で、旧checkpointからの互換resumeも許可しない。

全回帰は `CUDA_VISIBLE_DEVICES='' PYTHONPATH=.:lewm .venv/bin/python -m pytest mylewm/tests -q` で **111合格・5スキップ**（14.35秒）。CUDA専用5件は未実行。fork/LanceとLightningのログ・再開に関する警告は残るが、上記CPU再開の実測一致を別途確認した。実manifestでの新CLI dry-run、Markdownリンク・見出し参照、Python構文、`git diff --check`も確認した。

実PushTデータでも学習を起動せずnative loaderを確認し、train 1,585,717クリップ、固定validation 256件、取得画像4×3×224×224・行動4×10を確認した。旧train 1,645,509クリップとは末尾条件が異なる。実データの確認は1クリップの読込までで、全クリップの内容監査・実データ学習・成功率試験ではない。

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

[学習記録](reports/PUSHT_TRAINING_100K.ja.md)・[評価レポート](reports/PUSHT_CHECKPOINT_EVALUATION.ja.md)・[実装監査履歴](reports/IMPLEMENTATION_AUDIT.ja.md)に証拠を分離しています。過去のテスト件数は実行時点の範囲を示し、現在のテスト件数や制御試行数と混同しません。

## 未完了・主張できないこと

- 同じ新規E/A/F初期値・データ順・100,000更新のRaw/TC/BT比較。
- LIBERO-10の本学習、共有モデルの平均・各タスク・下位タスクの制御成績。
- 複数学習seedの変動、非線形Tの効果と単なる尺度・アフィン効果の切り分け。
- 追加Tの学習計算量を含めた同実測計算予算での比較。
- 公式配布重みの正確な過去学習履歴とnative Lance経路の再現。
- 上流eval今回50ケースでの公式配布重みの直接比較。

PushTだけでマルチタスク改善を証明しません。98%を論文の3学習seed平均96%への優越とはしません。固定confirm 200件も既使用の回帰集合であり、未使用最終テストではありません。有限試行で任意の条件に対する非劣化を保証しません。

## Issueと運用

最後に記録したGitHub対応は2026-09-07の #1・#3 close、その他は未完了です。今回GitHubの最新状態は照会・変更していません。監査履歴にある過去の状態と現在のissue状態を同一視しません。

追加学習・評価は明示依頼時のみ。定期監視・自動評価予約は行いません。手動の進捗確認は `bash mylewm/tools/monitor_training.sh --once`、評価手順は[PushT](EVALUATE_PUSHT.ja.md)／[LIBERO](EVALUATE_LIBERO.ja.md)です。完了済み学習のログが増えないことを障害とは扱いません。

今回の構成整理と回帰確認は[整理記録](CLEANUP.ja.md)へ記録します。新しい性能試験は行いません。
