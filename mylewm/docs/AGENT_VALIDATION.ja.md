# 評価・比較の制約

評価、データ loader、依存、比較条件を変更するときの資料。パスはリポジトリルート基準。新規評価の実行にはユーザーの明示依頼が必要。

PushT評価のepisode列は`src/mylewm/data/pusht_eval_data.py`で正規化する。ロード対象の列一覧だけから保存済みepisode列の有無を推測しない。SWMのep_len/ep_offsetからの補完はメモリ上で行い、保存済みHDF5へ列を書き足さない。

PushT評価launcherも通常は全量データhashを走査しない。`--verify-data`指定時のみ再走査し、未計算hashはnull、prepare時hashは別フィールドに保存する。起動端末とconsole.logへ処理段階・CEM開始・環境step進捗を逐次出す。

PushT評価のHDF5は`--dataset`、manifestの`dataset`の順で解決する。固定cacheやsymlinkを要求しない。移転時はサイズを確認し、明示した全量検証時だけ新規prepareのSHA-256と照合する。評価のprovenanceとloaderに同じ解決済みパスを使用する。

ユーザー指定の途中checkpoint評価は本学習終了前でも可能。`mylewm/docs/EVALUATION.ja.md`に従い、保存完了済み`step_N_object.ckpt`を使用する。共有メモリ型GB10ではRAMに余裕があれば同一GPUで学習と評価を併走できる。使用量の実測は`mylewm/docs/reports/CACHED_CEM.ja.md`を参照する。最終評価のcompleted.json条件を途中評価へ適用しない。文書更新だけを理由に評価は開始せず、共有学習環境の依存を同期しない。

依存更新時は既存object checkpointの読込も検証する。現在はTransformers 4.57.6を固定（5系では旧ViTEncoderの復元失敗）。稼働中の学習用`.venv`にsyncせず、`UV_PROJECT_ENVIRONMENT`で隔離して検証する。CPU限定テストのCUDA5件スキップは理由と件数を報告し、GPU合格とは扱わない。

評価は起動だけで完了扱いしない。明示依頼された評価は終了コード・結果・launcherの`status.json`を照合して報告する。SIGKILL等ではstatus更新ができないため、実プロセスも確認する。廃止済みの`watch_evaluation.sh`は削除済みで、定期監視service・エージェントを起動しない。通知経路は未成立であり「無人でも必ず気付く」と主張しない。失敗時はログを保持して原因を確認し、無限再起動や他GPUサービス停止をしない。

PushT評価の初心者用シェルは`bash scripts/evaluate_pusht.sh --help`。既定はdry-run、`--execute`だけがGPU評価を起動する。手順は`mylewm/docs/EVALUATION.ja.md`。新規`output/`子ディレクトリへ出力し、信頼済み`*_object.ckpt`だけを入力する。GB10の対象データclean cache解放は明示フラグで行い、他プロセス停止や全体cache削除はしない。既存評価と重複起動しない。公式配布checkpointと途中checkpointの差を、同更新予算の方式の優劣と呼ばない。

- 主比較の計画はRaw/TC/BTを同じ新規E/A/F初期値、データ順、100,000更新で学習するもの。計画と実施済みを区別する。
- データ分割、前処理、行動座標、精度、optimizer、計画予算、開始状態、Goal、環境成功関数を揃える。追加Tの訓練計算量も報告する。
- PushTだけでマルチタスクを実証しない。LIBERO-10の平均・各タスク・下位タスクと学習seed間の変動を報告する。
- TC論文の凍結表現＋BCと、CEMによる計画成功率を直接順位付けしない。公式checkpoint再現と同予算での再学習も別の比較である。
- 生の潜在MSE低下を性能向上と呼ばない。尺度・アフィン対照、実制御成績、信頼区間を確認する。有限試験で無条件の非劣化保証をしない。
- コード変更後は関連テストを実行する。通常の回帰確認は `.venv/bin/python -m pytest mylewm -q`。旧ABCの検算はGit履歴に保存済み。回帰テスト合格をBTの学習・制御評価と呼ばない。
- プッシュ前に `git diff --check`、差分、追加ファイル、秘密情報・生成物の混入を確認する。失敗や未検証部分を隠さない。
