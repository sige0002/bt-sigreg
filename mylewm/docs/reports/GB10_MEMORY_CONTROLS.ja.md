# GB10のCUDA初期化OOM：PushT／LIBERO対照試験

2026-09-11。ユーザーの対照試験依頼に従い、停止済みの本学習とは別の短期runで調べた。元のPushT・LIBERO本学習は再開していない。

**同一GPUでの併走時に、元の障害と同じCUDA context確保OOMを再現した。PushTの`pin_memory`だけを無効にした条件では観測中の失敗がなく、有効に戻すと再発した。** また最初の併走試験では、PushTを動かしたままLIBEROだけを停止すると、新規CUDA初期化が3回連続で成功した。

今回特定できたのは、PushTの大きな固定CPUメモリ保持と、LIBERO追加によるメモリ負荷が重なる構成での障害である。LIBERO単独の試験では再現しておらず、LIBERO実装のメモリリークを原因とする証拠ではない。利用可能RAMが約40GiBあっても、ドライバが要求する連続領域を確保できない状態になる。要求サイズ・ドライバ経路の先行調査は[CUDA Driver API診断](LIBERO_BC.ja.md#追加の原因調査cuda-driver-apiと要求サイズ)を参照する。

## 条件と観測方法

環境はGB10、Linux上のRAM約119.6GiB、driver 580.95.05、CUDA 13.0、PyTorch 2.9.1+cu130。試験時のGitは`424a49750a2208942e967c2a111a80f1c79a01ac`。共有`.venv`・学習ソース・元runのconfig／manifest／checkpointは変更していない。

| 項目 | PushT | LIBERO-10 |
|---|---|---|
| CLI | `mylewm.training.train --mode bt` | `mylewm.training.train_libero train --mode bt` |
| batch / workers | 256 / 8 | 128 / 4 |
| 精度・compile | bf16-mixed、encoder compile有効 | 元runの既定条件 |
| 固定CPUメモリ | 有効／対照だけ無効 | 全条件で有効 |
| 画像 | 4フレーム、224×224、float32 | 4フレーム×2カメラ、128×128、uint8 |
| manifest | `output/manifests/pusht/manifest.json` | `output/manifests/libero10/manifest.json` |

seed 3072、LR 5e-5、warmup 500、100,000更新の学習率スケジュールを維持した。観測終了時に外側から短期診断プロセスを停止し、短い総更新数に合わせてスケジュールを変えていない。観測予算は各学習128更新、上限480秒。併走条件はPushTが16更新以上になってからLIBEROを追加する。

独立した新規プロセスで約15秒ごとに`cuDevicePrimaryCtxRetain`を実行し、終了コード・CUDAエラーを保存した。システムRAM・buddyinfoと、各学習プロセス内のdevice／host allocator統計を約2秒ごとに記録した。学習には読取専用の観測スレッドを付けた通常CLIを使い、モデル・データ処理・逆伝播・optimizerを置換していない。

初期化失敗が2回連続した場合は、その診断のLIBEROだけを止め、PushTを継続して新規CUDA初期化を3回観測する。MemAvailableが8GiB未満なら終了する条件も設けた。OS再起動、手動compaction、全体cache解放、他のサービス停止は行っていない。

## 結果

14:12〜14:41 JSTに5条件を実行し、全coordinatorが終了コード0・`status.json=finished`で終了した。

| 条件（実施順） | 最後の同時稼働観測の更新数 PushT / LIBERO | 両者が更新開始後の観測時間¹ | 新規CUDA初期化の失敗 / 試行¹ |
|---|---:|---:|---:|
| PushT単独、pin有効 | 129 / — | 141秒 | 0 / 8 |
| LIBERO単独、pin有効 | — / 129 | 155秒 | 0 / 9 |
| 併走、両者pin有効 | 154 / 93 | 282秒 | **2 / 17** |
| 併走、PushTだけpin無効 | 148 / 128 | 307秒 | **0 / 18** |
| 併走、PushTのpinを再有効化 | 180 / 128 | 388秒 | **1 / 23** |

¹ 単独条件ではその1本の更新開始後。起動前・他方の起動待ち・停止後のプローブは表に含めない。観測時間は約2秒周期の最初と最後の有効sample間で、初期化から停止までのwall timeとは異なる。

最初の併走試験では、14:24:59のPushT 149／LIBERO 88更新と、14:25:15の154／93更新で`cuDevicePrimaryCtxRetain`が`CUDA_ERROR_OUT_OF_MEMORY (2)`になった。直前のMemAvailableはそれぞれ37.65／38.38GiB。各失敗プローブ直後のNormal zoneはorder 9以上が0で、kernel logも`_memdescAllocInternal`、`kgrctxAllocMainCtxBuffer`の`NV_ERR_NO_MEMORY (0x51)`を記録した。

この2連続失敗後、LIBEROだけを停止した。PushTを継続したまま新規初期化は**3/3成功**し、Normal zoneのorder 9以上の空きも数千ブロックへ回復した。その後にPushTも停止した。これは2本をまとめて止めた先行調査と異なり、LIBERO追加負荷の寄与を分離した観測である。

PushTのpinを無効にした条件は、LIBERO 128更新までの併走中に失敗がなかった。試験全体の最小MemAvailableは44.35GiB、通常のシステム観測におけるNormal zoneのorder 9以上の最小空きは15ブロックだった。

再有効化条件では14:35:53〜54、PushT 64／LIBERO 23更新で同じOOMが再発した。直前のMemAvailableは38.52GiB。失敗プローブ直後のorder 9以上は0、kernel logも同じmain context確保失敗だった。以後の定期初期化は成功し、180／128更新まで観測して終了した。したがって**元設定で常に失敗するのではなく、同じ構成で確保可否が変動する**。

試験間の開始時order 9以上の空きブロック数は、併走pin有効7,124、無効8,983、再有効化11,595と異なる。再有効化は開始時に最も多くの連続領域があっても途中で再発した。ブロック数はorderごとの個数の和であり、2MiB単位へ換算した空き容量ではない。

停止処理中にも他方が進むため、最終ログの更新数は同時稼働の終了点と異なる。最初の併走試験のPushTは回復観測を含め207更新、pin無効条件のLIBEROは139更新、再有効化条件のLIBEROは143更新まで記録された。これらを全て両者同時稼働の更新数に数えていない。

## メモリの実測と解釈

PushTの固定host allocatorの予約量は、有効条件で約16.001GiB、無効条件で実質0だった。LIBEROは全条件で約0.564GiB。以前の「先読み16枠なら約16GiB」という計算を、今回の実際のDataLoader稼働中にも確認できた。ただしvalidation前の観測であり、全期間の総上限ではない。

GPU allocatorの最大予約量はPushTが約22.9〜23.4GiB、LIBEROが約25.4GiB。LIBEROのpeak allocatedは約25.09GiBで、先行する本学習の記録とも一致した。GPU予約量・host予約量・RSS・cacheは異なる指標であり、そのまま合算して物理RAM総使用量とはしない。

PushTの`--no-pin-memory`は固定host領域の保持を減らすが、workerが先読みする画像のRAM自体は引き続き必要になる。メモリ総量の低下と、ページを移動可能にする効果を今回の試験だけで数量分解してはいない。最小worker数・最小pin削減量、長時間の性能・安定性も未測定である。

無効化にはデータ転送待ちが増えて学習速度が下がる可能性がある。一般的なデメリットと、このGB10で速度差をまだ確定していない点は[学習手順のpin-memory説明](../TRAINING.ja.md#pin-memory-tradeoffs)に記載した。本試験のwall timeや異なる同時稼働区間を、そのまま速度比較として使わない。

このPyTorch環境では、反復中のhost統計`allocated_bytes.current`が`reserved_bytes.current`や物理RAM量を超えて増える記録があった。一方で予約量と`num_host_alloc`は安定していた。このactiveカウンタを実使用量やリークの証拠にせず、reservedとプロセス・システム観測を使用した。raw値は証拠に保持している。`VmPin`／`VmLck`が0でも固定メモリなしとは判断できないことは[単一バッファ実測](LIBERO_BC.ja.md#memory-attribution)で確認済み。

## 比較の限界と本学習の状態

同じ学習種別の各試験で、初期モデル・manifest・学習ソースhashの一致を確認した。保存configの差分はPushTが`pin_memory`だけ、LIBEROが出力先だけだった。試験間でOSの物理配置を初期化していないため、開始時の連続空き領域と失敗時点は異なる。無効条件を有効条件で挟んだ再確認により設定変更の寄与を調べたが、短期1seedの結果を長時間のOOM不発保証や学習性能の比較とはしない。

診断の学習子プロセスは、規定の観測後またはOOM切り分けのためにSIGINTで停止した。記録された終了コード1はこの意図的な中断によるもので、学習中の自発的クラッシュや100,000更新完了として数えない。外側のcoordinatorの終了コードと`status.json`を照合済み。新規contextのOOM時にも、先に確保済みのcontextを使う2本の学習は更新を続けていた。

元の本学習はPushTが保存済み30,000更新、LIBEROが保存済み2,000更新の状態で停止を維持する。再開条件や既定のpin設定を勝手に変更しない。今回の緩和条件を既存runの厳密再開へ適用する検証は行っていない。

終了後は各条件の新規CUDA初期化が成功し、全診断PID・DataLoader worker・GPU compute processの残留がないことを確認した。元LIBERO serviceはinactive/dead、MainPID=0。元のPushT `step_30000.ckpt`／`last.ckpt`とLIBERO `resume.pt`のSHA-256・サイズ・mtimeも停止前の記録と一致した。今回のGit変更は記録と手順のみで、通常回帰pytestの追加実行はしていない。

## 証拠

ローカル保存先は`output/diagnostics/gb10_memory_controls_20260911/`。生成ログ・初期重みはGitへ追加しない。

- `protocol.json`：観測予算、適応的な切り分け、元runを停止維持する範囲。
- `control.py`・`train_observed.py`：診断用wrapper。初回PushT単独時のwrapperは`control_initial_pusht.py`にも保存。以後の変更はLIBEROの`CUBLAS_WORKSPACE_CONFIG`を元run同様に未指定へ揃えたもの。
- 各条件の`*_launch.json`、`*_run/config.json`：実コマンド、環境、初期モデル・manifest・ソースhash。
- 各条件の`system.jsonl`、`*_observation/memory.jsonl`、`context_probe_*.log`、`context_probes.jsonl`、`events.jsonl`、`status.json`：メモリ・更新数・エラー・停止対象。
- `summary.json`・`audit.json`：集計、同時稼働区間だけのプローブ、失敗時点、比較条件、元checkpointの不変性・終了後の状態。
- 失敗した併走条件の`kernel.log`：元の障害と同じドライバのcontext確保失敗。

最初の併走条件では追加の小規模GPU MLPも4更新成功したが、その約26秒後から新規contextのOOMが出た。可否が時間で変化することを示す補助観測である。その後のブラウザ確認はPlaywright実行ファイルの場所が合わず起動前に失敗し、Chromiumは起動していない。OOMはその試行より先に発生した。再有効化条件にはこのMLP・ブラウザ試行を追加しておらず、今回の再現にブラウザ起動は必要なかった。固定メモリ無効条件の補助MLPは終了付近で成功したが停止処理と重なるため、両学習継続中の成否は定期Driver APIプローブで判定する。
