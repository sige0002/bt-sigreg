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

### GPU試験のOOMを追加診断（同日）

「RAM容量が足りない」という説明は不正確でした。**モデルを作らず、決定論モードも使わない新規プロセスの`torch.cuda.mem_get_info()`だけで再現**しました。CUDAコンテキストの初期化バッファ確保が失敗しており、BCの重み・活性・逆伝播が消費する容量によるOOMとは区別します。

| 実測 | 対象cache解放前 | 解放直後 |
|---|---:|---:|
| MemAvailable | 40.38 GiB | 40.64 GiB |
| MemFree | 1.62 GiB | 7.79 GiB |
| Cached（Shmemを含む） | 57.71 GiB | 51.68 GiB |

既存PushT HDF5の先頭8GiB範囲だけへ`POSIX_FADV_DONTNEED`を明示的に実行し、約6GiBのclean file cacheを解放しました。ファイルのsize・mtimeは不変です。解放後もモデルなしのCUDA初期化は同じエラーとなり、単なるcache量だけでは解消しませんでした。共有`.venv`の更新、全体`drop_caches`、OS設定変更、本学習の停止はしていません。

ドライバ580.95.05のkernel logは`kgrctxAllocMainCtxBuffer`→`kernel_graphics_context.c:1178`で`NV_ERR_NO_MEMORY`を記録。該当版の[コンテキスト確保処理](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/580.95.05/src/nvidia/src/kernel/gpu/gr/kernel_graphics_context.c#L1115)と[HAL定義](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/580.95.05/src/nvidia/generated/g_kernel_graphics_nvoc.h#L862)はmain contextに連続領域を要求しています。解放直後の`/proc/buddyinfo`ではNormal zoneのorder 9以上が0（4KiB pageなので2MiB以上の空きブロックなし）。**物理メモリの断片化による確保失敗が有力ですが、失敗要求の実サイズ・割当経路の直接トレースは未実施で、根本原因を完全確定したとはしません。** CMA空き約1.78MiBも観測しましたが、それが当該要求の原因とは断定しません。プロセスのcgroupにmemory.max/high上限はありませんでした。

NVIDIAも[容量内で起きるUMA／cache関連のメモリ問題](https://nvidia.custhelp.com/app/answers/detail/a_id/5776)を案内しています。ただし今回の部分cache解放は不成功で、この一般的案内だけから原因確定・解消済みとはしません。検証結果は引き続き「通常BC GPU試験1合格、追加の決定論GPU試験は初期化で未検証」です。

証拠は`output/libero10/bc_implementation_check/memory_diagnosis_20260911/`のprobe、JSON行付きログ、kernel log、buddyinfo、参照した同版ドライバソースに保存しました。RAMの余裕は同一GPU併走の必要条件ですが、GB10では新規CUDAコンテキストが作れるかも別途確認します。

### 追加の原因調査：CUDA Driver APIと要求サイズ

ユーザーの追加調査依頼に従い、PyTorchを一切importせず`libcuda.so.1`へ直接問い合わせました。**新規CUDAコンテキストの初期化に必要な連続メモリを確保できないこと**が失敗原因と判断できます。RAM総容量やBC計算量による不足ではありません。

| 実測した呼び出し・状態 | 結果 |
|---|---|
| `cuInit(0)` | CUDA_SUCCESS |
| `cuDeviceGetCount` | CUDA_SUCCESS、1 GPU |
| `cuDevicePrimaryCtxRetain` | CUDA_ERROR_OUT_OF_MEMORY（2） |
| RMのgraphics engine contextサイズ照会（`0x801707`、engine 0） | NV_OK、基本サイズ1,145,600 bytes、alignment 4,096 |
| 最大subcontext数照会（`0x20801201`、info index `0x2c`） | NV_OK、64 |
| CUDA計算オブジェクトclass `0xcec0` のRM allocation | NV_ERR_NO_MEMORY（`0x51`） |
| 失敗区間の`/proc/vmstat`差分 | compact_stall +3、compact_fail +3、compact_success +0、allocstall_normal +2 |
| 同時点のNormal zone空きブロック | order 9以上が0（2MiB以上なし） |

診断プロセスだけに`LD_PRELOAD`で`ioctl`記録を加え、元の要求を変更せず、同プロセスが所有するdevice/subdevice handleへの読み取り専用照会を追加しました。既存学習プロセスへのattachや照会はしていません。サイズ照会だけなら終了コード0、コンテキスト作成まで進むと終了コード2で再現します。`cuInit`成功だけをコンテキスト作成成功と扱いません。

要求サイズと失敗の対応は次の通りです。

1. ドライバ580.95.05の`kgraphicsGetMainCtxBufferSize`は基本サイズからmain contextを構成します。subcontext headerを含める経路では`align_up(1,145,600,4096) + 4096×64 = 1,409,024 bytes`です。ヘッダ有無にかかわらず1MiBを超え2MiB以下です。[同版ソース](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/580.95.05/src/nvidia/src/kernel/gpu/gr/kernel_graphics.c#L1705)
2. `kgraphicsShouldForceMainCtxContiguity`はtrueを返します。Linux側の`nv_alloc_contig_pages`は`get_order(page_count×PAGE_SIZE)`で切り上げて`__get_free_pages`を呼ぶため、このサイズは**order 9＝2MiBの連続領域**を必要とします。[Linux側ソース](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/580.95.05/kernel-open/nvidia/nv-vm.c#L358)
3. その空きブロックがなく、失敗区間では連続空き領域を作るcompactionも成功していません。先の約6GiBの部分cache解放でもNormal zoneのorder 9以上は増えず、同じ失敗が続きました。

`vmstat`はシステム全体のカウンタなので、3件全てを診断プロセスに帰属させたカーネルトレースとは区別します。どの常駐・移動不能ページが断片化を固定しているか、既存GPU allocationとpin-memoryの寄与の内訳は未特定です。CMAやBC実装を単独原因と断定しません。BPF/perfによる割当引数・戻り値の直接追跡は管理者権限が必要で、このセッションの`sudo -n`では実行できませんでした。

主な追加証拠は同じ診断ディレクトリの`driver_probe.py`、`trace_ioctl.c`、`driver_probe_compaction.log`、`ioctl_trace_compaction.log`、`ioctl_query_only.log`です。shared libraryはその場で`cc -shared -fPIC ... -ldl`により作成した診断専用物で、学習環境へ導入していません。診断後もPushTとLIBEROの元PID・更新進行を確認しました。解消のためのOS設定変更、手動compaction、学習停止・再起動、ドライバ更新は未実施です。

### 新規学習コマンドでの確認（同日12:39〜12:41 JST）

「本当に新規の学習ができないか」という依頼に対し、既存のPushT・LIBERO学習を動かしたまま、新規プロセスで実際の学習コマンドを実行しました。**この時点では新しいGPU学習は開始できず、CPUの小規模学習と既存GPU学習は動作しました。** 確認対象はこの環境状態における新規CUDAプロセスの起動です。

| 新規プロセス | 指定予算 | 完了更新数 | 終了コード・所要時間 | 結果 |
|---|---|---:|---|---|
| 小規模MLP、CPU | batch 8、4更新 | 4 | 0・3.71秒 | 逆伝播・SGD更新が完了 |
| 同じMLP、CUDA 0 | batch 8、4更新 | 0 | 1・2.58秒 | `.to('cuda:0')`でOOM |
| BC CLI、実データmanifest・凍結BT ViT | batch 2、2更新、workers 0 | 0 | 1・9.22秒 | データ・encoder由来の照合後、方策のCUDA転送でOOM |
| BT LIBERO世界モデルCLI、新規初期値 | batch 2、2更新、workers 0 | 0 | 1・14.68秒 | データmetadata確認・ViT生成後、モデルのCUDA転送でOOM |

MLPは2,372パラメータ（float32の重み合計9,488 bytes）です。GPU側の3試行ともoptimizer更新より前に失敗し、同時刻のkernel logに`kgrctxAllocMainCtxBuffer`の`NV_ERR_NO_MEMORY`が残りました。BCの`status.json`も`failed`です。BT CLIは出力ディレクトリ作成前に終了したため、外側で実コマンド・終了コード・ログを保存しました。2更新の指定を、2更新完了として扱いません。

試行後のMemAvailableは40.90 GiBでしたが、Normal zoneのorder 9以上の空きブロックは0でした。先のDriver API診断と整合し、モデルやbatchを小さくするだけでは今回の初期化失敗を回避できませんでした。OS・ドライバ状態を回復させた後も新規学習できない、という意味ではありません。回復操作は今回未実施です。

12:40:59時点で元PIDは両方生存し、PushTは32,675更新（CSVの`step=32674`、`update=32675`）、LIBEROは2,048更新まで進行し、両ログは約2秒前に更新されていました。新規の長時間学習や再試行キューは追加していません。

証拠は`output/libero10/bc_implementation_check/memory_diagnosis_20260911/fresh_training_123954/`に保存しました。`results.json`、`bt_cuda_result.json`にコマンド・終了コード・時間、各`*.log`に実行結果、`kernel.log`に該当時間帯のドライバ記録、`state_after.json`にRAM・buddyinfo・既存学習の進捗を記録しています。

<a id="gpu-recovery-after-stop"></a>

### 学習停止後の回復確認（同日12:55〜12:56 JST）

ユーザーの「状態の解消しよう。とりあえず学習止めてから」に従い、PushTへSIGINTを送り、LIBEROのuser serviceを停止しました。**停止後に新規CUDA初期化が回復し、直前と同じ条件のBT世界モデルGPU学習が2更新・validation・checkpoint保存まで正常終了しました。** OS再起動・ドライバ変更・手動compaction・全体cache解放は行っていません。

| 停止後の確認 | 結果 |
|---|---|
| 元の2 PID・DataLoader worker | 終了、LIBERO serviceはinactive/dead |
| MemAvailable | 約115 GiB |
| Normal zoneのorder 9空きブロック | 停止前0から、停止直後5,298個（各2MiB）に回復 |
| 小規模MLPの新規GPU学習 | 4更新、終了コード0、2.06秒 |
| 新規BT LIBERO学習、batch 2・workers 0 | 2更新、終了コード0、8.02秒、`completed.json`もstep 2 |
| BT確認runのGPU利用 | configのdeviceはcuda、metricsのpeak allocatedは726,093,312 bytes |

MLPとBTの確認はユーザーの「まだ再実行しなくていい」が届く前に終了しました。以後は追加の学習起動・再開を行わず、両本学習を停止したままにしています。GPU初期化の回復を確認した短期試験であり、長時間併走時の再発防止は未検証です。GPU専用回帰の追加実行も行っていません。

| 停止した本学習 | 最後に記録された更新 | 保存済み再開点 |
|---|---:|---:|
| PushT `bt_compiled_100k_s3072` | 33,016（CSV stepは33,015） | 30,000（`step_30000.ckpt`・`last.ckpt`） |
| LIBERO `bt_spectral_v2_100k_s3072` | 2,351 | 2,000（`resume.pt`） |

上記3つの再開ファイルはCPUで読み取り、更新数・optimizer状態の存在と停止前後のSHA-256不変を確認しました。停止直前の未保存分は再開点からやり直しになります。元runのconfig・manifest・checkpoint・metricsは改変していません。再開時は各runの開始時コード・環境を使用し、照合を無効化しません。

証拠は同じ診断ディレクトリの`recovery_125512/`です。`before_stop.json`、`stop_actions.json`、`checkpoint_verification.json`、`results.json`、`after_recovery.json`と各実行ログを保持しています。LIBERO停止時にsystemdはcontrol groupのkillについて`Invalid argument`を1件記録しましたが、その後の実PID・worker・GPU compute process一覧で残留なしを確認しました。

<a id="memory-attribution"></a>

### LIBERO追加との因果関係と固定メモリの確認（同日追記）

「後から起動したLIBEROが原因では」という依頼について、**LIBERO追加によるメモリ負荷増は確認できましたが、LIBERO単独の問題とは特定できませんでした。PushT側に大きな固定メモリ負荷があることを追加で実測しました。** 両本学習は停止したまま、保存済み記録・ソースの照合、CPUで各データの1clip読込、モデルなしの固定CPUバッファ確保だけを行いました。

時系列は次の通りです。

| JST | 記録 |
|---|---|
| 11:02:29 | PushTに加えてLIBEROの本学習を起動 |
| 11:03:29 | LIBEROの最初の更新。Torch GPU peak 26,934,989,312 bytes（25.09 GiB） |
| 11:19:02 | 別プロセスのBC GPUテストが1件合格、16.88秒 |
| 11:31:50 | Playwrightによるviewer確認の最初のHTTPアクセス |
| 11:31:51 | 調査したkernel log内の最初のmain context OOM |

LIBEROのGPU peakは1〜2,351更新の全記録で同じでした。起動直後から新規CUDAプロセスが全て失敗したわけではなく、GPU peakが継続増大した証拠もありません。ただしpeak allocatedだけでは、予約メモリ・固定CPUメモリ・物理配置の変化を追跡できず、全種のメモリリーク不在を証明しません。最初のOOMはブラウザ起動と重なりますが、ブラウザが断片化を作ったか、既存状態を顕在化させたかもこの時系列だけでは分かりません。

両runのconfigに記録された`training/train.py`・`training/train_libero.py`・`training/loop.py`のhashは、今回調査したソースと一致しました。実データのCPU読込で形状・dtypeを確認し、各runと同じ1バッチ分の画像を`torch.empty(..., device='cpu', pin_memory=True)`で確保しました。CUDAの初期化は使いますが、モデル・DataLoader worker・GPUへの画像転送・逆伝播・optimizer更新は含みません。

| 画像の負荷要因 | PushT本学習の設定 | LIBERO本学習の設定 |
|---|---:|---:|
| batch / workers | 256 / 8 | 128 / 4 |
| CPU上の画像 | 4フレーム、224×224、float32 | 4フレーム×2カメラ、128×128、uint8 |
| 1バッチの画像payload | 588 MiB | 48 MiB |
| 1バッチの固定メモリ確保実測 | **1 GiB** | **64 MiB** |
| 既定prefetch=2による先読み枠 | 16バッチ | 8バッチ |
| 全先読み枠がpin済みの場合の画像確保量（計算値） | 約16 GiB | 約0.5 GiB |
| 停止直前の親PIDのRssShmem（実測） | 10.18 GiB | 0.43 GiB |

最後から2行目は1バッチ実測からの計算であり、停止前に全枠がpin済みだったという測定ではありません。現在処理中のバッチ・validation・allocatorの未使用cache等を含む総上限でもありません。RssShmemも全てをpin-memoryへ分類した値ではなく、GPU観測値やシステムcacheと単純加算しません。

PyTorchの固定メモリallocatorは要求を2の累乗へ切り上げます。インストール済みヘッダと[PyTorch v2.9.1の処理](https://github.com/pytorch/pytorch/blob/v2.9.1/aten/src/ATen/core/CachingHostAllocator.h)を照合し、[host_memory_stats](https://docs.pytorch.org/docs/2.9/generated/torch.cuda.memory.host_memory_stats.html)で上記1GiB／64MiBを実測しました。各バッファを削除してGCした後も、activeは0になる一方、reservedは同量のままで`num_host_free=0`でした。プロセス内の再利用cacheとして保持されます。両診断プロセスは終了コード0（1.66秒／1.20秒）で終了し、確保領域もプロセス終了で解放しました。

さらに**固定メモリを確保した診断プロセスでもVmPin・VmLckは0**でした。この環境では、それらの0だけを根拠にpin-memoryなしとは判断できません。以前の停止前snapshotでも両PIDのVmPin・VmLckは0ですが、固定CPUメモリの寄与を除外できません。

現時点の作業仮説は、**PushTの大きな画像先読み・固定メモリ保持に、LIBEROの約25GiBのGPU確保が加わり、両者の併走中に連続空き領域が不足した**というものです。起動順だけからLIBERO実装のバグと結論付けず、これを因果の完全特定とも呼びません。2本を同時に止めたため、停止による回復だけでは片方の寄与を分離できません。

この調査時点で残った条件は、各学習の単独動作と併走、同じ計画予算でPushTの先読み／pin-memoryを変えた対照でした。**この時点では未実行・未予約で、同日後続のユーザー依頼により別runの対照試験を実施しました。** 併走時の同じOOM、LIBEROだけの停止による回復、PushTのpin-memory無効時の不発と再有効化時の再発を確認しています。[対照試験の条件・実測・限界](GB10_MEMORY_CONTROLS.ja.md)を参照してください。元runの再開条件やsource/config照合は変更していません。

追加証拠は`memory_diagnosis_20260911/attribution_20260911/`の`summary.json`、`batch_payloads.json`、`config_comparison.json`、`kernel_context_errors.json`、`libero_timed_steps.json`、`pin_allocation_probe.py`と各`pin_*.log`です。解析途中の補助スクリプトに出力フォルダ作成順・JSON型の扱いの失敗があり、修正後の出力を保存しました。メモリ確保の診断自体は両方正常終了しています。終了後も元PIDなし・GPU compute processなし・LIBERO service inactiveを確認しました。

先に行ったCPUの実データ確認は`output/libero10/bc_implementation_check/train/`に保存。凍結ViTは5,501,376パラメータ、方策は5,985,287パラメータです。固定validation flow lossは2更新時2.60468、4更新時2.43697でした。短期動作確認であり、学習性能の結論ではありません。

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

同日12:55 JSTにユーザー指示で停止しました。最後の記録は2,351更新、保存済み再開点は2,000更新です。現在は停止したままで、再開の明示依頼を待ちます。

```bash
systemctl --user status bt-libero10-100k-s3072 --no-pager
journalctl --user -u bt-libero10-100k-s3072 -n 5 --no-pager
tail -n 1 output/libero10/bt_spectral_v2_100k_s3072/metrics.jsonl
```
