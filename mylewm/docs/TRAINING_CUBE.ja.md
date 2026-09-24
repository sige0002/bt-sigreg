# OGBench Cube：Raw・Cayley v2 BT・ノルム保存BT v1の学習ガイド

このページは、LeWMが用いた**Cube single expert**の画像・行動データで、3方式を同条件で学習するための手順です。構成、データ、設定、起動、再開、結果の読み方をここにまとめます。PushTやSceneの設定を流用しません。コマンドはBashで、同じ端末・リポジトリのルートから実行します。

## 1. 比較するものとモデルの構成

```text
224×224画像 → ViT-Tiny → 状態 z（192次元）
                              │
5次元行動×5時刻 → 25次元 → 行動埋め込み → 予測器 F → 未来の状態
                              │                        │
                              │          次画像からの教師状態とのMSE
                              ↓
                  Raw：SIGReg(z)
          Cayley v2 BT：SIGReg(T_cayley(z))
        ノルム保存BT v1：SIGReg(T_norm_preserving(z))
```

学習損失は3方式とも `1段潜在予測MSE + 0.09 × SIGReg` です。違いは正則化前の変換Tです。未来教師側のエンコーダにも勾配を通し、予測・Goal距離はzで計算します。Tは学習専用で、推論モデルには含めません。DINOや再構成損失、多段予測損失は追加していません。

| 名前 | 設定ファイル | 正則化前の処理 |
|---|---|---|
| Raw | `mylewm/configs/cube_raw.yaml` | 恒等写像。直接SIGReg |
| Cayley v2 BT | `mylewm/configs/cube_cayley.yaml` | `cayley_spectral_v2`、2層、kappa=0.2 |
| ノルム保存BT v1 | `mylewm/configs/cube_norm_preserving.yaml` | `mixed_pair_mobius_norm_preserving_v1`、2層、kappa=0.2 |

共通の推論モデルは18,034,628パラメータです。学習専用Tの追加はRawが0、Cayley v2が147,840、ノルム保存BT v1が37,056で、推論時は3方式ともTを除きます。これはパラメータ数であり、学習時間・推論速度の実測値ではありません。

ノルム保存BT v1は実数上で`||T(z)||=||z||`を満たし、Tによる一様な拡大縮小を禁止します。非線形に方向を変える自由度を残しますが、有用な非線形性の獲得、エンコーダの尺度、タスク内情報の保持、成功率は保証しません。ガウス分布との半径分布の不一致をTだけで解消することもできません。

### BTの世代と、この比較に含めるもの

「旧BT→新型BT→ノルム保存BT」という開発経緯はありますが、過去の「新型」は複数の方式を指していました。今後は次の実装識別子を併記します。v1/v2は各方式内の版であり、BT全体を通した通し番号ではありません。

| 開発上の呼び方 | 方式・版 | 変更点と今回の扱い |
|---|---|---|
| 初期試作 | Frobenius制約BT | Cayley以前の試作。旧checkpointはCayley v2と非互換。今回の対象外 |
| 旧／従来BT | Cayley v2：`cayley_spectral_v2` | 大域的距離境界の範囲で拡大縮小を許す。今回の比較対象 |
| 新版試作S+O | `student_observed_transport_v1` | 白色化による形状正則化＋固定DINOへの観測損失。Sceneで270更新後に共分散Cholesky計算が失敗。今回の対象外 |
| ノルム保存BT | T：`mixed_pair_mobius_norm_preserving_v1` | 各入力のノルムを厳密に保存する式へ変更。今回の比較対象 |

ノルム保存Tを最初に組み込んだSceneの全体方式名は`norm_preserving_observed_v1`で、DINO観測損失を含みます。今回のCubeと最近のPushT比較は**同じノルム保存T＋元の潜在予測MSE**であり、Scene方式全体の移植ではありません。最近のPushT表の「従来／新型」は、それぞれCayley v2／ノルム保存BT v1です。学習runの新旧やbatchサイズの違いを、アルゴリズムの別世代と数えません。

| 共通部分 | 明示した条件 |
|---|---|
| 視覚 | ViT-Tiny、patch14、224px、事前学習なし |
| 状態・履歴 | 192次元・履歴3・1段先予測 |
| 予測器 | ARPredictor、6層、16 heads、head幅64、MLP幅2048 |
| 行動 | 物理5次元（相対XYZ・yaw・gripper）×5時刻＝入力25次元 |
| 行動中間幅 | **10**。公式LeWMのEmbedder既定値をYAMLへ明記。入力25とは別 |
| 正則化 | 元の時刻別SIGReg、投影1024、knots17、重み0.09 |
| 更新 | AdamW、weight decay0.001、LR5e-5、勾配clip1.0 |

行動中間幅10は、過去のScene比較で使った25→25とは異なります。今回は公式Cubeモデルの構成へ合わせる選択です。PushTの行動入力10をCubeへ渡すと設定検査で拒否します。

## 2. 公式ベンチマークと今回の比較範囲

元データはHFの`quentinll/lewm-cube`、revision `02a19a67a0dc8c9d6215f89c19e0a597691e152a`の`cube_single_expert.tar.zst`です。OGBenchの名前が似た別データに置き換えません。公開アーカイブは46,184,624,478 bytes、LFS SHA-256は`3725d6a01abd492164441ef0a27e588f52b94a118fab56b96987b1a34a6c2600`です。

今回の主比較はepisode単位の80%学習・10%検証・10%評価とし、学習episodeだけで行動の正規化統計を計算します。3方式は同一manifest・初期モデル・データ順・予算を使用します。公式コードの90/10 clip分割、全データの正規化統計、epoch単位の学習率スケジュールとは異なります。**同じCubeデータ・モデル構成での方式比較であり、公式スコアの完全再現ではありません。**

LeWMのCube評価は、データ中の状態を開始点とし、25環境ステップ先の画像をGoalにする短区間制御です。OGBench標準の5タスク評価とは区別します。後で成功率を比較する際は、3方式で同じケース・Goal・CEM予算を使います。予定する論文準拠の探索条件は候補300・反復10・elite30、horizon5×action_block5、receding_horizon5、環境予算50です。**PushTの30反復を無断で流用しません。**

## 3. 環境・データ・保存先

Linux、NVIDIA GPUと対応ドライバ、Git、uv、zstd、Python3.12を使います。リポジトリのlockfileはPyTorch2.9.1／CUDA13.0用です。初回はリポジトリをcloneし、ルートへ移動してください。

Ubuntu/Debianで必要なツールが未導入の場合だけ、先に次を実行します。これはGPUドライバの導入ではありません。

```bash
sudo apt-get update
sudo apt-get install -y git curl zstd tmux
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

```bash
git clone https://github.com/sige0002/bt-sigreg.git
cd bt-sigreg
export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
uv python install 3.12
uv sync --locked
export CUBLAS_WORKSPACE_CONFIG=:4096:8
.venv/bin/python -c 'import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available()'
```

既存の学習環境には`uv sync`を実行しません。このマシンでは既存`.venv`をそのまま使用します。以後は`.venv/bin/python`を使います。長時間実行は開始前に`tmux new -s cube-train`で端末を保持し、同じ変数をその端末内で設定します。

このマシンでのデータ保存先は外付けSSDです。別マシンでは自分のSSDマウント先へ変更し、未接続のパスへ内蔵ディスクで代替保存しないでください。

```bash
export CUBE_ROOT="/media/$USER/SSD1/bt-sigreg/ogbench/cube"
df -h "/media/$USER/SSD1"
```

取得済みデータは再取得しません。別マシンで新規取得する場合だけ、十分な空き容量を確保して実行します。圧縮約46GB＋展開後HDF5約102GBに加え、Python環境・checkpointの容量が必要です。

```bash
mkdir -p "$CUBE_ROOT/download" "$CUBE_ROOT/extracted"
uv run --no-sync hf download quentinll/lewm-cube cube_single_expert.tar.zst \
  --repo-type dataset --revision 02a19a67a0dc8c9d6215f89c19e0a597691e152a \
  --local-dir "$CUBE_ROOT/download"
sha256sum "$CUBE_ROOT/download/cube_single_expert.tar.zst"
```

上記のLFS SHA-256と一致してから、未使用の展開先へ展開します。

```bash
tar --zstd -xf "$CUBE_ROOT/download/cube_single_expert.tar.zst" \
  -C "$CUBE_ROOT/extracted"
```

## 4. manifestを準備する

manifestは、データの場所、サイズ・mtime・SHA-256、列契約、episode分割、正規化統計、検証・評価ケースを固定するJSONです。新規prepareはHDF5を読み取り専用で検査し、列の追加や元データの書き換えをしません。

実際の展開先HDF5を`CUBE_HDF5`に指定します。`CUBE_MANIFEST`は未使用パスです。既存manifestがある場合はそれを指定し、prepareコマンドを飛ばします。

```bash
export CUBE_HDF5="$CUBE_ROOT/extracted/cube_single_expert.h5"
export CUBE_MANIFEST="$PWD/output/manifests/ogbench/cube/comparison_v1/manifest.json"
.venv/bin/python -m mylewm.data.cube_data \
  --dataset "$CUBE_HDF5" --manifest "$CUBE_MANIFEST" --seed 3072 \
  --validation-cases 256 --confirm-cases 200
```

画像224×224 RGB uint8、5次元行動、episode境界、qpos/qvel、CubeのGoal用位置・姿勢を検査します。終端行の番兵行動を除き、非有限・範囲外の行動を拒否します。通常の学習起動はサイズ・mtimeを検査し、`--verify-data`を付けた場合だけ全量SHA-256を再計算します。

## 5. 3方式の条件を決める

1更新は1バッチで重みを更新する処理、1epochは学習集合を1周する量です。以下は**batch128・10epochs相当の新規比較**を用意する例です。現在のPushTの「1万終了後に7万へ延長」とは別のスケジュールです。

今回取得した実データは10,000軌道・2,010,000画像です。分割は学習8,000／検証1,000／評価1,000軌道、学習クリップは1,456,000件。batch128で1周11,375更新、10周では**113,750更新／方式**です。

```bash
export CUBE_STEPS=$(.venv/bin/python - <<'PY'
import json, os
m = json.load(open(os.environ['CUBE_MANIFEST']))
print(10 * (m['train_clips'] // 128))
PY
)
export CUBE_RUN_ROOT="$PWD/output/ogbench/cube/comparison_s3072"
echo "$CUBE_STEPS"
```

| 設定 | 3方式に共通 |
|---|---|
| batch・seed | 128・3072 |
| LR | 5e-5、warmup500更新、その後総更新数で0へcosine減衰 |
| 保存・検証 | 5,000更新ごと、最終推論モデルも保存 |
| 実行 | BF16 mixed、workers4、pinなし、encoder compileなし |
| 検証 | 同じmanifestの固定256クリップ |

短期動作検証の4更新・warmup1は、本学習の設定へ流用しません。YAMLは必須で、方式名だけでは起動できません。

## 6. dry-runと学習開始

`FAMILY`は`raw`、`cayley`、`norm_preserving`のいずれかです。まず設定を確認します。

```bash
export FAMILY=raw
cube_args=(
  --manifest "$CUBE_MANIFEST" --model-config "$PWD/mylewm/configs/cube_${FAMILY}.yaml"
  --steps "$CUBE_STEPS" --warmup-steps 500 --batch-size 128 --workers 4
  --save-every 5000 --val-every 5000 --seed 3072 --lr 5e-5
  --no-pin-memory --no-compile-encoder --accelerator gpu --precision bf16-mixed
)
.venv/bin/python -m mylewm.training.train_cube \
  "${cube_args[@]}" --output "$CUBE_RUN_ROOT/$FAMILY"
```

`Dry-run only`なら設定検査の合格です。まだ学習・CUDA初期化はしていません。次は実際の学習開始です。

```bash
.venv/bin/python -m mylewm.training.train_cube \
  "${cube_args[@]}" --output "$CUBE_RUN_ROOT/$FAMILY" --execute
echo $?
```

他の方式は`FAMILY`を変更して第6節を繰り返します。方式を変えたら配列`cube_args`も作り直します。出力先は方式ごとに分け、同じ場所を再利用しません。まずは逐次実行し、並列化は実測メモリを確認してから行います。

本学習はまだ開始していません。準備・短期検証の完了と、成功率の獲得を混同しないでください。

## 7. 進捗・保存・再開

```bash
tail -n 3 "$CUBE_RUN_ROOT/$FAMILY/metrics/version_0/metrics.csv"
cat "$CUBE_RUN_ROOT/$FAMILY/completed.json"
```

別端末では変数を設定し直します。CSVの`update`が更新数、lossは学習誤差で、制御成功率ではありません。完了時は終了コード0、`completed.json`の`state=completed`と`step=$CUBE_STEPS`を照合します。

| ファイル | 内容 |
|---|---|
| `config.json` | YAML・解決済み設定・初期重みhash・コード/依存hash・データ識別情報 |
| `step_5000_object.ckpt`等 | 推論用。Tなし、5次元行動の正規化統計と入力契約を保持 |
| `step_5000.ckpt`・`last.ckpt` | 再開用。T・optimizer・学習率・乱数・データ位置を保持 |

中断した場合は、元プロセス終了を確認し、開始時のソース・環境・YAML・manifest・全引数を維持して、未使用出力先へ再開します。

```bash
.venv/bin/python -m mylewm.training.train_cube \
  "${cube_args[@]}" --output "$CUBE_RUN_ROOT/${FAMILY}_resume1" \
  --resume "$CUBE_RUN_ROOT/$FAMILY/step_5000.ckpt" --execute
```

総更新数は残り回数に変えません。`*_object.ckpt`では再開できません。設定やソースの不一致は拒否されるため、照合を解除せず元の条件を復元します。開始後は同じcheckoutへコード修正や依存同期を行わず、別作業は別checkoutで進めます。

## 8. 検証の範囲と参照元

2026-09-24に次を確認済みです。本学習は開始していません。

| 確認項目 | 結果 |
|---|---|
| CPU回帰 | Cube・PushT transport・ノルム保存T・予算延長の関連47件が合格。コミット対象だけの隔離checkoutでも47件合格 |
| データ | 公式配布revisionのアーカイブを取得し、公開LFS SHA-256と一致。展開HDF5の形式・値・全量SHA-256を記録 |
| 実GPU学習 | Raw／Cayley v2／ノルム保存BT v1それぞれbatch128で4更新、固定256クリップ検証、保存までexit0 |
| 初期値・更新・export | 3方式のモデル初期値hashが一致。モデルとBTのTが更新され、損失は有限。推論重みは学習側と一致し、Tを含まない |
| 実GPU再開 | 各方式の2更新目から4更新目へ再開しexit0。連続4更新とのモデル・T・optimizer・LR・乱数状態が完全一致 |
| 環境 | `swm/OGBCube-v0`のreset、224px RGB描画、5次元行動1回を確認。保存軌道の画像再現・制御評価ではない |
| 本学習設定 | 3方式とも113,750更新・warmup500のdry-run合格。実行・予約なし |

CPU回帰では分割の非重複、学習データだけの統計、行動25次元への時系列連結、誤った次元・domain・複数Cubeデータの拒否、初期Tでの損失・勾配一致、未来教師への勾配も確認しています。4更新は学習接続の検証であり、収束や制御成功率の証拠ではありません。

このマシンの実データは`/media/sadasue/SSD1/bt-sigreg/ogbench/cube/extracted/cube_single_expert.h5`、manifestは同SSDの`bt-sigreg/output/manifests/ogbench/cube/comparison_v1/manifest.json`です。証拠は`bt-sigreg/output/ogbench/cube/readiness_20260924/`の`gpu_smoke/verified.json`、`gpu_resume/verified.json`、各status・log、`comparison_plan.json`に保存しました。初回テストの検算コード修正と環境確認レポート生成の失敗ログも保持しています。実行ソースは同ディレクトリの`source_v2`へ固定し、既存PushT学習の環境・ソースは変更していません。

今回の完了目標は学習準備です。CubeのCEM成功率は未測定で、PushT用評価launcherの流用をCube正式対応と扱いません。qpos/qvel・Goal位置/姿勢とconfirmケースは準備し、制御評価の描画・復元・成功関数の監査は別工程です。Cube singleだけでマルチタスク改善を主張しません。

一次資料：[公式学習データ設定](https://github.com/lucas-maes/le-wm/blob/main/config/train/data/ogb.yaml)、[公式Cube評価設定](https://github.com/lucas-maes/le-wm/blob/main/config/eval/cube.yaml)、[公式Embedder](https://github.com/lucas-maes/le-wm/blob/main/module.py)、[LeWM論文](https://arxiv.org/html/2603.19312v1)、[配布データ](https://huggingface.co/datasets/quentinll/lewm-cube)。確認時の公式ソースrevisionは`8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`です。
