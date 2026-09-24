# ノルム保存BT v1：構成からPushT学習・再開・評価まで

このページは、新型BTを初めて使う人向けの手順書です。上から順に、仕組みを理解し、環境とデータを用意し、1万更新の学習、途中再開、7万への延長、成功率の評価まで進めます。通常の作業に別ページを読む必要はありません。コマンドは**Bashの同じ端末で、リポジトリのルートから**実行します。学習・評価のコマンドは、実行すると実際にGPUを使います。

## 1. 何を学習するのか

**版の対応：** このページと最近の比較表の「新型BT」は、Tの識別子が`mixed_pair_mobius_norm_preserving_v1`の**ノルム保存BT v1**です。「従来BT」は`cayley_spectral_v2`（Cayley特異値制約v2）です。それ以前のFrobenius試作、およびSceneで試した新版S+O（`student_observed_transport_v1`）とは区別します。S+Oは白色化＋DINO観測損失の試作で、270更新後に共分散計算で停止しました。その後のScene修正版`norm_preserving_observed_v1`には同じノルム保存Tと観測損失を使いますが、本ページのPushTは元の潜在予測MSEを使います。v1/v2は方式内の版で、全体を通した世代番号ではありません。

PushTは、操作する点でT字の物体を押し、目標の位置・向きへ動かす環境です。学習に使うのは記録済みの画像と行動です。画像を直接描き直すモデルではなく、画像から取り出した192次元の状態を予測します。報酬・言語指示・タスクIDは学習入力に使いません。制御時には目標画像を与えます。

```text
画像（224×224） → ViT-Tiny画像エンコーダ → 状態 z（192次元）
                                           │
        行動（2次元×5時刻）→ 行動埋め込み ────┤
                                           ↓
                              小型の自己回帰予測器 F
                                           ↓
                                      未来の状態 ẑ
                                           │
次時刻の画像 → 同じエンコーダ → 教師状態 z ────┴→ 予測誤差

状態 z → 学習専用の可逆変換 T → u → SIGReg（分布の正則化）
```

| 部品 | この実験の構成・役割 |
|---|---|
| 画像エンコーダ | ViT-Tiny、パッチ14、画像224。事前学習なしで学ぶ |
| 状態 | 192次元。元のBatchNorm付きMLP headを使用 |
| 行動エンコーダ | 2次元の行動5個を10次元にまとめ、192次元へ埋め込む。中間幅も10 |
| 未来予測器 | `ARPredictor`、6層、隠れ幅192、16 heads、MLP幅2048、head幅64。履歴長3 |
| 学習専用のT | 192次元→192次元。混合pair Möbius写像、2層、kappa=0.2 |
| 推論 | エンコーダ・行動埋め込み・予測器を使用。Tは使用しない |

学習クリップは5環境ステップ間隔の4画像と、その間の行動です。予測は潜在状態の1段先を学び、計画時は繰り返し予測して先まで進みます。学習時の1段と、CEMの25環境ステップの計画区間は別です。

損失は `L = 元の1段潜在予測MSE + 0.09 × SIGReg(T(z))`。SIGRegは元の時刻別方式を用います。未来画像を教師状態へ変換するエンコーダにも勾配を流します。DINO、再構成損失、多段予測損失は加えていません。

新型では、実数上で **`||T(z)|| = ||z||`、`T(0)=0`** が成り立ちます。Tだけが一様な拡大縮小で正則化を満たす自由度をなくす設計です。非線形な方向の変換は許します。これはエンコーダ自身の尺度、中心化分散、タスク情報の保持、成功率を保証する条件ではありません。比較対象の従来BTはCayley特異値制約v2で、その他の構成を揃えます。

## 2. 環境とSSD保存先を用意する

必要なものはLinux、NVIDIA GPUと対応ドライバ、Git、uv、Zstandardです。Pythonは3.12、lockfileはPyTorch 2.9.1／CUDA 13.0です。現行のGB10ではbatch256の学習1本あたりGPU割当が約25GiBでしたが、必要量は機種・並列数で変わります。

未導入のツールだけ導入します。

```bash
sudo apt-get update
sudo apt-get install -y git curl zstd tmux
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

**次のSSDパスは自分のマウント先に置き換えます。** 既存実験と重ならない新しいフォルダを使います。`df`で実際にSSDがマウントされていることと空き容量を確認してください。圧縮データ約13GB、展開後約46GBに加え、環境・checkpoint・動画の容量が必要です。

```bash
export EXPERIMENT_ROOT="/media/$USER/SSD1/pusht-newbt-example"
df -h "$(dirname "$EXPERIMENT_ROOT")"
mkdir -p "$EXPERIMENT_ROOT"
git clone https://github.com/sige0002/bt-sigreg.git "$EXPERIMENT_ROOT/source"
cd "$EXPERIMENT_ROOT/source"
```

既にclone済みならclone行を飛ばし、そのルートへ移動します。**開始したコードと環境は学習・再開が終わるまで変更しません。** 別作業には別checkoutを使います。SSH越しの場合はここで`tmux new -s pusht-newbt`を実行し、その中で以降のコマンドを入力します。`Ctrl-b`、続けて`d`で離れ、`tmux attach -t pusht-newbt`で戻れます。

```bash
export BT_SIGREG_ROOT="$PWD"
export PYTHONPATH="$PWD/src:$PWD/lewm"
export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
uv python install 3.12
uv sync --locked
.venv/bin/python -c 'import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(), "CUDAが利用できません"'
```

`uv sync`は初回構築時だけです。既に学習で使用中の環境には実行しません。以降は`.venv/bin/python`または`uv run --no-sync`を使用します。

## 3. データと分割を固定する

```bash
export PUSHT_DATA_ROOT="$EXPERIMENT_ROOT/data"
export PUSHT_HDF5="$PUSHT_DATA_ROOT/pusht_expert_train.h5"
export PUSHT_MANIFEST="$EXPERIMENT_ROOT/manifest.json"
mkdir -p "$PUSHT_DATA_ROOT"
uv run --no-sync hf download quentinll/lewm-pusht --repo-type dataset \
  --include pusht_expert_train.h5.zst --local-dir "$PUSHT_DATA_ROOT/download"
zstd -d --keep "$PUSHT_DATA_ROOT/download/pusht_expert_train.h5.zst" \
  -o "$PUSHT_HDF5"
```

取得済みならダウンロード・展開は飛ばし、`PUSHT_HDF5`を既存HDF5の絶対パスに設定します。上書き展開はしません。

manifestはデータの場所・分割・行動の正規化統計・評価ケースを保存するJSONです。この新型比較は**episode単位の80/10/10分割**を使用します。別のRaw／BTガイドにあるclip90分割とは異なります。

```bash
.venv/bin/python -m mylewm.training.loop prepare \
  --dataset "$PUSHT_HDF5" --manifest "$PUSHT_MANIFEST"
```

この`prepare`は学習を開始しません。分割seedは実装で20260906に固定、学習seed3072とは別です。元の公開データでは学習1,585,717クリップ、固定validation256クリップ、confirm200ケースを使います。初回は全データのSHA-256計算に時間がかかります。既存manifestは作り直さず、両方式で同じものを使います。データやmanifestを途中で移動・編集しません。

## 4. 設定を選び、学習前に確認する

**1ステップ＝1バッチで重みを1回更新すること**です。batch256なら1万更新で延べ256万例を提示します。同じデータを繰り返すため、別々の256万件という意味ではありません。

| 設定 | 今回の1万更新比較 |
|---|---|
| 更新数・バッチ | 10,000・256 |
| 学習率 | 5e-5、warmup500、その後cosineで10,000時点に0 |
| 乱数 | seed3072。両方式で初期値・データ順を揃える |
| 保存・validation | 5,000更新ごと |
| 実行 | BF16 mixed、workers4、pinなし、encoder compileなし |
| モデル設定 | YAMLを必ず明示。省略・暗黙の次元補完なし |

新型を学ぶ設定です。`train_args`は長い引数を同じ内容で再利用するBash配列です。

```bash
export FAMILY=norm_preserving
export MODEL_YAML="$PWD/mylewm/configs/pusht_transport_${FAMILY}.yaml"
export TRAIN_RUN="$EXPERIMENT_ROOT/runs/${FAMILY}_10k"
train_args=(
  --manifest "$PUSHT_MANIFEST" --model-config "$MODEL_YAML"
  --steps 10000 --warmup-steps 500 --batch-size 256 --workers 4
  --save-every 5000 --val-every 5000 --seed 3072 --lr 5e-5
  --no-pin-memory --no-compile-encoder --accelerator gpu --precision bf16-mixed
)
.venv/bin/python -m mylewm.training.train_pusht_transport \
  "${train_args[@]}" --output "$TRAIN_RUN"
```

`Dry-run only`で終われば、引数とYAMLの照合が通っています。この時点では学習・GPU検証・データ全量検証はしていません。YAMLの型・次元・損失等はこの比較用の厳密な契約で検査します。任意のモデル変更に使える汎用設定ではありません。

## 5. 学習を開始し、保存結果を読む

次の`--execute`付きコマンドが実際の学習開始です。

```bash
.venv/bin/python -m mylewm.training.train_pusht_transport \
  "${train_args[@]}" --output "$TRAIN_RUN" --execute
echo $?
```

最初の保存は5,000更新です。進捗は別端末で、実際のrunパスを指定して確認します。

```bash
tail -n 3 "$TRAIN_RUN/metrics/version_0/metrics.csv"
```

別端末では変数は引き継がれないため、先に`TRAIN_RUN`等を同じ値に設定してください。CSVの`update`が更新数です。lossやvalidation MSEは環境の成功率ではありません。

| 保存物 | 用途 |
|---|---|
| `config.json` | 解決済みYAML、batch、学習率、初期値・コード・依存の識別情報 |
| `step_5000_object.ckpt`、`step_10000_object.ckpt` | 評価用モデル。学習専用Tを除く |
| `step_5000.ckpt`、`last.ckpt` | 再開用。T・optimizer・学習率・乱数等を含む |
| `completed.json` | 総更新数へ到達した記録 |

```bash
cat "$TRAIN_RUN/completed.json"
ls -lh "$TRAIN_RUN/step_10000_object.ckpt" "$TRAIN_RUN/last.ckpt"
```

終了コード0、`completed.json`の`state=completed`・`step=10000`、最終checkpointを確認します。途中ファイルがあるだけでは完了ではありません。

従来BTも比較する場合は第4節へ戻り、`FAMILY=cayley`に替えて第4〜5節を実行します。別の出力先になり、モデルYAMLだけが切り替わります。manifest・batch・更新数・seed・学習率は揃えます。通常は順番に実行し、並列の場合は十分なメモリを確認したうえで別端末を使います。各`config.json`の`initial_model_sha256`と共通条件が一致することを確認します。

## 6. 中断から再開する

元の学習プロセスが終了している場合だけ行います。第2〜4節で設定した変数・配列と、開始時のソース・環境をそのまま使います。5,000更新からなら次のとおりです。

```bash
export RESUME_RUN="$EXPERIMENT_ROOT/runs/${FAMILY}_10k_resume1"
.venv/bin/python -m mylewm.training.train_pusht_transport \
  "${train_args[@]}" --output "$RESUME_RUN" \
  --resume "$TRAIN_RUN/step_5000.ckpt" --execute
```

総更新数は10,000のままです。`*_object.ckpt`では再開しません。batch・YAML・保存間隔・ソース等を変えると照合で拒否されます。照合を解除せず、開始時の条件を復元してください。再開後の保存物は`RESUME_RUN`に入ります。

## 7. 完了した1万から7万へ延長する場合

これは第6節の通常再開と違い、**将来の学習率スケジュールを変える追加実験**です。現在の比較は、10,000で学習率0まで学んだ後、70,000終端へ延長しています。初めから70,000で学ぶ結果とは同じではありません。単に`--steps`を書き換えて元checkpointを渡すと不一致になります。

第5節の完了済み`last.ckpt`から、元ファイルを書き換えず派生checkpointを作ります。途中再開を経た場合は、まず`TRAIN_RUN`を完了したrun（例：`$RESUME_RUN`）へ設定し直します。学習開始時と同じコード・環境・YAML・manifestを使います。

```bash
export EXTENSION_ROOT="$EXPERIMENT_ROOT/extensions/${FAMILY}_70k"
.venv/bin/python - <<'PY'
import json, os
from pathlib import Path
import torch
from mylewm.training.extend_pusht_budget import extend_completed_checkpoint
parent = Path(os.environ['TRAIN_RUN'])
out = Path(os.environ['EXTENSION_ROOT'])
assert json.loads((parent/'completed.json').read_text())['step'] == 10000
recipe = json.loads((parent/'config.json').read_text())
checkpoint = torch.load(parent/'last.ckpt', map_location='cpu', weights_only=False)
extended = extend_completed_checkpoint(checkpoint, recipe, 70000)
out.mkdir(parents=True, exist_ok=False)
torch.save(extended, out/'continuation_start.ckpt')
(out/'parent.json').write_text(json.dumps({'parent': str(parent.resolve()),
    'start_step': 10000, 'target_step': 70000}, indent=2))
PY
.venv/bin/python -m mylewm.training.train_pusht_transport \
  --manifest "$PUSHT_MANIFEST" --model-config "$MODEL_YAML" \
  --output "$EXTENSION_ROOT/train" --resume "$EXTENSION_ROOT/continuation_start.ckpt" \
  --steps 70000 --warmup-steps 500 --batch-size 256 --workers 4 \
  --save-every 5000 --val-every 5000 --seed 3072 --lr 5e-5 \
  --no-pin-memory --no-compile-encoder --accelerator gpu --precision bf16-mixed \
  --execute
```

重み・T・optimizer moment・更新数・乱数・消費済みデータ位置を保持します。10,001回目の学習率は約4.773e-5、その後70,000で0へ減衰し、warmupは繰り返しません。追加は60,000更新です。派生checkpointでも再開時の全設定照合は有効です。従来BTも同じ手順・予算で延長します。

過去の実験checkpointを再開する場合、このページの最新checkoutではなく**その実験開始時の凍結ソース・環境**が必要です。現在稼働中の実験は、外付けSSDの`output/pusht/norm_preserving_10k_20260922_v2/source/`を用い、継続先は`output/pusht/norm_preserving_continue70k_20260923/`です。この記載は再起動の指示ではありません。

## 8. 保存済みモデルの成功率を評価する

1万更新の例です。途中評価も可能ですが、保存が完了した`*_object.ckpt`を使います。7万への継続で保存した45,000を調べるなら`CHECKPOINT`を`$EXTENSION_ROOT/train/step_45000_object.ckpt`へ変更し、未使用の評価出力先を選びます。

```bash
export CHECKPOINT="$TRAIN_RUN/step_10000_object.ckpt"
export EVAL_RUN="$PWD/output/pusht/${FAMILY}_10000_confirm50"
bash scripts/evaluate_pusht.sh \
  --checkpoint "$CHECKPOINT" --manifest "$PUSHT_MANIFEST" \
  --output "$EVAL_RUN" --num-eval 50 --offset 0 --seed 42 --cem-seed 42
```

ここまでは設定確認です。次は実際に評価します。

```bash
bash scripts/evaluate_pusht.sh \
  --checkpoint "$CHECKPOINT" --manifest "$PUSHT_MANIFEST" \
  --output "$EVAL_RUN" --num-eval 50 --offset 0 --seed 42 --cem-seed 42 --execute
echo $?
cat "$EVAL_RUN/status.json"
```

GB10で対象データのclean cache解放を使う場合は実行コマンドに`--gb10-cache-workaround`を付けます。現在の比較では両方式に付け、評価は1本ずつ実行しています。

評価は目標画像と予測状態の距離を使ってCEMで行動を選びます。標準設定は候補300・反復30・上位30、計画5ブロック×5行動＝25環境ステップ、環境予算50、Goal間隔25です。予測とGoal距離はzで計算し、Tは使いません。成功はSWMの位置・角度の閾値判定で、画像の重なり率ではありません。

`status.json`の`state=succeeded`・`cases=50`、終了コード0を確認します。`results.txt.json`の`successes`は50件の真偽値で、その合計が成功数です。`viewer/index.html`と動画で失敗内容を見られます。失敗・中断した評価を0%と数えず、ログを残して原因を確認します。

2方式を評価したら、対応する結果を指定して条件一致と成否差を確認できます。

```bash
.venv/bin/python -m mylewm.evaluation.compare_paired \
  --baseline "$PWD/output/pusht/cayley_10000_confirm50/results.txt.json" \
  --candidate "$PWD/output/pusht/norm_preserving_10000_confirm50/results.txt.json" \
  --output "$PWD/output/pusht/paired_newbt_10000.json"
```

## 9. 今回の記録と解釈

2026-09-24に照合した同じ50ケース・CEM seed42の結果です。下表は両方batch256、同じ初期値・データ順・スケジュールです。

| 更新数 | 従来BT | 新型BT |
|---:|---:|---:|
| 5,000 | 12% | 20% |
| 10,000 | 22% | 28% |
| 15,000 | 42% | 50% |
| 20,000 | 54% | 64% |
| 25,000 | 68% | 78% |
| 30,000 | 84% | 78% |
| 35,000 | 84% | 86% |
| 40,000 | 84% | 86% |
| 45,000 | 84% | 94% |

45kの新型は47/50、従来は42/50。単一学習seed・繰り返し使った50ケースの結果で、一般的な優位性やマルチタスク80%を実証したものではありません。旧batch128の比較は提示例数を揃え、今回45kには旧90kを対応させます。学習率履歴やデータ抽出は別なので、同条件の方式比較とは区別します。

## 10. よくあるつまずきと実装の場所

| 症状・疑問 | 対応 |
|---|---|
| 変数が空、パスが見つからない | 別端末なら第2〜4節の変数を設定し直す。SSDマウントも確認 |
| 出力先が存在すると停止する | 未使用の新しい名前へ変更。既存の成果は削除しない |
| 再開が設定不一致になる | 元のソース・環境・YAML・manifest・全引数を復元。hash照合を解除しない |
| lossは下がったが成功率がない | lossは成功率ではない。第8節の環境評価が必要 |
| 設定はどこか | `mylewm/configs/pusht_transport_norm_preserving.yaml`。対照は`pusht_transport_cayley.yaml` |
| 学習処理はどこか | `src/mylewm/training/train_pusht_transport.py`。共有処理は`training/train.py` |
| 新型Tの数式実装はどこか | `src/mylewm/algorithms/norm_preserving_transport.py` |
| 予算延長はどこか | `src/mylewm/training/extend_pusht_budget.py` |

このページは実行手順の正本です。既存実験の生ログ・checkpoint・manifestをGitに含めず、各実験の出力先で保持します。
