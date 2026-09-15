# 4. 評価：何を測り、どう結果を読むか

[アルゴリズム](ALGORITHM.ja.md) → [実装](IMPLEMENTATION.ja.md) → [学習](TRAINING.ja.md) → 評価

BT-SIGRegの効果を調べる場合は、先に[比較手順](COMPARISON_PROTOCOL.ja.md)でRaw／TC／BTの共通条件を決め、[既存実験一覧](EXPERIMENTS.ja.md)で再利用できる証拠を確認します。このページはその条件を実行するための操作ガイドです。CEM改善とBCは主比較の代わりにしません。

OGBench Cube single expertは[LeWMに揃えたOGBench手順](OGBENCH.ja.md)を参照してください。

## 最初に評価経路を選ぶ

| 経路 | 入力checkpoint | 測るもの |
|---|---|---|
| 世界モデル＋CEM | `step_N_object.ckpt` | 予測器と行動探索を組み合わせた制御能力 |
| 凍結encoder＋BC（補助） | `step_N_bc.pt` | 画像表現と学習済み模倣方策を組み合わせた制御能力 |
| オフライン診断 | 世界モデルcheckpointと保持デモ | 行動依存の予測、候補順位、誤差。環境成功率ではない |

**lossが低いことと、タスクに成功することは別です。** CEMの候補数を増やして予測コストが下がっても、実環境で成功した証拠にはなりません。BC評価ではCEMを使いません。

<a id="libero"></a>
## LIBERO：環境準備と入力条件

LIBERO-10は10タスクを環境内で実行し、`env.check_success()`で成功を判定します。デモ画像との一致率や予測lossを成功条件には使いません。

学習時と評価時で、カメラ・画像の向きと前処理・行動の意味・正規化統計を一致させます。LIBERO本体とOSMesaを用意し、[画像監査](reference/EVALUATION.ja.md#libero-2-osmesa画像監査)を先に通してください。実環境を呼ぶコマンドは`scripts/run_libero.sh`経由で実行します。

BC評価器は保存データの全デモ・両カメラから描画解像度（128／256px）を判定し、混在を拒否します。256pxモデルには同じデータで取得した256pxの画像監査が必要です。`audit_libero_images --regenerated --dataset <再生成data> --task-id N --output <未使用出力>`を`run_libero.sh`経由で全タスク実行してください。再生成データは元デモ順のresetを再現し、実測状態が保存されているt=1,4,8を監査します。t=0は元の初期状態が保存され、settling後画像と一致しないため使いません。閾値は旧経路と同じです。

### 世界モデル＋CEMを評価する

`mylewm.evaluation.evaluate_libero`へ世界モデルのobject checkpointを渡します。BC checkpointは渡せません。Goal画像を使って、予測上でGoalに近づく行動列を探索します。[CEMの実行例と出力確認](reference/EVALUATION.ja.md#libero-4-実環境cem評価)

比較するときは候補数・反復数・horizon・実行行動数を固定し、初期状態・Goal・前処理も照合します。探索量が違う結果を、そのまま世界モデルの優劣としません。CEMとBCの成功率も同条件比較にはなりません。

現行の通しCEM評価器は128px描画を前提とする箇所が残っています。新256pxモデルの比較に進む前に、この経路の解像度対応と画像監査を確認してください。BC評価器・native診断の256px対応だけでは代用できません。

### BCを評価する

以下は**旧128pxデータで学習したBCモデル**の評価例です。checkpointが存在すること、出力名が未使用であること、指定したauditが入力条件に合うことを確認します。

```bash
export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
export PYTHONPATH="$PWD/output/libero10/bc_implementation_check/runtime${PYTHONPATH:+:$PYTHONPATH}"
bash scripts/run_libero.sh -m mylewm.evaluation.evaluate_libero_bc \
  --checkpoint output/libero10/bc_bt100k_40k_s3072/step_40000_bc.pt \
  --manifest output/manifests/libero10/manifest.json \
  --output output/libero10/eval_bc_example \
  --task-ids 0 1 2 3 4 5 6 7 8 9 --episodes 50 --offset 0 \
  --budget 520 --execute-actions 8 --settling-steps 5 --seed 42 \
  --device cuda:0 \
  --render-audit-dir output/libero10/eval_bt100k_20260914/render_audit
```

既定はdry-runです。実行する場合だけ同じコマンドに`--execute`を追加します。各タスク50初期状態で計500試行、1試行最大520行動です。通常reset→固定初期状態→ゼロ行動5回の後、8行動ずつ生成・実行します。

少数の接続確認なら`--task-ids 0 --episodes 1 --budget 9`に変え、別出力で実行できます。少数試行の成績から全タスク平均を推定しません。BCのタスクIDは名前を経由してcheckpoint内の順序へ対応付けます。

<a id="pusht"></a>
## PushTを評価する

PushTは学習経路によってcheckpoint・行動変換の扱いが異なります。新しいRaw／BT runは[新経路の評価](reference/EVALUATION.ja.md#pusht-2-新しいrawbt-runを評価する)、保存済み旧checkpointは[既存checkpointの確認](reference/EVALUATION.ja.md#pusht-3-既存checkpointを評価する前の確認)から進みます。

```bash
bash scripts/evaluate_pusht.sh --help
```

LIBEROの7次元行動やBCをPushTへ流用しません。固定評価ケースと訓練データから引いたケースの違いにも注意します。[PushTの詳細手順](reference/EVALUATION.ja.md#pusht)

## 終了と結果の確認

LIBERO評価では、次の順に確認します。

1. プロセスの終了コードが0で、`status.json`が`succeeded`になっている。
2. `summary.json`の試行数と`episodes.jsonl`の行数が指定数と一致する。
3. タスク別成功率と全10タスク平均を読む。
4. 保存された初期・最終画像と動画を見て、失敗の様子を確認する。

| 保存物 | 内容 |
|---|---|
| `config.json` | checkpoint、manifest、seed、制御・描画条件、hash |
| `episodes.jsonl` | タスク・初期状態ごとの成否、行動数、時間 |
| `summary.json` | タスク別成功率とmacro平均 |
| `taskN_initM.npz` | 行動・初期状態・初期／最終画像など |
| `viewer/index.html` | 画像・結果の閲覧ページ。BCはMP4動画も表示 |

BC viewerの成功デモ画像は閲覧用の参考です。方策には渡していません。動画は環境内の20Hzに対応し、推論や保存を待った実時間を表していません。

途中で停止した評価は「完了」でも「未実施分が失敗」でもありません。タスク順に進んでいる途中の成功割合は、全10タスクの平均ではありません。例えば最初の2タスクだけが終わっていても、残り8タスクの性能は分かりません。

## 低成功率の原因を切り分ける

最初に画像・行動・初期化が学習と一致しているかを確認します。その後、CEMなら予測と候補順位、BCなら学習／検証loss・実行中の失敗動作を調べます。checkpointを変える比較は、同じ初期状態・画像・seedで行います。

保持デモの予測とnative候補順位を調べる入口は`mylewm.evaluation.diagnose_libero_model`です。過去の[診断例](reports/LIBERO_MODEL_DIAGNOSTICS.ja.md)では、デモの一段予測と長い候補列の実環境順位が一致しない場合を確認しました。補助診断と通しのタスク成功率を混同しません。

<a id="intermediate"></a>
## 途中checkpointと厳密な比較

保存完了済みの途中checkpointも評価できます。最終学習の`completed.json`は不要ですが、対象checkpointが書込み中でないこと、別の未使用出力を使うことを確認します。[GPU・途中評価の詳細](reference/EVALUATION.ja.md#intermediate)

Raw／TC／BTの主比較では、データ分割、初期値、予算、前処理、GoalまたはBC条件、成功関数を揃えます。複数学習seedの平均・ばらつきと下位タスクも報告します。TC-LeWMの3学習seed×3評価seed×各タスク50試行は計4,500試行であり、単一モデルの500試行と同じ検証規模ではありません。

[検証状況](VALIDATION.ja.md)には、実施済み範囲と未実証の主張を記録しています。

<details>
<summary>従来の詳細手順へのリンク（旧URL互換）</summary>

<a id="pusht-1-何を測るのか"></a>

[pusht-1-何を測るのか](reference/EVALUATION.ja.md#pusht-1-何を測るのか)

<a id="pusht-2-新しいrawbt-runを評価する"></a>

[pusht-2-新しいrawbt-runを評価する](reference/EVALUATION.ja.md#pusht-2-新しいrawbt-runを評価する)

<a id="pusht-3-既存checkpointを評価する前の確認"></a>

[pusht-3-既存checkpointを評価する前の確認](reference/EVALUATION.ja.md#pusht-3-既存checkpointを評価する前の確認)

<a id="pusht-4-既存checkpointの設定確認だけを行う"></a>

[pusht-4-既存checkpointの設定確認だけを行う](reference/EVALUATION.ja.md#pusht-4-既存checkpointの設定確認だけを行う)

<a id="pusht-5-評価を実行しログを見る"></a>

[pusht-5-評価を実行しログを見る](reference/EVALUATION.ja.md#pusht-5-評価を実行しログを見る)

<a id="pusht-6-終了と結果を確認する"></a>

[pusht-6-終了と結果を確認する](reference/EVALUATION.ja.md#pusht-6-終了と結果を確認する)

<a id="pusht-動かない極端に低い成功率を調べる"></a>

[pusht-動かない極端に低い成功率を調べる](reference/EVALUATION.ja.md#pusht-動かない極端に低い成功率を調べる)

<a id="pusht-7-公式モデルと比較する"></a>

[pusht-7-公式モデルと比較する](reference/EVALUATION.ja.md#pusht-7-公式モデルと比較する)

<a id="pusht-8-世界モデルrolloutの速度を測る"></a>

[pusht-8-世界モデルrolloutの速度を測る](reference/EVALUATION.ja.md#pusht-8-世界モデルrolloutの速度を測る)

<a id="pusht-9-別ステップケース数とトラブル"></a>

[pusht-9-別ステップケース数とトラブル](reference/EVALUATION.ja.md#pusht-9-別ステップケース数とトラブル)

<a id="libero-比較の位置付け"></a>

[libero-比較の位置付け](reference/EVALUATION.ja.md#libero-比較の位置付け)

<a id="libero-1-分割を作る"></a>

[libero-1-分割を作る](reference/EVALUATION.ja.md#libero-1-分割を作る)

<a id="libero-2-osmesa画像監査"></a>

[libero-2-osmesa画像監査](reference/EVALUATION.ja.md#libero-2-osmesa画像監査)

<a id="libero-3-rawbtを新規学習する"></a>

[libero-3-rawbtを新規学習する](reference/EVALUATION.ja.md#libero-3-rawbtを新規学習する)

<a id="libero-4-実環境cem評価"></a>

[libero-4-実環境cem評価](reference/EVALUATION.ja.md#libero-4-実環境cem評価)

<a id="libero-5-ブラウザの可視ui"></a>

[libero-5-ブラウザの可視ui](reference/EVALUATION.ja.md#libero-5-ブラウザの可視ui)

<a id="libero-実行確認した範囲"></a>

[libero-実行確認した範囲](reference/EVALUATION.ja.md#libero-実行確認した範囲)

<a id="intermediate-実行前の確認"></a>

[intermediate-実行前の確認](reference/EVALUATION.ja.md#intermediate-実行前の確認)

<a id="intermediate-pusht"></a>

[intermediate-pusht](reference/EVALUATION.ja.md#intermediate-pusht)

<a id="intermediate-libero-10"></a>

[intermediate-libero-10](reference/EVALUATION.ja.md#intermediate-libero-10)

<a id="intermediate-結果の扱いと確認範囲"></a>

[intermediate-結果の扱いと確認範囲](reference/EVALUATION.ja.md#intermediate-結果の扱いと確認範囲)

</details>
