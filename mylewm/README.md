# myLeWM

提案するマルチタスクLeWM拡張の実装領域です。`multitask_lewm.py` は公式LeWMの共有E/A/Fに接続できる、正規化付き未来予測・対応未来識別損失を提供します。

公式比較対象は sibling directory の `../lewm/` に分離しています。公式コードは変更せず、提案側だけを変更します。

## Smoke test

```bash
PYTHONPATH=mylewm .venv/bin/python -m pytest -q mylewm/test_multitask_lewm.py
```

現在は損失モジュールと合成テンソルの勾配経路を検証済みです。実データによる学習にはLeWMのHDF5データセットが必要です。

## 公式LeWMとの接続

`MultiTaskJEPAObjective` は、公式LeWMの `JEPA` インスタンスを受け取り、共有エンコーダ・行動エンコーダ・予測器を使って損失を計算します。

```python
from multitask_jepa import MultiTaskJEPAObjective

objective = MultiTaskJEPAObjective(
    model=official_lewm,
    temperature=0.1,
    cross_weight=1.0,
    target_detach=True,
)

loss, metrics = objective.loss_from_observations(
    contexts={"view_a": context_a, "view_b": context_b},
    futures={"view_a": future_a, "view_b": future_b},
    horizons=(1, 2, 4),
    positive=positive_mask,
)
loss.backward()
optimizer.step()
```

`context_*` は `pixels: [B,T,C,H,W]` と `action: [B,T,A]` を持つ辞書、`future_*` は `pixels: [B,T,C,H,W]` を持つ辞書です。同じバッチ行・同じ時刻が対応するように並べます。`horizons` はモデル系列中の時刻インデックスで、各時刻について別々に候補識別損失を計算します。

`positive` は `[B,B]` の真偽マスクです。行 `b` の予測に対して、列 `c` を物理的に同じ結果とみなせる場合に `True` にします。デフォルトでは対角も正例として追加されます。独立サンプルを混ぜる場合や対応が不明な場合は、誤った対角対応を作らないよう `cross_task_loss(..., diagonal_positive=False)` を使い、対応マスクを明示してください。

## スモークテスト

```bash
PYTHONPATH=mylewm .venv/bin/python -m pytest -q mylewm/test_multitask_lewm.py
```

このテストは、損失の有限性、複数正例、target detach、複数ホライズンの公式JEPA風接続、勾配経路を確認します。実際の画像エンコーダやロボット制御の性能は検証しません。

## 公式LeWMを使う場合

公式コードは `../lewm/` にあります。依存環境を作成した後、公式LeWMの設定に従って `JEPA` を生成し、上記の `MultiTaskJEPAObjective` に渡します。

```bash
uv pip install --python .venv/bin/python 'stable-worldmodel[train,env]'
PYTHONPATH=lewm:mylewm .venv/bin/python -m pytest -q mylewm/test_multitask_lewm.py
```

公式の学習データとチェックポイントは自動では含めていません。LeWMのHDF5データを取得した場合は、公式データローダで読み出した各ビューの画像・行動系列を `contexts` と `futures` に変換してから学習します。学習時には、物理的に対応する未来の正例マスクをデータの並びと同じ順序で作成してください。

## 研究目的との対応

この実装が提供するのは、複数タスクの経験を同じE/A/Fへ通し、対応する未来潜在を共有損失で学習する最小経路です。研究目的である「一つのタスクで認識した状態を別タスクの予測へ再利用すること」を確認するには、実データで次を別々に評価する必要があります。

- 一つの状態から複数操作の未来を予測できるか
- ある操作の予測終点から、未来画像を再入力せず別操作を予測できるか
- 未学習の配置・操作順序でもモデルを固定して再利用できるか

現段階では、公式LeWMの完全なデータセット学習ループ、赤操作から青操作へつなぐ実機・シミュレータ評価、CEMによる制御評価は未実装です。損失単体のテスト合格を、共有世界モデルの形成結果とは解釈しません。
