# PushT固定ケース失敗監査

実施：2026-09-09。対象は固定confirm先頭50ケース、CEM/environment seed 42、BT 100,000更新と公式配布LeWM。制御評価済みのcheckpoint・manifest・ケースhashを照合し、デモ軌道上で一段teacher-forced予測と5遷移自由rolloutを測った。

## 結論

固定ケースの失敗を、直ちに「世界モデルが学習できていない」とは扱えない。両モデルが失敗した3件では遷移誤差上昇との関連が見えるが、BT/公式で成否が入れ替わった5件は、expert軌道上の遷移MSEだけでは説明できない。CEM探索、計画した軌道でのモデル誤差、Goal latent costの順位付けを次段階で測る必要がある。

| モデル内のケース群 | 件数 | 一段MSE平均 | 5遷移終端MSE平均 |
|---|---:|---:|---:|
| BT・両方成功 | 42 | 0.002912 | 0.02001 |
| BT・両方失敗 | 3 | 0.006102 | 0.06184 |
| BT・BTだけ失敗 | 3 | 0.003202 | 0.02742 |
| 公式・両方成功 | 42 | 0.006235 | 0.04780 |
| 公式・両方失敗 | 3 | 0.009316 | 0.06167 |
| 公式・公式だけ失敗 | 2 | 0.005181 | 0.03294 |

BTの両方失敗群は共通成功群に対し、一段で約2.10倍、5遷移で約3.09倍。ただしn=3で、平均は外れ値の影響を受ける。BTだけ失敗群は約1.10倍／1.37倍。公式だけ失敗群は約0.83倍／0.69倍であり、「遷移MSEが大きいからそのモデルだけ失敗した」という単純な対応はない。

BTと公式の生MSEを直接順位付けしない。潜在尺度・分布が異なるため、主に各モデル内部の成功群対失敗群を見る。さらに本監査はexpert軌道を使い、実際にCEMが選んだ軌道の反実仮想誤差を測っていない。関連は原因を証明しない。

## 測定定義

- 一段：実画像4枚（5物理step間隔）と実行動3 chunkから、3個の次latentをteacher forcingで予測したMSEの平均。
- 5遷移：実画像3枚を履歴とし、続く実行動7 chunkで自由rolloutし、35物理step後の実画像latentとの終端MSEを測る。CEMの計画horizon 5に対応する。
- BTはcheckpointに保存されたtrain-only行動統計、公式LeWMは上流evalと同じ全データ行動統計を使う。
- 生結果：`output/pusht/failure_audit_bt_official_seed42.json`（Git対象外）。

## 再実行

```bash
PYTHONPATH=.:lewm .venv/bin/python mylewm/tools/audit_pusht_failure_modes.py \
  --bt-checkpoint .cache/stable-wm/pusht/bt_spectral_v2_100k_s3072/step_100000_object.ckpt \
  --official-checkpoint .cache/stable-wm/pusht/lewm_object.ckpt \
  --bt-result output/pusht/repeated_eval/fixed_bt_seed42/results.txt.json \
  --official-result output/pusht/repeated_eval/fixed_official_seed42/results.txt.json \
  --manifest .cache/stable-wm/pusht/rbg_v0/manifest.json \
  --output output/pusht/failure_audit_bt_official_seed42_new.json
```

次段階では固定された失敗・対照ケースだけを計測付きで再評価し、各再計画でCEM候補、予測cost、選択候補、実現したGoal距離、予測した次latentと実観測latentを保存する。これにより「遷移」「cost」「探索」を分ける。本監査だけでCEMが原因と断定しない。
