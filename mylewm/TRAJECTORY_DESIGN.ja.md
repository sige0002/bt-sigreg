# 通常軌道から学ぶmyLeWM：方式選択と評価計画

2026-09-06。ユーザーの『交差対応表なしで経験から世界を学んでほしい』という
指示に基づく設計変更。旧crossed_v1は現行の学習方式ではない。

## 方式の選択

| 先行研究 | 利用できる原理 | 今回の判断 |
|---|---|---|
| [LeWorldModel](https://arxiv.org/html/2603.19312v3) | 次潜在予測＋SIGRegによる非崩壊 | 基本目的関数・共有E/A/Fを維持 |
| [SPR](https://arxiv.org/abs/2007.05929) | 実軌道上の複数ステップ潜在予測、画像拡張 | 自己回帰の多段予測原理を採用。SPR全体の再実装ではない |
| [PlaNet](https://planetrl.github.io/) | latent overshootingで多段遷移の整合を学ぶ | 多段学習の参考。確率状態空間モデル・画像復号器への置換はしない |
| [VICReg](https://arxiv.org/abs/2105.04906) | 分散・共分散による非崩壊 | 代替候補だが、まずSIGRegを残して変更を限定 |

上記の組合せがPushTで改善することは文献から保証できない。今回は
LeWM＋open-loop予測の追加学習が基本性能を維持するかを調べる実験である。
独自損失やマルチタスク能力の新規性を実証したという主張ではない。

## 実装

trajectory_experiment.pyを使う。公式JEPAのE/A/F・潜在尺度・CEMを維持する。
単位長正規化、交差表、複製ビュー、タスク埋め込み、画像復号器は追加しない。

画像は実データのt=0,5,10,15,20,25,30。各画像間の5個の実行動を連結する。
最初の3画像がcontextで、残り4画像が未来教師。
次潜在予測は公式と同じ1ステップシフトを用いる。追加項はcontext終端から
H=1,2,4モデルステップ先の自己回帰誤差。実測未来は予測器へ渡さない。

L = L_one_step + 0.09 SIGReg(z) + 0.1 L_open_loop

教師エンコーダにも勾配を流す。画像ImageNet正規化、行動標準化、5×2の
行動ブロックは訓練・評価でそろえ、学習統計をcheckpointへ保存する。
一般の複数タスクを混合できる計算原理だが、この実験のデータはPushTだけ。

## 初回実験

公式checkpointから全パラメータを追加学習（ランダム初期化実験ではない）。
初期SHA256とmanifestを固定。旧myLeWMの重みは使用しない。
1,000更新、batch=16、AdamW lr=1e-6、weight decay=1e-3、clip norm=1。
SIGRegは公式の17 knots/1024 projections。1000更新で保存・検証。
学習量の少ない予備的な保持試験であり、ゼロからの収束予算ではない。

同じ元HDF5から追加学習用18,307episode、検証128、pilot50、confirm200を
エピソード単位で分離。評価用の開始位置も学習前に固定。
公式初期checkpointはこのHDF5上で学習済みなので、評価episodeを公式にとって
未知と呼ばない。行動正規化統計も互換性のため同じHDF5全体から算出する。

## 判定

公式と固定候補を同じ開始状態・Goal・seed・CEM・50制御ステップで評価。
成功判定は使用中のswm/PushT-v1の状態距離基準（手先＋物体位置の差のノルム<20、
物体角度差<π/9）。標準固定T目標へのcoverage95%成功率とは同一視しない。

pilot50は診断、confirm200は固定候補の最終確認。confirmを見て候補を調整しない。
50ずつ実行し、JSONにcheckpoint SHA256、episode/start、個別成否、全configを保存。
学習コード・データ分割の独立レビューで重大な時刻ずれや分割漏洩なし。

観測された成功率差が0以上かと、母成功率差が0以上と主張できるかを分ける。
compare_paired.pyはcandidateだけ成功した確率の片側下限とbaselineだけ成功した
確率の片側上限をClopper–Pearsonで求め、Bonferroniで保守的な95%下限を作る。
下限>=0なら厳密な非劣性の判定を満たす。両者が同率でも通常は下限<0となり、
有限試行で『絶対に落ちない』ことは保証できない。許容低下幅を勝手に緩めない。

この検証は固定checkpointと当該評価分布の比較である。複数学習seed、
未見タスク、未見配置、順序接続での性能は別途検証が必要。

## 実行

```bash
PYTHONPATH=lewm:mylewm .venv/bin/python mylewm/trajectory_experiment.py prepare
PYTHONPATH=lewm:mylewm .venv/bin/python mylewm/trajectory_experiment.py train
PYTHONPATH=lewm:mylewm .venv/bin/python mylewm/run_trajectory_evaluation.py
```

prepareはmanifest上書きを拒否。trainは既存ログがあれば拒否。
結果は.cache/stable-wm/pusht/trajectory_v1/に保存。
