# A/B/C 先行研究ノート（2026-09-07）

対象は、通常の観測・行動・次観測から、軽量で共有の LeWM を学び、現在非操作の物体を後続操作まで予測状態に残すこと（PushT と LIBERO-10）である。これは文献探索メモであり、実装、学習、Issue 更新はしていない。大規模 WAM の結果を小型 LeWM 案の有効性の証拠には用いない。

## 読み方と資料の状態

2026-09-07 に一次資料を確認した。`arXiv:2607.26924v3` は著者プレプリント（ICLR 等の査読採録の記載を確認できない）、`arXiv:2107.11676` は arXiv プレプリント、RIM は ICLR 2021 論文、C-SWM は ICLR 2020 論文である。以下で「著者ら」は資料の報告、「本ノートの含意」はそこからの限定的な推論である。

|資料（版・読んだ範囲）|直接確認した内容|本件に対する境界|
|---|---|---|
|Liu et al., [TC-LeWM, arXiv:2607.26924v3](https://arxiv.org/abs/2607.26924v3), abstract と HTML の方法・実験節|全潜在でなく時系列中心化残差へ SIGReg を掛け、持続／残差の分散配分を分離すると報告。下流は主に凍結表現の方策学習。|A の残差枝は直接の先行。CEM、PushT、提案の平均側帯域・PR 下限、非操作物体保持はこの論文だけからは出ない。|
|Biza, van der Pol, Kipf, [Negative Sampling and C-SWM, arXiv:2107.11676](https://arxiv.org/abs/2107.11676), abstract、§4.1--4.3、結論|系列外ばかりの負例では不動物体の配置で episode を識別でき、遷移を学ばない反例を報告。同一 episode 負例を混ぜると、当該 gridworld の10段 Hits@1 が 12%→95%（2D）等に回復。|B の in/out 分離と背景ショートカット監査の直接先例。C-SWM は object slots/GNN、gridworld/Atari、対照的遷移損失であり、LeWM の MSE+SIGReg、LIBERO、CEM の性能根拠ではない。|
|Kipf et al., [C-SWM, arXiv:1911.12247v2 / ICLR 2020](https://arxiv.org/abs/1911.12247v2), abstract、§4、実験設定|画像から object representations と関係を対照的に学習し、独立に操作できる複数物体等で評価。|「非操作物体を構造として残す」という目標には近いが、object slot/関係 GNN と追加構造を導入するため、E/A/F を維持する A--C の簡易介入とは別設計。|
|Goyal et al., [RIM, arXiv:1909.10893 / ICLR 2021](https://arxiv.org/abs/1909.10893), abstract と方法の導入|複数の recurrent mechanism を attention で疎に通信させ、各時刻に関係する mechanism だけ更新する。|C の「選択的更新」には概念上の先例。ただし C は座標ごとの `g⊙(p-z)` 一回混合であり、RIM の複数セル、競合、attention 通信、因子化を主張しない。|
|Cho et al., [GRU, arXiv:1406.1078](https://arxiv.org/abs/1406.1078), 書誌・abstract|更新／リセット gate を持つ recurrent unit を提案。|保持・更新 gate 自体は既知。C の可能な差は、LeWM の予測終端で現状態との convex mixing を行い、通常の一段教師だけで「観測にはあるが rollout で失う」情報を診断する点に限る。|
|Zhou et al., [τ0-WM, arXiv:2606.01027v2](https://arxiv.org/abs/2606.01027v2), abstract、関連研究・実験の記述|多視点・言語・robot state を含む video diffusion 型の action-conditioned simulator。著者記載で約27,300時間のデータと test-time candidate sampling を使用。|action-conditioned future simulation という上位課題は一致するが、規模、入力、拡散、データ、評価が根本的に異なる。小型 LeWM、A/B/C、あるいは通常軌道の少データ効率の証拠にしない。|
|Zhang et al., [ResWM, arXiv:2603.11110](https://arxiv.org/abs/2603.11110), abstract|残差「行動」表現と隣接観測差 encoder を Dreamer 系に入れる著者プレプリント。|C の residual **state update** とは異なる。行動表現変更、Dreamer、RL であり、LeWM の gate の先例／対照には弱い。|

## A: 時間残差 SIGReg + 持続平均の分散帯域、A+ PR 下限

Issue #8 の式は、窓内平均 `m_b=T^{-1}Σ_t z_{b,t}` と残差 `r_{b,t}=z_{b,t}-m_b` を作り、

```math
L_A=L_{pred}+\lambda_r SIGReg(R)+\lambda_m\{[\gamma-v_M]_+^2+[v_M-\Gamma]_+^2\},
\quad v_M=(BD)^{-1}\sum_b\|m_b-\bar m\|^2.
```

とする。有限標本の `Var(z)=v_M+v_R` は分解恒等式であり、TC-LeWM の「残差へ SIGReg」の直接拡張は平均成分を無制約に戻す点ではない。A は平均の**総量だけ**を帯域に置く。

* 重なり：TC-LeWM は残差へ Gaussian 正則化を移すため、A の主幹は既知である。VICReg 型の variance/covariance regularisation も「崩壊を尺度・相関から抑える」一般先例であり、帯域のみを別名で主張できない。
* A の残余：TC が直接採点しない系列一定 offset に `v_M≥γ` の非崩壊圧を加える。だが trace 下限は rank 1 でも満たせ、背景や task/episode の恒常要因でも満たせる。静止窓では `r=0` で、残差 SIGReg が観測ノイズを増幅し得る。完全崩壊で共分散損失の勾配が零となり得る点も回避しない。
* A+：`q(C_M)=(tr C_M)^2/(||C_M||_F^2+ε)` により `q(C_M)≥q0` を緩く要求する。`ε→0` の非零共分散では effective rank / participation ratio で、rank 1 は `q=1`、等分散 k 方向は `q=k`。従って `q0>1` なら rank-1 解は**罰則を満たせない**。これは TC の式にはなく、全方向の白色化とも異なる。`q0` は物体数・タスク数ではなく、`q0≤min(D,B−1)` が必要。
* A+ の限界：罰則が rank-1 解から新方向へ実際に脱出させる保証はない（完全崩壊や厳密対称点では共分散由来の勾配が零になり得る）。複数の背景方向や task cluster だけで PR を満たせもする。従って A/A+ の採否は、平均・残差のスペクトルに加え、**task 内**の統計、非操作物体／gripper 読出し、行動入替応答、実測未来を再入力しない rollout と同一 CEM 成功率で判定する。

## B: 系列内外の一段未来識別

Issue #12 の候補は、予測 `q_{b,t}` と実測 future key `y_c` に

```math
s(q,y)=-||q-y||^2/(D\tau),\qquad
\ell_G=log\sum_{c\in C_G}e^{s(q,y_c)}-log\sum_{p\in P}e^{s(q,y_p)}
```

を、同一 episode 別時刻の `C_in` と別 episode の `C_out` へ別々に加えるもの。既知の複数正例には log-sum-exp を使い、空群は0、正例を含まない非空群は入力エラーとする仕様である。

* 重なり：Biza et al. の C-SWM-ER は、同一 episode 負例を混ぜることで episode 固有の不動物体だけを読むショートカットを壊す。したがって「in 負例が変化を、out 負例が持続を必ず学ばせる」という強い解釈は不適切で、負例設計が表現を変えるという既知知見の適用である。
* B の有望な差分：一段 action-conditioned LeWM の予測 `q` を query とし、同じ batch から固定個数の in/out 候補をマスクして、追加 E/F forward・全ペア `Q²` を要求しない。task ID を model に渡さず、候補抽出だけを同 task に制限する設計は背景/task ID shortcut を弱める試みである。**B は SIGReg 等の正則化の置換ではなく、初期仕様では Raw SIGReg を残した補助識別目的である。**
* 限界：別 episode/別時刻は物理的負例の証明ではない。停止、復帰、重複 frame は false negative、同 task 内でも camera/background/trajectory ID が shortcut になる。全 latent 一様では `ell=log K` でも勾配ゼロになり得るため、SIGReg を残す理由はあるが、両者の併用が必要情報を保証はしない。まず `β=0`、in-only、out-only、通常 CPC、両群を候補数・GPU 時間・軌道集合まで揃える。

## C: 状態・行動依存の保持更新 gate

Issue #13 の挿入は predictor の candidate `p_t=P(F_core(...))` の**後**に

```math
g_t=\sigma(W_g h_t+b_g),\qquad \hat z_{t+1}=z_t+g_t\odot(p_t-z_t)
```

と置く。`g=0` は copy、`g=1` は既存 candidate で、未来観測を予測器入力にしない。C はAの正則化変更・Bの補助識別目的とは別の出力構造実験である。

* 重なり：GRU は gate、RIM は selective update の既存先例。従って「gate で保持更新する」こと、新しい `g` 値を物理的更新確率／物体発見とみなすことは新規主張にできない。
* C の限定的な研究問い：同じ一段 MSE+SIGReg の下で、観測 `z` に読出せる非操作対象が `\hat z` / rollout で失われるなら、copy path がその**予測側の劣化**を改善するか。これは gate 値の可視化ではなく、観測対予測の probe、行動開始時の更新、衝突・空振り・役割逆転、CEM で検査する。
* 限界と必須対照：`(p,g)` は非同定（例：同じ出力を複数組が作る）。`p` の増幅と小さい `g` は相殺でき、copy path だけで長期安定性は保証しない。固定 `g=.9`、単純残差、同程度 parameter 増量、学習 gate を、同じ rollout 実装・保存復元・計画予算で比較する。固定/残差が同等なら gate の複雑さは不採用。

## 結論：優先順位と主張可能範囲

1. A は TC の直接近接で新規性が最も弱い一方、SIGReg 適用範囲という主題への因果的切り分けが最も明確。A+ は trace-only の rank-1 反例に正の罰則を与えるが、最適化での脱出も意味的保持も保証しない。TC を超える主張には Raw/TC/A/A+ の同条件比較が必要。
2. B は「負例の時間・episode 構造が世界モデルを大きく変える」という強い既知の警告に沿う。ただし新規性は in/out 二群化と LeWM の小型一段・通常軌道という制約下で、ショートカットを減らし実際の control/reuse を改善するかにある。
3. C は既存 gated recurrent dynamics に関連する軽量な挿入であり、観測表現には情報があるがpredictor/rolloutで失われると診断した場合に優先度が上がる。まずRawを含む観測`z`と予測後の情報保持を比較する。A/Bの学習を先に必須化するものではない。

いずれも、巨大 WAM の著者報告を小型案の証明へ移さない。最小の成功主張は、同一初期化・軌道・更新数・計画器・評価で Raw（必要なら TC）より、PushT を落とさず LIBERO-10 の平均・下位 task・非操作対象の予測保持を改善した、までである。

## 追加確認：Gaussian 識別可能性、直接的な residual dynamics、WAM の語

### Gaussian を平均側で緩める A が継承できないもの

Klindt, LeCun, Balestriero, [*When Does LeJEPA Learn a World Model?*, arXiv:2605.26379v1](https://arxiv.org/html/2605.26379v1) は著者プレプリントである。本追記では abstract、§3--§5、§7、Appendix D.2 を読んだ。

同論文の線形識別可能性は、(i) 真の潜在の各成分が独立、(ii) 正例対で定常、独立な加法ノイズ遷移、(iii) 特に `z~N(0,I)` と OU 型 `z'=ρz+sqrt(1-ρ²)η`、`0<ρ<1`、(iv) encoder 出力次元と真の潜在次元が一致、(v) population の global optimum で alignment と**出力全体の** Gaussianity を満たす、という共同仮定で `h(z)=Qz`（直交 Q）を導く。Gaussian では Hermite の次数 d が `ρ^d` で減衰するため、alignment は非線形な次数を厳密に不利にする。

これは A/A+ へ次を保証しない。

* A は `r` のみを Gaussian 化し、`m` は trace/PR だけで、全出力 `h(z)~N(0,I)` も whitening も要求しない。従って同定定理の feasible set が違い、直交的な線形回復は継承しない。
* 接触・切替・多物体相互作用、goal/専門家軌道の偏った訪問分布、有限 batch は独立な定常 OU 正例対ではない。同論文自身も次元一致、有限標本と最適化速度を未解決と明記し、人口 global optimum の結果である。
* 理論は encoder 側だけである。著者らは action-conditioned transition `p(ẑ'|ẑ,a)` は別途データから学ぶ必要があり、同定を証明していないと明記する。制御遷移の同定には action が全潜在方向を励起する条件が必要になり得る。よって A の一段 F、rollout、CEM、非操作物体の保持には保証がない。
* Gaussian 緩和を「常に悪化」と読むことも誤りである。この理論は上の世界クラスで Gaussian が線形識別可能性に一意、という条件付き結論であり、A は TC の実証仮説として同じ小型・通常軌道の比較で判定する。

### C に近い action-conditioned residual dynamics（最大2件）

厳密に `ẑ=z+sigmoid(g)⊙(p-z)`、画像から一段 JEPA 教師だけで学ぶ**小型** latent dynamics の先例は、今回確認した範囲では見つからなかった。次の二つは近いが、C の新規性を強く支える証拠にはしない。

1. Jiang et al., [*FlowMo-WM*, arXiv:2606.13817](https://arxiv.org/abs/2606.13817)（著者プレプリント、abstract、§4.1、limitations を確認）は、画像・行動履歴から短期 motion latent と長期 drift context を作り、`z_{t+1}=z_t+F(z_t,a_t)+R(z_t,a_t,c_t)-R(z_t,a_t,0)` と明示的 residual state transition を用いる。保持したい外生文脈を residual で分ける点は C に近い。しかし 32-step history、pose supervision、60-step rollout、2D aquatic simulatorであり、座標ごとの学習 gate、JEPA、PushT/LIBERO の先例ではない。
2. Zhang et al., [*DWM: Separating World Effects from Actions in Latent World Models*, arXiv:2607.18715v1](https://arxiv.org/abs/2607.18715v1)（著者プレプリント、abstract を確認）は、次 latent の変化を action-invariant world effect と補完的 action-driven residual に加法分解する auxiliary world head/contrastive objective を提案する。著者は PushT-W 等の自作 W-variant で CEM 成功の平均絶対改善 13.1% と報告するが、これは原資料の著者報告であり本案へ移植しない。C の copy/update gate でも小型性の証拠でもなく、「action によらず続く変化」と「action 効果」を分離する競合設計として扱う。

### WAM / world action model は同一の実験カテゴリーではない

`world action model` / WAM は標準化された単一のアーキテクチャ名ではない。文脈により、(a) 観測から action をも出力する video-action model、(b) action を条件として未来を生成する video simulator、(c) latent action を推定するモデル、(d) planning/policy 評価まで含む複合系を指し得る。例えば τ0-WM は一つの著者プレプリント内で video-action model と action-conditioned video simulator の二接口を併せ、約27,300時間、multi-view/言語/robot state、diffusion、test-time sampling を用いる。

従って WAM の名称や大規模成績は、A/B/C の E/A/F・一段予測・通常軌道・LeWM 規模・PushT/LIBERO-10 における有効性の証明ではない。引用する場合も「action-conditioned future simulation を大規模には研究している」という背景に限定し、規模、入力、訓練目的、評価、計画器が一致する小型対照を代替しない。
