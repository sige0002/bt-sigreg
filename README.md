# Task-Centered SIGReg for Multi-Task LeWorldModel
## マルチタスク LeWM におけるタスク条件付き潜在正則化の研究アイデアメモ

**ステータス:** 研究アイデア整理段階  
**目的:** LeWorldModel (LeWM) の軽量・高速な JEPA 世界モデルという利点を維持しながら、マルチタスク学習で生じる潜在表現の干渉を緩和する。

---

## 1. 背景

LeWorldModel (LeWM) は、画像から潜在表現を生成し、行動を条件として次時刻の潜在表現を予測する世界モデルである。

基本構造は、

\[
z_t = E_\theta(o_t)
\]

\[
\hat{z}_{t+1}
=
F_\psi(z_{\le t}, a_{\le t})
\]

で表される。

- \(o_t\): 時刻 \(t\) の画像観測
- \(a_t\): 行動
- \(E_\theta\): 画像エンコーダ
- \(z_t\): 潜在表現
- \(F_\psi\): 行動条件付き予測器

LeWM の主要な特徴は、画像再構成を行わず、潜在空間で未来予測を行うことである。

学習損失は概ね、

\[
\mathcal{L}_{\mathrm{LeWM}}
=
\mathcal{L}_{\mathrm{pred}}
+
\lambda
\mathcal{L}_{\mathrm{SIGReg}}
\]

で構成される。

予測損失は、

\[
\mathcal{L}_{\mathrm{pred}}
=
\mathbb{E}
\left[
\|\hat z_{t+1}-z_{t+1}\|^2
\right]
\]

である。

一方、SIGReg は潜在表現の周辺分布を等方ガウス分布

\[
p(z)
\approx
\mathcal N(0,I)
\]

へ近づけることで、全画像が同一潜在表現へ写像されるような**表現崩壊**を防止する。

---

## 2. JEPA における基本的な表現崩壊

予測損失だけの場合、

\[
E_\theta(o)=c
\]

という定数写像が存在しうる。

すると、

\[
z_t=c,
\qquad
z_{t+1}=c,
\qquad
\hat z_{t+1}=c
\]

となるため、

\[
\mathcal L_{\mathrm{pred}}=0
\]

を達成できる。

しかし、この潜在表現には状態情報が存在しない。

SIGReg は、

\[
z\sim \mathcal N(0,I)
\]

という非退化な分布を要求することで、この自明解を禁止する。

---

# 3. マルチタスク LeWM における問題

複数タスクを同じ LeWM で学習することを考える。

タスク変数を

\[
T\in\{1,\ldots,K\}
\]

とする。

例えば、

- Task A: 赤いキューブを右へ置く
- Task B: 青いキューブを左へ置く
- Task C: ボタンを押す

などである。

各タスクの潜在分布が、

\[
z\mid T=k
\sim
\mathcal N(\mu_k,\Sigma_k)
\]

のような異なるクラスタを形成すると仮定する。

全体の潜在分布は、

\[
p(z)
=
\sum_{k=1}^{K}
p(T=k)\,p(z\mid T=k)
\]

となる。

これは一般には**混合分布**であり、単一のガウス分布とは限らない。

---

## 4. Raw LeWM の SIGReg とマルチタスク構造の衝突

Raw LeWM はタスクを区別せず、

\[
p(z)\approx \mathcal N(0,I)
\]

を要求する。

しかしマルチタスクでは、

```text
潜在空間

 Task A                 Task B                 Task C

  ○○○                    ○○○                    ○○○
 ○ μA ○                  ○ μB ○                  ○ μC ○
  ○○○                    ○○○                    ○○○
```

のように、タスクごとに異なる潜在クラスタを持つことが自然である。

にもかかわらず、Raw LeWM では、

```text
Task A ─┐
Task B ─┼──> 全潜在表現を混合 ──> SIGReg ──> N(0,I)
Task C ─┘
```

となる。

そのため、タスク間のクラスタ中心

\[
\mu_A,\mu_B,\mu_C
\]

が離れていること自体が、「単一ガウスからのずれ」として正則化される可能性がある。

---

## 5. 分散分解による理解

全潜在分布の共分散は、

\[
\mathrm{Cov}(z)
=
\mathbb E_T
[
\mathrm{Cov}(z\mid T)
]
+
\mathrm{Cov}_T
(
\mathbb E[z\mid T]
)
\]

と分解できる。

すなわち、

\[
\boxed{
\mathrm{Cov}(z)
=
\text{タスク内分散}
+
\text{タスク間分散}
}
\]

である。

Raw LeWM の SIGReg は全体として、

\[
\mathrm{Cov}(z)\approx I
\]

を要求する。

そのため、

- タスク間の違い
- タスク内部の状態変化

が同じ分散の「予算」を共有することになる。

極端には、

```text
タスク間差       : 非常に大きく表現
タスク内状態変化 : 小さく表現
```

でも、全体として十分に分散していれば SIGReg を満たしうる。

これは、マルチタスク世界モデルにおいて望ましくない可能性がある。

---

# 6. 提案アイデア: Task-Centered SIGReg

## 6.1 基本発想

Raw LeWM の

\[
\mathrm{SIGReg}(Z)
\]

を、

\[
\mathrm{SIGReg}(Z_k-\mu_k)
\]

へ変更する。

つまり、

> 全タスクをまとめて1個のガウス分布にするのではなく、各タスクのクラスタ内部だけに SIGReg を適用する。

---

## 6.2 タスク中心

タスク \(k\) に属する潜在表現集合を、

\[
Z_k
=
\{z_i\mid T_i=k\}
\]

とする。

その中心を、

\[
\boxed{
\mu_k
=
\frac{1}{|Z_k|}
\sum_{z_i\in Z_k}
z_i
}
\]

と定義する。

各サンプルについて、

\[
\boxed{
r_i
=
z_i-\mu_{T_i}
}
\]

というタスク中心からの残差を作る。

---

## 6.3 イメージ

### 元の潜在空間

```text
Task A                           Task B

   ○ ○ ○                           ○ ○ ○
 ○   μA  ○                       ○  μB   ○
   ○ ○ ○                           ○ ○ ○
```

### SIGReg を計算するときだけ中心を引く

```text
Task A residual                 Task B residual

    ○ ○ ○                           ○ ○ ○
  ○   0   ○                       ○   0   ○
    ○ ○ ○                           ○ ○ ○
```

重要なのは、**元の潜在空間で \(\mu_A\) と \(\mu_B\) を同じ位置へ移動するわけではない**ことである。

SIGReg を計算するときだけ、各クラスタの局所座標系へ変換している。

---

# 7. タスク別 SIGReg

各タスクについて、

\[
\mathcal L_k
=
\mathrm{SIGReg}
\left(
Z_k-\mu_k
\right)
\]

を計算する。

例えば3タスクなら、

\[
\mathcal L_A,
\qquad
\mathcal L_B,
\qquad
\mathcal L_C
\]

を独立に計算する。

その後、

\[
\boxed{
\mathcal L_{\mathrm{TaskSIG}}
=
\frac{1}{K}
\sum_{k=1}^{K}
\mathcal L_k
}
\]

とする。

---

## 8. 「損失を平均する」の意味

ここで平均しているのは**潜在表現ではない**。

例えば、

\[
\mathcal L_A=0.10,
\qquad
\mathcal L_B=0.20,
\qquad
\mathcal L_C=0.15
\]

なら、

\[
\mathcal L_{\mathrm{TaskSIG}}
=
0.15
\]

とするだけである。

つまり、

```text
Task A の「クラスタ内部が正常か」の採点 → 0.10
Task B の「クラスタ内部が正常か」の採点 → 0.20
Task C の「クラスタ内部が正常か」の採点 → 0.15

                       ↓

                  損失値だけ平均
```

である。

以下のように潜在表現そのものを平均するわけではない。

\[
\frac{z_A+z_B+z_C}{3}
\]

したがって、

- 赤いキューブを右へ置く
- 青いキューブを左へ置く

という異なる意味が、損失平均によって混ざるわけではない。

---

## 9. なぜ最後にまとめるのか

ニューラルネット学習では、最終的に

```python
loss.backward()
```

へ渡す1つのスカラー目的関数を作る必要がある。

そのため、

\[
\mathcal L_A,\mathcal L_B,\mathcal L_C
\]

という複数の要求を、

\[
\mathcal L_{\mathrm{TaskSIG}}
=
\frac{\mathcal L_A+\mathcal L_B+\mathcal L_C}{3}
\]

として1つへまとめる。

平均そのものが研究アイデアなのではない。

本質は、

\[
\boxed{
\mathrm{SIGReg}(Z_A\cup Z_B\cup Z_C)
}
\]

を、

\[
\boxed{
\mathrm{SIGReg}(Z_A-\mu_A),
\quad
\mathrm{SIGReg}(Z_B-\mu_B),
\quad
\mathrm{SIGReg}(Z_C-\mu_C)
}
\]

へ変更することである。

---

# 10. 最終学習目的

提案法の最終損失は、

\[
\boxed{
\mathcal L
=
\mathcal L_{\mathrm{pred}}
+
\lambda
\mathcal L_{\mathrm{TaskSIG}}
}
\]

である。

展開すると、

\[
\boxed{
\mathcal L
=
\mathcal L_{\mathrm{pred}}
+
\frac{\lambda}{K}
\sum_{k=1}^{K}
\mathrm{SIGReg}
\left(
Z_k-\mu_k
\right)
}
\]

となる。

---

# 11. 学習アルゴリズム全体

```text
                           multi-task dataset

        Task A                Task B                Task C
      image/action          image/action          image/action
          │                     │                     │
          └─────────────────────┼─────────────────────┘
                                ▼

                         Shared Encoder Eθ
                                │
                                ▼
                          latent z_t
                                │
                ┌───────────────┴────────────────┐
                │                                │
                ▼                                ▼

       Shared LeWM Predictor              task ID で分割
       Fψ(z, action)                           │
                │                     ┌─────────┼─────────┐
                │                     ▼         ▼         ▼
                │                   Task A    Task B    Task C
                │                     │         │         │
                │                    μA        μB        μC
                │                     │         │         │
                │                  z-μA      z-μB      z-μC
                │                     │         │         │
                │                  SIGReg    SIGReg    SIGReg
                │                     │         │         │
                │                     └────┬────┴────┬────┘
                │                          │         │
                │                          └────┬────┘
                │                               ▼
                │                          task average
                │                               │
                ▼                               ▼

          prediction loss                 TaskSIG loss
                │                               │
                └───────────────┬───────────────┘
                                ▼

               L = L_pred + λ L_TaskSIG
                                │
                                ▼

                    backpropagation
                                │
                                ▼

                Shared Encoder + Predictor
```

---

# 12. なぜ JEPA の表現崩壊を防げるのか

仮に Task A 内で、

\[
z_i=c_A
\]

となり、全サンプルが同一表現へ崩壊したとする。

その場合、

\[
\mu_A=c_A
\]

なので、

\[
z_i-\mu_A=0.
\]

全残差が0になる。

したがって残差分布は、

\[
\delta(0)
\]

という一点集中になる。

しかし SIGReg の目標は、

\[
\mathcal N(0,I)
\]

であるため、損失が大きくなる。

したがって、

\[
\boxed{
\text{各タスク内部で全潜在表現が同一点になる表現崩壊を防止できる}
}
\]

と期待できる。

---

# 13. なぜ Raw LeWM のマルチタスク問題を緩和できるのか

Task A 全体を潜在空間で \(\Delta\) だけ平行移動したとする。

\[
z_i'
=
z_i+\Delta
\]

すると、

\[
\mu_A'
=
\mu_A+\Delta.
\]

したがって、

\[
z_i'-\mu_A'
=
(z_i+\Delta)
-
(\mu_A+\Delta)
\]

より、

\[
\boxed{
z_i'-\mu_A'=z_i-\mu_A
}
\]

となる。

つまり Task A のクラスタ中心の絶対位置は SIGReg 損失に影響しない。

そのため、

\[
\|\mu_A-\mu_B\|
\]

を SIGReg が直接縮める圧力を除去できる。

これは、

\[
\boxed{
\text{タスク間構造}
}
\]

と、

\[
\boxed{
\text{表現崩壊防止}
}
\]

を切り離すことを意味する。

---

# 14. もう一つの重要な効果

Task-centered SIGReg では、タスク間差だけで潜在分散を稼ぐことができない。

Raw LeWM:

\[
\mathrm{Var}(z)
=
\mathrm{Var}_{\mathrm{within-task}}
+
\mathrm{Var}_{\mathrm{between-task}}
\]

であるため、タスク間差が大きければ、タスク内状態表現が弱くても全体分散は大きくなりうる。

提案法では、

\[
r=z-\mu_T
\]

なので、タスク間の平均差が除去され、

\[
\mathrm{Var}(r)
\]

は主にタスク内変動を反映する。

したがって、

\[
\boxed{
\text{各タスク内部でも非退化な状態表現を形成する必要がある}
}
\]

という、Raw LeWM より強い制約になる。

---

# 15. 世界モデルの予測結果は混ざらないのか

今回共有するのは世界の状態変化モデル、

\[
F(z_t,a_t)
\rightarrow
z_{t+1}
\]

である。

これは、

\[
z_t\rightarrow a_t
\]

という行動方策ではない。

例えば、

\[
F(z_{\mathrm{red}},a_{\mathrm{right}})
=
z_{\mathrm{red,right}}
\]

と、

\[
F(z_{\mathrm{blue}},a_{\mathrm{left}})
=
z_{\mathrm{blue,left}}
\]

は、同じ \(F\) で同時に学習できる。

入力となる状態と行動が異なるためである。

したがって、

```text
Task A: 赤を右
Task B: 青を左

               ↓

     「赤を左」に平均される
```

という意味平均は、TaskSIG の損失平均によって直接発生するものではない。

---

# 16. LeWM のメリットは維持できるか

今回変更するのは**学習時の正則化**だけである。

そのため以下は原理的に維持できる。

- 画像再構成器不要
- EMA 教師エンコーダ不要
- stop-gradient に依存しない
- 事前学習済み視覚モデル不要
- 単一の共有世界モデル
- 潜在空間での高速ロールアウト
- CEM/MPC の既存構造
- 推論時の追加コストほぼなし
- ViT-Tiny + 192次元 CLS 潜在という軽量構造を維持可能

推論時には task-centered SIGReg 自体を使う必要がないため、Raw LeWM とほぼ同じ推論経路を維持できる。

---

# 17. LeWM の潜在表現が強く抽象化される理由

LeWM の潜在状態は ViT-Tiny の CLS token 1個を用いている。

224×224画像、patch size 14 では、

\[
16\times16=256
\]

個のパッチが存在する。

ViT 内部では、

```text
256 patch tokens
+
CLS token
```

が存在するが、LeWM は最終的に CLS token だけを使う。

ViT-Tiny では CLS は192次元である。

したがって、

```text
224×224 RGB image
       ↓
256 spatial patches
       ↓
ViT
       ↓
CLS token only
       ↓
192-dimensional latent
```

となる。

この強い情報圧縮には、

1. ViT-Tiny による小さな表現容量
2. CLS 1 token のみ利用
3. 画像再構成損失が存在しない
4. 未来潜在予測に不要な情報を保持する必要がない

という複数要因が関係する。

単に「ViT-Tinyだから」だけではない。

---

# 18. 本提案で解決できない可能性がある問題

Task-centered SIGReg は万能ではない。

特に、

\[
\text{潜在表現全体が非崩壊}
\]

であっても、

- グリッパ開閉
- 接触状態
- 微小な物体位置
- 行動によって変化する状態
- 長期タスク進捗

などの重要な状態成分が十分強く表現される保証はない。

SIGReg は、

\[
\text{「何かが分散している」}
\]

ことを要求するが、

\[
\text{「何が分散するべきか」}
\]

までは指定しないためである。

したがって、本提案の主張は、

> JEPA のすべての潜在表現問題を解決する

ではなく、

> Raw LeWM のマルチタスクにおける周辺分布ガウス化とタスククラスタ構造の衝突を緩和しつつ、SIGReg の表現崩壊防止能力を維持する

とするのが妥当である。

---

# 19. 実験1

## 仮説

\[
\boxed{
\text{Raw LeWM のマルチタスク性能劣化の一因は、
全タスクの潜在周辺分布を単一等方ガウスへ正則化することにある}
}
\]

これを検証する。

---

## 19.1 比較手法

最低限、以下を比較する。

### A. Raw LeWM

\[
\mathcal L
=
\mathcal L_{\mathrm{pred}}
+
\lambda
SIGReg(Z)
\]

### B. Temporal-Centered LeWM

時間局所平均を除去した残差へ SIGReg を適用する既存方向。

\[
r_t
=
z_t-\bar z_t^{\mathrm{temporal}}
\]

\[
\mathcal L
=
\mathcal L_{\mathrm{pred}}
+
\lambda SIGReg(r)
\]

### C. Proposed Task-Centered LeWM

\[
r_i
=
z_i-\mu_{T_i}
\]

\[
\boxed{
\mathcal L
=
\mathcal L_{\mathrm{pred}}
+
\frac{\lambda}{K}
\sum_{k=1}^K
SIGReg(Z_k-\mu_k)
}
\]

---

# 20. バッチ設計

Task-centered SIGReg は、各タスク内で分布統計を計算するため、通常の完全ランダムバッチでは不利になる可能性がある。

例えば50タスク、batch size 128なら、

\[
128/50\approx2.56
\]

サンプル/タスクとなり、SIGReg の統計推定には少なすぎる。

そこで task-balanced batch を用いる。

例:

- 1 batch に8タスク
- 各タスクから16系列
- batch size = 128

とする。

さらに各系列から複数時刻を利用できるなら、SIGReg に使える潜在点数を増やせる。

---

# 21. タスク損失の重み付け

基本実験では、

\[
\alpha_k=\frac1K
\]

として、

\[
\mathcal L_{\mathrm{TaskSIG}}
=
\sum_k
\alpha_k
\mathcal L_k
\]

とする。

これは、データ量が多いタスクだけが損失を支配するのを避けるためである。

ただし将来的には、

- データ数比例
- sqrt データ数比例
- uncertainty weighting
- GradNorm
- gradient conflict-aware weighting

などとの比較も可能である。

---

# 22. 評価項目

単純な成功率だけではなく、潜在表現そのものを評価する。

## 制御性能

- task success rate
- long-horizon success
- unseen configuration generalization
- multi-task average success

## 潜在表現

- タスク間クラスタ中心距離
- タスク内分散
- between-task / within-task ratio
- linear probe による状態復号
- グリッパ状態復号
- 物体位置復号
- 行動変化に対する潜在変化量

## マルチタスク干渉

タスク別勾配を、

\[
g_k
=
\nabla_\theta \mathcal L_k
\]

として、

\[
\cos(g_i,g_j)
=
\frac{
g_i^\top g_j
}{
\|g_i\|\|g_j\|
}
\]

を測定する。

Raw LeWM と比較して負の cosine similarity が減少するなら、SIGReg が引き起こしていたタスク間干渉を減らせた可能性を示せる。

---

# 23. 重要なアブレーション

1. **中心を引かない task-wise SIGReg**
   \[
   \frac1K\sum_k SIGReg(Z_k)
   \]

2. **中心を引く Task-Centered SIGReg**
   \[
   \frac1K\sum_k SIGReg(Z_k-\mu_k)
   \]

3. **Raw SIGReg**
   \[
   SIGReg(\cup_k Z_k)
   \]

4. **Temporal-Centered SIGReg**

5. **Task-Centered + Temporal-Centered**

これにより、

- タスク分割そのもの
- 中心除去
- 時間中心化

のどの要素が性能改善に寄与したか分離できる。

---

# 24. 将来拡張1: Goal-Conditioned Center

離散 task ID を使わず、Goal画像から条件中心を生成する。

\[
z_{\mathrm{goal}}
=
E_\theta(o_{\mathrm{goal}})
\]

\[
\mu
=
h(z_{\mathrm{goal}})
\]

として、

\[
SIGReg(z-\mu(z_{\mathrm{goal}}))
\]

を考える。

これにより、固定された task ID ではなく連続的なGoal条件で潜在空間を構造化できる可能性がある。

---

# 25. 将来拡張2: Language-Conditioned Center

命令文 \(c\) を言語エンコーダへ入力し、

\[
e_c
=
E_{\mathrm{text}}(c)
\]

\[
\mu(c)
=
h(e_c)
\]

として、

\[
SIGReg(z-\mu(c))
\]

を適用する。

この場合、

\[
p(z\mid c)
\]

という条件付き潜在分布として扱える。

ただし外部言語モデルを導入すると、LeWM の「外部事前学習なし」という簡潔さは弱くなるため、実験1とは分離して扱う。

---

# 26. 将来拡張3: 階層的潜在分解

より一般的には、

\[
z_t
=
z^{\mathrm{task}}
+
z_t^{\mathrm{slow}}
+
z_t^{\mathrm{fast}}
\]

と分解することを考える。

- \(z^{\mathrm{task}}\): タスク・Goalなど長期文脈
- \(z_t^{\mathrm{slow}}\): タスク進捗やシーン構造
- \(z_t^{\mathrm{fast}}\): 行動に応じた短期状態変化

SIGReg を高速成分だけに適用する。

\[
SIGReg(z_t^{\mathrm{fast}})
\]

これは短期・中期・長期の階層世界モデルへの拡張につながる。

---

# 27. 現時点での研究仮説

本研究アイデアを一文で表すと、

\[
\boxed{
\text{
LeWM の SIGReg を全潜在周辺分布ではなく
タスク中心化された条件付き残差へ適用することで、
タスク間構造を潰さず各タスク内部の表現崩壊を防止できるのではないか
}
}
\]

である。

数式では、

### Raw LeWM

\[
\boxed{
p(z)
\rightarrow
\mathcal N(0,I)
}
\]

### Proposed Task-Centered LeWM

\[
\boxed{
p(z-\mu_k\mid T=k)
\rightarrow
\mathcal N(0,I)
}
\]

すなわち、

\[
\boxed{
p(z\mid T=k)
\approx
\mathcal N(\mu_k,I)
}
\]

という、タスクごとの局所ガウス構造を許容する。

全体としては、

\[
p(z)
=
\sum_k
p(T=k)
p(z\mid T=k)
\]

となり、単一ガウスではなく、タスク構造を持つ混合分布を許容する。

---

# 28. このアイデアの新規性を検証するときの注意

「task-conditioned Gaussian」や「class-conditional Gaussian」自体は一般的な機械学習の概念であり、それ自体を新規性として主張するべきではない。

新規性候補は、

1. LeWM / SIGReg のマルチタスク問題への適用
2. 全潜在ガウス化ではなく task-centered residual のみを Gaussianize すること
3. LeWM の推論構造・軽量性を変えないこと
4. task-center removal が latent geometry とマルチタスク干渉へ与える効果の解析
5. Temporal-Centered SIGReg との系統的比較

に置くべきである。

---

# 29. 関連研究

## LeWorldModel

Lucas Maes, Quentin Le Lidec, Damien Scieur, Yann LeCun, Randall Balestriero,  
**LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels**, 2026.

- arXiv: https://arxiv.org/abs/2603.19312
- Code: https://github.com/lucas-maes/le-wm

LeWM は、次潜在表現予測と Gaussian latent regularization の2項のみで、画像から端から端まで JEPA 世界モデルを学習する。

## Temporally Centered SIGReg

Chang Liu et al.,  
**Temporally Centered SIGReg Improves Multi-Task LeWorldModel Learning: From Analysis to Method**, 2026.

- arXiv: https://arxiv.org/abs/2607.26924

マルチタスク LeWM において、全潜在周辺分布を単一ガウスへ正則化することによるタスククラスタ圧縮を分析し、時間中心化残差へ SIGReg を適用する。

## SCALE

Jiaming Hu et al.,  
**SCALE: State-Calibrated Latent Embeddings for JEPA Planning in the Right Geometry**, 2026.

- arXiv: https://arxiv.org/abs/2608.16287

LeWM の潜在表現に状態情報が存在するだけでなく、計画に適した潜在幾何を形成することの重要性を議論する。

---

# 30. 次に実施するべき最小実験

最初の実装では複雑な言語条件やGoal条件を導入しない。

**検証したい仮説を1つに絞る。**

> Raw LeWM における単一周辺 Gaussian SIGReg が、マルチタスク潜在構造の悪化要因になっているか。

そのため、

```text
Raw LeWM
   vs
Temporal-Centered LeWM
   vs
Task-Centered LeWM
```

を同一条件で比較する。

Task-Centered LeWM の変更点は原則として、

```python
for task in tasks_in_batch:
    z_k = z[task_id == task]
    mu_k = z_k.mean(dim=0, keepdim=True)
    residual_k = z_k - mu_k
    loss_k = sigreg(residual_k)

loss_task_sig = mean(loss_k for each task)

loss = prediction_loss + lambda_sig * loss_task_sig
```

のみとする。

これにより、ネットワーク容量、ViT、Predictor、CEM、データセットなどを固定したまま、**SIGReg の適用対象だけを変えた因果的比較**が可能になる。

---

## 31. 現時点での要点

> **平均することが研究アイデアではない。**
>
> 重要なのは、全タスクを混ぜた潜在分布へ SIGReg をかけるのをやめ、各タスクの中心を除いたタスク内残差へ独立に SIGReg をかけることである。
>
> これにより、各タスク内では表現崩壊を防止しつつ、タスク間のクラスタ中心距離には SIGReg が直接干渉しない潜在空間を学習できる可能性がある。
>
> また、タスク間差だけを利用して SIGReg の分散条件を満たす「逃げ道」を減らし、各タスク内部でも十分な状態表現を持つことを促せる可能性がある。
>
> 一方、状態変化に必要な全情報が必ず保持されること、共有Encoderによる負の転移、部分観測、長期予測誤差などは別問題として残る。
