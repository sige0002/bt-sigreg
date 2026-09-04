# Goal-Weighted Transition Geometry for Multi-Task LeWorldModel

## 研究アイデアメモ

**ステータス:** 仮説設計段階  
**目的:** LeWorldModel (LeWM) の軽量・高速な潜在世界モデルという利点を維持しながら、マルチタスク環境で「タスクに必要な状態変化」を潜在表現に残し、Goal に応じて正しく計画できるようにする。

> リポジトリ名 `task-centered-sigreg-lewm` は初期案に由来する。現在の本命案は、Task ごとの混合ガウスを作る方式ではなく、**共有世界表現 + Goal-weighted transition geometry + 非崩壊制約**である。

---

# 1. 問題設定

LeWM は画像を潜在表現へ変換し、行動を条件として未来の潜在状態を予測する。

\[
z_t = E_\theta(o_t)
\]

\[
\hat z_{t+1} = F_\psi(z_{\le t}, a_{\le t})
\]

- \(o_t\): 現在画像
- \(a_t\): 行動
- \(z_t\): 潜在状態
- \(E_\theta\): Encoder
- \(F_\psi\): LeWM predictor

元の LeWM は SIGReg により潜在表現を等方ガウスへ近づけ、JEPA の表現崩壊を防ぐ。

\[
p(z) \approx \mathcal N(0,I)
\]

しかしマルチタスクでは、潜在表現全体を単一の等方ガウスへ押し込むことが、

- タスクごとの多峰構造
- 状態進捗
- 接触・把持などの局所状態変化
- 低次元多様体構造

と衝突する可能性がある。

重要なのは、**世界の真の潜在分布がガウスである必要はない**という点である。

---

# 2. 初期案からの変更

初期案では、Task \(k\) ごとに

\[
p(z\mid T=k) \approx \mathcal N(\mu_k,I)
\]

とし、Task 内部だけに SIGReg を適用することを考えた。

しかしこの方式には、

1. 1つの Task 内部にも approach / grasp / carry / place など多峰構造がありうる
2. Task ごとに単一ガウスを仮定する必要性がない
3. 同じ物理状態を Task が違うだけで分離するのは不自然な場合がある

という問題がある。

したがって、本命案では **Task ごとの分布形を人間が決めない**。

---

# 3. 研究の中心仮説

本研究では次を狙う。

> **世界状態と世界の状態変化は全 Task で共有する。**  
> **Goal によって「どの潜在状態・状態変化が重要か」だけを変える。**  
> **その Goal に必要な状態変化が潰れないよう、分布形を指定しない非崩壊制約を加える。**

つまり、

```text
世界の表現          -> 全 Task で共有
世界の状態変化      -> 全 Task で共有
何を見るか          -> Goal ごとに変える
重要な状態変化      -> 潰れないよう制約する
```

である。

---

# 4. 世界モデル本体は変更しない

LeWM predictor 自体には Task ID や Goal を入れない。

\[
\hat z_{t+1}=F_\psi(z_t,a_t)
\]

理由は、同じ世界状態 \(z_t\) と同じ行動 \(a_t\) なら、物理的な次状態は Task に依存しないはずだからである。

```text
current image
     |
     v
  Encoder
     |
     v
    z_t ---- action
     |          |
     +----------+
          |
          v
   Shared LeWM
          |
          v
   predicted z
```

---

# 5. Goal によって「見る潜在方向」を変える

Goal 画像を同じ Encoder へ入れる。

\[
z_g = E_\theta(o_{goal})
\]

さらに小さなネットワーク \(W_\eta\) から Goal-dependent weight を生成する。

\[
w_g = W_\eta(z_g)
\]

ここで、

\[
w_g \in [0,1]^D
\]

とする。

\(D=192\) なら、192次元それぞれについて「この Goal ではどの程度重要か」を表す重みになる。

例:

```text
Task A: 赤いキューブを右へ

robot       0.8
red cube    1.0
blue cube   0.1
gripper     0.8
background  0.0

Task B: 青いキューブを左へ

robot       0.8
red cube    0.1
blue cube   1.0
gripper     0.8
background  0.0
```

実際には各潜在次元が人間に解釈可能である必要はない。

---

# 6. Goal-weighted latent distance

元の LeWM は Goal との距離を、

\[
d(z,z_g)=\|z-z_g\|^2
\]

で評価する。

本提案では、

\[
\boxed{
d_g(z,z_g)
=
\|w_g\odot(z-z_g)\|^2
}
\]

とする。

つまり、Goal に関係する潜在方向だけを強く見る。

```text
shared latent z
      |
      +-------------------+
      |                   |
   Goal A              Goal B
      |                   |
      v                   v
    w_A                  w_B
      |                   |
red-related dims     blue-related dims
   important            important
```

---

# 7. Expert trajectory から Goal の「進捗」を学ぶ

成功軌道

\[
z_0,z_1,z_2,\ldots,z_g
\]

では、時間が進むにつれて Goal に近づくはずである。

したがって、

\[
d_g(z_{t+1},z_g) < d_g(z_t,z_g)
\]

を要求する。

margin ranking loss の例:

\[
\boxed{
\mathcal L_{progress}
=
[m+d_g(z_{t+1},z_g)-d_g(z_t,z_g)]_+
}
\]

これにより、\(w_g\) は「Goal を達成するうえで重要な潜在方向」を学習する。

---

# 8. 非崩壊制約

## 8.1 なぜ必要か

Goal weight が

\[
w_g=0
\]

になれば、

\[
d_g(z,z_g)=0
\]

となり、どの状態も Goal と同じになってしまう。

また、一部の潜在方向だけに依存して Goal に必要な状態変化を失う可能性もある。

そこで、**Goal によって重要だと判断された状態変化が潰れない**ことだけを要求する。

---

## 8.2 状態変化を見る

\[
\Delta z_t=z_{t+1}-z_t
\]

Goal-weighted transition を、

\[
\boxed{
\Delta u_t=w_g\odot\Delta z_t
}
\]

とする。

ここで重要なのは、潜在状態 \(z\) の分布ではなく、**行動によって生じた状態変化 \(\Delta z\)** を見ることである。

---

## 8.3 最小版: 分散非崩壊

バッチ内の Goal-weighted transition を集め、各潜在次元の標準偏差を

\[
\sigma_j = Std(\Delta u_{:,j})
\]

とする。

完全に潰れた方向では、

\[
\sigma_j\approx0
\]

となる。

最小実装では、例えば

\[
\boxed{
\mathcal L_{var}
=
\frac1D\sum_j[\gamma-\sigma_j]_+^2
}
\]

を用いる。

意味は単純である。

```text
Goal に必要な状態変化が全部同じ / 0
        -> variance ≈ 0
        -> penalty

Goal に必要な状態変化が残っている
        -> variance > threshold
        -> no penalty
```

この制約は、潜在分布をガウスにすることを要求しない。

- Gaussian: OK
- multimodal: OK
- skewed: OK
- curved manifold: OK
- collapse: NG

---

# 9. 有効ランクは最初は使わない

より強い制約として、Goal-weighted transition covariance の有効ランクを使うこともできる。

しかし、

\[
r_{eff}\ge q
\]

とすると、最低ランク \(q\) が新しいハイパーパラメータになる。

そのため最初の実験では使わない。

研究の順序は、

```text
Step 1: 分散非崩壊のみ

もし
「完全崩壊はしないが 1〜2 次元しか使わない」
という問題が実際に観測されたら

Step 2: 有効ランク制約を追加
```

とする。

---

# 10. 最終損失

最小版では、

\[
\boxed{
\mathcal L
=
\mathcal L_{pred}
+
\lambda_p\mathcal L_{progress}
+
\lambda_v\mathcal L_{var}
}
\]

とする。

### 1. LeWM prediction loss

\[
\mathcal L_{pred}
=
\|\hat z_{t+1}-z_{t+1}\|^2
\]

### 2. Goal progress loss

\[
\mathcal L_{progress}
\]

Goal に近づくほど潜在距離が小さくなるようにする。

### 3. Goal-weighted transition variance loss

\[
\mathcal L_{var}
\]

Goal に必要な状態変化が潜在表現から消えるのを防ぐ。

---

# 11. 学習フロー

```text
Multi-task trajectory
        |
        v
     Encoder
        |
        +-----------------------+
        |                       |
       z_t                   z_{t+1}
        |                       |
        +-------- Δz -----------+
        |
        | + action
        v
 Shared LeWM predictor
        |
        v
 prediction loss

Goal image
    |
    v
 Encoder
    |
    v
  z_goal
    |
    v
 W(z_goal)
    |
    v
   w_goal
    |
    +--------------------------+
    |                          |
    v                          v
Goal-weighted              Goal-weighted
latent distance            transition
    |                          |
    v                          v
progress loss             variance loss

             all losses
                 |
                 v
             backward
```

---

# 12. 赤・青キューブ例

## Task A

赤いキューブを右へ置く。

Goal weight は主に、

- 赤キューブ状態
- ロボット状態
- グリッパ状態

に関係する潜在方向を重視する。

## Task B

青いキューブを左へ置く。

Goal weight は主に、

- 青キューブ状態
- ロボット状態
- グリッパ状態

に関係する潜在方向を重視する。

しかし世界モデル \(F(z,a)\) は両 Task で同じである。

```text
                 shared world latent
                         |
            +------------+------------+
            |                         |
          Task A                    Task B
      red -> right              blue -> left
            |                         |
           w_A                       w_B
            |                         |
       red-related              blue-related
       transitions              transitions
       stay visible             stay visible
```

---

# 13. 研究のアイデンティティ

本研究の主張は、

\[
\boxed{
\text{LeWM の潜在分布を正則化するのではなく、}
\text{Goal に必要な潜在状態変化を正則化する}
}
\]

ことである。

Raw LeWM の問いは、

> 潜在表現 \(z\) はどの分布であるべきか？

だった。

本提案の問いは、

> この Goal を達成するために、どの状態変化を潜在表現に残すべきか？

である。

したがって、潜在分布そのものに Gaussian / mixture Gaussian といった強い仮定を置かない。

---

# 14. 既存研究との位置付け

## LeWorldModel

- 潜在未来予測 + SIGReg
- 等方ガウスによる非崩壊

https://arxiv.org/abs/2603.19312

## TC-LeWM

- 時間的に持続する成分を除いた残差に SIGReg
- マルチタスク LeWM の直接的 baseline

https://arxiv.org/abs/2607.26924

## UR-JEPA

- 等方ガウスではなく局所的な低次元多様体構造を許す

https://arxiv.org/abs/2606.01443

## No Gaussian Required / Contrastive Inverse Dynamics

- 潜在状態変化から action を識別できるようにし、分布仮定なしで崩壊を防ぐ

https://arxiv.org/abs/2608.17542

本提案は、単に action を復号するのではなく、**Goal に応じて重要な状態変化方向を選び、その方向の非崩壊と Goal progress を同時に学習する**ことを狙う。

---

# 15. 計算コスト

## 学習時

追加する主な処理は、

1. Goal weight network \(W_\eta\)
2. Goal-weighted difference \(w_g\odot\Delta z\)
3. batch 内標準偏差
4. progress ranking loss

である。

これらは ViT Encoder や LeWM Predictor に比べて軽量であると予想されるが、実際の wall-clock overhead は実測する。

## 推論時

非崩壊 loss は不要。

追加されるのは Goal weight の生成と、

\[
\|w_g\odot(\hat z-z_g)\|^2
\]

の計算だけである。

そのため、LeWM の潜在空間での高速 planning という利点はほぼ維持できると期待する。

---

# 16. 最小実験

最初は以下を比較する。

1. Raw LeWM
2. TC-LeWM
3. No Gaussian Required / Action-NCE 系
4. Proposed: Goal-weighted transition geometry

提案法では、まず有効ランクを使わず、

\[
\mathcal L_{pred}
+
\lambda_p\mathcal L_{progress}
+
\lambda_v\mathcal L_{var}
\]

だけで評価する。

---

# 17. 評価項目

## 制御性能

- multi-task success rate
- long-horizon success
- unseen object configuration
- unseen Goal

## 潜在表現

- Goal progress と潜在距離の順位相関
- object position linear probe
- gripper state probe
- robot state probe
- Goal-weighted transition variance

## 計算量

- training step time
- GPU memory
- planning latency
- CEM throughput

---

# 18. 今後の拡張

最小版で状態変化が少数方向へ潰れることが確認された場合のみ、

- effective rank regularization
- local manifold regularization
- counterfactual action perturbation
- shared / private transition subspace

を追加検討する。

最初から複雑な制約を積み上げず、**本当に必要な制約だけを実験で追加する**。

---

# 19. 現時点の一文要約

> **マルチタスク LeWM において、世界モデル自体は全 Task で共有したまま、Goal ごとに重要な潜在方向を学習し、その Goal に必要な状態変化が潜在空間で潰れないよう正則化する。潜在分布そのものにはガウス性を要求しない。**
