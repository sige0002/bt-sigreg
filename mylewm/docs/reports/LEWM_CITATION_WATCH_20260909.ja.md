# LeWM引用研究の取り込み候補とウォッチリスト

調査日：2026-09-09。対象は、LeWorldModel（LeWM）を直接の出発点・比較対象にする公開論文、およびLeWM系のSIGReg・小型潜在ダイナミクスの設計判断を変えうる近接研究である。Semantic ScholarのLeWM（arXiv:2603.19312）citation graphは、この日に最初の100件を返した。以下はそのうち本研究と関係する**26件**を精査対象に選んだもの（LpWM等を含む）。全てpreprintを含むため、ここでの数値は採択済みの事実や再現済みの性能ではない。

## 結論

BT-SIGRegに今もっとも直接効くのは **TC-LeWM** と **LpWM** である。ただし、両者をBTへ直ちに合成しない。

- TC-LeWMは「全潜在を一つのGaussianへ寄せること」がマルチタスクで問題になりうる、というBTの問題設定に最も近い独立仮説である。BTとの主比較に必ず入れる。
- LpWMはGaussianを保ったまま座標を変えるBTとは別に、**参照分布そのものを疎な非負分布へ変える**。BTが勝つべき強い幾何学的対照であり、次の研究段階での最優先ベースラインである。
- Fast-LeWMは主に計画器の計算とrollout誤差を扱う。多段／prefix予測を研究の中心にしないという本研究の制約と衝突するため、BTの学習目的には今は取り込まない。ただしBTがLIBEROで精度を示せた後の「同じ表現で計画を速くする」独立のsystems拡張として監視する。

従って、次の実験順は **Raw / TC / BT を同条件で完了 → LpWM（またはRDMReg）を別の表現幾何として追加 → Fast-LeWMを必要なら計画側だけで別比較** が妥当である。これにより「Gaussianをかける座標の問題」と「Gaussian以外の幾何が良い問題」と「rollout計算の問題」を混同しない。

## 優先度A：主比較へ取り込む

### TC-LeWM — 時間中心化SIGReg

[TC-LeWM](https://arxiv.org/abs/2607.26924) は、局所時間平均を引いた残差へSIGRegをかけ、潜在全体のcluster中心間隔を直接Gaussian化しない。著者らはLIBEROで、Raw LeWMより良い凍結表現＋BCの成績を報告している。これはCEM成功率ではないため、本リポジトリの計画成績と数字を直接並べてはならない。

BTとの関係は明確である。

| 観点 | TC-LeWM | BT-SIGReg |
|---|---|---|
| Gaussianをかける対象 | 時間残差 `r=z-mean_time(z)` | 共有可逆写像後 `u=T(z)` |
| cluster中心への直接圧力 | 残差化により弱める | Tに移すがGaussian目標は残す |
| 静的・文脈情報 | 残差から消えうる | zに残し、Tは可逆 |
| 新しい推論計算 | なし | なし（Tを除外） |

**取り込み方：** TCをBT内部へ足さず、Raw・TC・BTの三方式を同じ初期E/A/F、データ順、更新数、CEM、開始状態で比較する。LIBERO-10でmacro平均・各タスク・最悪タスクを必ず出す。TCがBT以上なら、BTの「座標分離」が時間中心化以上の利点を実証できなかったことになる。

**警戒点：** 残差SIGRegは軌道内一定成分をほぼ見ない。これは本研究の「非操作物体保持」を目的化しない方針と矛盾しないが、Goal距離や操作に必要な文脈が失われないことは制御で確認する必要がある。

## 優先度A：次段階の別幾何ベースライン

### LpWM — 疎な非負表現

[LpWM](https://arxiv.org/abs/2608.22764) は、LeWMの密な等方Gaussian表現を、整流化一般化Gaussianに合わせるRDMRegで置換し、非負・疎な表現を学ぶ。著者らは、PushTで中程度の予測器容量において密なLeWMより最大57%高い計画成功率を報告し、supportが離散的なdynamical regime、値がその内部の連続状態を表す傾向を主張する。[公式実装](https://github.com/YilunKuang/lpworldmodel) はRDMRegとGaussian対照を含む。

これは「Gaussian制約を緩める」だけではなく、**表現の幾何を密な符号付き球状表現から疎な錐へ替える**案である。BTの可逆Tでは、非負性・真の疎性を一般には作れない。従ってBTとLpWMを合成して成功しても、どちらの要因かは分からない。

**取り込み方：** まずBTと混ぜず、同じE/A/F・一段予測・CEMで、Raw Gaussian SIGReg、TC、BT、RDMReg/LpWMの4条件にする。LpWM原論文は予測器容量との相互作用を主張しているため、最低でも本研究の小型予測器容量と一段上の容量を同じ探索予算で対照にする。推論パラメータ数、CEM時間、訓練時間も報告する。

**研究上の分岐：**

- BTがLpWMを上回るなら、「Gaussianを捨てず、予測座標と正則化座標を分ける」ことに価値がある。
- LpWMが上回るなら、問題の中心は座標の配置でなくGaussian目標そのものか、疎性が小型予測器を助けることにある。
- 両者がTCに負けるなら、複雑な分布幾何より時間方向の分散配分が主要因かもしれない。

**採用しない要素：** LpWMの「疎なfeatureが物理因子を発見する」という解釈を、本研究の成功条件にしない。supportの可視化や非操作物体probeは補助診断であり、主判定は共有LIBERO-10の制御成績である。

> 注：LPWMという略語には別の [Latent Particle World Models](https://openreview.net/forum?id=lTaPtGiUUc) もある。こちらはkeypoint・mask・動画生成・確率的particle dynamicsを使う別系列で、LpWorldModelとは別物である。後者は物体中心・生成・より大きな構成を導入するため、現在のLeWM程度の小型共有JEPAという目的には取り込まない。ウォッチ対象ではあるが優先度は低い。

## 優先度B：強いGaussian対照として監視

### Sub-JEPA — 固定ランダム部分空間のGaussian化

[Sub-JEPA](https://arxiv.org/abs/2605.09241) は、全空間のSIGRegを複数の固定直交ランダム部分空間でのSIGRegに替える。論文と[公式実装](https://github.com/intcomp/Sub-JEPA)は、LeWMの4制御環境での改善を報告している。

BTへの直結は弱い。BTは可逆Tを通して全次元のGaussian目標を保つ一方、Sub-JEPAは拘束する同時依存構造を弱める。したがってこれは**Gaussian適用座標を変えるBT**と、**Gaussian拘束の強さを変えるSub-JEPA**を区別する有効な対照である。

優先度をTC/LpWMより一段下げる理由は、Sub-JEPAがマルチタスクLIBERO-10を主題にしている証拠を今回確認できず、固定ランダム射影を増やすだけでは本研究の分散配分仮説を直接検証しないためである。Raw/TC/BT/LpWMの結果が出た後、同じ射影計算量を記録した第5条件として加える。

### AC-MTM — 分布自由のaction-contrastive anti-collapse

[No Gaussian Required / AC-MTM](https://arxiv.org/abs/2608.17542) は、前向き潜在予測を保ちつつ、学習時だけの逆ダイナミクスheadにAction-NCEを入れ、定数表現が行動を識別できないことを非崩壊圧力にする。著者らは、難しいmulti-object環境でSIGRegより高い成績を報告している。

これはBTの仮説に対する最も強い反証候補である。もし分布形状を一切指定しないanti-collapseが小型マルチタスクで勝つなら、「Gaussianをどの座標に適用するか」を改善するより、Gaussianそのものを必要としない可能性がある。

ただし最初から取り込まない。Action-NCEはbatch内の負例・行動の重複・観測可能性に新たな仮定を持ち込み、BTと損失が異なる。BTの同条件比較後に、学習時のみのhead・同じ前向きE/A/F・同じCEMという独立ベースラインにする。行動空間が連続で類似行動の多いLIBEROではfalse negativeの監査が必須である。

## 優先度C：systemsのウォッチ対象

### Fast-LeWM — action-prefix予測

[Fast-LeWM](https://arxiv.org/abs/2606.26217) は、候補行動列のprefixを符号化し、各prefix到達潜在を並列予測する。逐次one-step rolloutを置換し、著者らはLeWMより平均成功率とplanning速度を改善したと報告する。[公式実装](https://github.com/Yuntian-Gao/Fast-LeWorldModel)にはLeWMと同じデータ配置・評価入口がある。

Fast-LeWMは価値があるが、BT-SIGRegの主問題とは別である。

- 解くもの：CEMの逐次rollout時間、長いhorizonでの誤差蓄積。
- 解かないもの：混合タスク潜在に全体Gaussianをかける圧力、あるいはBTのbi-Lipschitzな尺度逃避。
- 本研究との衝突：prefix全horizonへの教師あり損失は、現在採用しない多段予測損失に当たる。

したがって今は**実装しない**。将来、Raw/TC/BT/LpWMの表現比較でBTがLIBERO-10の精度を示した後、固定した学習済み表現にFast-LeWM型のplanner/predictorを別途つなぐ研究として検討する。そのときは「BTで性能が上がった」と「prefix planningで速くなった」を別の表・別の主張にする。報告する指標もdynamics-module時間、CEM全体時間、成功率、horizon別の誤差を分ける。

### 階層計画と物理補助head

[Hi-LeWM](https://arxiv.org/abs/2607.12547) は長horizon向けの階層計画を扱うが、論文自身が高レベル探索と低レベル制御の分布ずれを重要な失敗要因として示す。BTの表現仮説が未検証の段階で階層を足すと、性能の原因を追えなくなるため保留する。

[Spectral-Target Physical Latent Structuring](https://arxiv.org/abs/2609.04264) は学習時だけのFourier補助headで「物理表現の怠惰」を抑える案である。推論コストを増やさない点は魅力的だが、補助目標がどの観測由来か、PushT/LIBERO-10の通常軌道だけで成立するかを精査するまで採用しない。外部の物理ラベルやタスク仕様を必要とするなら、本研究の条件外である。

## 再現性を先に固定する文献

[Independent Reproduction of LeWM on TwoRoom](https://arxiv.org/abs/2608.10145) は新アルゴリズムではないが、LeWM比較で最優先で監視すべきである。画像正規化、frameskip内の行動取得、行動z-score、Goalの作り方、評価budgetが成功率を大きく変えうると報告している。

本リポジトリでは、この警告を次の比較契約として扱う。

1. Raw/TC/BT/LpWMで、manifest、episode split、画像前処理、行動正規化、初期E/A/F、データ順、更新数、optimizer、CEM、Goal、開始状態、成功関数を固定する。
2. 予測lossやSIGReg値で方式を順位付けせず、対応する環境初期状態での成功差と信頼区間を主に報告する。
3. 公式配布checkpointとの比較は「配布物の再現」とし、同予算で新規初期化したRaw/TC/BT/LpWMの比較とは分離する。
4. 学習時BN統計、eval mode、checkpointの行動正規化bufferと、評価が使う物理行動範囲を保存・照合する。

## 推奨する監視順と採否表

| 優先 | 研究 | 今回の位置付け | 次に確認すべき事実 |
|---:|---|---|---|
| 1 | TC-LeWM | BTの直接競合ベースライン | LIBERO-10をCEMで同条件比較できるか |
| 2 | LpWM | Gaussian目標そのものを替える強い対照 | RDMRegを同じ小型E/A/Fへ移したときの容量・計算量・CEM成績 |
| 3 | Sub-JEPA | Gaussian拘束強度の対照 | 多タスク条件・コード・射影予算の再現性 |
| 4 | AC-MTM | Gaussian不要という反証候補 | 連続行動での負例定義、LIBEROでのfalse-negative影響 |
| 5 | Fast-LeWM | 表現ではなくplanning systems拡張 | BT表現を固定したときの独立な速度/精度トレードオフ |
| 6 | Hi-LeWM / spectral auxiliary | 長horizon・物理補助の探索候補 | 本研究の通常軌道・小型共有モデル条件を満たすか |

この表は学習実行の指示ではない。現時点で実施済みなのはBTのPushT単一seed実験だけであり、Raw/TC/BTの同予算比較、LIBERO-10本学習、LpWMやSub-JEPA/AC-MTMの再現は未実施である。

## LeWMから何を変えるか：引用研究26件の競合地図

ここでいう「競合」は、同じ小型・報酬なし・画像行動軌道からの世界モデルという目的を、BTとは異なる要因で改善しようとすることを意味する。各論文が実際にBTを比較したことを意味しない。`採用候補`はBTへ即座に足す意味ではなく、同条件の独立ベースラインまたは次の仮説として優先する、という意味である。

### A. SIGReg／表現幾何を直接変える研究

| # | 論文 | LeWMからの変更と主張する寄与 | BT-SIGRegとの競合・差異 | 本研究での扱い |
|---:|---|---|---|---|
| 1 | [TC-LeWM](https://arxiv.org/abs/2607.26924) | 全潜在でなく時間中心化残差へSIGRegを適用し、persistent成分と残差の分散配分を分ける。 | BTは`z`を保存し可逆`T(z)`にGaussianを移す。TCは時間方向に対象を選び、Tを持たない。最も直接のマルチタスク競合。 | **必須主比較**。Raw/TC/BTを同条件にする。 |
| 2 | [LpWM](https://arxiv.org/abs/2608.22764) | RDMRegで整流化一般化Gaussianへ合わせ、非負・疎な表現を作る。小型predictorの必要複雑性を下げることを主張。 | BTはGaussianを残して座標を分離、LpWMは参照分布・符号・疎性を替える。Gaussian仮定自体への強い競合。 | **最優先の次段階ベースライン**。BTと合成しない。 |
| 3 | [Sub-JEPA](https://arxiv.org/abs/2605.09241) | 複数の固定ランダム部分空間でGaussian正則化し、全空間の強い拘束を緩める。 | BTは全次元Gaussianを可逆座標で保つ。Sub-JEPAは拘束されない同時依存を増やす。 | 主要比較後の第5条件。 |
| 4 | [QQWorld](https://arxiv.org/abs/2607.28415) | Epps–Pulleyのtail勾配減衰を問題化し、射影分位点とGaussian quantileのQQ一致へ置換する。 | BTの問題はGaussianをかける**座標**、QQWorldはGaussian一致の**数値的推定器**。両者は直交するが、BTの改善をEPの弱さと誤認しないため必要。 | BTのEP-SIGRegが勝った後の数値対照。 |
| 5 | [VIScore / VISReg比較](https://arxiv.org/abs/2608.11174) | 同じGaussian targetでもcenter/scale/shapeの重み付けとbatchが計画へ異なる影響を持つとし、plannerまで含む診断を提案。 | BTのTはscale逃避を境界で制限するが、SIGReg実装のshape/scale寄与は別問題。 | z/u診断とCEM評価へ取り込む。損失置換は後。 |
| 6 | [SCALE](https://arxiv.org/abs/2608.16287) | plannerが使う距離幾何をstate-calibrateする軽量正則化を提案。 | BTは分布幾何と予測しやすさを分離するが、Goal距離が進捗を順位付ける保証はない。SCALEはその隙を突く競合。 | Goal-costの対照・ウォッチ。 |
| 7 | [No Gaussian Required / AC-MTM](https://arxiv.org/abs/2608.17542) | Action-NCEの訓練時逆ダイナミクスheadをanti-collapse圧力にしてGaussian目標を外す。 | BTの前提「Gaussianの利点を残す」への最強の反証。推論構成は同じだが損失は別。 | Raw/TC/BT/LpWM後の独立基準。 |
| 8 | [Delta-JEPA](https://arxiv.org/abs/2606.31232) | endpointでなく潜在差分から行動を復元し、action-sensitiveな遷移幾何を作る。 | BTは分布制約を変えるだけでaction sensitivityを直接保証しない。 | 行動シャッフル診断が悪い場合の有力対照。 |
| 9 | [Physically Grounded JEPA](https://arxiv.org/abs/2609.03565) | inverse dynamicsとstate alignmentを追加し、transitionの実効次元と物理整合を高める。 | task/物理state alignmentの定義が必要なら、通常画像・行動のみというBTの制約を越える可能性がある。 | ラベル不要な部分だけを監視。現段階では採用しない。 |
| 10 | [Spectral-Target Physical Latent Structuring](https://arxiv.org/abs/2609.04264) | 訓練時Fourier補助headでnon-collapseでも物理を表さない「laziness」を抑える。 | BTは可逆TゆえにEが捨てた物理情報を回復できない。同じ弱点を別の補助信号で扱う。 | 補助信号が通常軌道だけから作れる場合にのみ対照。 |
| 11 | [JEPA-x](https://arxiv.org/abs/2608.24044) | physical trajectoryを特権viewとしてcross-predictし、共通遷移を学ぶ。 | 物理stateという特権教師を使うので、主研究の条件外。性能比較に混ぜると不公平。 | **目的外**。上限参考のみ。 |
| 12 | [UniJEPA](https://arxiv.org/abs/2608.07409) | photometric予測とtemporal予測を一つのlatentで共同学習する。 | BTは通常の行動軌道の一段動力学に集中する。別タスクを増やすため原因帰属ができない。 | 事前学習なし条件を保つ限り採用しない。 |

### B. planner cost・検索を替える研究

| # | 論文 | LeWMからの変更と主張する寄与 | BT-SIGRegとの競合・差異 | 本研究での扱い |
|---:|---|---|---|---|
| 13 | [Fast-LeWM](https://arxiv.org/abs/2606.26217) | one-step反復rolloutをaction-prefixの並列到達潜在予測で置換し、速度と長horizon誤差を改善。 | 表現正則化でなくdynamics query単位の変更。prefix全horizon教師は本研究で採用しない多段予測。 | **systemsのみ監視**。主学習へ入れない。 |
| 14 | [LEAP](https://arxiv.org/abs/2609.03294) | frozen LeWMに対してCEMでなく微分可能な行動最適化＋terminal state energyを使う。 | 同じworld modelでもplannerだけで大きく成功率が変わる、という警告。BTの優位をplanner変更で作ってはならない。 | 固定CEM主比較を守る。後にplanner-only比較。 |
| 15 | [Hi-LeWM](https://arxiv.org/abs/2607.12547) | frozen low-level LeWMに高level latent subgoalと階層searchを追加。 | 長horizonの別問題。macro-actionのtrain/inference分布ずれを論文自身が報告。 | BTの表現比較後まで保留。 |
| 16 | [Traj-LeWM](https://arxiv.org/abs/2608.14125) | endpoint距離だけでなくtrajectory-level costと訓練時preferenceを加える。 | 予測器・Goal costの両方を変えるためBTと同時に加えると因果が崩れる。 | endpoint CEMを固定し、後段のplanner競合。 |
| 17 | [LeFlow](https://arxiv.org/abs/2608.24855) | latent trajectory priorとinverse dynamicsで反復action最適化をamortizeする。 | 追加の生成planner/逆モデルを持ち、小型E/A/Fの純粋な比較を越える。 | systems上限として監視のみ。 |
| 18 | [ACID](https://arxiv.org/abs/2607.02403) | inverse dynamicsで予測遷移のaction cycle-consistencyを測り、planning costに入れる。 | BTは学習表現、ACIDはdecision-timeの実現可能性cost。相補的だが別因子。 | CEMの候補が非現実的なときの後段対照。 |
| 19 | [Objective Is the Bottleneck](https://arxiv.org/abs/2608.12959) | latent距離が遠距離の真の進捗を順位付けないことを示し、objectiveだけの交換を検証。 | BTのTが情報を保っても、`||z_goal-z||²`がcontrol metricである保証はない。 | 必須のmetric audit。成功率を潜在MSEと混同しない。 |
| 20 | [Decision-Metric Alignment](https://arxiv.org/abs/2608.18746) | random/CEM candidateのlatent costと実costの順位一致を測り、action-conditioned補助目標を提案。 | BTのGoal距離仮定を直接検査する。補助目標は本研究の二項損失を変えるため別方式。 | Plan-Real rank監査を取り込む。DA-LeWMは後段対照。 |
| 21 | [Monotone Planning Costs](https://arxiv.org/abs/2608.09073) | image-goal navigationで単調なplanning costを持つworld modelを扱う。 | Goal metricを構造的に制限する別研究で、BTの分布仮説とは別。 | 長距離でcost非単調性が出た場合に読む。 |
| 22 | [ProWorld](https://arxiv.org/abs/2608.01926) | long-horizon visual goal reachingへprogress-aware hyperbolic geometryを導入。 | 双曲幾何とprogressはBTのEuclidean z＋可逆Gaussian headとは異なる強いmetric仮説。 | 目的外寄りのplanner/geometry対照として監視。 |

### C. 何を測るべきか、何が学習されるかを問う研究

| # | 論文 | LeWMからの変更または寄与 | BT-SIGRegとの競合・差異 | 本研究での扱い |
|---:|---|---|---|---|
| 23 | [LeWM独立再現](https://arxiv.org/abs/2608.10145) | 正規化、action chunk、Goal offset、BNなどの設定差で成功率が大きく変わることを示す。 | アルゴリズム競合ではないが、BTの公正比較を成立させる前提。 | **必須の実験契約**。 |
| 24 | [Intervention Gap](https://arxiv.org/abs/2608.29998) | latentが現在状態を捉えても、imagined intervention効果が環境と合わない失敗を直接監査する。 | BTはnon-collapse/可逆性だけで介入忠実性を保証しない。 | PushT/LIBEROでaction intervention auditを追加。 |
| 25 | [ACPC](https://arxiv.org/abs/2608.12939) | 視覚摂動下のaction-conditioned rollout一貫性を測り、invarianceとstate separationを同時診断。 | Gaussian/BTの分散指標だけでは見えない頑健性・aliasingを測る。 | 画像摂動の補助診断として採用候補。 |
| 26 | [VIScore](https://arxiv.org/abs/2608.11174) | encoder、predictor、plannerのreachability/capacity/hallucinationを合わせた成功率診断を提案。 | BTのz/u統計が良くてもplanningが良いとは限らないという直接の注意。 | 既存のloss・probeに加える評価候補。 |

## BT-SIGRegが実際に寄与しうる場所と、競合に負ける条件

BTの固有の主張は限定する必要がある。LeWMの`z`をそのままGaussianにする代わりに、同じ共有・有界可逆`T`後の`u`だけをGaussianへ近づけ、予測・rollout・Goal costは`z`に残す。これは次の一点にだけ寄与しうる。

> 小型の共有E/A/Fが複数タスクの遷移を表すのに良い座標と、SIGRegが要求するGaussian座標を分けることで、Gaussian anti-collapseを捨てずに分散配分の衝突を弱められるか。

そのため、BTは以下を主張してはいけない。

- TCより一般にマルチタスクに強い（TCとの同条件LIBERO比較がまだない）。
- LpWMより表現幾何が良い（Gaussian targetを替えた比較がまだない）。
- action sensitivity、physical grounding、intervention fidelity、Goal-distance alignmentを自動的に得る（Delta-JEPA、AC-MTM、Intervention Gap、Decision-Metric Alignmentが別の要因を示す）。
- Fast-LeWM/LEAP/Hi-LeWMより速い・長horizonで強い（これらは主にplanner/dynamics queryを変える）。

反対に、BTが研究として残る最低条件は、同一のE/A/F・データ・計画器・予算下でRawおよびTCを上回るか同等に保ち、少なくともLIBERO-10で平均改善が一部タスクの大きな悪化と引換えでないことを示すことである。次にLpWMへ勝てるか、または「Gaussianを残すBT」と「疎なLpWM」が異なるタスク群で勝つ理由を示せれば、座標変換研究として意味が立つ。

## 参考文献・確認範囲

- [LeWorldModel](https://arxiv.org/abs/2603.19312) — 原方式の目的、二項損失、約15M規模の基準。
- [TC-LeWM](https://arxiv.org/abs/2607.26924) — 時間中心化SIGReg、LIBERO表現＋BCの報告。CEMと同一ではない。
- [LpWM](https://arxiv.org/abs/2608.22764)、[公式コード](https://github.com/YilunKuang/lpworldmodel) — RDMReg、疎な非負表現、PushTの容量依存結果。
- [Sub-JEPA](https://arxiv.org/abs/2605.09241)、[公式コード](https://github.com/intcomp/Sub-JEPA) — 固定ランダム部分空間へのGaussian正則化。
- [Fast-LeWM](https://arxiv.org/abs/2606.26217)、[公式コード](https://github.com/Yuntian-Gao/Fast-LeWorldModel) — action-prefix予測とplanning高速化。
- [AC-MTM](https://arxiv.org/abs/2608.17542) — 分布自由のanti-collapse対照。
- [LeWM独立再現](https://arxiv.org/abs/2608.10145) — 比較プロトコルの感度。
- [Latent Particle World Models](https://openreview.net/forum?id=lTaPtGiUUc) — LpWMと同名に近いが別系列であることの確認用。

本レポートは2026-09-09に取得した公開abstract・公式リポジトリ・公開ページに基づく。各論文の全実験設定・付録・コードの再実行は未実施であり、著者報告の数値を本研究で再現済みとは扱わない。
