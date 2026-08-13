# Packing 决策逻辑 V2 简要说明

同步下 packing 决策逻辑 v2 的想法,按几个问题来梳理:

**背景问题**
因为 neat packing 限制了 per_device_batch_size = 1,此时 cutoff_len 会直接影响吞吐量(每步每卡的 token 数就等于 cutoff_len)。按之前的方案,cutoff_len 设成"数据集分析得到的最长样本长度",可能偏小,无法发挥 GPU 的最大性能。

**那能不能把 cutoff_len 尽量加长?**
可以,但此时还不知道 GPU 型号,没法直接定最大值。所以做法是:在候选区间内取若干个 cutoff_len 值,再逐一筛选。
区间下界是数据集最长样本长度;上界不宜直接取 max_position_embeddings——主要不是因为吞吐(neat packing 用块对角 attention,加长基本只会让吞吐上升到饱和、不会掉),而是**加长会先撞上显存、GBS、step 数这几个限制**。所以上界实际取「max_position_embeddings、显存可行上限、GBS 可控上限」三者的最小值,显存可行上限由规划器预测给出。

**global batch size 怎么保证?会不会影响效果?**
packing 之后,每个 pack 装的样本数是变的,所以样本级的 GBS 天然会有波动——部分 step 会小于/超过目标 GBS,对效果的影响需要实测确认。
逼近目标 GBS 的做法是靠梯度累积:先估每个 pack 的平均样本数 `n_pack ≈ cutoff × pack利用率 / 平均样本长度`(用**均值**,不建议用中位数——长尾数据下中位数会低估),再取 `梯度累积 = round(GBS / (n_pack × GPU数))`。
这里有个硬约束:当一个优化器步(梯度累积=1)装的样本已经超过目标 GBS 时,因为梯度累积至少为 1,就**再也降不到目标 GBS 了**。所以必须满足 `n_pack × GPU数 ≤ GBS`。对小模型,这个上界往往比显存上限更紧。

**step 数量怎么防止过小?**
packing 之后,总 pack 数 ≈ num_samples / n_pack。真实的**优化器步数**是:
`opt_steps = 总pack数 × epochs / (GPU数 × 梯度累积)`。
要防止步数过小(设至少要 K 步),约束是 `opt_steps ≥ K`。注意:cutoff 越长 → n_pack 越大 → pack 数越少 → step 越少,所以光控 n_pack 不够,GPU 数和梯度累积都在分母里,得一起算。K 具体设多少会影响效果(warmup/decay 需要的最小步数、epoch 杠杆),需要进一步调研。

**一个关键点**
cutoff 加长能提吞吐,但同时会让每个 pack 装的样本变多、总 step 数变少——这两件事是同一个 cutoff 的取舍,方向相反,不能拆成两条独立规则各自约束。正确做法是联合决策:在同时满足「显存可行 + GBS 可控 + step 数达标」三个门槛的候选里,选吞吐最高的那个 cutoff。

**补充:规划器的已知边界**
显存可行上限依赖预测规划器,而规划器目前对 LoRA + 标准 dense 模型预测可信且偏保守;对**全参微调大模型、以及 VL 多模态模型会低估显存**。所以对这两类,选 cutoff 上界时要额外留余量,别踩规划器的盲区。
