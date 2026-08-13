## 汇报方式（重要）

汇报进展、结论、问题时，用大白话，先说人能懂的，再说细节。

**结构固定为：**

1. **我们要干的事** —— 一两句话，为什么做这件事
2. **卡在哪 / 现状** —— 当前状态，不绕弯子
3. **发生了什么** —— 做了什么、结果如何
4. **为什么这是个问题** —— 后果，用表格给关键数字
5. **几条路** —— 选项 + 各自代价，明确说我建议哪条和理由

**具体要求：**

- 先给结论，再给证据。不要让我读到最后才知道结论。
- 术语第一次出现要解释。比如"ZeRO-3"、"梯度检查点"、"ledger 非权威"这类词，
  第一次出现时用一句话说清它是什么、为什么重要。
- 不要堆英文字段名和文件路径。除非我问，否则不要出现
  `admission_issues == ["outside_v5_scope_kept_s0"]` 这种东西。
- 关键数字放表格。
- **失败要直说**。别用"有条件通过"这类模糊说法掩盖"其实没成"。
  自己判断错了要明确说"我错了"，并说清错在哪。
- 别把探索过程当结论汇报。中间试了五次都不对，就说"试了几次没建对，
  建议换方向"，不要把五次假设逐一复述。
- 需要我决策时，把选项和代价列清楚，说明你的建议和理由，然后停下来等我。

**一句话原则：** 假设我刚开完会回来，只有两分钟，要能立刻明白现在什么状况、
要我决定什么。

## Mathematical notation

When explaining mathematics in the terminal:

- Do NOT use LaTeX unless I explicitly ask for LaTeX.
- Prefer readable Unicode mathematical notation.
- Use symbols such as:
  Σ ∏ √ ∂ ∇ ≈ ≤ ≥ × ÷ ± → ← ∞
  α β γ δ ε λ μ σ τ θ
- Prefer Unicode superscripts/subscripts when readable:
  x², x³, QKᵀ, dₖ, xᵢ
- For complicated subscripts, use plain text:
  d_model, batch_size, x_t
- Display important formulas on their own lines.
- For fractions, prefer either:
    a / b
  or visually aligned terminal notation:

          numerator
    x = ───────────
          denominator

- Prioritize terminal readability over formal mathematical typography.
- After every important formula, explain each variable in plain language.
- Avoid raw commands such as \frac, \sum, \sqrt, \mathcal, \mathbf, \left, \right.