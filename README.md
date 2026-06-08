# Agent Memory System

三层记忆架构 + Token 预算管理，为长对话场景下的 LLM Agent 提供上下文管理能力。

---

## 架构总览

```
用户请求
    ↓
ContextManager
    ├── ShortTermMemory    短期记忆（会话级滚动窗口）
    ├── LongTermMemory     长期记忆（用户级结构化知识）
    ├── MilvusMemory       向量记忆（跨会话语义检索）
    └── PromptBudget       Token 预算管理（LLM 调用前压缩）
```

---

## 模块说明

### 1. ShortTermMemory — 短期记忆

**文件**：`short_term_memory.py`

维护单会话的对话滚动窗口，兼顾细节保留与 token 经济性。

**核心策略**：
- 保留最近 `full_detail_rounds`（默认 5）轮完整对话内容
- 当全量轮数达到阈值时，逆序定位压缩边界，将最早一轮异步压缩为单句摘要
- 压缩后只更新被修改的 2 条消息（Redis LSET 原地替换），避免全量覆盖写放大

**压缩触发逻辑**：

```
第1~4轮：正常追加，不触发压缩
第5轮写入：触发压缩 → 将 round_1 压缩为摘要
第6轮写入：触发压缩 → 将 round_2 压缩为摘要
...
始终保持最近 5 轮为完整细节，更早的轮次全部为摘要
```

**数据结构**：

```python
@dataclass
class _Msg:
    role:       str    # "user" | "assistant"
    content:    str
    round_id:   int
    compressed: bool
    summary:    str    # 压缩后的摘要文本
    message_id: int    # 对应 DB Message.id
```

---

### 2. LongTermMemory — 长期记忆

**文件**：`long_term_memory.py`

跨会话的用户级结构化知识库，以 JSON 形式持久化。

**JSON Schema**：

```json
{
  "schema_version": "1.0",
  "updated_at": "ISO8601",
  "user_profile": {
    "name", "role", "department", "language_preference", "expertise_level"
  },
  "task_preferences": {
    "preferred_output_format", "preferred_chart_type", "auto_download"
  },
  "domain_knowledge": [
    {"topic", "content", "confidence", "last_seen_round"}
  ],
  "recurring_tasks": [
    {"task_name", "frequency", "typical_instruction", "last_seen_round"}
  ],
  "corrections": [{"wrong", "right", "round_id"}],
  "session_continuity": {
    "last_active_dataset", "pending_tasks"
  }
}
```

**核心策略**：
- 每隔 N 轮（默认 3 轮）从短期历史中调用 LLM 抽取结构化知识
- 按 topic / task_name 去重合并，最近优先保留
- 超出字符上限时按字段优先级淘汰（corrections → session_continuity → 旧 domain_knowledge）

---

### 3. MilvusMemory — 向量记忆

**文件**：`milvus_memory.py`

基于 Milvus 的跨会话语义检索记忆，支持 Dense + BM25 混合检索。

**写入策略（Sliding Window 分段）**：

```
每轮对话原文 "问：{query}\n答：{reply}"
    ↓ 滑动窗口切分（chunk ≤ 300 token，重叠率 20%）
    ↓ BGE-large-zh-v1.5（Infinity ONNX INT8）生成稠密向量
    ↓ Milvus 内置 BM25 Function 自动生成稀疏向量
    ↓ 写入 Milvus Collection
```

**检索策略（混合检索 RRF 融合）**：

```
查询文本
    ├── Dense 路：BGE 语义向量 → IVF_FLAT ANN
    └── Sparse 路：Milvus BM25 → SPARSE_INVERTED_INDEX
        ↓ RRFRanker 融合重排
        ↓ pre-filtering: user_id + conv_id（STL_SORT 标量索引）
```

**Collection Schema**：

| 字段 | 类型 | 说明 |
|------|------|------|
| id | INT64 | 主键，自增 |
| text | VARCHAR | 原文，自动分词（BM25） |
| metadata | JSON | 完整元信息（round_id / chunk_seq 等） |
| user_id | INT64 | 独立标量字段，建 STL_SORT 索引 |
| conv_id | INT64 | 独立标量字段，建 STL_SORT 索引 |
| dense | FLOAT_VECTOR(1024) | BGE 稠密向量 |
| sparse | SPARSE_FLOAT_VECTOR | BM25 稀疏向量（自动填充） |

> `user_id` / `conv_id` 独立为标量字段（而非 JSON 内嵌）的原因：支持 Milvus pre-filtering，在 ANN 检索前先缩减候选集，显著降低召回延迟。

---

### 4. PromptBudget — Token 预算管理

**文件**：`prompt_budget.py`

每次 LLM 调用前对 prompt 各模块做 token 估算与自适应压缩，确保不超出模型上下文限制。

**Token 预算分配（默认，基于 model_context_limit）**：

| 模块 | 比例 | 说明 |
|------|------|------|
| system_prompt | 20% | 角色声明 + 工具描述 |
| short_term_history | 35% | 近期对话轮次 |
| long_term_memory | 10% | 结构化长期记忆 JSON |
| recall_context | 8% | Milvus 语义召回片段 |
| df_info | 10% | DataFrame 表结构预览 |
| user_query | 5% | 当前用户输入（永不压缩） |
| output_reserve | 12% | 留给模型输出 |

**压缩优先级（超限时按序执行）**：

```
1. recall_context   → 裁减到 top-1 召回片段
2. df_info          → 仅保留列名行，去掉数据预览行
3. long_term_memory → 按字段优先级裁剪（保留 user_profile + task_preferences）
4. short_term_history → 从最旧轮次开始删减
5. system_prompt    → 压缩为工具名列表
6. （兜底）强制截断 short_term_history
```

**Token 估算规则**（轻量，无需加载 tokenizer）：
- 中文字符：1 字 ≈ 1.5 token
- 其余字符：1 字符 ≈ 0.25 token
- 误差 ±15%，足够预算调度使用

**使用示例**：

```python
budget = PromptBudget(BudgetConfig(model_context_limit=8192))
sections = PromptSections(
    system_prompt      = system_text,
    short_term_history = ctx.get_short_history(),
    long_term_memory   = ctx.get_long_term_text(),
    recall_context     = await ctx.get_recall_context(query),
    df_info            = df_info_text,
    user_query         = user_query,
)
sections, report = budget.fit(sections)
# sections 已压缩到预算内，report 包含各模块 token 用量及压缩记录
```

---

## 依赖

```
pymilvus>=2.5.0
httpx>=0.25.0
redis[asyncio]>=4.0.0
```

Embedding 服务（BGE-large-zh-v1.5 ONNX INT8）：

```bash
docker run -p 7997:7997 michaelf34/infinity:latest \
  v2 --model-id BAAI/bge-large-zh-v1.5 \
  --engine optimum --dtype int8 --port 7997
```

环境变量：

```bash
MILVUS_HOST=localhost
MILVUS_PORT=19530
INFINITY_URL=http://localhost:7997
```
