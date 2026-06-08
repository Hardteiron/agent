"""
Milvus 向量记忆（Milvus Memory）
==================================
职责：
  - 提供基于语义向量的召回记忆检索能力（可选模块）
  - 支持 Dense（BGE 嵌入）和 Sparse（Milvus 内置 BM25）混合检索（RRF 融合）
  - 全局单集合设计：所有用户 / 会话共享一个 Collection，
    通过 metadata.user_id / conv_id 过滤隔离，避免集合数爆炸

依赖：
  pip install pymilvus>=2.5.0 httpx

Embedding 后端（Infinity ONNX + INT8 量化）：
  - 使用 Infinity Embedding Server 提供 BGE-large-zh-v1.5 向量服务
  - 服务端以 ONNX Runtime + INT8 量化运行，显著降低内存占用和推理延迟
  - 客户端通过 HTTP 调用 /v1/embeddings，兼容 OpenAI Embeddings API 格式
  - 同步路径（add / add_batch / search）使用 httpx.Client
  - 异步路径（add_turn_async / search_async）使用 httpx.AsyncClient，零线程池开销
  - 通过环境变量 INFINITY_URL 配置服务地址，默认 http://localhost:7997

  启动 Infinity 服务示例：
    docker run -p 7997:7997 michaelf34/infinity:latest \
      v2 --model-id BAAI/bge-large-zh-v1.5 \
      --engine optimum --dtype int8 \
      --port 7997

变更说明（sentence-transformers → Infinity ONNX + INT8）：
  - 删除 SentenceTransformer 进程内加载（~2 GB 显存 / 内存）
  - 改为 HTTP 客户端请求独立 Infinity 服务，支持多进程 / Celery Worker 并发安全调用
  - 向量已在服务端 L2 归一化（normalize=True），客户端无需额外处理
  - INT8 量化相比 FP32 内存节省 75%，延迟降低约 30-50%，召回率损失 < 0.5%

要求：
  - Milvus Server >= 2.5.0（BM25 Function 在 2.4 引入，建议 2.5+）
  - pymilvus >= 2.5.0
  - httpx >= 0.25.0
  - Infinity Embedding Server（见上方 docker 启动示例）

Chunk 策略（完整原文分段，不压缩）：
  每轮对话（user_query + assistant_reply）的完整原文按 token 边界分段写入，
  每段 ≤ _CHUNK_TOKEN_LIMIT token（默认 380），同一轮多段共享 round_id。

  分段规则：
    - 第一段前缀固定为 "问：{user_query}\\n答："
    - 后续段追加 reply 分段内容，前缀为 "（续）："
    - 不足一段则整体写入，无需分段

  分段 metadata 字段（在原有基础上新增）：
    "chunk_seq":     int  # 分段序号，从 0 开始
    "total_chunks":  int  # 本轮总分段数

Metadata 字段规范：
  {
    "user_id":      int,   # 必填，用于跨会话检索用户级记忆
    "conv_id":      int,   # 必填，用于会话级过滤
    "round_id":     int,   # 轮次编号，同一轮所有分段相同
    "chunk_seq":    int,   # 分段序号（0-based）
    "total_chunks": int,   # 本轮总分段数
    "type":         str,   # "dialogue"（原文）
    "timestamp":    float  # Unix 时间戳
  }

对外接口（供 ContextManager 调用）：
  async add_turn_async(user_id, conv_id, user_query, reply, round_id) -> List[int]
      写入完整原文（自动分段），返回所有 chunk 的 Milvus ID 列表

  async get_by_round_id(user_id, conv_id, round_id) -> str
      精确按 round_id 召回，按 chunk_seq 顺序拼回完整对话原文，不走向量搜索

  async search_async(query, user_id, top_k, conv_id, round_id_gte, round_id_lte)
      语义检索，可选 round_id 范围过滤

  add(text, meta) -> int            同步写入单条（Celery 任务中使用）
  add_batch(items) -> List[int]     批量写入
  search(query, top_k, expr) -> List[MilvusMemoryItem]
  delete(memory_id)
  count() -> int
  close()
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Optional

import logging

logger = logging.getLogger(__name__)

try:
    import httpx
    from pymilvus import (
        AnnSearchRequest,
        Collection,
        CollectionSchema,
        DataType,
        FieldSchema,
        Function,
        FunctionType,
        Index,
        RRFRanker,
        connections,
        utility,
    )
    _MILVUS_AVAILABLE = True
except ImportError:
    _MILVUS_AVAILABLE = False


# ─────────────────────────────────────────────
# 常量
# ─────────────────────────────────────────────

BGE_LARGE_ZH_DIM = 1024

_DEFAULT_INDEX_PARAMS = {
    "metric_type": "IP",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 128},
}
_DEFAULT_SEARCH_PARAMS = {
    "metric_type": "IP",
    "params": {"nprobe": 10},
}

# ── Chunk 分段参数 ────────────────────────────────────────────────────────────
# BGE-large-zh-v1.5 最大 512 tokens，CJK ≈ 1.5 token/字，英文 ≈ 0.3 token/字
_CHUNK_TOKEN_LIMIT  = 300   # 每段主体 token 上限（约 200 汉字）
_CHUNK_OVERLAP_RATE = 0.2   # 相邻段重叠比例（取前段尾部 20% token 作为后段头部）

# 旧常量保留向后兼容
_CHUNK_TEXT_LIMIT = 150
_MAX_EMBED_TOKENS = 480


def _estimate_tokens(text: str) -> int:
    """轻量 token 数估算（无需 tokenizer）：CJK×1.5 + 其他×0.3。"""
    cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    return int(cjk * 1.5 + (len(text) - cjk) * 0.3)


def _char_token(ch: str) -> float:
    """单字符 token 估算权重。"""
    return 1.5 if "\u4e00" <= ch <= "\u9fff" else 0.3


def _split_with_overlap(
    text: str,
    token_limit: int  = _CHUNK_TOKEN_LIMIT,
    overlap_rate: float = _CHUNK_OVERLAP_RATE,
) -> List[str]:
    """
    将 text 按 sliding window 切分为若干段，相邻段之间有内容重叠。

    算法：
      1. 逐字累积 token 估算，到达 token_limit 时记录一段
      2. 下一段从当前段尾部回退 (overlap_rate × token_limit) token 处开始
      3. 重叠部分保证上下文连贯，避免语义在边界处断裂

    参数：
      token_limit:  每段主体最大 token 数（不含重叠头）
      overlap_rate: 重叠占 token_limit 的比例，默认 0.2（即 60 token 重叠）

    示例（token_limit=10, overlap_rate=0.2）：
      原文: AAAAABBBBBCCCCC
      段 0: AAAAABB         (tokens ≈ 10)
      段 1:    BBBBBCCC     (头部 AB 与段 0 重叠)
      段 2:       CCCCC     (头部 BC 与段 1 重叠)
    """
    if not text:
        return [""]

    overlap_tokens = int(token_limit * overlap_rate)  # 重叠 token 数
    segments: List[str] = []

    n = len(text)
    seg_start = 0     # 当前段的起始字符位置

    while seg_start < n:
        budget = 0.0
        seg_end = seg_start

        # 向前扫描直到达到 token_limit
        while seg_end < n:
            budget += _char_token(text[seg_end])
            seg_end += 1
            if budget >= token_limit:
                break

        segments.append(text[seg_start:seg_end])

        if seg_end >= n:
            break  # 已到文本末尾

        # 计算下一段起始位置：从 seg_end 向前回退 overlap_tokens 个 token
        back_budget = 0.0
        next_start = seg_end
        while next_start > seg_start and back_budget < overlap_tokens:
            next_start -= 1
            back_budget += _char_token(text[next_start])

        # 防止死循环：若无法回退（文本太短），直接从 seg_end 继续
        if next_start <= seg_start:
            seg_start = seg_end
        else:
            seg_start = next_start

    return segments or [""]


def _safe_truncate(text: str, half_token_budget: int = 200) -> str:
    """按估算 token 数截断文本（兼容旧调用）。"""
    budget = 0.0
    for i, ch in enumerate(text):
        budget += _char_token(ch)
        if budget >= half_token_budget:
            return text[:i] + "…"
    return text


# ─────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────

@dataclass
class MilvusMemoryItem:
    """Milvus 检索返回的一条记忆。"""
    text:  str
    meta:  Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    id:    int   = 0


# ─────────────────────────────────────────────
# MilvusMemoryManager
# ─────────────────────────────────────────────

class MilvusMemoryManager:
    """
    基于 Milvus + BGE 稠密向量 + Milvus 内置 BM25 稀疏向量的记忆管理器。

    BM25 实现说明：
      - 使用 Milvus 服务端内置 BM25 Function（需 Milvus >= 2.5.0）
      - 写入时只需提供文本，Milvus 自动计算并存储 BM25 稀疏向量
      - 检索时 sparse 路直接传文本查询，Milvus 自动 BM25 编码后检索
      - IDF 统计由 Milvus 服务端管理，随数据持久化，不受进程重启影响

    全局单集合设计：
      - 所有用户 / 会话共享 Collection，通过 metadata 字段过滤
      - search_async(user_id=1) 检索用户全局记忆
      - search_async(user_id=1, conv_id=42) 检索特定会话记忆

    Chunk 粒度：
      - 每轮对话（user_query + assistant_reply）作为一个 chunk
      - 长文本按滑动窗口分段，保证向量语义密度

    Args:
        host, port:         Milvus 地址
        collection_name:    集合名（hybrid 模式自动加 _hybrid 后缀）
        embedding_model:    SentenceTransformer 模型名
        dimension:          稠密向量维度
        hybrid_search:      是否启用 BM25 + Dense 混合检索
        auto_create:        是否自动建集合
        analyzer_params:    BM25 分词器参数（默认标准分词器，适合中英文混合）
    """

    def __init__(
        self,
        host:             Optional[str] = None,
        port:             Optional[int] = None,
        collection_name:  str = "agent_memory_store",
        embedding_model:  str = "BAAI/bge-large-zh-v1.5",
        dimension:        int = BGE_LARGE_ZH_DIM,
        index_params:     Optional[Dict[str, Any]] = None,
        search_params:    Optional[Dict[str, Any]] = None,
        hybrid_search:    bool = True,
        auto_create:      bool = True,
        analyzer_params:  Optional[Dict[str, Any]] = None,
        infinity_url:     Optional[str] = None,
    ) -> None:
        if not _MILVUS_AVAILABLE:
            raise ImportError(
                "pymilvus >= 2.5.0 和 httpx 未安装。\n"
                "请执行：pip install 'pymilvus>=2.5.0' 'httpx>=0.25.0'"
            )

        self.host   = host or os.environ.get("MILVUS_HOST", "localhost")
        self.port   = port or int(os.environ.get("MILVUS_PORT", "19530"))
        self.hybrid = hybrid_search
        self.dim    = dimension
        self._idx_params  = index_params  or dict(_DEFAULT_INDEX_PARAMS)
        self._srch_params = search_params or dict(_DEFAULT_SEARCH_PARAMS)

        # BM25 分词器参数：默认使用标准分词器，支持中英文
        # 可通过 analyzer_params 覆盖，例如使用 jieba 分词器：
        #   {"tokenizer": "jieba"}
        self._analyzer_params = analyzer_params or {"type": "standard"}

        self.collection_name = (
            f"{collection_name}_hybrid" if hybrid_search else collection_name
        )

        # ── Infinity Embedding Server 配置 ────────────────────────────────
        # 优先级：构造参数 > 环境变量 INFINITY_URL > 默认值
        self._infinity_url  = (infinity_url or os.environ.get("INFINITY_URL", "http://localhost:7997")).rstrip("/")
        self._embed_model   = embedding_model
        self._embed_endpoint = f"{self._infinity_url}/v1/embeddings"

        # 同步 httpx 客户端（供 add / add_batch / search 使用）
        # timeout=60s 应对大批量编码；limits 限制最大连接数避免资源耗尽
        self._http_client = httpx.Client(
            timeout=60.0,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )

        logger.info(
            f"[MilvusMemory] Infinity embedding backend: {self._embed_endpoint} "
            f"model={self._embed_model}"
        )

        self._connect()

        if auto_create:
            self._ensure_collection()

    # ── 连接 ──────────────────────────────────

    def _connect(self) -> None:
        connections.connect(alias="default", host=self.host, port=self.port)
        logger.info(f"[MilvusMemory] connected to {self.host}:{self.port}")

    def _ensure_collection(self) -> None:
        if utility.has_collection(self.collection_name):
            self._col = Collection(self.collection_name)
            if not self._col.indexes:
                self._create_indexes()
            self._col.load()
            logger.info(
                f"[MilvusMemory] loaded '{self.collection_name}' "
                f"({self._col.num_entities} items, hybrid={self.hybrid})"
            )
            return

        # ── 字段定义 ──────────────────────────────────────────────────────
        fields = [
            FieldSchema("id",       DataType.INT64,        is_primary=True, auto_id=True),
            # enable_analyzer=True：Milvus 对该字段执行内置分词，为 BM25 Function 提供 token 流
            FieldSchema(
                "text",
                DataType.VARCHAR,
                max_length=65535,
                enable_analyzer=True,
                analyzer_params=self._analyzer_params,
            ),
            FieldSchema("metadata", DataType.JSON),
            # 独立标量字段，支持 pre-filtering 索引
            FieldSchema("user_id",  DataType.INT64),
            FieldSchema("conv_id",  DataType.INT64),
            FieldSchema("dense",    DataType.FLOAT_VECTOR,  dim=self.dim),
        ]
        if self.hybrid:
            # sparse 字段由 BM25 Function 自动填充，无需手动插入
            fields.append(
                FieldSchema("sparse", DataType.SPARSE_FLOAT_VECTOR)
            )

        schema = CollectionSchema(
            fields, description=f"Agent memory (hybrid={self.hybrid}, bm25=milvus_builtin)"
        )

        # ── BM25 Function 声明 ─────────────────────────────────────────────
        # Milvus 服务端在写入时自动将 text 字段的分词结果计算为 BM25 稀疏向量，
        # 存入 sparse 字段；检索时同样自动处理查询文本。
        if self.hybrid:
            bm25_fn = Function(
                name="bm25",
                function_type=FunctionType.BM25,
                input_field_names=["text"],
                output_field_names=["sparse"],
            )
            schema.add_function(bm25_fn)

        self._col = Collection(self.collection_name, schema)
        self._create_indexes()
        self._col.load()
        logger.info(f"[MilvusMemory] created '{self.collection_name}' with Milvus built-in BM25")

    def _create_indexes(self) -> None:
        Index(self._col, "dense", self._idx_params)
        if self.hybrid:
            Index(self._col, "sparse",
                  {"metric_type": "BM25", "index_type": "SPARSE_INVERTED_INDEX"})
        # 标量索引：加速 pre-filtering
        Index(self._col, "user_id", {"index_type": "STL_SORT"})
        Index(self._col, "conv_id", {"index_type": "STL_SORT"})

    # ── 向量生成（Infinity ONNX + INT8）────────

    def _dense(self, text: str) -> List[float]:
        """同步：单条文本 → 稠密向量。调用 Infinity /v1/embeddings。"""
        return self._dense_batch([text])[0]

    def _dense_batch(self, texts: List[str]) -> List[List[float]]:
        """同步：批量文本 → 稠密向量列表。Infinity 服务端已 L2 归一化。"""
        payload = {
            "model": self._embed_model,
            "input": texts,
        }
        resp = self._http_client.post(self._embed_endpoint, json=payload)
        resp.raise_for_status()
        data = resp.json()
        # OpenAI 兼容格式：{"data": [{"embedding": [...], "index": 0}, ...]}
        embeddings = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in embeddings]

    async def _dense_batch_async(self, texts: List[str]) -> List[List[float]]:
        """异步：批量文本 → 稠密向量列表。使用 httpx.AsyncClient，无线程池开销。"""
        payload = {
            "model": self._embed_model,
            "input": texts,
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(self._embed_endpoint, json=payload)
        resp.raise_for_status()
        data = resp.json()
        embeddings = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in embeddings]

    # ── 同步写入（供 Celery 任务调用） ────────

    def add(self, text: str, meta: Optional[Dict[str, Any]] = None) -> int:
        """
        写入一条记忆，返回 Milvus 主键 ID。
        meta 应包含 user_id / conv_id / round_id / type 字段。

        注意：hybrid 模式下无需传 sparse 字段，Milvus BM25 Function 自动计算。
        """
        meta = {**(meta or {}), "timestamp": time.time()}
        # hybrid 模式：只传 text + metadata + dense，BM25 Function 自动填充 sparse
        entities: List[Any] = [
            [text],
            [json.dumps(meta, ensure_ascii=False)],
            [meta.get("user_id", 0)],
            [meta.get("conv_id", 0)],
            [self._dense(text)],
        ]

        result = self._col.insert(entities)
        self._col.flush()
        return int(result.primary_keys[0])

    def add_batch(self, items: List[Dict[str, Any]]) -> List[int]:
        """
        批量写入。items 格式：[{"text": ..., "meta": {...}}, ...]

        注意：hybrid 模式下无需传 sparse 字段，Milvus BM25 Function 自动计算。
        """
        texts = [it["text"] for it in items]
        parsed_metas = [{**(it.get("meta") or {}), "timestamp": time.time()} for it in items]
        metas = [json.dumps(m, ensure_ascii=False) for m in parsed_metas]
        user_ids = [m.get("user_id", 0) for m in parsed_metas]
        conv_ids  = [m.get("conv_id", 0)  for m in parsed_metas]
        # 只传 text + metadata + user_id + conv_id + dense，BM25 Function 自动填充 sparse
        entities: List[Any] = [texts, metas, user_ids, conv_ids, self._dense_batch(texts)]

        result = self._col.insert(entities)
        self._col.flush()
        return [int(pk) for pk in result.primary_keys]

    # ── 异步写入（供 ContextManager 调用） ────

    async def add_turn_async(
        self,
        user_id:    int,
        conv_id:    int,
        user_query: str,
        reply:      str,
        round_id:   int = 0,
        summary:    str = "",   # 保留参数，向后兼容，不再影响 chunk 内容
    ) -> List[int]:
        """
        将一轮对话的完整原文（user_query + reply）分段写入 Milvus。

        分段策略（Sliding Window，有重叠）：
          - 将 "问：{user_query}\\n答：{reply}" 整体拼接后做 sliding window 切分
          - 每段主体 ≤ _CHUNK_TOKEN_LIMIT token（约 300 token / 200 汉字）
          - 相邻段头部重叠 _CHUNK_OVERLAP_RATE × _CHUNK_TOKEN_LIMIT token（约 60 token）
          - 重叠保证语义不在边界断裂，同时向量能覆盖跨段关键词

        Metadata 字段：
          user_id      — 用户 ID，过滤隔离
          conv_id      — 会话 ID，过滤隔离
          round_id     — 轮次编号（与 ShortTermMemory._round_id 一致）
          chunk_seq    — 本轮内分段序号，0-based
          total_chunks — 本轮总分段数
          overlap_rate — 实际重叠比例（冗余存储，便于调试）
          type         — "dialogue"（原文，非摘要）
          timestamp    — Unix 时间戳

        精确召回：
          使用 get_by_round_id_async(conv_id, round_id) 按 chunk_seq 升序
          拼回完整原文（重叠部分通过 chunk_start_offset 字段恢复非重叠拼接）

        Returns:
            写入成功的 Milvus ID 列表（长度 = total_chunks）
        """
        # ── 拼接全文 ──────────────────────────────────────────
        full_text = f"问：{user_query}\n答：{reply}"

        # ── Sliding window 切分 ────────────────────────────────
        raw_segs = _split_with_overlap(
            full_text,
            token_limit=_CHUNK_TOKEN_LIMIT,
            overlap_rate=_CHUNK_OVERLAP_RATE,
        )

        # ── 计算每段非重叠起始偏移（用于精确拼回时去重）────────
        start_offsets: List[int] = []
        pos = 0
        for seg in raw_segs:
            start_offsets.append(pos)
            seg_end = pos + len(seg)
            overlap_tokens = int(_CHUNK_TOKEN_LIMIT * _CHUNK_OVERLAP_RATE)
            back = 0.0
            next_start = seg_end
            while next_start > pos and back < overlap_tokens:
                next_start -= 1
                back += _char_token(full_text[next_start])
            pos = next_start if next_start > pos else seg_end

        total = len(raw_segs)
        ts = time.time()

        # ── 异步获取稠密向量（直接调用 Infinity，无需 run_in_executor）────
        dense_vecs = await self._dense_batch_async(raw_segs)

        items_meta = [
            json.dumps(
                {
                    "user_id":            user_id,
                    "conv_id":            conv_id,
                    "round_id":           round_id,
                    "chunk_seq":          seq,
                    "total_chunks":       total,
                    "chunk_start_offset": start_offsets[seq],
                    "overlap_rate":       _CHUNK_OVERLAP_RATE,
                    "type":               "dialogue",
                    "timestamp":          ts,
                },
                ensure_ascii=False,
            )
            for seq in range(total)
        ]

        # 只传 text + metadata + user_id + conv_id + dense，BM25 Function 自动填充 sparse
        entities: List[Any] = [
            raw_segs,
            items_meta,
            [user_id] * total,
            [conv_id] * total,
            dense_vecs,
        ]

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: (self._col.insert(entities), self._col.flush()),
        )
        ids: List[int] = [int(pk) for pk in result[0].primary_keys]
        logger.debug(
            f"[MilvusMemory] add_turn_async: user={user_id} conv={conv_id} "
            f"round={round_id} → {total} chunk(s) ids={ids}"
        )
        return ids

    # ── 精确按 round_id 召回（不走向量搜索）──

    def get_by_round_id(
        self,
        user_id:  int,
        conv_id:  int,
        round_id: int,
    ) -> str:
        """
        精确召回指定轮次的完整对话原文（去除重叠后拼合）。

        流程：
          1. expr 精确过滤 user_id + conv_id + round_id
          2. 按 chunk_seq 升序排列所有分段
          3. 利用 chunk_start_offset 去除每段头部重叠，还原无重复完整原文

        Returns:
            完整原文（"问：...\\n答：..." 格式）；未找到时返回 ""
        """
        self._col.load()
        expr = (
            f'user_id == {user_id} '
            f'and conv_id == {conv_id} '
            f'and metadata["round_id"] == {round_id}'
        )
        dummy = [0.0] * self.dim
        res = self._col.search(
            data=[dummy],
            anns_field="dense",
            param=self._srch_params,
            limit=200,    # 单轮 chunk 数理论上限（长文本最多约 50 段）
            expr=expr,
            output_fields=["text", "metadata"],
        )
        items = self._parse(res)
        if not items:
            return ""

        # 按 chunk_seq 升序排列
        items.sort(key=lambda x: x.meta.get("chunk_seq", 0))

        # ── 利用 chunk_start_offset 去重拼合 ────────────────────
        parts: List[str] = []
        offsets = [item.meta.get("chunk_start_offset", 0) for item in items]

        for idx, item in enumerate(items):
            seg_text = item.text
            if idx == 0:
                parts.append(seg_text)
            else:
                delta = offsets[idx] - offsets[idx - 1]
                skip = min(delta, len(seg_text))
                parts.append(seg_text[skip:])

        return "".join(parts)

    async def get_by_round_id_async(
        self,
        user_id:  int,
        conv_id:  int,
        round_id: int,
    ) -> str:
        """get_by_round_id 的异步版本。"""
        loop = asyncio.get_event_loop()
        result: str = await loop.run_in_executor(
            None, partial(self.get_by_round_id, user_id, conv_id, round_id)
        )
        logger.debug(
            f"[MilvusMemory] get_by_round_id_async: user={user_id} "
            f"conv={conv_id} round={round_id} → {len(result)} chars"
        )
        return result

    # ── 同步检索 ──────────────────────────────

    def search(
        self,
        query:  str,
        top_k:  int = 3,
        expr:   Optional[str] = None,
    ) -> List[MilvusMemoryItem]:
        """语义检索（hybrid 时使用 RRF 融合 dense + Milvus 内置 BM25 sparse）。"""
        self._col.load()
        fields = ["text", "metadata"]
        if self.hybrid:
            return self._hybrid_search(query, top_k, expr, fields)
        return self._dense_search(query, top_k, expr, fields)

    def _dense_search(
        self, query: str, top_k: int,
        expr: Optional[str], fields: List[str],
    ) -> List[MilvusMemoryItem]:
        res = self._col.search(
            data=[self._dense(query)],
            anns_field="dense",
            param=self._srch_params,
            limit=top_k,
            expr=expr,
            output_fields=fields,
        )
        return self._parse(res)

    def _hybrid_search(
        self, query: str, top_k: int,
        expr: Optional[str], fields: List[str],
    ) -> List[MilvusMemoryItem]:
        """
        混合检索：Dense（BGE 语义）+ Sparse（Milvus 内置 BM25），RRF 融合。

        关键变化（与本地 BM25 版本对比）：
          - sparse_req 的 data 直接传文本字符串列表，无需手动 BM25 编码
          - Milvus 服务端通过 BM25 Function 自动将查询文本转为稀疏向量后检索
          - metric_type 改为 "BM25"（内置 BM25 索引的标准度量）
        """
        k2 = top_k * 3

        # ── Dense 路：BGE 语义相似度 ──────────────────────────────
        dense_req = AnnSearchRequest(
            data=[self._dense(query)],
            anns_field="dense",
            param=self._srch_params,
            limit=k2,
            expr=expr,
        )

        # ── Sparse 路：Milvus 内置 BM25 ───────────────────────────
        # data 直接传原始文本，Milvus 服务端自动调用 BM25 Function 编码
        # metric_type 必须与索引创建时一致（"BM25"）
        sparse_req = AnnSearchRequest(
            data=[query],
            anns_field="sparse",
            param={"metric_type": "BM25"},
            limit=k2,
            expr=expr,
        )

        # ── RRF 融合重排 ───────────────────────────────────────────
        res = self._col.hybrid_search(
            reqs=[dense_req, sparse_req],
            rerank=RRFRanker(),
            limit=top_k,
            output_fields=fields,
        )
        return self._parse(res)

    # ── 异步检索（供 ContextManager 调用） ────

    async def search_async(
        self,
        query:          str,
        user_id:        int,
        top_k:          int = 3,
        conv_id:        Optional[int] = None,
        round_id_gte:   Optional[int] = None,
        round_id_lte:   Optional[int] = None,
    ) -> List[MilvusMemoryItem]:
        """
        异步语义检索，可附加 round_id 范围过滤。

        过滤优先级（从严到宽）：
          user_id + conv_id + round_id 范围  →  会话内指定轮次范围
          user_id + conv_id                  →  会话内全局召回
          user_id only                       →  跨会话全局召回

        Args:
            query:        当前用户输入，用于向量相似度排序
            user_id:      用户 ID
            top_k:        返回最相关的 k 条 chunk
            conv_id:      会话 ID（None 则跨会话全局召回）
            round_id_gte: 只召回 round_id ≥ 该值的 chunk（None 表示不限）
            round_id_lte: 只召回 round_id ≤ 该值的 chunk（None 表示不限）

        Returns:
            List[MilvusMemoryItem]，按相似度降序
        """
        conditions = [f'user_id == {user_id}']
        if conv_id is not None:
            conditions.append(f'conv_id == {conv_id}')
        if round_id_gte is not None:
            conditions.append(f'metadata["round_id"] >= {round_id_gte}')
        if round_id_lte is not None:
            conditions.append(f'metadata["round_id"] <= {round_id_lte}')
        expr = " and ".join(conditions)

        loop = asyncio.get_event_loop()
        items = await loop.run_in_executor(
            None, partial(self.search, query, top_k, expr)
        )
        logger.debug(
            f"[MilvusMemory] search_async: user={user_id} conv={conv_id} "
            f"round=[{round_id_gte},{round_id_lte}] "
            f"query='{query[:40]}' → {len(items)} results"
        )
        return items

    # ── 解析结果 ──────────────────────────────

    @staticmethod
    def _parse(results) -> List[MilvusMemoryItem]:
        items: List[MilvusMemoryItem] = []
        for hits in results:
            for hit in hits:
                meta: Dict[str, Any] = {}
                raw = hit.entity.get("metadata")
                if raw:
                    try:
                        meta = json.loads(raw)
                    except Exception:
                        meta = {"raw": str(raw)}
                items.append(MilvusMemoryItem(
                    text=hit.entity.get("text", ""),
                    meta=meta,
                    score=hit.score,
                    id=hit.id,
                ))
        return items

    # ── 删除 / 维护 ───────────────────────────

    def delete(self, memory_id: int) -> None:
        self._col.delete(f"id in [{memory_id}]")
        self._col.flush()

    def delete_by_expr(self, expr: str) -> None:
        self._col.delete(expr)
        self._col.flush()

    def drop_collection(self) -> None:
        self._col.drop()
        logger.info(f"[MilvusMemory] dropped '{self.collection_name}'")

    def count(self) -> int:
        self._col.load()
        return self._col.num_entities

    def close(self) -> None:
        self._http_client.close()
        connections.disconnect("default")
        logger.info("[MilvusMemory] disconnected")

    def __enter__(self) -> "MilvusMemoryManager":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
