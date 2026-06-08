"""
ContextManager — 三层记忆统一管理器
======================================
职责：
  - 短期记忆（ShortTermMemory）：Redis LIST，会话级滚动窗口
  - 长期记忆（LongTermMemory）：Redis STRING，用户级文本摘要
  - 召回记忆（MilvusMemoryManager）：向量数据库，跨会话语义检索

存储策略：
  - Redis 作为主存储层：首次获取时由 DB 回填，更新后同步写入 Redis
  - PostgreSQL 持久化：全部通过 Celery 异步任务完成，不阻塞响应流
  - Milvus：每轮对话结束后异步写入（可选，config.milvus_enabled 控制）

Redis Key 规范：
  mem:conv:{conv_id}:messages     →  LIST   每项为 Message 序列化 JSON
  mem:user:{user_id}:long_term    →  STRING  长期记忆全文

Chunk 策略（Milvus）：
  每轮对话（一问一答）作为一个 chunk，格式：
    "问：{user_query[:300]}\n答：{summary or reply[:300]}"
  通过 metadata.user_id / conv_id 过滤隔离，全局单集合设计。

DB 模型（写操作均为 Celery 异步）：
  Message          — 唯一事实来源，is_compressed/summary 承担压缩状态
  LongTermMemory   — 长期记忆全文快照（upsert）

公开接口：
  await load()
  get_short_history() -> List[Dict]
  get_long_term_text() -> str
  await get_recall_context(query) -> str      Milvus 召回，返回格式化文本
  await update(user_query, asst_reply, summary, user_msg_id, asst_msg_id)
  await close()

工厂函数：
  await build_context_manager(user_id, conv_id, llm, redis_url) -> ContextManager
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import redis.asyncio as aioredis

from agent.short_term_memory import ShortTermMemory, _Msg
from agent.long_term_memory import LongTermMemory

logger = logging.getLogger(__name__)

_EXTRACT_EVERY_N_ROUNDS = 3


class ContextManager:
    """
    三层记忆管理器（每请求实例化一次，无进程级共享状态）。

    使用方式：
        ctx = await build_context_manager(user_id=1, conv_id=42, llm=llm)
        history    = ctx.get_short_history()          # 短期对话历史
        long_text  = ctx.get_long_term_text()         # 长期记忆文本
        recall_ctx = await ctx.get_recall_context(q)  # Milvus 召回上下文
        await ctx.update(user_query, reply, summary,
                         user_msg_id=u_id, asst_msg_id=a_id)
        await ctx.close()
    """

    def __init__(
        self,
        redis:             aioredis.Redis,
        user_id:           int,
        conv_id:           int,
        llm:               Optional[Any] = None,
        short_max_rounds:  int = 20,
        short_full_detail: int = 5,
        long_max_chars:    int = 2000,
        ttl:               int = 7 * 24 * 3600,
        extract_every_n:   int = _EXTRACT_EVERY_N_ROUNDS,
        milvus_manager:    Optional[Any] = None,   # MilvusMemoryManager | None
        milvus_top_k:      int = 3,
    ) -> None:
        self.redis   = redis
        self.user_id = user_id
        self.conv_id = conv_id
        self.llm     = llm
        self.ttl     = ttl
        self.extract_every_n = extract_every_n

        # Milvus 召回记忆（可选）
        self._milvus:    Optional[Any] = milvus_manager
        self._milvus_top_k = milvus_top_k

        # Redis Key
        self._k_msgs = f"mem:conv:{conv_id}:messages"
        self._k_long = f"mem:user:{user_id}:long_term"

        # 内存记忆对象
        self._short = ShortTermMemory(
            max_rounds=short_max_rounds,
            full_detail_rounds=short_full_detail,
            llm=llm,
        )
        self._long = LongTermMemory(max_chars=long_max_chars, llm=llm)

        self._last_extract_at: int = 0

    # ──────────────────────────────────────────
    # 加载（Redis → 内存；Redis Miss → DB → Redis）
    # ──────────────────────────────────────────

    async def load(self) -> None:
        """
        从 Redis 加载短期 + 长期记忆。
        Redis 未命中时回源 DB，并将结果写入 Redis（缓存预热）。
        """
        await self._load_short()
        await self._load_long()

    async def _load_short(self) -> None:
        """
        从 Redis 加载短期记忆。

        优先从 Redis 获取消息列表；Redis 未命中时从数据库回填，
        并将结果写入 Redis 实现缓存预热。
        """
        raw = await self.redis.lrange(self._k_msgs, 0, -1)
        if raw:
            records = []
            for item in raw:
                try:
                    records.append(json.loads(item))
                except Exception:
                    pass
            self._short.load_from_records(records)
            logger.debug(
                f"ContextManager[conv={self.conv_id}]: "
                f"loaded {len(records)} messages from Redis"
            )
            return

        # Redis miss → DB 回填 → 写入 Redis
        records = await self._fetch_messages_from_db()
        if records:
            self._short.load_from_records(records)
            await self._flush_short_to_redis(records)
            logger.debug(
                f"ContextManager[conv={self.conv_id}]: "
                f"loaded {len(records)} messages from DB → Redis"
            )

    async def _load_long(self) -> None:
        text = await self.redis.get(self._k_long)
        if text:
            self._long.set_text(text)
            logger.debug(
                f"ContextManager[user={self.user_id}]: "
                f"loaded long-term ({len(text)} chars) from Redis"
            )
            return

        # Redis miss → DB 回填 → 写入 Redis
        text = await self._fetch_long_from_db()
        if text:
            self._long.set_text(text)
            await self.redis.set(self._k_long, text, ex=self.ttl)
            logger.debug(
                f"ContextManager[user={self.user_id}]: "
                f"loaded long-term ({len(text)} chars) from DB → Redis"
            )

    # ──────────────────────────────────────────
    # 读（供 agent/base.py / context.py 构建 prompt）
    # ──────────────────────────────────────────

    async def build_prompt_sections(
        self,
        user_query:    str,
        system_prompt: str = "",
        df_info:       str = "",
    ) -> "PromptSections":
        """
        组装并返回 PromptSections，供 PromptBudget.fit() 做 token 预算检查。

        Args:
            user_query:    当前用户输入
            system_prompt: 角色声明 + 工具描述文本
            df_info:       DataFrame 预览文本（可选）

        Returns:
            PromptSections — 已填充各记忆层内容，尚未做 token 压缩
        """
        from agent.prompt_budget import PromptSections
        recall = await self.get_recall_context(user_query, conv_id=self.conv_id)
        return PromptSections(
            system_prompt      = system_prompt,
            short_term_history = self.get_short_history(),
            long_term_memory   = self.get_long_term_text(),
            recall_context     = recall,
            df_info            = df_info,
            user_query         = user_query,
        )

    def get_prompt_budget(self) -> "PromptBudget":
        """
        返回与当前配置绑定的 PromptBudget 实例。

        从全局 config 读取 model_context_limit / prompt_budget_ratios，
        方便在 base.py 中一行取用。
        """
        from config import config as cfg
        from agent.prompt_budget import PromptBudget, BudgetConfig
        return PromptBudget(BudgetConfig(
            model_context_limit = getattr(cfg, "model_context_limit", 8192),
            ratios              = getattr(cfg, "prompt_budget_ratios", {}),
            min_short_rounds    = getattr(cfg, "prompt_budget_min_short_rounds", 1),
        ))

    def get_short_history(self) -> List[Dict[str, str]]:
        """
        返回格式化对话历史。
        - 最近 full_detail_rounds 轮：完整 content
        - 更早轮：摘要行（role=system, content="[历史摘要] ..."）
        """
        return self._short.get_history()

    def get_long_term_text(self) -> str:
        """返回长期记忆全文。"""
        return self._long.get_text()

    async def get_recall_context(
        self,
        query:          str,
        conv_id:        Optional[int] = None,
        round_id_gte:   Optional[int] = None,
        round_id_lte:   Optional[int] = None,
    ) -> str:
        """
        语义召回：从 Milvus 检索与 query 语义相关的历史对话 chunk。

        Args:
            query:        当前用户输入，用于向量检索
            conv_id:      非 None 时限定在该会话内检索；None 则跨会话全局召回
            round_id_gte: 只召回 round_id ≥ 该值的 chunk（None 表示不限）
            round_id_lte: 只召回 round_id ≤ 该值的 chunk（None 表示不限）

        Returns:
            格式化文本，可直接注入 prompt；Milvus 不可用时返回 ""
        """
        if self._milvus is None:
            return ""
        try:
            items = await self._milvus.search_async(
                query=query,
                user_id=self.user_id,
                top_k=self._milvus_top_k,
                conv_id=conv_id,
                round_id_gte=round_id_gte,
                round_id_lte=round_id_lte,
            )
            if not items:
                return ""
            lines = [f"- {item.text}" for item in items]
            return "【相关历史记忆（语义召回）】\n" + "\n".join(lines)
        except Exception as e:
            logger.warning(f"ContextManager.get_recall_context: {e}")
            return ""

    async def get_round_detail(self, round_id: int) -> str:
        """
        精确召回：按 round_id 从 Milvus 取回指定轮次的完整对话原文。

        适用场景：
          - 短期记忆中该轮已被压缩（只剩摘要），需要查看原文
          - 需要精确引用某一轮的具体内容

        Args:
            round_id: 目标轮次编号（与 ShortTermMemory._round_id 一致）

        Returns:
            完整原文字符串（"问：...\n答：..." 格式）；Milvus 不可用或未找到时返回 ""
        """
        if self._milvus is None:
            return ""
        try:
            text = await self._milvus.get_by_round_id_async(
                user_id=self.user_id,
                conv_id=self.conv_id,
                round_id=round_id,
            )
            return text
        except Exception as e:
            logger.warning(f"ContextManager.get_round_detail: {e}")
            return ""

    # ──────────────────────────────────────────
    # 更新（大模型回答后调用）
    # ──────────────────────────────────────────

    async def update(
        self,
        user_query:      str,
        assistant_reply: str,
        summary:         str = "",
        user_msg_id:     int = 0,
        asst_msg_id:     int = 0,
    ) -> None:
        """
        在大模型回答后统一更新三层记忆：
          1. 写内存 ShortTermMemory（附带 DB message_id）
          2. 若触发压缩：异步生成摘要 → 同步刷写 Redis → Celery 回写 DB
          3. 全量刷写 Redis（同步）
          4. 每 extract_every_n 轮抽取长期记忆 → 更新 Redis → Celery 持久化 DB
          5. 异步写入 Milvus（完整原文分段，不压缩）

        Args:
            user_query:      用户输入
            assistant_reply: LLM 完整回复（Milvus 存原文）
            summary:         LLM 生成的简短摘要（用于短期记忆压缩和长期记忆抽取）
            user_msg_id:     DB Message.id（user），0 表示未持久化
            asst_msg_id:     DB Message.id（assistant），0 表示未持久化
        """
        # ── 1. 写内存短期记忆 ──
        next_round = self._short._round_id + 1
        self._short._messages.append(_Msg(
            role="user",
            content=user_query,
            round_id=next_round,
            message_id=user_msg_id,
        ))
        self._short._round_id = next_round

        self._short._messages.append(_Msg(
            role="assistant",
            content=assistant_reply if assistant_reply else summary,
            round_id=next_round,
            message_id=asst_msg_id,
        ))

        # ── 2. 压缩检查（异步 LLM 生成摘要）──
        # 当全量轮数达到 full_detail_rounds 时立即压缩最早一轮
        indexed_msgs = []
        if self._short._round_id >= self._short.full_detail_rounds:
            compressed_ids, indexed_msgs = await self._short.compress_oldest_async()
            if indexed_msgs:
                summary_text = indexed_msgs[0][1].summary
                self._dispatch_update_compressed(
                    [{"message_ids": compressed_ids, "summary": summary_text}]
                )

        # ── 3. Redis 写入：有压缩时 LSET 原地修改被压缩的 2 条，然后追加新轮 ──
        if indexed_msgs:
            await self._patch_compressed_to_redis(indexed_msgs)
        await self._append_turn_to_redis(
            user_query=user_query,
            assistant_reply=assistant_reply if assistant_reply else summary,
            round_id=next_round,
            user_msg_id=user_msg_id,
            asst_msg_id=asst_msg_id,
        )

        # ── 4. 长期记忆抽取 ──
        current_round = self._short.round_count
        if current_round - self._last_extract_at >= self.extract_every_n:
            await self._extract_and_update_long()
            self._last_extract_at = current_round

        # ── 5. 异步写入 Milvus（完整原文分段，不压缩）──
        if self._milvus is not None:
            try:
                await self._milvus.add_turn_async(
                    user_id=self.user_id,
                    conv_id=self.conv_id,
                    user_query=user_query,
                    reply=assistant_reply,
                    round_id=next_round,
                )
            except Exception as e:
                logger.warning(f"ContextManager Milvus write failed: {e}")

    # ──────────────────────────────────────────
    # Redis 刷写
    # ──────────────────────────────────────────

    async def _append_turn_to_redis(
        self,
        user_query:      str,
        assistant_reply: str,
        round_id:        int,
        user_msg_id:     int,
        asst_msg_id:     int,
    ) -> None:
        """
        仅追加本轮新增的 2 条消息到 Redis LIST 尾部（RPUSH），不删除历史。

        适用场景：正常新增一轮，无压缩发生。
        原子性：pipeline 保证 2 条同时写入，同时刷新 TTL。
        """
        user_rec = json.dumps({
            "role":       "user",
            "content":    user_query,
            "round_id":   round_id,
            "compressed": False,
            "summary":    "",
            "message_id": user_msg_id,
        }, ensure_ascii=False)
        asst_rec = json.dumps({
            "role":       "assistant",
            "content":    assistant_reply,
            "round_id":   round_id,
            "compressed": False,
            "summary":    "",
            "message_id": asst_msg_id,
        }, ensure_ascii=False)

        pipeline = self.redis.pipeline()
        pipeline.rpush(self._k_msgs, user_rec, asst_rec)
        pipeline.expire(self._k_msgs, self.ttl)
        await pipeline.execute()

    async def _patch_compressed_to_redis(self, indexed_msgs) -> None:
        """
        用 LSET 原地更新 Redis LIST 中被压缩的消息（仅修改 2 条），避免全量覆盖。
        indexed_msgs: [(list_index, _Msg), ...]，由 compress_oldest_async 返回。
        """
        pipeline = self.redis.pipeline()
        for idx, msg in indexed_msgs:
            pipeline.lset(self._k_msgs, idx, json.dumps({
                "role":       msg.role,
                "content":    msg.content,
                "round_id":   msg.round_id,
                "compressed": msg.compressed,
                "summary":    msg.summary,
                "message_id": msg.message_id,
            }, ensure_ascii=False))
        await pipeline.execute()

    async def _flush_to_redis(self) -> None:
        """
        全量覆盖写入 Redis LIST（DELETE + 全量 RPUSH）。

        仅在发生压缩（旧消息 content/compressed 字段被修改）时调用，
        正常新增走 _append_turn_to_redis 以减少写放大。
        使用 pipeline 的 MULTI/EXEC 保证原子性，消除 DELETE 后空窗期。
        """
        records = [
            {
                "role":       msg.role,
                "content":    msg.content,
                "round_id":   msg.round_id,
                "compressed": msg.compressed,
                "summary":    msg.summary,
                "message_id": msg.message_id,
            }
            for msg in self._short.get_messages()
        ]
        await self._flush_short_to_redis(records)

    async def _flush_short_to_redis(self, records: List[Dict]) -> None:
        """
        全量覆盖写入 Redis LIST（原子 pipeline）。
        DELETE + RPUSH 在同一 pipeline 内执行，避免空窗期。
        """
        pipeline = self.redis.pipeline()
        pipeline.delete(self._k_msgs)
        if records:
            # pipeline.rpush 支持一次传多个值，减少 round-trip
            pipeline.rpush(
                self._k_msgs,
                *[json.dumps(rec, ensure_ascii=False) for rec in records],
            )
            pipeline.expire(self._k_msgs, self.ttl)
        await pipeline.execute()

    # ──────────────────────────────────────────
    # 长期记忆抽取 + 压缩 + Redis 同步
    # ──────────────────────────────────────────

    async def _extract_and_update_long(self) -> None:
        history = self._short.get_history()
        current_round = self._short.round_count
        patch = await self._long.extract_from_short_term_async(history, round_id=current_round)
        if not patch:
            return

        await self._long.merge_patch_async(patch, round_id=current_round)
        logger.info(
            f"ContextManager[user={self.user_id}]: "
            f"extracted patch keys={list(patch.keys())}"
        )

        if self._long.needs_compress():
            await self._long.compress_async()
            logger.info(f"ContextManager[user={self.user_id}]: long-term compressed")

        text = self._long.get_text()

        # 同步更新 Redis（存 JSON 字符串，方便 set_text() 回填）
        await self.redis.set(self._k_long, text, ex=self.ttl)

        # 异步 Celery 持久化到 DB（传结构化 dict，写入 JSONField）
        self._dispatch_persist_long(self._long.get_data())

    # ──────────────────────────────────────────
    # DB 读操作（仅 load 时使用，走 sync_to_async）
    # ──────────────────────────────────────────

    async def _fetch_messages_from_db(self) -> List[Dict[str, Any]]:
        """从 DB Message 表加载该会话消息（Redis miss 时回源）。"""
        try:
            from asgiref.sync import sync_to_async
            from api.models import Message as DBMessage

            @sync_to_async
            def _query() -> List[Dict]:
                qs = (
                    DBMessage.objects
                    .filter(conversation_id=self.conv_id)
                    .order_by("timestamp")
                    .values(
                        "id", "role", "content", "round_id",
                        "is_compressed", "summary",
                    )
                )
                return [
                    {
                        "message_id": row["id"],
                        "role":       row["role"],
                        "content":    row["summary"] if row["is_compressed"] else row["content"],
                        "round_id":   row["round_id"],
                        "compressed": row["is_compressed"],
                        "summary":    row["summary"],
                    }
                    for row in qs
                ]

            return await _query()
        except Exception as e:
            logger.warning(f"ContextManager._fetch_messages_from_db: {e}")
            return []

    async def _fetch_long_from_db(self) -> str:
        """从 DB LongTermMemory 取该用户最新快照（Redis miss 时回源）。

        DB 存储为 JSONField（dict），回填前序列化为 JSON 字符串供 set_text() 使用。
        """
        try:
            from asgiref.sync import sync_to_async
            from api.models import LongTermMemory as DBLong

            @sync_to_async
            def _query() -> str:
                row = (
                    DBLong.objects
                    .filter(user_id=self.user_id)
                    .order_by("-created_at")
                    .values("text")
                    .first()
                )
                if not row:
                    return ""
                val = row["text"]
                # JSONField 读出的是 dict，序列化为字符串供 set_text() 反序列化
                if isinstance(val, dict):
                    import json as _json
                    return _json.dumps(val, ensure_ascii=False)
                return val if val else ""

            return await _query()
        except Exception as e:
            logger.warning(f"ContextManager._fetch_long_from_db: {e}")
            return ""

    # ──────────────────────────────────────────
    # Celery 任务调度（DB 写操作全部异步）
    # ──────────────────────────────────────────

    def _dispatch_update_compressed(self, compressed_rounds: List[Dict]) -> None:
        """触发 Celery 任务异步回写 DB 压缩状态。"""
        try:
            from agent.tasks import update_compressed_messages
            update_compressed_messages.delay(compressed_rounds)
            logger.debug(
                f"ContextManager[conv={self.conv_id}]: "
                f"dispatched update_compressed_messages ({len(compressed_rounds)} rounds)"
            )
        except Exception as e:
            logger.warning(f"ContextManager._dispatch_update_compressed: {e}")

    def _dispatch_persist_long(self, data: dict) -> None:
        """触发 Celery 任务异步持久化长期记忆到 DB（传结构化 dict）。"""
        try:
            from agent.tasks import persist_long_term_memory
            persist_long_term_memory.delay(
                user_id=self.user_id,
                data=data,
            )
            logger.debug(
                f"ContextManager[user={self.user_id}]: "
                f"dispatched persist_long_term_memory (keys={list(data.keys())})"
            )
        except Exception as e:
            logger.warning(f"ContextManager._dispatch_persist_long: {e}")

    # ──────────────────────────────────────────
    # 资源释放
    # ──────────────────────────────────────────

    async def close(self) -> None:
        try:
            await self.redis.aclose()
        except Exception:
            pass

    async def __aenter__(self) -> "ContextManager":
        await self.load()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()


# ──────────────────────────────────────────────
# 工厂函数
# ──────────────────────────────────────────────

async def build_context_manager(
    user_id:   int,
    conv_id:   int,
    llm:       Optional[Any] = None,
    redis_url: Optional[str] = None,
) -> ContextManager:
    """
    工厂函数：创建并已 load 的 ContextManager（含 Milvus 可选集成）。

    Milvus 由 config.milvus_enabled 控制，False 时优雅跳过，不影响其他功能。

    Args:
        user_id:   Django User.id
        conv_id:   Conversation.id
        llm:       LLM 实例（摘要 / 长期记忆抽取）
        redis_url: redis://... ；None 时从 config 读取

    Returns:
        已调用 load() 的 ContextManager 实例
    """
    from config import config as cfg

    if redis_url is None:
        auth = f":{cfg.redis_password}@" if cfg.redis_password else ""
        redis_url = f"redis://{auth}{cfg.redis_host}:{cfg.redis_port}/{cfg.redis_db}"

    redis = aioredis.from_url(
        redis_url, encoding="utf-8", decode_responses=True
    )

    # ── 可选：初始化 Milvus 管理器 ──
    milvus_manager = None
    if getattr(cfg, "milvus_enabled", False):
        try:
            from agent.milvus_memory import MilvusMemoryManager
            milvus_manager = MilvusMemoryManager(
                host=getattr(cfg, "milvus_host", "localhost"),
                port=getattr(cfg, "milvus_port", 19530),
                collection_name=getattr(cfg, "milvus_collection", "agent_memory_store"),
                embedding_model=getattr(cfg, "milvus_embedding_model", "BAAI/bge-large-zh-v1.5"),
                hybrid_search=getattr(cfg, "milvus_hybrid_search", True),
            )
            logger.info("ContextManager: Milvus enabled")
        except Exception as e:
            logger.warning(f"ContextManager: Milvus init failed, disabled → {e}")
            milvus_manager = None

    ctx = ContextManager(
        redis=redis,
        user_id=user_id,
        conv_id=conv_id,
        llm=llm,
        short_max_rounds=cfg.short_term_max_turns,
        short_full_detail=getattr(cfg, "short_full_detail_rounds", 5),
        long_max_chars=getattr(cfg, "long_term_max_chars", 2000),
        ttl=cfg.redis_ttl,
        extract_every_n=getattr(cfg, "long_term_extract_every_n", _EXTRACT_EVERY_N_ROUNDS),
        milvus_manager=milvus_manager,
        milvus_top_k=getattr(cfg, "milvus_top_k", 3),
    )
    await ctx.load()
    return ctx
