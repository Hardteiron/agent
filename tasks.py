"""
agent/tasks.py — 记忆系统 Celery 异步任务
==========================================
所有 PostgreSQL 写操作（Message 持久化、压缩状态回写、长期记忆快照）
都在此处以 Celery 任务形式异步执行，不阻塞 SSE 响应流。

任务列表：
  persist_messages            保存一轮 user + assistant Message 到 DB
  update_compressed_messages  批量更新 Message.is_compressed / summary
  persist_long_term_memory    保存长期记忆文本快照到 DB LongTermMemory 表
  persist_milvus_turn         将一轮对话写入 Milvus 向量数据库（仅 milvus_enabled=True 时生效）

调用约定：
  所有任务通过 .delay(**kwargs) 触发，返回 AsyncResult（通常忽略）。
  任务失败自动重试 3 次，间隔 30 秒（指数退避）。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from celery import shared_task

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 1. 持久化一轮对话消息（user + assistant）
# ──────────────────────────────────────────────

@shared_task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=30,
    max_retries=3,
    name="agent.tasks.persist_messages",
)
def persist_messages(
    self,
    conv_id: int,
    user_query: str,
    assistant_reply: str,
    round_id: int,
    user_role: str = "user",
    asst_role: str = "assistant",
    asst_type: str = "text",
    user_msg_id: int = 0,
    asst_msg_id: int = 0,
) -> Dict[str, int]:
    """
    将一轮对话的 user / assistant 消息写入 DB Message 表。

    如果调用方已提前创建了 Message 行（user_msg_id / asst_msg_id > 0），
    则跳过创建，直接返回已有 ID（幂等）。

    Returns:
        {"user_msg_id": int, "asst_msg_id": int}
    """
    import django
    django.setup()  # Worker 进程内确保 Django 初始化

    from api.models import Conversation, Message
    from django.utils import timezone

    try:
        conversation = Conversation.objects.get(id=conv_id)
    except Conversation.DoesNotExist:
        logger.error(f"persist_messages: Conversation {conv_id} not found")
        raise

    result: Dict[str, int] = {}

    # ── user 消息 ──
    if user_msg_id:
        result["user_msg_id"] = user_msg_id
        user_message = Message.objects.filter(id=user_msg_id).first()
    else:
        user_message = Message.objects.create(
            conversation=conversation,
            role=user_role,
            content=user_query,
            round_id=round_id,
            timestamp=timezone.now(),
        )
        result["user_msg_id"] = user_message.id
        logger.debug(f"persist_messages: created user Message id={user_message.id}")

    # ── assistant 消息 ──
    if asst_msg_id:
        result["asst_msg_id"] = asst_msg_id
    else:
        asst_message = Message.objects.create(
            conversation=conversation,
            role=asst_role,
            content=assistant_reply,
            type=asst_type,
            round_id=round_id,
            reply_to=user_message,
            timestamp=timezone.now(),
        )
        result["asst_msg_id"] = asst_message.id
        logger.debug(f"persist_messages: created asst Message id={asst_message.id}")

    return result


# ──────────────────────────────────────────────
# 2. 批量回写压缩状态到 Message 表
# ──────────────────────────────────────────────

@shared_task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=30,
    max_retries=3,
    name="agent.tasks.update_compressed_messages",
)
def update_compressed_messages(
    self,
    compressed_rounds: List[Dict[str, Any]],
) -> int:
    """
    批量 UPDATE Message 表中被压缩轮次的 is_compressed / summary 字段。

    Args:
        compressed_rounds: [{"message_ids": [int, ...], "summary": "..."}, ...]

    Returns:
        更新的总行数
    """
    import django
    django.setup()

    from api.models import Message

    total = 0
    for group in compressed_rounds:
        ids: List[int] = group.get("message_ids", [])
        summary: str = group.get("summary", "")
        if ids:
            updated = Message.objects.filter(id__in=ids).update(
                is_compressed=True,
                summary=summary,
            )
            total += updated
            logger.debug(
                f"update_compressed_messages: updated {updated} rows "
                f"(ids={ids[:5]}{'…' if len(ids) > 5 else ''})"
            )
    return total


# ──────────────────────────────────────────────
# 3. 持久化长期记忆快照
# ──────────────────────────────────────────────

@shared_task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=30,
    max_retries=3,
    name="agent.tasks.persist_long_term_memory",
)
def persist_long_term_memory(
    self,
    user_id: int,
    data: Dict[str, Any],
) -> int:
    """
    将长期记忆最新结构化快照写入 DB LongTermMemory 表（upsert 语义：
    先尝试更新同用户最新一条，若不存在则创建）。

    Args:
        user_id: Django User.id
        data:    长期记忆结构化字典（与 LongTermMemory.get_data() 返回值一致）

    Returns:
        DB LongTermMemory.id
    """
    import django
    django.setup()

    import json as _json
    from api.models import LongTermMemory as DBLong
    from django.contrib.auth.models import User

    try:
        user = User.objects.get(id=user_id)
    except User.DoesNotExist:
        logger.error(f"persist_long_term_memory: User {user_id} not found")
        raise

    # token_count 按 JSON 序列化长度计算
    token_count = len(_json.dumps(data, ensure_ascii=False))

    # upsert：更新最新一条快照（避免无限追加行）
    latest = DBLong.objects.filter(user=user).order_by("-created_at").first()
    if latest:
        latest.text = data
        latest.token_count = token_count
        latest.save(update_fields=["text", "token_count"])
        logger.info(
            f"persist_long_term_memory: updated LongTermMemory id={latest.id} "
            f"for user={user_id} ({token_count} chars)"
        )
        return latest.id
    else:
        obj = DBLong.objects.create(
            user=user,
            text=data,
            token_count=token_count,
        )
        logger.info(
            f"persist_long_term_memory: created LongTermMemory id={obj.id} "
            f"for user={user_id} ({token_count} chars)"
        )
        return obj.id


# ──────────────────────────────────────────────
# 4. 写入 Milvus 向量数据库（完整原文分段写入）
# ──────────────────────────────────────────────

@shared_task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=30,
    max_retries=3,
    name="agent.tasks.persist_milvus_turn",
)
def persist_milvus_turn(
    self,
    user_id:    int,
    conv_id:    int,
    user_query: str,
    reply:      str,
    round_id:   int = 0,
    summary:    str = "",   # 保留参数，向后兼容，不再影响 chunk 内容
) -> List[int]:
    """
    将一轮对话的完整原文分段写入 Milvus（供 Celery Worker 调用）。

    分段策略（与 add_turn_async 一致）：
      - 完整 reply 按 _CHUNK_TOKEN_LIMIT token 切分，每段独立写入
      - 每段 metadata 带 chunk_seq / total_chunks，支持精确按序拼回
      - 不再用 summary 替代 reply，保留原文用于精确召回

    Args:
        user_id:    Django User.id
        conv_id:    Conversation.id
        user_query: 用户输入原文
        reply:      LLM 完整回复（存原文）
        round_id:   轮次编号
        summary:    保留参数（不再使用）

    Returns:
        写入的 Milvus ID 列表；Milvus 不可用时返回 []
    """
    from config import config as cfg

    if not getattr(cfg, "milvus_enabled", False):
        return []

    try:
        import time
        from agent.milvus_memory import (
            MilvusMemoryManager,
            _CHUNK_TOKEN_LIMIT,
            _CHUNK_OVERLAP_RATE,
            _split_with_overlap,
            _estimate_tokens,
            _char_token,
        )

        mgr = MilvusMemoryManager(
            host=getattr(cfg, "milvus_host", "localhost"),
            port=getattr(cfg, "milvus_port", 19530),
            collection_name=getattr(cfg, "milvus_collection", "agent_memory_store"),
            embedding_model=getattr(cfg, "milvus_embedding_model", "BAAI/bge-large-zh-v1.5"),
            hybrid_search=getattr(cfg, "milvus_hybrid_search", True),
            auto_create=False,
        )

        # ── 与 add_turn_async 完全一致的分段逻辑 ──
        full_text = f"问：{user_query}\n答：{reply}"
        raw_segs = _split_with_overlap(
            full_text,
            token_limit=_CHUNK_TOKEN_LIMIT,
            overlap_rate=_CHUNK_OVERLAP_RATE,
        )

        # 计算 chunk_start_offset（与 add_turn_async 相同）
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
        items = [
            {
                "text": seg,
                "meta": {
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
            }
            for seq, seg in enumerate(raw_segs)
        ]

        ids = mgr.add_batch(items)
        logger.info(
            f"persist_milvus_turn: user={user_id} conv={conv_id} "
            f"round={round_id} → {total} chunk(s), ids={ids}"
        )
        mgr.close()
        return ids
    except Exception as e:
        logger.warning(f"persist_milvus_turn failed: {e}")
        raise   # 触发 Celery 自动重试
