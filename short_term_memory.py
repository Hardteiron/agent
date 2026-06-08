"""
短期记忆（Short-Term Memory）
==============================
职责：
  - 维护单个会话（conversation）维度的对话窗口
  - 按轮次（round）存储 user/assistant 消息对
  - 超出 max_rounds 时保留最新 full_detail_rounds 轮完整内容，
    更早的轮次压缩为单行摘要

存储位置：
  - 纯内存（Python list），由 ContextManager 负责从 Redis/DB 加载并持久化
  - 本文件不依赖 Redis 或 Django ORM，便于单独测试

对外接口（供 ContextManager 调用）：
  get_history() -> List[Dict]       获取格式化历史（{"role", "content"}）
  get_messages() -> List[_Msg]      获取原始消息列表（含 round_id / compressed）
  load_from_records(records)        从外部记录（DB / Redis）批量回填
  get_compressed_rounds() -> List[Dict]  获取压缩轮摘要，供 Celery 任务同步 DB
  clear()                           清空
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 内部数据结构
# ─────────────────────────────────────────────

@dataclass
class _Msg:
    role: str           # "user" | "assistant"
    content: str
    round_id: int       # 从 1 开始，同一轮 user+assistant 共享
    compressed: bool = False
    summary: str = ""   # 压缩后的摘要文本
    message_id: int = 0 # 对应 DB Message.id（0 表示尚未持久化）


# ─────────────────────────────────────────────
# ShortTermMemory
# ─────────────────────────────────────────────

class ShortTermMemory:
    """
    单会话短期记忆，滚动窗口 + 异步压缩策略。

    Args:
        max_rounds:         最多保留的总轮数（超出后压缩最早轮）
        full_detail_rounds: 最近 N 轮保持完整内容，更早轮压缩为摘要
        llm:                LLM 实例，用于生成压缩摘要；为 None 时截断兜底
    """

    def __init__(
        self,
        max_rounds: int = 20,
        full_detail_rounds: int = 5,
        llm: Optional[Any] = None,
    ) -> None:
        self.max_rounds = max_rounds
        self.full_detail_rounds = full_detail_rounds
        self.llm = llm

        self._messages: List[_Msg] = []
        self._round_id: int = 0

    # ─── 读 ───────────────────────────────────

    def get_history(self) -> List[Dict[str, str]]:
        """
        返回格式化对话历史，供 prompt 注入。
        压缩轮以摘要行呈现，保留轮以原文呈现。
        格式：[{"role": "user"/"assistant"/"system", "content": "..."}]
        """
        out: List[Dict[str, str]] = []
        for msg in self._messages:
            if msg.compressed:
                # 每压缩轮只输出一条摘要（取 user 行，避免重复）
                if msg.role == "user" and msg.summary:
                    out.append({"role": "system", "content": f"[历史摘要] {msg.summary}"})
            else:
                out.append({"role": msg.role, "content": msg.content})
        return out

    def get_messages(self) -> List[_Msg]:
        """返回原始消息列表（供 ContextManager 序列化到 Redis）。"""
        return list(self._messages)

    @property
    def round_count(self) -> int:
        """当前已记录的轮次数。"""
        return self._round_id

    # ─── 批量回填（从 Redis/DB 加载） ─────────

    def load_from_records(self, records: List[Dict[str, Any]]) -> None:
        """
        从外部序列化记录批量回填（幂等，会先清空）。

        records 格式：
            [{"role": "user"|"assistant", "content": "...",
              "round_id": int, "compressed": bool, "summary": "...",
              "message_id": int}, ...]
        """
        self.clear()
        for r in records:
            msg = _Msg(
                role=r.get("role", "user"),
                content=r.get("content", ""),
                round_id=r.get("round_id", 0),
                compressed=r.get("compressed", False),
                summary=r.get("summary", ""),
                message_id=r.get("message_id", 0),
            )
            self._messages.append(msg)
            if msg.round_id > self._round_id:
                self._round_id = msg.round_id

    def clear(self) -> None:
        self._messages.clear()
        self._round_id = 0

    # ─── 压缩 ─────────────────────────────────

    async def compress_oldest_async(self) -> tuple[List[int], List[tuple[int, "_Msg"]]]:
        """
        逆序遍历找 oldest_round_id：
          - 当前消息未压缩且下一条已压缩（或当前是第一条）→ 该轮即为最早全量轮
        """
        oldest_round_id = None
        msgs = self._messages
        for i in range(len(msgs) - 1, -1, -1):
            if not msgs[i].compressed:
                if i == 0 or msgs[i - 1].compressed:
                    oldest_round_id = msgs[i].round_id
                    break
        if oldest_round_id is None:
            return [], []
        return await self._compress_round_async(oldest_round_id)

    async def _compress_round_async(self, round_id: int) -> tuple[List[int], List[tuple[int, "_Msg"]]]:
        user_msg = asst_msg = None
        user_idx = asst_idx = -1
        for i, m in enumerate(self._messages):
            if m.round_id != round_id:
                continue
            if m.role == "user":
                user_msg, user_idx = m, i
            else:
                asst_msg, asst_idx = m, i

        if user_msg is None:
            return [], []

        summary = await self._make_summary_async(
            user_msg.content, asst_msg.content if asst_msg else ""
        )

        user_msg.compressed = True
        user_msg.summary = summary

        indexed_msgs = [(user_idx, user_msg)]
        compressed_ids = [user_msg.message_id] if user_msg.message_id else []

        if asst_msg is not None:
            asst_msg.compressed = True
            asst_msg.summary = summary
            indexed_msgs.append((asst_idx, asst_msg))
            if asst_msg.message_id:
                compressed_ids.append(asst_msg.message_id)

        logger.debug(f"ShortTermMemory: compressed round {round_id} → '{summary[:40]}…'")
        return compressed_ids, indexed_msgs

    def get_compressed_rounds(self) -> List[Dict]:
        """
        返回所有已压缩轮次的信息（供 Celery 任务批量同步 DB）。
        格式：[{"message_ids": [...], "summary": "..."}]
        """
        groups: Dict[int, Dict] = {}
        for m in self._messages:
            if not m.compressed:
                continue
            rid = m.round_id
            if rid not in groups:
                groups[rid] = {"message_ids": [], "summary": m.summary}
            if m.message_id:
                groups[rid]["message_ids"].append(m.message_id)
        return list(groups.values())

    async def _make_summary_async(self, user_text: str, asst_text: str) -> str:
        """异步调用 LLM 生成摘要，失败时截断兜底。"""
        if self.llm:
            try:
                prompt = (
                    "将下面一轮对话压缩为一句话摘要（限 30 字内）：\n"
                    f"问：{user_text[:150]}\n"
                    f"答：{asst_text[:150]}"
                )
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, partial(self.llm.complete, prompt)
                )
                text = result if isinstance(result, str) else str(result)
                return text.strip()[:80]
            except Exception as e:
                logger.warning(f"ShortTermMemory summary LLM failed: {e}")
        # 截断兜底
        return f"Q:{user_text[:20]}… A:{asst_text[:20]}…"
