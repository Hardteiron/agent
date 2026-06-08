"""
长期记忆（Long-Term Memory）—— JSON 结构化存储版
=================================================
职责：
  - 维护用户级别（跨会话）的长期记忆，以结构化 JSON 存储
  - 从短期记忆历史中异步抽取重要事实，按字段合并写入
  - 超出字符上限时，按"最近优先"策略压缩/淘汰历史信息
  - 与 Redis / DB 交互时统一用 JSON 字符串序列化

JSON Schema（memory_data 字典）：
  {
    "schema_version": "1.0",
    "updated_at":     str,        # ISO8601 时间戳
    "user_profile": {
      "name": str, "role": str, "department": str,
      "language_preference": str, "expertise_level": str, "notes": str
    },
    "task_preferences": {
      "preferred_output_format": str, "preferred_chart_type": str,
      "default_eval_dimension": list[str], "auto_download": bool, "notes": str
    },
    "domain_knowledge": [           # 按 topic 去重合并，保留最近 N 条
      {"topic": str, "content": str, "confidence": str, "last_seen_round": int}
    ],
    "recurring_tasks": [            # 按 task_name 去重，低频旧条目优先淘汰
      {"task_name": str, "frequency": str, "typical_instruction": str,
       "last_seen_round": int}
    ],
    "corrections": [                # 用户明确纠正 AI 的行为，保留最新 N 条
      {"wrong": str, "right": str, "round_id": int}
    ],
    "session_continuity": {
      "last_active_dataset": str, "last_active_columns": list[str],
      "pending_tasks": str
    }
  }

对外接口（供 ContextManager 调用）：
  get_text() -> str                   序列化为 JSON 字符串（供 Redis/DB 存储 + prompt 注入）
  get_data() -> dict                  返回原始字典
  set_text(text: str)                 从 Redis/DB 回填时反序列化
  is_empty() -> bool
  needs_compress() -> bool
  async merge_patch_async(patch: dict, round_id: int)  合并 LLM 抽取结果
  async compress_async() -> str       JSON 结构化压缩/淘汰，超限时调用
  async extract_from_short_term_async(turns, round_id) -> dict  从对话抽取
"""
from __future__ import annotations

import asyncio
import json
import logging
from copy import deepcopy
from datetime import datetime, timezone
from functools import partial
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_MAX_CHARS        = 3000   # JSON 序列化后的字符上限
DEFAULT_EXTRACT_ROUNDS   = 5      # 每次取最近 N 轮对话做抽取
MAX_DOMAIN_KNOWLEDGE     = 20     # domain_knowledge 最大条目数
MAX_RECURRING_TASKS      = 10     # recurring_tasks 最大条目数
MAX_CORRECTIONS          = 10     # corrections 最大条目数
STALE_ROUNDS_THRESHOLD   = 30     # 距今超过此轮次的低频任务视为过期


# ─────────────────────────────────────────────
# 提示词常量
# ─────────────────────────────────────────────

_EXTRACT_PROMPT_TEMPLATE = """\
你是一个记忆抽取助手。请从以下最近对话中，抽取值得长期保存的信息，\
以严格的 JSON 格式输出，只输出 JSON，不要任何解释或 Markdown 代码块。

【当前长期记忆（已有内容，不要重复抽取相同信息）】
{existing_memory}

【最近对话（当前轮次：{current_round}）】
{dialogue}

【输出要求】
返回一个 JSON 对象，只包含本次对话中**新增或需要修改**的字段，其余字段省略。
没有任何新信息时，输出空对象 {{}}。

可包含以下字段（均可选）：
{{
  "user_profile": {{
    "name": "...",
    "role": "...",
    "department": "...",
    "language_preference": "...",
    "expertise_level": "初级/中级/高级",
    "notes": "..."
  }},
  "task_preferences": {{
    "preferred_output_format": "...",
    "preferred_chart_type": "...",
    "default_eval_dimension": ["..."],
    "auto_download": true,
    "notes": "..."
  }},
  "domain_knowledge": [
    {{
      "topic": "...",
      "content": "...",
      "confidence": "high/medium/low",
      "last_seen_round": {current_round}
    }}
  ],
  "recurring_tasks": [
    {{
      "task_name": "...",
      "frequency": "高频/中频/低频",
      "typical_instruction": "...",
      "last_seen_round": {current_round}
    }}
  ],
  "corrections": [
    {{
      "wrong": "...",
      "right": "...",
      "round_id": {current_round}
    }}
  ],
  "session_continuity": {{
    "last_active_dataset": "...",
    "last_active_columns": ["..."],
    "pending_tasks": "..."
  }}
}}

注意：
- domain_knowledge 和 recurring_tasks 中若已有相同 topic/task_name，只输出需要更新的条目
- corrections 只在用户明确纠正 AI 行为时输出
- user_profile / task_preferences 只在对话中有明确信息时输出
- 所有字符串字段内容精简，每条不超过 100 字
"""

_COMPRESS_PROMPT_TEMPLATE = """\
你是一个记忆压缩助手。以下用户长期记忆的 JSON 条目过多，已超出存储上限。
请按规则精简后，输出完整的长期记忆 JSON，只输出 JSON，不要任何解释或 Markdown 代码块。

【当前长期记忆】
{current_memory_json}

【压缩规则（按优先级执行）】
1. domain_knowledge：
   - 保留 last_seen_round 最大（最近）的条目
   - 超出 {max_knowledge} 条时删除 last_seen_round 最小的
   - confidence=low 且 last_seen_round 较旧的优先删除
   - 含义相近的条目合并为一条，content 取并集精简版（不超过 80 字）
2. recurring_tasks：
   - 保留 frequency=高频 的全部
   - frequency=低频 且 last_seen_round < {stale_round_threshold} 的删除
   - 超出 {max_tasks} 条时删除最旧的中频/低频条目
3. corrections：
   - 保留 round_id 最大的 {max_corrections} 条，其余删除
4. session_continuity.pending_tasks：
   - 超过 100 字则截断至 100 字
5. user_profile / task_preferences：
   - 不压缩，原样保留
6. 压缩后确保 JSON 序列化长度不超过 {max_chars} 字符

输出完整的长期记忆 JSON，包含所有未修改的字段，确保 JSON 语法合法。
"""


# ─────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _empty_data() -> Dict[str, Any]:
    """返回空白的记忆字典骨架。"""
    return {
        "schema_version": "1.0",
        "updated_at": _now_iso(),
        "user_profile": {},
        "task_preferences": {},
        "domain_knowledge": [],
        "recurring_tasks": [],
        "corrections": [],
        "session_continuity": {},
    }


def _safe_json_loads(text: str) -> Optional[Dict]:
    """安全解析 JSON，失败返回 None。"""
    if not text:
        return None
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _merge_list_by_key(
    existing: List[Dict],
    incoming: List[Dict],
    key: str,
) -> List[Dict]:
    """
    按 key 字段去重合并两个列表。
    incoming 中与 existing 同 key 的条目覆盖更新；新 key 追加。
    """
    merged = {item[key]: item for item in existing if key in item}
    for item in (incoming or []):
        if key in item:
            merged[item[key]] = item   # 覆盖/新增
    return list(merged.values())


def _merge_patch(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    """
    将 patch 合并到 base，返回新字典（不修改原对象）。

    合并规则：
      - user_profile / task_preferences / session_continuity：dict 字段级 update
      - domain_knowledge：按 topic 去重合并
      - recurring_tasks：按 task_name 去重合并
      - corrections：追加（不去重）
    """
    result = deepcopy(base)
    result["updated_at"] = _now_iso()

    for flat_key in ("user_profile", "task_preferences", "session_continuity"):
        if flat_key in patch and isinstance(patch[flat_key], dict):
            if not isinstance(result.get(flat_key), dict):
                result[flat_key] = {}
            result[flat_key].update(
                {k: v for k, v in patch[flat_key].items() if v not in (None, "", [])}
            )

    if "domain_knowledge" in patch:
        result["domain_knowledge"] = _merge_list_by_key(
            result.get("domain_knowledge", []),
            patch["domain_knowledge"],
            "topic",
        )

    if "recurring_tasks" in patch:
        result["recurring_tasks"] = _merge_list_by_key(
            result.get("recurring_tasks", []),
            patch["recurring_tasks"],
            "task_name",
        )

    if "corrections" in patch and patch["corrections"]:
        existing = result.get("corrections", [])
        result["corrections"] = existing + patch["corrections"]

    return result


# ─────────────────────────────────────────────
# LongTermMemory
# ─────────────────────────────────────────────

class LongTermMemory:
    """
    用户级 JSON 结构化长期记忆。

    Args:
        max_chars: JSON 序列化后超出此字符数时触发压缩（默认 3000）
        llm:       LLM 实例（用于抽取和压缩）；为 None 时静默降级
    """

    def __init__(self, max_chars: int = DEFAULT_MAX_CHARS, llm: Optional[Any] = None) -> None:
        self.max_chars = max_chars
        self.llm = llm
        self._data: Dict[str, Any] = _empty_data()

    # ─── 读 / 写 ──────────────────────────────────────────────────────────

    def get_text(self) -> str:
        """序列化为 JSON 字符串，用于 Redis/DB 存储或注入 prompt。"""
        return json.dumps(self._data, ensure_ascii=False, indent=2)

    def get_data(self) -> Dict[str, Any]:
        """返回原始字典（只读副本）。"""
        return deepcopy(self._data)

    def set_text(self, text: str) -> None:
        """从 Redis 或 DB 回填时调用（反序列化 JSON）。"""
        parsed = _safe_json_loads(text)
        if parsed:
            # 兼容旧版纯文本：若解析成功但缺少 schema_version，视为旧格式丢弃
            if "schema_version" in parsed:
                self._data = parsed
            else:
                # 旧版纯文本迁移：写入 domain_knowledge 兜底保留
                self._data = _empty_data()
                self._data["domain_knowledge"].append({
                    "topic": "历史记忆（迁移）",
                    "content": text[:200],
                    "confidence": "low",
                    "last_seen_round": 0,
                })
        else:
            # 纯文本兜底
            self._data = _empty_data()
            if text and text.strip():
                self._data["domain_knowledge"].append({
                    "topic": "历史记忆（迁移）",
                    "content": text[:200],
                    "confidence": "low",
                    "last_seen_round": 0,
                })

    def is_empty(self) -> bool:
        d = self._data
        return (
            not d.get("user_profile")
            and not d.get("task_preferences")
            and not d.get("domain_knowledge")
            and not d.get("recurring_tasks")
            and not d.get("corrections")
            and not d.get("session_continuity")
        )

    def needs_compress(self) -> bool:
        return len(self.get_text()) > self.max_chars

    # ─── 合并 patch ───────────────────────────────────────────────────────

    async def merge_patch_async(self, patch: Dict[str, Any], round_id: int = 0) -> None:
        """
        将 LLM 抽取的 patch 字典合并到当前记忆。
        不调用 LLM，纯内存操作，可在 async 上下文直接 await。
        """
        if not patch:
            return
        self._data = _merge_patch(self._data, patch)
        logger.debug(f"LongTermMemory: merged patch (round={round_id}), keys={list(patch.keys())}")

    # ─── 压缩（LLM 驱动，超限时调用）────────────────────────────────────

    async def compress_async(self) -> str:
        """
        异步压缩长期记忆：
          1. 优先让 LLM 按语义规则精简
          2. LLM 失败时降级到本地硬截断
        """
        current_json = self.get_text()

        if self.llm:
            try:
                current_round = max(
                    (item.get("last_seen_round", 0) for item in self._data.get("domain_knowledge", [])),
                    default=0,
                )
                stale_threshold = max(0, current_round - STALE_ROUNDS_THRESHOLD)
                prompt = _COMPRESS_PROMPT_TEMPLATE.format(
                    current_memory_json=current_json,
                    max_knowledge=MAX_DOMAIN_KNOWLEDGE,
                    stale_round_threshold=stale_threshold,
                    max_tasks=MAX_RECURRING_TASKS,
                    max_corrections=MAX_CORRECTIONS,
                    max_chars=self.max_chars,
                )
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, partial(self.llm.complete, prompt)
                )
                result_str = (result if isinstance(result, str) else str(result)).strip()
                parsed = _safe_json_loads(result_str)
                if parsed and "schema_version" in parsed:
                    self._data = parsed
                    self._data["updated_at"] = _now_iso()
                    logger.info(
                        f"LongTermMemory: LLM compressed to {len(self.get_text())} chars"
                    )
                    return self.get_text()
                logger.warning("LongTermMemory: LLM compress returned invalid JSON, falling back")
            except Exception as e:
                logger.warning(f"LongTermMemory compress LLM failed: {e}")

        # ── 本地降级：硬截断各列表 ────────────────────────────────────────
        self._local_compress()
        logger.info(
            f"LongTermMemory: local compressed to {len(self.get_text())} chars"
        )
        return self.get_text()

    def _local_compress(self) -> None:
        """本地硬截断：各列表按最近优先保留上限条目。"""
        dk = self._data.get("domain_knowledge", [])
        dk.sort(key=lambda x: x.get("last_seen_round", 0), reverse=True)
        self._data["domain_knowledge"] = dk[:MAX_DOMAIN_KNOWLEDGE]

        rt = self._data.get("recurring_tasks", [])
        rt.sort(key=lambda x: x.get("last_seen_round", 0), reverse=True)
        self._data["recurring_tasks"] = rt[:MAX_RECURRING_TASKS]

        co = self._data.get("corrections", [])
        co.sort(key=lambda x: x.get("round_id", 0), reverse=True)
        self._data["corrections"] = co[:MAX_CORRECTIONS]

        self._data["updated_at"] = _now_iso()

    # ─── 抽取 ────────────────────────────────────────────────────────────

    async def extract_from_short_term_async(
        self,
        turns: List[dict],
        round_id: int = 0,
    ) -> Dict[str, Any]:
        """
        异步从最近对话轮次中抽取结构化长期记忆 patch。

        Args:
            turns:    get_history() 返回的消息列表
            round_id: 当前轮次编号（写入 last_seen_round）

        Returns:
            dict patch（可直接传给 merge_patch_async），无内容时返回 {}
        """
        recent = turns[-(DEFAULT_EXTRACT_ROUNDS * 2):]
        if not recent:
            return {}

        dialogue_lines = [
            f"{'用户' if m['role'] == 'user' else '助手'}：{m['content'][:400]}"
            for m in recent
            if m.get("role") in ("user", "assistant")
        ]
        if not dialogue_lines:
            return {}
        dialogue = "\n".join(dialogue_lines)

        # 为节省 token，existing_memory 只传非空字段的精简版
        existing_summary = _summarize_for_prompt(self._data)

        if not self.llm:
            return {}

        try:
            prompt = _EXTRACT_PROMPT_TEMPLATE.format(
                existing_memory=existing_summary,
                current_round=round_id,
                dialogue=dialogue,
            )
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, partial(self.llm.complete, prompt)
            )
            result_str = (result if isinstance(result, str) else str(result)).strip()

            # 去除 LLM 可能包裹的 ```json ... ``` 代码块
            if result_str.startswith("```"):
                lines = result_str.splitlines()
                result_str = "\n".join(
                    l for l in lines if not l.strip().startswith("```")
                ).strip()

            patch = _safe_json_loads(result_str)
            if not patch:
                return {}
            return patch

        except Exception as e:
            logger.warning(f"LongTermMemory extract LLM failed: {e}")
            return {}


# ─────────────────────────────────────────────
# 辅助：为 prompt 生成已有记忆的精简摘要
# ─────────────────────────────────────────────

def _summarize_for_prompt(data: Dict[str, Any]) -> str:
    """
    将当前记忆字典转为紧凑的文本摘要，用于注入抽取 prompt（节省 token）。
    """
    parts: List[str] = []

    up = data.get("user_profile", {})
    if up:
        kv = ", ".join(f"{k}={v}" for k, v in up.items() if v)
        parts.append(f"用户信息: {kv}")

    tp = data.get("task_preferences", {})
    if tp:
        kv = ", ".join(f"{k}={v}" for k, v in tp.items() if v)
        parts.append(f"任务偏好: {kv}")

    dk = data.get("domain_knowledge", [])
    if dk:
        topics = ", ".join(item.get("topic", "") for item in dk[:10])
        parts.append(f"已有知识主题: {topics}")

    rt = data.get("recurring_tasks", [])
    if rt:
        names = ", ".join(item.get("task_name", "") for item in rt[:5])
        parts.append(f"常见任务: {names}")

    sc = data.get("session_continuity", {})
    if sc.get("pending_tasks"):
        parts.append(f"未完成任务: {sc['pending_tasks'][:80]}")

    return "\n".join(parts) if parts else "（暂无记忆）"
