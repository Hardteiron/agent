"""
PromptBudget — 上下文 Token 预算管理器
=========================================

职责：
  - 在每次 LLM 调用前，对 prompt 各模块做 token 估算
  - 按预设比例分配 token 预算，超限时按优先级逐级压缩
  - 对外暴露 fit() 接口，返回压缩后可直接发送的各模块内容

Token 预算分配（默认，基于 model_context_limit）：
  ┌──────────────────────────┬────────┬──────────────────────────────┐
  │ 模块                     │ 比例   │ 说明                         │
  ├──────────────────────────┼────────┼──────────────────────────────┤
  │ system_prompt            │ 20%    │ 角色声明 + 工具描述（固定）  │
  │ short_term_history       │ 35%    │ 近期对话轮次                 │
  │ long_term_memory         │ 10%    │ 结构化长期记忆 JSON          │
  │ recall_context           │  8%    │ Milvus 语义召回片段          │
  │ df_info                  │ 10%    │ DataFrame 表结构预览         │
  │ user_query               │  5%    │ 当前用户输入（不压缩）       │
  │ output_reserve           │ 12%    │ 留给模型输出（不用于输入）   │
  └──────────────────────────┴────────┴──────────────────────────────┘
  合计输入侧：88%；输出保留：12%

压缩策略（按优先级，从低到高）：
  1. recall_context   → 裁减到 top-1（从 top-k）
  2. df_info          → 仅保留列名行（去掉数据预览行）
  3. long_term_memory → 取最高优先字段（user_profile + task_preferences + 最近 3 条 domain_knowledge）
  4. short_term_history → 缩减保留轮数（full_detail_rounds -= 1，最低 1 轮）
  5. system_prompt    → 去掉可选的工具详细说明（保留工具名列表）
  （user_query 永不压缩）

对外接口：
  count_tokens(text: str) -> int
  estimate_section(content) -> int          支持 str / list[dict]
  fit(sections: PromptSections) -> PromptSections  原地压缩，返回压缩后的 sections
  BudgetReport   压缩过程的诊断信息
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# 默认预算比例（可通过 BudgetConfig 覆盖）
# ─────────────────────────────────────────────

DEFAULT_RATIOS: Dict[str, float] = {
    "system_prompt":      0.20,
    "short_term_history": 0.35,
    "long_term_memory":   0.10,
    "recall_context":     0.08,
    "df_info":            0.10,
    "user_query":         0.05,
    "output_reserve":     0.12,   # 不参与输入 prompt，留给模型输出
}

# 压缩优先级（越小越先压缩）
_COMPRESS_PRIORITY = [
    "recall_context",
    "df_info",
    "long_term_memory",
    "short_term_history",
    "system_prompt",
    # user_query 不压缩
]

# 中文字符 token 近似系数（1 汉字 ≈ 1.5 token；1 ASCII ≈ 0.25 token）
_CN_COEFF  = 1.5
_ASCII_COEFF = 0.25


# ─────────────────────────────────────────────
# Token 估算
# ─────────────────────────────────────────────

def count_tokens(text: str) -> int:
    """
    轻量 token 估算（无需加载 tokenizer）。

    规则：
      - 中文字符（\u4e00-\u9fff）：1 字 ≈ 1.5 token
      - 其余字符（ASCII、标点等）：每字符 ≈ 0.25 token
    误差约 ±15%，足够预算调度使用。
    """
    if not text:
        return 0
    cn = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    other = len(text) - cn
    return int(cn * _CN_COEFF + other * _ASCII_COEFF)


def estimate_section(content: Any) -> int:
    """
    估算任意 prompt 片段的 token 数。
    支持：str、list[dict]（messages 格式）、dict
    """
    if content is None:
        return 0
    if isinstance(content, str):
        return count_tokens(content)
    if isinstance(content, list):
        total = 0
        for item in content:
            if isinstance(item, dict):
                total += count_tokens(item.get("content", ""))
                total += count_tokens(item.get("role", ""))
            else:
                total += count_tokens(str(item))
        return total
    if isinstance(content, dict):
        return count_tokens(json.dumps(content, ensure_ascii=False))
    return count_tokens(str(content))


# ─────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────

@dataclass
class BudgetConfig:
    """
    预算配置。通常从 Config 注入，或使用默认值。

    Args:
        model_context_limit: 模型支持的最大 token 数（输入+输出总和）
        ratios:              各模块 token 比例，覆盖 DEFAULT_RATIOS
        min_short_rounds:    短期历史最少保留轮数（压缩时不低于此值）
    """
    model_context_limit: int = 8192
    ratios: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_RATIOS))
    min_short_rounds: int = 1

    def input_limit(self) -> int:
        """输入侧总 token 上限（去掉 output_reserve 后）。"""
        reserve_ratio = self.ratios.get("output_reserve", 0.12)
        return int(self.model_context_limit * (1.0 - reserve_ratio))

    def budget_for(self, section: str) -> int:
        """返回指定模块的 token 预算（绝对值）。"""
        ratio = self.ratios.get(section, 0.0)
        return int(self.model_context_limit * ratio)


@dataclass
class PromptSections:
    """
    prompt 各模块内容容器。

    由 agent/base.py 在构建 prompt 前填充，传入 fit() 后得到压缩版本。
    """
    system_prompt:      str             = ""   # 角色声明 + 工具描述（固定部分）
    short_term_history: List[Dict]      = field(default_factory=list)  # get_short_history()
    long_term_memory:   str             = ""   # get_long_term_text()（JSON 字符串）
    recall_context:     str             = ""   # get_recall_context()
    df_info:            str             = ""   # DataFrame 预览文本
    user_query:         str             = ""   # 当前用户输入

    def total_tokens(self) -> int:
        return (
            estimate_section(self.system_prompt)
            + estimate_section(self.short_term_history)
            + count_tokens(self.long_term_memory)
            + count_tokens(self.recall_context)
            + count_tokens(self.df_info)
            + count_tokens(self.user_query)
        )

    def section_tokens(self) -> Dict[str, int]:
        return {
            "system_prompt":      estimate_section(self.system_prompt),
            "short_term_history": estimate_section(self.short_term_history),
            "long_term_memory":   count_tokens(self.long_term_memory),
            "recall_context":     count_tokens(self.recall_context),
            "df_info":            count_tokens(self.df_info),
            "user_query":         count_tokens(self.user_query),
        }


@dataclass
class BudgetReport:
    """fit() 执行后的诊断报告。"""
    original_tokens:  int
    final_tokens:     int
    input_limit:      int
    within_budget:    bool
    compressed:       List[str] = field(default_factory=list)   # 被压缩的模块名
    section_tokens:   Dict[str, int] = field(default_factory=dict)


# ─────────────────────────────────────────────
# PromptBudget
# ─────────────────────────────────────────────

class PromptBudget:
    """
    上下文 Token 预算管理器。

    Usage::

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
        # → sections 已压缩到预算内，report 包含诊断信息

        prompt = build_prompt(sections)   # 用压缩后的 sections 构建最终 prompt
    """

    def __init__(self, config: Optional[BudgetConfig] = None) -> None:
        self.config = config or BudgetConfig()

    # ──────────────────────────────────────────
    # 主入口
    # ──────────────────────────────────────────

    def fit(self, sections: PromptSections) -> tuple[PromptSections, BudgetReport]:
        """
        检查各模块 token 用量，按压缩优先级逐级缩减，直到总量在预算内。

        不修改传入对象，返回新的 PromptSections。
        """
        import copy
        s = copy.deepcopy(sections)
        input_limit   = self.config.input_limit()
        original_tok  = s.total_tokens()
        compressed    = []

        logger.debug(
            f"PromptBudget.fit: original={original_tok} tokens, limit={input_limit}"
        )

        for section_name in _COMPRESS_PRIORITY:
            if s.total_tokens() <= input_limit:
                break
            over = s.total_tokens() - input_limit
            budget = self.config.budget_for(section_name)
            current = estimate_section(getattr(s, section_name, ""))

            if current == 0:
                continue

            # 只在该模块确实超出自身预算或整体仍超限时压缩
            if current > budget or s.total_tokens() > input_limit:
                logger.debug(
                    f"  compress [{section_name}]: {current} → budget={budget}, over={over}"
                )
                new_val = self._compress(section_name, getattr(s, section_name), budget, s)
                setattr(s, section_name, new_val)
                compressed.append(section_name)

        final_tok = s.total_tokens()
        within = final_tok <= input_limit

        if not within:
            logger.warning(
                f"PromptBudget.fit: still over limit after compression "
                f"({final_tok} > {input_limit}), force-truncating short_term_history"
            )
            s.short_term_history = self._force_truncate_history(
                s.short_term_history, input_limit - s.total_tokens() + estimate_section(s.short_term_history)
            )
            if "short_term_history" not in compressed:
                compressed.append("short_term_history(force)")

        report = BudgetReport(
            original_tokens = original_tok,
            final_tokens    = s.total_tokens(),
            input_limit     = input_limit,
            within_budget   = s.total_tokens() <= input_limit,
            compressed      = compressed,
            section_tokens  = s.section_tokens(),
        )
        logger.info(
            f"PromptBudget: {report.original_tokens}→{report.final_tokens} tokens "
            f"(limit={input_limit}, compressed={compressed})"
        )
        return s, report

    # ──────────────────────────────────────────
    # 各模块压缩策略
    # ──────────────────────────────────────────

    def _compress(
        self,
        name: str,
        value: Any,
        budget: int,
        sections: PromptSections,
    ) -> Any:
        """分派到对应的压缩函数。"""
        fn = {
            "recall_context":     self._compress_recall,
            "df_info":            self._compress_df_info,
            "long_term_memory":   self._compress_long_term,
            "short_term_history": self._compress_short_term,
            "system_prompt":      self._compress_system_prompt,
        }.get(name)
        if fn is None:
            return value
        return fn(value, budget)

    # ── recall_context ────────────────────────
    def _compress_recall(self, text: str, budget: int) -> str:
        """
        保留前 N 条召回片段，直到 token 数 ≤ budget。
        召回片段格式："- <text>\n"
        """
        if not text or count_tokens(text) <= budget:
            return text
        lines = [l for l in text.splitlines() if l.strip()]
        # 保留标题行
        header_lines = [l for l in lines if l.startswith("【")]
        body_lines   = [l for l in lines if not l.startswith("【")]

        result = "\n".join(header_lines)
        for line in body_lines:
            candidate = result + "\n" + line
            if count_tokens(candidate) <= budget:
                result = candidate
            else:
                break
        return result

    # ── df_info ───────────────────────────────
    def _compress_df_info(self, text: str, budget: int) -> str:
        """
        压缩策略（按级别）：
          Level-1：去掉数据预览行（sample rows），只保留行数/列数/列名
          Level-2：仅保留行数 + 列名一行
        """
        if not text or count_tokens(text) <= budget:
            return text

        lines = text.splitlines()
        # Level-1：去掉样本数据行（通常是制表对齐的数据行）
        filtered = [
            l for l in lines
            if not _looks_like_data_row(l)
        ]
        candidate = "\n".join(filtered)
        if count_tokens(candidate) <= budget:
            return candidate

        # Level-2：只保留前 3 行（行数/列数/列名）
        return "\n".join(lines[:3])

    # ── long_term_memory ──────────────────────
    def _compress_long_term(self, text: str, budget: int) -> str:
        """
        按字段优先级裁剪 JSON 长期记忆：
          保留：schema_version、updated_at、user_profile、task_preferences
          减少：domain_knowledge（最近 3 条）、recurring_tasks（最近 2 条）
          丢弃：corrections、session_continuity（超限时）
        """
        if not text or count_tokens(text) <= budget:
            return text

        try:
            data = json.loads(text)
        except Exception:
            # 非 JSON 格式直接截断
            return _truncate_to_budget(text, budget)

        # 逐步裁剪，直到 ≤ budget
        steps = [
            # (field, 裁剪方式)
            ("domain_knowledge",  lambda d: d[:3]),
            ("domain_knowledge",  lambda d: d[:1]),
            ("recurring_tasks",   lambda d: d[:2]),
            ("recurring_tasks",   lambda d: []),
            ("corrections",       lambda _: []),
            ("session_continuity",lambda _: {}),
        ]
        for fname, reducer in steps:
            candidate = dict(data)
            if fname in candidate:
                candidate[fname] = reducer(candidate[fname])
            compressed_text = json.dumps(candidate, ensure_ascii=False)
            if count_tokens(compressed_text) <= budget:
                return compressed_text
            data = candidate

        # 最终兜底：只保留 user_profile + task_preferences
        minimal = {
            k: data[k]
            for k in ("schema_version", "user_profile", "task_preferences")
            if k in data
        }
        return _truncate_to_budget(
            json.dumps(minimal, ensure_ascii=False), budget
        )

    # ── short_term_history ────────────────────
    def _compress_short_term(
        self,
        history: List[Dict],
        budget: int,
    ) -> List[Dict]:
        """
        按轮次从最旧开始裁剪，直到 ≤ budget。
        最少保留 min_short_rounds 轮（config 控制）。

        压缩顺序：
          1. 优先丢弃 role=system（摘要行，信息密度低）
          2. 再从最旧的 user/assistant 轮开始丢弃
        """
        if not history or estimate_section(history) <= budget:
            return history

        min_rounds = max(1, self.config.min_short_rounds)

        # 先移除摘要行（role=system）
        result = [m for m in history if m.get("role") != "system"]
        if estimate_section(result) <= budget:
            return result

        # 按轮次分组（两条为一轮：user + assistant），从最旧开始删
        pairs: List[List[Dict]] = []
        buf: List[Dict] = []
        for msg in result:
            buf.append(msg)
            if msg.get("role") == "assistant":
                pairs.append(buf)
                buf = []
        if buf:
            pairs.append(buf)   # 不完整的最新轮

        while len(pairs) > min_rounds and estimate_section(
            [m for pair in pairs for m in pair]
        ) > budget:
            pairs.pop(0)   # 删除最旧一轮

        return [m for pair in pairs for m in pair]

    # ── system_prompt ─────────────────────────
    def _compress_system_prompt(self, text: str, budget: int) -> str:
        """
        压缩系统 prompt：
          Level-1：截断工具描述中的详细 JSON schema，只保留工具名和一行描述
          Level-2：硬截断到 budget token 上限
        """
        if not text or count_tokens(text) <= budget:
            return text

        # Level-1：找到【工具】区块，仅保留工具名列表
        marker = "你可以使用以下工具来完成用户任务："
        idx = text.find(marker)
        if idx != -1:
            header = text[:idx + len(marker)]
            # 提取工具名（以 "### " 开头的行）
            tool_section = text[idx:]
            tool_names = [
                line.strip().lstrip("# ").split("(")[0].strip()
                for line in tool_section.splitlines()
                if line.strip().startswith("###") or (
                    line.strip().startswith("\"name\"") and ":" in line
                )
            ]
            if tool_names:
                compressed = header + "\n（工具列表）" + "、".join(tool_names[:20])
                if count_tokens(compressed) <= budget:
                    return compressed

        # Level-2：硬截断
        return _truncate_to_budget(text, budget)

    # ── 强制截断 history（最后兜底）────────────
    @staticmethod
    def _force_truncate_history(history: List[Dict], budget: int) -> List[Dict]:
        """保留最新的若干条消息，直到 token 数 ≤ budget。"""
        result: List[Dict] = []
        for msg in reversed(history):
            candidate = [msg] + result
            if estimate_section(candidate) <= budget:
                result = candidate
            else:
                break
        return result


# ─────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────

def _truncate_to_budget(text: str, budget: int) -> str:
    """硬截断文本到 budget token 以内（按字符二分逼近）。"""
    if count_tokens(text) <= budget:
        return text
    # 粗略估算每字符平均 token 数
    ratio = len(text) / max(count_tokens(text), 1)
    target_chars = int(budget * ratio * 0.9)
    truncated = text[:target_chars]
    # 微调
    while count_tokens(truncated) > budget and len(truncated) > 0:
        truncated = truncated[:-10]
    return truncated + "…（已截断）"


def _looks_like_data_row(line: str) -> bool:
    """启发式判断是否为 DataFrame 的数据预览行（非表头/描述行）。"""
    stripped = line.strip()
    if not stripped:
        return False
    # 数据行特征：以数字开头，或包含大量空白分隔的列值
    if stripped[0].isdigit() and "  " in stripped:
        return True
    # 分隔线
    if set(stripped) <= set("-+|="):
        return True
    return False
