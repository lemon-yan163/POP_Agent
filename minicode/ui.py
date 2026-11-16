"""终端对话界面。

目标不是做成全屏 TUI，而是在普通 REPL 上把信息分层：

    AI(deepseek-flash) > read src/main.c
                         edit src/main.c
                         - #include          │ + #include <stdio.h>
                         shell $ make
                         ↳ exit 0 · 3 lines
    ─────────────────────────────────────────────────────────────
    >> 继续
    ─────────────────────────────────────────────────────────────

- 每个模型回合只打印一次 ``AI(model) >``，工具调用缩进对齐在下面。
- 文件编辑显示并排 diff；敏感文件（.env / 私钥）不显示内容。
- shell 输出显示退出码和少量输出行，完整内容仍然只交给模型。
- 底部输入框用两条横线框住 ``>>``，与正文分开。
"""

from __future__ import annotations

import difflib
import json
import os
import re
import shutil
import sys
import unicodedata
from pathlib import Path

from .permissions import PermissionRequest, Rule
from .term import stdin
from .tools import ROOT

DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
BOLD = "\033[1m"
RESET = "\033[0m"

MAX_DIFF_ROWS = 18
MAX_DIFF_FILE = 4 * 1024 * 1024
MAX_NAME_WIDTH = 24
MAX_SHELL_LINES = 8

_SENSITIVE_NAMES = ("id_rsa", "id_dsa", "id_ecdsa", "id_ed25519", ".netrc", ".npmrc", ".pypirc", "credentials")
_SENSITIVE_SUFFIXES = (".pem", ".key", ".p12", ".pfx")

# 工具输出和模型输出都可能夹带控制字符。直接打到终端上，轻则把光标/颜色弄乱，
# 重则能用 OSC 改标题、写剪贴板。展示前统一剥掉 CSI/OSC 和孤立控制字符。
_ANSI_RE = re.compile(
    r"\x1b\[[0-9;?]*[ -/]*[@-~]"
    r"|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)"
    r"|\x1b[()][A-Za-z0-9]"
    r"|\x1b[<=>]"
)
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _sanitize(s: str) -> str:
    return _CONTROL_RE.sub("", _ANSI_RE.sub("", s)).replace("\r", "")


def _char_width(ch: str) -> int:
    if unicodedata.combining(ch):
        return 0
    if ch in ("\u200b", "\ufeff"):
        return 0
    return 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1


def _disp_width(s: str) -> int:
    return sum(_char_width(c) for c in s)


def _clip(s: str, width: int) -> str:
    if width <= 0:
        return ""
    s = _sanitize(s)
    out: list[str] = []
    used = 0
    for ch in s:
        w = _char_width(ch)
        if used + w > width:
            if width - used >= 1:
                out.append("…")
            break
        out.append(ch)
        used += w
    return "".join(out)


def _pad(s: str, width: int) -> str:
    return s + " " * max(0, width - _disp_width(s))


def _short(value: object, fallback: str = "?") -> str:
    if value is None:
        return fallback
    text = str(value)
    return text if text else fallback


def _is_sensitive_path(path: str) -> bool:
    name = Path(path).name
    if name.endswith(".example"):
        return False
    if name == ".env" or name.startswith(".env"):
        return True
    if name in _SENSITIVE_NAMES:
        return True
    return name.endswith(_SENSITIVE_SUFFIXES)


def _describe_tool(name: str, args: dict, width: int) -> tuple[str, str]:
    path = _short(args.get("path"), "")
    if name == "read_file":
        label = "read"
        detail = path
        start = args.get("start_line")
        end = args.get("end_line")
        if start or end:
            detail += f":{start or 1}-{end or 'end'}"
    elif name == "write_file":
        label = "write"
        detail = path
    elif name == "edit_file":
        label = "edit"
        detail = path
    elif name == "run_shell":
        label = "shell"
        detail = "$ " + " ".join(_short(args.get("command"), "").split())
    elif name == "web_search":
        label = "search"
        detail = _short(args.get("query"), "")
    else:
        label = name
        try:
            detail = json.dumps(args, ensure_ascii=False)
        except (TypeError, ValueError):
            detail = ""

    limit = max(8, width - len(label) - 1)
    return label, _clip(detail.replace("\n", " ⏎ "), limit)


def _format_rule(rule: Rule) -> str:
    suffix = " (exact)" if rule.literal else ""
    return f"{rule.action}: {rule.resource}{suffix}"


class ChatUI:
    def __init__(self, model: str, name: str | None = None) -> None:
        self.model = _sanitize(model)
        self.out = sys.stdout
        self.tty = self.out.isatty()
        self.color = bool(
            self.tty
            and not os.getenv("NO_COLOR")
            and os.getenv("TERM", "") != "dumb"
        )
        try:
            columns = shutil.get_terminal_size((100, 24)).columns
        except OSError:
            columns = 100
        self.width = max(40, min(columns, 160))

        # 自定义名字只在当前运行里生效；空串表示使用默认的 AI(模型名)。
        self.display_name = ""
        self.set_name(name)
        self.show_reasoning = True

        self._header_printed = False
        self._text_open = False
        self._reasoning_open = False
        self._tools_in_step = False

    def set_name(self, name: str | None) -> str:
        """设置标题行显示名；None 或空串恢复默认 ``AI(模型名)``。

        名字只影响本进程的终端显示，不写入磁盘。返回实际显示的标题。
        """
        raw = " ".join(_sanitize(name or "").split())
        if raw:
            width = _disp_width(raw)
            if width > MAX_NAME_WIDTH:
                raise ValueError(
                    f"名字最长 {MAX_NAME_WIDTH} 列（中文按 2 列），当前 {width} 列"
                )
            tag = raw
        else:
            raw = ""
            tag = f"AI({self.model})"
        self.display_name = raw
        self.prefix = self._c(BOLD, tag) + self._c(DIM, " > ")
        # 工具行要和标题行对齐；中文/emoji 是双列宽，不能用 len()。
        self.indent = " " * (_disp_width(tag) + 3)
        return tag

    # ---- 输出基础 ---------------------------------------------------------

    def _c(self, code: str, text: str) -> str:
        if not self.color or not text:
            return text
        return f"{code}{text}{RESET}"

    def _write(self, text: str = "", end: str = "\n") -> None:
        print(text, end=end, file=self.out, flush=True)

    def _ensure_header(self) -> bool:
        if self._header_printed:
            return False
        self._write(self.prefix, end="")
        self._header_printed = True
        return True

    # ---- 回合/步骤 --------------------------------------------------------

    def step_start(self) -> None:
        """模型每发起一次请求（可能带工具）就开一个新块。"""
        produced = self._header_printed
        self.flush()
        if produced:
            self._write()
        self._header_printed = False
        self._tools_in_step = False

    def flush(self) -> None:
        self._close_reasoning()
        self._close_text()

    def end_turn(self) -> None:
        """一轮用户输入结束：收尾并把块状态清干净，下一轮不再多打空行。"""
        self.flush()
        self._header_printed = False
        self._tools_in_step = False

    def _close_text(self) -> None:
        if self._text_open:
            self._write()
            self._text_open = False

    def _close_reasoning(self) -> None:
        if self._reasoning_open:
            self._write()
            self._reasoning_open = False

    # ---- 流式内容 ---------------------------------------------------------

    def reasoning(self, delta: str) -> None:
        if not self.show_reasoning or not delta:
            return
        delta = _sanitize(delta)
        if not delta:
            return
        self._close_text()
        if self._ensure_header():
            self._write()
        if not self._reasoning_open:
            self._write(self.indent, end="")
            self._write(self._c(DIM, "💭 "), end="")
            self._reasoning_open = True
        self._write(self._c(DIM, delta), end="")

    def text(self, delta: str) -> None:
        if not delta:
            return
        delta = _sanitize(delta)
        if not delta:
            return
        self._close_reasoning()
        if not self._text_open:
            already_open = not self._ensure_header()
            if already_open and self._tools_in_step:
                # 工具列表和正文之间空一行，别让正文看起来像工具参数
                self._write()
                self._tools_in_step = False
            self._text_open = True
        self._write(delta, end="")

    # ---- 工具 -------------------------------------------------------------

    def tool_start(self, name: str, args: dict, call_id: str = "") -> None:
        self._close_reasoning()
        self._close_text()
        fresh = self._ensure_header()
        if not fresh and not self._tools_in_step and self._header_printed:
            self._write()
        if not fresh:
            self._write(self.indent, end="")
        label, detail = _describe_tool(name, args, self.width - len(self.indent))
        line = self._c(CYAN, label)
        if detail:
            line += " " + self._c(DIM, detail)
        self._write(line)
        self._tools_in_step = True

        diff = self._diff_preview(name, args)
        if diff:
            self._write(diff, end="")

    def tool_end(self, name: str, output: str, call_id: str = "") -> None:
        summary, detail, color = _summarize_tool(name, output)
        if not summary:
            return
        self._write(self.indent + self._c(color, "↳ " + summary))
        limit = max(10, self.width - len(self.indent) - 2)
        for line in detail[:MAX_SHELL_LINES]:
            self._write(self.indent + "  " + self._c(DIM, _clip(line, limit)))
        extra = len(detail) - MAX_SHELL_LINES
        if extra > 0:
            self._write(self.indent + "  " + self._c(DIM, f"… +{extra} lines"))

    # ---- diff -------------------------------------------------------------

    def _diff_preview(self, name: str, args: dict) -> str:
        if name not in ("write_file", "edit_file"):
            return ""
        raw_path = args.get("path")
        if not isinstance(raw_path, str) or not raw_path or _is_sensitive_path(raw_path):
            return ""
        try:
            p = (ROOT / raw_path).resolve()
            p.relative_to(ROOT)
            existed = p.exists()
            if existed and (not p.is_file() or p.stat().st_size > MAX_DIFF_FILE):
                return ""
            old = p.read_text(encoding="utf-8") if existed else ""
            if name == "edit_file":
                old_s = args.get("old_string")
                new_s = args.get("new_string")
                if not isinstance(old_s, str) or not isinstance(new_s, str) or not old_s:
                    return ""
                count = old.count(old_s)
                if count == 0:
                    return ""
                if count > 1 and not args.get("replace_all"):
                    return ""
                new = (
                    old.replace(old_s, new_s)
                    if args.get("replace_all")
                    else old.replace(old_s, new_s, 1)
                )
            else:
                new = args.get("content")
                if not isinstance(new, str) or len(new) > MAX_DIFF_FILE:
                    return ""
            if old == new:
                return ""
            return self._render_diff(old.splitlines(), new.splitlines(), new_file=not existed)
        except (OSError, UnicodeDecodeError, ValueError):
            return ""

    def _render_diff(self, old: list[str], new: list[str], new_file: bool = False) -> str:
        matcher = difflib.SequenceMatcher(None, old, new, autojunk=False)
        avail = max(24, self.width - len(self.indent))
        left_width = max(10, (avail - 3) // 2)
        right_width = max(10, avail - left_width - 3)
        content_left = max(4, left_width - 2)
        content_right = max(4, right_width - 2)

        rows: list[str] = []
        truncated = False
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            if tag == "delete":
                for i in range(i1, i2):
                    if len(rows) >= MAX_DIFF_ROWS:
                        truncated = True
                        break
                    rows.append(self._diff_single(old[i], "-", RED, content_left + content_right + 3))
                if truncated:
                    break
                continue
            if tag == "insert":
                for j in range(j1, j2):
                    if len(rows) >= MAX_DIFF_ROWS:
                        truncated = True
                        break
                    rows.append(self._diff_single(new[j], "+", GREEN, content_left + content_right + 3))
                if truncated:
                    break
                continue
            a = old[i1:i2]
            b = new[j1:j2]
            n = max(len(a), len(b))
            pairs: list[tuple[str | None, str | None]] = [
                (a[k] if k < len(a) else None, b[k] if k < len(b) else None)
                for k in range(n)
            ]
            for left, right in pairs:
                if len(rows) >= MAX_DIFF_ROWS:
                    truncated = True
                    break
                rows.append(
                    self._diff_row(left, right, left_width, right_width, content_left, content_right)
                )
            if truncated:
                break

        if not rows:
            return ""
        if truncated:
            rows.append(self.indent + self._c(DIM, "… diff truncated"))
        if new_file:
            rows.insert(0, self.indent + self._c(DIM, "new file"))
        return "\n".join(rows) + "\n"

    def _diff_row(
        self,
        left: str | None,
        right: str | None,
        left_width: int,
        right_width: int,
        content_left: int,
        content_right: int,
    ) -> str:
        if left is not None:
            ltext = _clip(left.expandtabs(4).rstrip(), content_left)
            lcell = self._c(RED, "- " + _pad(ltext, content_left))
        else:
            lcell = " " * left_width
        if right is not None:
            rtext = _clip(right.expandtabs(4).rstrip(), content_right)
            rcell = self._c(GREEN, "+ " + rtext)
        else:
            rcell = ""
        if right is None:
            return self.indent + lcell
        return self.indent + lcell + self._c(DIM, " │ ") + rcell

    def _diff_single(self, line: str, sign: str, color: str, width: int) -> str:
        text = _clip(line.expandtabs(4).rstrip(), max(4, width - 2))
        return self.indent + self._c(color, f"{sign} {text}")

    # ---- 审批提示 / 输入框 ------------------------------------------------

    async def ask_permission(self, req: PermissionRequest, retry: bool = False) -> str:
        self.flush()
        self._write()
        self._write(self._c(YELLOW, f"  🔐 授权 · {req.action}"))
        preview = req.preview or req.resource
        for line in preview.splitlines() or [""]:
            self._write(self._c(DIM, "    " + _clip(line, self.width - 6)))
        if req.patterns and req.allow_always:
            joined = ", ".join(_format_rule(r) for r in req.patterns)
            self._write(self._c(DIM, "    选 a 将保存： " + _clip(joined, self.width - 16)))
        elif not req.allow_always:
            self._write(self._c(DIM, "    此命令没有 always 选项（动态语法或高风险操作）"))
        if retry:
            if req.allow_always:
                self._write(self._c(RED, "    没听懂：y/yes 允许本次 · n/no 拒绝 · a/always 总是允许"))
            else:
                self._write(self._c(RED, "    没听懂：此命令只能 y/yes 允许本次 或 n/no 拒绝"))
        choices = "[y] 允许本次   [n] 拒绝"
        if req.allow_always:
            choices += "   [a] 总是允许"
        self._write("    " + _clip(choices, self.width - 4))
        return await stdin.read_line("    > ")

    def rule(self) -> None:
        if self.tty:
            self._write(self._c(DIM, "─" * self.width))

    def banner(self, session_id: str, budget: int, note: str = "") -> None:
        self._write()
        self._write(self._c(BOLD, " minicode ") + self._c(DIM, f"· {self.model}"))
        self._write(self._c(DIM, f" session {session_id} · context ~{budget:,} tokens"))
        self._write(self._c(DIM, " /help 查看命令 · Ctrl-C 中断生成"))
        if note:
            self._write(self._c(YELLOW, " ⚠ " + _clip(note, max(10, self.width - 3))))


def _summarize_tool(name: str, output: str) -> tuple[str, list[str], str]:
    """把工具输出压成一行摘要 + 少量可展示的行。"""
    output = output or ""
    lines = output.splitlines()
    first = lines[0] if lines else ""

    if first.startswith("[Permission denied") or first.startswith("[User denied"):
        return _clip(first, 160), [], RED
    if first.startswith("Error:"):
        return _clip(first, 160), [], RED

    if name == "read_file":
        return f"{len(lines)} lines", [], DIM
    if name == "write_file":
        return "ok", [], DIM
    if name == "edit_file":
        return "ok", [], DIM
    if name == "web_search":
        if output.startswith("No results"):
            return "no results", [], YELLOW
        count = output.count("\n\n") + 1
        return f"{count} results", [], DIM

    if name == "run_shell":
        approval = next((l for l in lines if l.startswith("[")), "")
        error = next((l for l in lines if l.startswith("Error:")), "")
        if error:
            return _clip(error, 160), [], RED
        # 输出格式：审批说明 / exit_code: N / 命令输出...
        exit_line = next((l for l in lines if l.startswith("exit_code:")), "")
        code = exit_line.split(":", 1)[1].strip() if ":" in exit_line else "?"
        if "killed: timeout" in output:
            return "timeout", [], YELLOW
        body_start = 0
        for idx, line in enumerate(lines):
            if line.startswith("exit_code:"):
                body_start = idx + 1
                break
        body = [l for l in lines[body_start:] if l.strip()]
        if body == ["(no output)"]:
            body = []
        summary = f"exit {code}"
        if "always; saved" in approval:
            summary += " · rule saved"
        elif "could not persist" in approval:
            summary += " · rule NOT saved"
        if body:
            summary += f" · {len(body)} lines"
        if "could not persist" in approval:
            color = YELLOW
        elif code == "0":
            color = GREEN
        else:
            color = RED
        if approval.startswith("[User denied") or approval.startswith("[Permission denied"):
            color = RED
        return summary, body, color

    return _clip(first or "ok", 160), [], DIM
