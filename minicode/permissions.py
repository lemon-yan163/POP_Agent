"""OpenCode 风格的权限系统。

核心语义照搬 OpenCode，刻意保持简单、可审计：

- 规则是 (action, resource, effect) 三元组，按顺序匹配，**最后一条命中的规则生效**；
  一条都没命中时默认 ``ask``（fail-closed）。
- effect 只有 ``allow`` / ``deny`` / ``ask``。
- 资源用通配符匹配：``*`` 匹配任意字符（包括 ``/``），``?`` 匹配一个字符。
  shell 规则如果以 `` *`` 结尾，也匹配「不带参数的那条命令」
  （``git status *`` 同时匹配 ``git status``）。
- 用户选 ``always`` 时保存的规则是项目级的，写在 ``.minicode/permissions.json``；
  它们只能放行，永远不能覆盖配置里的 ``deny``。
- shell 先由一个小型扫描器拆成子命令；任何一条被 deny 就整条拒绝，有任何一条需要
  ask 就整条询问一次。无法安全拆分的语法（``$(...)``、变量展开、反引号、子 shell、
  后台 ``&``）按整条命令处理，并且不提供 ``always``（同样的字符串下次可能展开成
  完全不同的命令）。

配置示例（``.minicode/permissions.json``）::

    {
      "version": 1,
      "permissions": [
        {"action": "shell", "resource": "*", "effect": "ask"},
        {"action": "shell", "resource": "git status *", "effect": "allow"},
        {"action": "shell", "resource": "rm *", "effect": "deny"},
        {"action": "read", "resource": "*.env", "effect": "ask"}
      ],
      "saved": []
    }

``saved`` 由程序维护，也可以手改。规则按文件里的顺序匹配，后写的优先。
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

EFFECTS = {"allow", "deny", "ask"}
ONCE = "once"
ALWAYS = "always"
REJECT = "reject"
RETRY = "retry"

# 主键是 y / a / n；同时接受常见英文全词和中文说法。o / r 保留给老习惯。
_ONCE_WORDS = {
    "y", "yes", "yeah", "yep", "yup", "ok", "okay", "sure", "allow", "approve",
    "go", "run", "o", "once", "1",
    "是", "好", "好的", "行", "可以", "同意", "允许", "执行", "执行一次",
    "仅本次", "这次", "继续",
}
_ALWAYS_WORDS = {
    "a", "always", "all", "forever", "save", "persist",
    "全部", "总是", "一直", "始终", "永远", "记住", "放行", "保存",
    "永久", "永久允许", "永远允许", "始终允许", "以后都允许", "不再询问",
}
_REJECT_WORDS = {
    "n", "no", "nope", "nah", "deny", "block", "stop", "r", "reject", "0",
    "否", "不", "不要", "不了", "不行", "不允许", "取消", "拒绝", "跳过",
    "停止", "终止", "别",
}
# 多词输入里只认 once/reject 词。always 必须整句就是它，否则 "all files gone" 会被
# 升级成永久放行 —— 这是最危险的方向。
_TOKEN_ONCE = _ONCE_WORDS
_TOKEN_REJECT = _REJECT_WORDS

# 这些命令的 "always" 只保存完整命令，不保存前缀。前缀放行 rm / sudo / sh -c
# 之类的命令等于把整个系统交出去。
_EXACT_ONLY = {
    "rm", "rmdir", "unlink", "shred", "truncate", "dd", "mkfs", "mkswap",
    "shutdown", "reboot", "poweroff", "halt", "init", "telinit",
    "sudo", "doas", "su", "pkexec",
    "chmod", "chown", "chgrp", "chattr", "setfacl", "setcap",
    "mount", "umount", "swapon", "swapoff", "losetup",
    "kill", "pkill", "killall", "renice",
    "passwd", "chpasswd", "useradd", "userdel", "usermod",
    "groupadd", "groupdel", "groupmod",
    "systemctl", "service", "journalctl", "timedatectl",
    "iptables", "ip6tables", "nft", "ufw", "firewall-cmd",
    "parted", "fdisk", "sfdisk", "gdisk", "wipefs",
    "modprobe", "insmod", "rmmod", "sysctl",
    "docker", "podman", "kubectl", "helm",
    "mv", "cp", "install", "ln", "tee", "sed", "awk",
    "find", "xargs", "env", "nohup", "timeout", "watch",
    "crontab", "at", "batch", "ssh", "scp", "rsync", "sftp",
    "sh", "bash", "zsh", "dash", "fish", "ksh",
}
_INTERPRETERS = {
    "sh", "bash", "zsh", "dash", "fish", "ksh",
    "python", "python2", "python3", "perl", "ruby", "node", "php", "lua",
}
# shell 里出现这些词时，前缀放行可能把「查询」变成「执行」。
_DYNAMIC_WORDS = {"eval", "exec", "source", ".", "command", "builtin"}


def _norm_answer(s: str) -> str:
    """把审批输入归一成 once / always / reject / retry。"""
    import unicodedata

    s = unicodedata.normalize("NFKC", s).strip().lower()
    if not s:
        return REJECT
    if s in _ALWAYS_WORDS:
        return ALWAYS
    if s in _ONCE_WORDS:
        return ONCE
    if s in _REJECT_WORDS:
        return REJECT

    hits: set[str] = set()
    has_always_word = False
    for w in s.split():
        if w in _ALWAYS_WORDS:
            has_always_word = True
        elif w in _TOKEN_ONCE:
            hits.add(ONCE)
        elif w in _TOKEN_REJECT:
            hits.add(REJECT)
    if has_always_word and hits:
        # "y a" / "no all" 同时出现两种意图，绝不能替用户猜。
        return RETRY
    if len(hits) == 1:
        return hits.pop()
    if hits:
        return RETRY
    # 不再做「在句子里找字母」的兜底：abort / afk / aha 里都有 a，
    # 那会把「我要中止」升级成永久放行。认不出来就重新问，三次后按拒绝处理。
    return RETRY


@dataclass(frozen=True)
class Rule:
    action: str
    resource: str
    effect: str
    literal: bool = False

    def as_dict(self) -> dict:
        d = {"action": self.action, "resource": self.resource, "effect": self.effect}
        if self.literal:
            d["literal"] = True
        return d

    def display(self) -> str:
        suffix = " (exact)" if self.literal else ""
        return f"{self.action}: {self.resource}{suffix}"


@dataclass
class PermissionRequest:
    action: str
    resource: str
    patterns: list[Rule] = field(default_factory=list)
    preview: str = ""
    allow_always: bool = True


class PermissionDenied(Exception):
    """用户或规则拒绝了操作。工具层会把它原样写进 tool 消息。"""


DEFAULT_RULES: list[Rule] = [
    # 与 OpenCode 的 base policy 一致：默认放行，但 .env 明确要问。
    Rule("*", "*", "allow"),
    Rule("read", "*.env", "ask"),
    Rule("read", "*.env.*", "ask"),
    Rule("read", "*.env.example", "allow"),
    # 覆盖/新建 .env 同样要问，避免模型顺手把密钥文件写坏。
    Rule("edit", "*.env", "ask"),
    Rule("edit", "*.env.*", "ask"),
    Rule("edit", "*.env.example", "allow"),
    # minicode 的 shell 没有沙箱，默认逐条询问。
    Rule("shell", "*", "ask"),
    # 把碰 .env 的命令单独标出来；用户可以在它后面追加更具体的 allow/deny。
    Rule("shell", "*.env*", "ask"),
]


def _glob_match(pattern: str, value: str) -> bool:
    parts: list[str] = []
    for ch in pattern:
        if ch == "*":
            parts.append(".*")
        elif ch == "?":
            parts.append(".")
        else:
            parts.append(re.escape(ch))
    try:
        return re.fullmatch("".join(parts), value, re.DOTALL) is not None
    except re.error:
        return False


def _action_matches(pattern: str, action: str) -> bool:
    return pattern == "*" or pattern == action


def _resource_matches(rule: Rule, resource: str) -> bool:
    if rule.literal:
        return resource == rule.resource
    if _glob_match(rule.resource, resource):
        return True
    # OpenCode 的 shell 规则：以 " *" 结尾时也匹配不带参数的命令。
    if rule.resource.endswith(" *") and resource == rule.resource[:-2]:
        return True
    return False


def _parse_rule(raw: object) -> Rule | None:
    if not isinstance(raw, dict):
        return None
    action = raw.get("action")
    resource = raw.get("resource")
    effect = raw.get("effect")
    if not all(isinstance(x, str) and x.strip() for x in (action, resource, effect)):
        return None
    effect = effect.strip().lower()
    if effect not in EFFECTS:
        return None
    return Rule(
        action.strip().lower(),
        resource.strip(),
        effect,
        bool(raw.get("literal", False)),
    )


def _parse_rules(raw: object) -> list[Rule]:
    if not isinstance(raw, list):
        return []
    out: list[Rule] = []
    for item in raw:
        rule = _parse_rule(item)
        if rule is not None:
            out.append(rule)
    return out


def normalize_shell(command: str) -> str:
    """把命令归一成便于规则匹配的形式：去掉多余空白和引号差异。

    只用于权限匹配，不用于执行。``echo "hello world"`` 和 ``echo hello world``
    对规则来说应当是同一条命令，否则用户刚选完 always，下一条同样的命令又会被问。
    """
    command = command.strip()
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        return command
    return " ".join(tokens)


def scan_shell(command: str) -> list[str] | None:
    """把复合命令拆成子命令。

    返回 None 表示无法安全分析（命令替换、反引号、子 shell、后台执行、引号不闭合），
    调用方应把整条命令当成一个不可拆分的资源，并禁止保存前缀规则。
    """
    command = command.strip()
    if not command:
        return []
    parts: list[str] = []
    buf: list[str] = []
    quote: str | None = None
    i = 0
    n = len(command)

    def flush() -> None:
        text = "".join(buf).strip()
        buf.clear()
        if text:
            parts.append(text)

    while i < n:
        ch = command[i]
        if quote:
            if quote == '"' and (ch == "$" or ch == "`"):
                return None
            buf.append(ch)
            if ch == "\\" and quote == '"' and i + 1 < n:
                buf.append(command[i + 1])
                i += 2
                continue
            if ch == quote:
                quote = None
            i += 1
            continue

        if ch in "'\"":
            quote = ch
            buf.append(ch)
            i += 1
            continue
        if ch == "\\":
            buf.append(ch)
            if i + 1 < n:
                buf.append(command[i + 1])
                i += 2
            else:
                return None
            continue
        if ch == "`" or (ch == "$" and i + 1 < n and command[i + 1] == "("):
            return None
        if ch == "$":
            # 变量展开只有在运行时才知道值，无法安全分析：整条按不可拆分处理，
            # 并且不允许保存 always（见 require_shell）。
            return None
        if ch in "(){}":
            return None
        if ch in ";\n":
            flush()
            i += 1
            continue
        if ch == "&":
            if i + 1 < n and command[i + 1] == ">":
                buf.append("&>")
                i += 2
                continue
            if i > 0 and command[i - 1] in "<>":
                buf.append("&")
                i += 1
                continue
            if i + 1 < n and command[i + 1] == "&":
                flush()
                i += 2
                continue
            # 后台执行：整条命令按不可拆分处理
            return None
        if ch == "|":
            if i + 1 < n and command[i + 1] in "|&":
                flush()
                i += 2
                continue
            flush()
            i += 1
            continue
        buf.append(ch)
        i += 1

    if quote:
        return None
    flush()
    return parts


def _has_redirect(command: str) -> bool:
    return ">" in command or "<" in command


def _touches_sensitive(command: str) -> bool:
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        return True
    for token in tokens:
        name = os.path.basename(token)
        if name == ".env" or name.startswith(".env.") or name.endswith(".env"):
            return True
        if name.endswith((".pem", ".key", ".p12", ".pfx")):
            return True
        if name in ("id_rsa", "id_dsa", "id_ecdsa", "id_ed25519", ".netrc", ".npmrc", ".pypirc", "credentials"):
            return True
    return False


def _is_dangerous(sub: str) -> bool:
    try:
        tokens = shlex.split(sub, posix=True)
    except ValueError:
        return True
    if not tokens:
        return True

    idx = 0
    while idx < len(tokens) and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", tokens[idx]):
        idx += 1
    if idx >= len(tokens):
        return True
    # 前置环境变量会改变后续命令的解析/加载路径（PATH、LD_PRELOAD、PYTHONPATH…），
    # 不适合用前缀长期放行。
    if idx > 0:
        return True
    exe = os.path.basename(tokens[idx])
    if exe in _DYNAMIC_WORDS:
        return True
    if exe in _EXACT_ONLY:
        return True
    if exe in _INTERPRETERS and any(t in ("-c", "-e", "-E") for t in tokens[idx + 1:]):
        return True
    if exe == "git" and any(
        t in ("-c", "--config-env", "--exec-path", "-C", "--git-dir", "--work-tree")
        for t in tokens[idx + 1:]
    ):
        return True
    if exe == "git" and idx + 1 < len(tokens) and tokens[idx + 1] in {
        "push", "clean", "reset", "checkout", "restore", "rebase",
    }:
        return True
    if exe == "find" and any(t in ("-exec", "-execdir", "-delete", "-ok") for t in tokens):
        return True
    return False


def _shell_needs_exact(sub: str) -> bool:
    if _has_redirect(sub) or _is_dangerous(sub) or _touches_sensitive(sub):
        return True
    # 通配符的展开结果取决于当前目录，保存下来的「精确」规则下次可能指向别的文件；
    # 不提供 always。
    if any(c in sub for c in "*?[]"):
        return True
    try:
        tokens = shlex.split(sub, posix=True)
    except ValueError:
        return True
    return bool(tokens) and tokens[0] in ("cd", "pushd", "popd")


def _propose_shell_rules(sub: str, force_exact: bool = False) -> list[Rule]:
    """给一条子命令生成 always 时保存的规则。

    默认保存「可执行文件 + 一个参数 + *」的前缀（接近 OpenCode 的
    ``git status *``）；遇到重定向、cd、解释器 -c、rm/sudo 等危险命令时只保存
    完整命令，避免一次 always 放行整类危险操作。
    """
    sub = normalize_shell(sub)
    if not sub:
        return []
    exact = Rule("shell", sub, "allow", literal=True)

    if force_exact or _shell_needs_exact(sub):
        return [exact]
    try:
        tokens = shlex.split(sub, posix=True)
    except ValueError:
        return [exact]
    if not tokens:
        return [exact]
    # 命令里出现通配符时，参数集合在运行时才展开；不能把省略通配符后的前缀
    # 当成 always 规则（例如 `python *.py` 不能放行 `python *`）。
    if any(any(c in t for c in "*?[]") for t in tokens):
        return [exact]
    idx = 0
    while idx < len(tokens) and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", tokens[idx]):
        idx += 1
    if idx >= len(tokens):
        return [exact]

    head = tokens[: idx + 1]
    if idx + 1 < len(tokens):
        head.append(tokens[idx + 1])
        exe = os.path.basename(tokens[idx])
        # `python -m pytest` / `npm run build` 的第二个参数才是真正要执行的东西；
        # 只保存到 `python -m *` 或 `npm run *` 会把「放行 pytest」扩大成任意模块/脚本。
        if exe in _INTERPRETERS and tokens[idx + 1] == "-m" and idx + 2 < len(tokens):
            head.append(tokens[idx + 2])
        elif (
            exe in {"npm", "yarn", "pnpm", "bun"}
            and tokens[idx + 1] in {"run", "exec", "x"}
            and idx + 2 < len(tokens)
        ):
            head.append(tokens[idx + 2])
    return [Rule("shell", " ".join(head) + " *", "allow")]


class PermissionManager:
    """加载规则、判定 effect、在需要时向用户询问，并保存 always 规则。"""

    def __init__(
        self,
        root: Path | None = None,
        ask: Callable[[PermissionRequest, bool], Awaitable[str]] | None = None,
    ) -> None:
        self.root = (root or Path.cwd()).resolve()
        self.config_path = self.root / ".minicode" / "permissions.json"
        self.ask = ask
        self._lock = asyncio.Lock()
        self._reject_all = False
        self._config_error: str | None = None
        self._save_error: str | None = None
        self._user_rules: list[Rule] = []
        self._saved: list[Rule] = []
        self._load()

    # ---- 规则加载/保存 -----------------------------------------------------

    def _load(self) -> None:
        try:
            raw = json.loads(self.config_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (OSError, json.JSONDecodeError) as e:
            self._config_error = f"{self.config_path}: {e}"
            return

        if isinstance(raw, list):
            self._user_rules = _parse_rules(raw)
            return
        if not isinstance(raw, dict):
            self._config_error = f"{self.config_path}: top-level must be an object or array"
            return
        self._user_rules = _parse_rules(raw.get("permissions"))
        self._saved = _parse_rules(raw.get("saved"))

    def _save(self) -> bool:
        if self._config_error:
            # 用户配置坏了时绝不覆盖，否则会把他的规则冲掉。
            self._save_error = "config file could not be parsed; approval not persisted"
            return False
        payload = {
            "version": 1,
            "permissions": [r.as_dict() for r in self._user_rules],
            "saved": [r.as_dict() for r in self._saved],
        }
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            try:
                os.chmod(self.config_path.parent, 0o700)
            except OSError:
                pass
            tmp = self.config_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            try:
                os.chmod(tmp, 0o600)
            except OSError:
                pass
            os.replace(tmp, self.config_path)
            self._save_error = None
            return True
        except OSError as e:
            self._save_error = str(e)
            return False

    # ---- 判定 -------------------------------------------------------------

    def _configured_effect(self, action: str, resource: str) -> str:
        effect = "ask"
        for rule in (*DEFAULT_RULES, *self._user_rules):
            if _action_matches(rule.action, action) and _resource_matches(rule, resource):
                effect = rule.effect
        return effect

    def effect(self, action: str, resource: str) -> str:
        """返回 allow / deny / ask。配置里的 deny 永远优先于已保存的 allow。"""
        configured = self._configured_effect(action, resource)
        if configured == "deny":
            return "deny"
        saved = "ask"
        for rule in self._saved:
            if _action_matches(rule.action, action) and _resource_matches(rule, resource):
                saved = rule.effect
        if saved == "deny":
            return "deny"
        if saved == "allow":
            return "allow"
        return configured

    def _denied_note(self, action: str, resource: str, by_rule: bool = False) -> str:
        why = "by rule" if by_rule else "by user"
        return (
            f"[Permission denied {why}: {action}: {resource}] "
            "The operation was NOT performed; no output is available. "
            "Do not retry the same operation without asking the user why."
        )

    # ---- 询问 -------------------------------------------------------------

    async def require(
        self,
        action: str,
        resource: str,
        *,
        patterns: list[Rule] | None = None,
        preview: str = "",
    ) -> str:
        """确认权限；allow 时返回一行审批说明，deny 时抛 PermissionDenied。"""
        effect = self.effect(action, resource)
        if effect == "allow":
            return f"[Permission allowed: {action}: {resource}]"
        if effect == "deny":
            raise PermissionDenied(self._denied_note(action, resource, by_rule=True))
        return await self._ask_and_apply(action, resource, patterns, preview)

    async def _ask_and_apply(
        self,
        action: str,
        resource: str,
        patterns: list[Rule] | None,
        preview: str,
        allow_always: bool = True,
    ) -> str:
        proposed = patterns or [Rule(action, resource, "allow", literal=True)]
        req = PermissionRequest(action, resource, proposed, preview or resource, allow_always)

        async with self._lock:
            if self._reject_all:
                raise PermissionDenied(self._denied_note(action, resource))
            if self.ask is None:
                raise PermissionDenied(
                    self._denied_note(action, resource)
                    + " (no interactive approver is configured)"
                )

            choice = ""
            retry = False
            for _ in range(3):
                try:
                    ans = await self.ask(req, retry)
                except EOFError:
                    choice = REJECT
                    break
                choice = _norm_answer(ans)
                if choice == ALWAYS and not allow_always:
                    # 与 OpenCode 的 policy 不同，动态命令/破坏性命令不提供持久放行：
                    # 同样的字符串下次可能展开成完全不同的命令。
                    choice = RETRY
                if choice != RETRY:
                    break
                retry = True
            else:
                choice = REJECT

            if choice == ONCE:
                return f"[User approved {action}: {resource}] (once; this time only)"
            if choice == ALWAYS:
                saved, note = self._remember(req)
                if saved:
                    return f"[User approved {action}: {resource}] (always; saved {note})"
                return (
                    f"[User approved {action}: {resource}] "
                    f"(always, but could not persist: {self._save_error or note})"
                )

            # OpenCode 的 reject 会拒绝本会话里其他还在等待的权限请求。
            self._reject_all = True
            raise PermissionDenied(self._denied_note(action, resource))

    def _remember(self, req: PermissionRequest) -> tuple[bool, str]:
        rules = req.patterns or [Rule(req.action, req.resource, "allow", literal=True)]
        added: list[Rule] = []
        for rule in rules:
            rule = Rule(rule.action, rule.resource, "allow", rule.literal)
            if rule not in self._saved:
                self._saved.append(rule)
                added.append(rule)
        if not added:
            return True, "rule already saved"
        ok = self._save()
        return ok, ", ".join(r.display() for r in added)

    # ---- shell ------------------------------------------------------------

    async def require_shell(self, command: str) -> str:
        """按子命令检查整条 shell 命令，任何子命令 deny 都整条拒绝。"""
        command = command.strip()
        if not command:
            raise PermissionDenied("[Permission denied: empty shell command]")
        parts = scan_shell(command)
        force_exact = parts is None
        if force_exact:
            parts = [command]
        assert parts is not None
        normalized = [normalize_shell(p) for p in parts]
        if not normalized or not any(normalized):
            raise PermissionDenied("[Permission denied: empty shell command]")

        denied: list[str] = []
        asks: list[str] = []
        for sub in normalized:
            eff = self.effect("shell", sub)
            if eff == "deny":
                denied.append(sub)
            elif eff == "ask":
                asks.append(sub)
        if denied:
            raise PermissionDenied(
                self._denied_note("shell", denied[0], by_rule=True)
                + f" (denied subcommand: {denied[0]})"
            )
        if not asks:
            return f"[Permission allowed: shell: {command}]"

        patterns: list[Rule] = []
        for sub in asks:
            patterns.extend(_propose_shell_rules(sub, force_exact=force_exact))
        allow_always = (
            not force_exact
            and "$" not in command
            and "`" not in command
            and not any(_shell_needs_exact(s) for s in asks)
        )
        return await self._ask_and_apply(
            "shell", command, patterns, preview=f"$ {command}", allow_always=allow_always
        )

    # ---- 展示 -------------------------------------------------------------

    def begin_turn(self) -> None:
        """每次新的用户回合重置 reject 的连锁标记。"""
        self._reject_all = False

    def describe(self) -> list[str]:
        lines = ["permissions（最后一条命中的规则生效；没有任何规则命中时询问）:"]
        for rule in (*DEFAULT_RULES, *self._user_rules):
            suffix = " (exact)" if rule.literal else ""
            lines.append(f"  {rule.action:<10} {rule.resource:<32} -> {rule.effect}{suffix}")
        if self._saved:
            lines.append("saved approvals（项目级，可手改）:")
            for rule in self._saved:
                suffix = " (exact)" if rule.literal else ""
                lines.append(f"  {rule.action:<10} {rule.resource:<32} -> {rule.effect}{suffix}")
        lines.append(f"config: {self.config_path}")
        if self._config_error:
            lines.append(f"warning: {self._config_error}")
        if self._save_error:
            lines.append(f"warning: {self._save_error}")
        return lines
