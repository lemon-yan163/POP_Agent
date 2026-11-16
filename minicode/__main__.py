import asyncio
import difflib
import getpass
import os
import shlex
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import termios
except ImportError:  # Windows 没有 termios；终端恢复退化为 no-op
    termios = None

# 必须在这里 import readline（只为副作用）：不导入的话 input() 走 Python 内置的最小行编辑器，
# 它把退格当作「擦掉一列」，而中文/emoji 是双列宽，于是按一次退格屏幕上会留下一个空格，
# 还会把 UTF-8 字节序列拆断导致 UnicodeDecodeError 把 REPL 打崩。
# GNU readline 同时还负责方向键：没有它时 ←/→/↑/↓ 会以 ^[[D / ^[[C / ^[[A / ^[[B 原样显示。
# GNU readline 是按字符宽度重绘的，中文退格才正常。
readline = None
try:
    import readline

    _hist = Path.cwd() / ".minicode" / "input_history"
    try:
        readline.read_history_file(_hist)
    except OSError:
        pass
    readline.set_history_length(1000)
except ImportError:
    pass

from .agent import Agent
from .permissions import PermissionManager
from .providers import (
    ROLE_MAX_BYTES,
    UnconfiguredProvider,
    create_provider,
    default_provider_name,
    provider_default_model,
)
from . import store
from .term import stdin
from .tools import create_tools
from .ui import ChatUI

_ANSI = __import__("re").compile(r"\x1b\[[0-9;]*[A-Za-z]|\x1b\][^\x07]*\x07|[\x00-\x08\x0b-\x1f\x7f]")


def clean_input(s: str) -> str:
    """兜底：万一 readline 没能接管（非 tty、精简环境），把残留的转义序列剥掉，
    免得 ^[[D 这种字符被当成用户正文发给模型。"""
    return _ANSI.sub("", s)


def _read_role_file(path: Path) -> str:
    """读取角色卡：只接受 UTF-8，限制大小，并剥掉终端控制字符。"""
    if not path.is_file():
        raise ValueError("不是普通文件")
    size = path.stat().st_size
    if size > ROLE_MAX_BYTES:
        raise ValueError(f"文件太大（{size} 字节，上限 {ROLE_MAX_BYTES}）")
    data = path.read_bytes()
    if len(data) > ROLE_MAX_BYTES:
        raise ValueError(f"文件太大（{len(data)} 字节，上限 {ROLE_MAX_BYTES}）")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError("不是 UTF-8 文本")
    return clean_input(text).strip()


def _open_editor(path: Path) -> bool:
    """用 $VISUAL / $EDITOR 打开角色卡；都没有时退回 vi。"""
    editor = os.getenv("VISUAL") or os.getenv("EDITOR") or "vi"
    try:
        cmd = shlex.split(editor)
    except ValueError:
        cmd = [editor]
    if not cmd:
        cmd = ["vi"]
    try:
        return subprocess.run([*cmd, str(path)]).returncode == 0
    except FileNotFoundError:
        print(
            f"找不到编辑器 {cmd[0]!r}；请设置 $EDITOR，或改用 /role set / /role load。",
            file=sys.stderr,
        )
        return False
    except KeyboardInterrupt:
        print("\n[编辑器已中断]")
        return False


def _mask_key(key: str) -> str:
    """状态里只显示长度和末四位，绝不回显完整 key。"""
    key = key or ""
    if not key:
        return "未设置"
    if len(key) <= 8:
        return f"已设置（{len(key)} 字符）"
    return f"已设置（{len(key)} 字符，末尾 {key[-4:]}）"


def _unquote(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        return s[1:-1].strip()
    return s


def _scrub_history(line: str) -> None:
    """把当前这条 /API 命令从 readline 历史里删掉，避免 key 落进 input_history。"""
    if readline is None:
        return
    try:
        n = readline.get_current_history_length()
        if n and readline.get_history_item(n) == line:
            readline.remove_history_item(n - 1)
    except Exception:
        pass


def _api_status(provider, session_keys: dict[str, str] | None = None) -> list[str]:
    name = getattr(provider, "provider_name", "?")
    source = "（本次运行设置）" if session_keys and name in session_keys else ""
    return [
        f"provider : {name}",
        f"model    : {provider.model}",
        f"base_url : {getattr(provider, 'base_url', '?')}",
        f"api_key  : {_mask_key(getattr(provider, 'api_key', ''))}{source}",
        f"reasoning: {'replay' if getattr(provider, 'replay_reasoning', False) else 'off'}",
    ]


def _api_brief(provider) -> str:
    return (
        f"{getattr(provider, 'provider_name', '?')} · {provider.model} · "
        f"key {_mask_key(getattr(provider, 'api_key', ''))}"
    )


_state = {
    "task": None,
    # 角色卡编辑器占用终端时，Ctrl-C 应该交给编辑器处理，而不是把 REPL 一起结束。
    "suspend_sigint": False,
}


def _save_terminal():
    """保存启动时的 termios 设置。

    readline/getpass 在等待输入时会把 ECHO 关掉自己回显；如果进程在 input()
    还阻塞时退出（典型场景：在 >> 提示符按 Ctrl-C），readline 来不及恢复，
    终端就会一直不回显。退出时用这份原始设置兜底恢复。
    """
    if termios is None:
        return None
    try:
        fd = sys.stdin.fileno()
        if os.isatty(fd):
            return fd, termios.tcgetattr(fd)
    except (OSError, ValueError, termios.error):
        pass
    return None


def _restore_terminal(saved) -> None:
    if not saved or termios is None:
        return
    try:
        fd, attrs = saved
        termios.tcsetattr(fd, termios.TCSADRAIN, attrs)
    except (OSError, ValueError, termios.error):
        pass

HELP = """commands:
  /help          show this help
  /history       list archived sessions
  /history <n>   resume archived session n (also accepts an id prefix)
  /reset         start a fresh session (current one is archived first)
  /think         toggle showing model reasoning (thinking) stream
  /permissions   show permission rules and saved approvals
  /API           show current provider / model / masked key
  /API set       set the API key with hidden input (session only)
  /API openai    switch to OpenAI (also /API deepseek)
  /API model <m> switch model   /API base <url>   switch base URL
  /AI-Name <n>   rename the AI header for this run (/AI-Name reset)
  /role          show the current role card
  /role set <t>  set a one-line role card
  /role edit [f] edit the role card in $EDITOR (session only)
  /role load <f> load a role card from a file (session only)
  /role reset    clear the role card
  /exit          quit
  Ctrl-C         cancel current generation, or quit when idle

The AI name and active role card are session-only: nothing is auto-saved,
and a restart goes back to AI(model) with no role card. /role edit <file>
edits a file you choose, but its content is still only applied this run.

Every session is archived automatically under .minicode/sessions/.
Startup always begins a NEW session; use /history to pick up an old one.

Permissions follow an OpenCode-style model: ordered allow/deny/ask rules,
last match wins, and no match means ask. Project rules live in
.minicode/permissions.json; "always" approvals are saved there."""

_COMMANDS = [
    "/help", "/history", "/reset", "/think", "/permissions", "/api",
    "/ai-name", "/role", "/exit",
]
_BARE_COMMANDS = [c[1:] for c in _COMMANDS]


def load_env() -> None:
    p = Path(".env")
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _on_sigint() -> None:
    if _state.get("suspend_sigint"):
        return
    task = _state.get("task")
    if task is not None:
        # 空闲时取消 REPL 任务，让它走到 input 的 CancelledError 分支正常收尾；
        # 直接 raise KeyboardInterrupt 会让 asyncio.run 在 input() 还阻塞时销毁任务，
        # 既会打印 "Task was destroyed"，readline 也来不及恢复终端回显。
        task.cancel()
    else:
        raise KeyboardInterrupt


async def repl() -> None:
    load_env()
    startup_note = ""
    try:
        provider = create_provider()
    except (RuntimeError, ValueError, TypeError) as e:
        # 没有 key 也先进入 REPL；输入消息时再提示用 /API 接入。
        provider = UnconfiguredProvider(str(e))
        startup_note = "还没有接入 API。用 /API set 或 /API <key> 接入（支持 DeepSeek / OpenAI）"
    ui = ChatUI(provider.model)
    permissions = PermissionManager(ask=ui.ask_permission)

    def on_event(e: dict) -> None:
        kind = e.get("type")
        try:
            if kind == "step_start":
                ui.step_start()
            elif kind == "reasoning_delta":
                ui.reasoning(e.get("text", ""))
            elif kind == "text_delta":
                ui.text(e.get("text", ""))
            elif kind == "tool_start":
                ui.tool_start(e.get("name", "?"), e.get("args") or {}, e.get("id", ""))
            elif kind == "tool_end":
                ui.tool_end(e.get("name", "?"), e.get("output", ""), e.get("id", ""))
        except Exception as exc:
            # UI 故障不能把整个 Agent 带崩；下一帧继续画。
            print(f"\n[ui] {exc}", file=sys.stderr)

    agent = Agent(provider, create_tools(permissions), on_event)

    # 启动永远是全新会话：上次的上下文不会自动灌进来，要恢复得自己 /history 选
    session_id = store.new_unique_id()
    model = provider.model

    # 角色卡和名字一样只活在这次运行里：不写 .minicode/，重启即恢复默认。
    role_card = ""
    # 每个 provider 在这次运行里通过 /API 设置的 key；切换 provider 时不会丢。
    session_keys: dict[str, str] = {}

    def apply_role(text: str, source: str = "") -> None:
        nonlocal role_card
        role_card = text
        setter = getattr(provider, "set_role", None)
        if callable(setter):
            setter(text)
        else:
            provider.role_card = text
        if not text:
            print("角色卡已清空（仅本次运行生效）")
            return
        lines = len(text.splitlines()) or 1
        where = f"（来源: {source}）" if source else ""
        print(f"角色卡已生效{where}：{len(text)} 字符 / {lines} 行（仅本次运行生效）")

    def is_unconfigured() -> bool:
        return getattr(provider, "provider_name", "") == "none"

    def current_api() -> dict:
        if is_unconfigured():
            # 占位 provider 没有真实配置；按环境默认选择目标 provider。
            name = default_provider_name()
            return {
                "name": name,
                "api_key": None,
                "base_url": None,
                "model": provider_default_model(name),
            }
        return {
            "name": getattr(provider, "provider_name", "deepseek"),
            "api_key": getattr(provider, "api_key", None),
            "base_url": getattr(provider, "base_url", None),
            "model": provider.model,
        }

    def rebuild_provider(**changes):
        """按当前配置重建 provider；失败时保留旧 provider，不影响正在进行的会话。"""
        nonlocal provider, model
        new_provider = create_provider(**changes)
        setter = getattr(new_provider, "set_role", None)
        if callable(setter):
            setter(role_card)
        else:
            new_provider.role_card = role_card
        provider = new_provider
        agent.provider = new_provider
        agent.compact_budget = new_provider.compact_budget
        model = new_provider.model
        ui.model = new_provider.model
        ui.set_name(ui.display_name or None)
        return new_provider

    def persist() -> None:
        try:
            store.save(session_id, agent.messages, model)
        except OSError as e:
            print(f"\n[warn] 会话存档失败: {e}", file=sys.stderr)

    def show_history(entries: list[dict]) -> None:
        if not entries:
            print("还没有任何存档会话。")
            return
        print(f"\n{len(entries)} 个存档会话（最近的在前）:")
        for i, e in enumerate(entries, 1):
            mark = "*" if e["id"] == session_id else " "
            print(
                f" {mark}{i:>2}  {e['updated']}  {e['turns']:>2}轮 {e['count']:>3}条  "
                f"{e['model']}  {e['preview']}"
            )
        print("\n用 /history <序号> 载入某个会话，/reset 开新会话")

    _state["task"] = asyncio.current_task()
    try:
        asyncio.get_running_loop().add_signal_handler(signal.SIGINT, _on_sigint)
    except (NotImplementedError, RuntimeError):
        pass

    ui.banner(session_id, agent.compact_budget, note=startup_note)

    while True:
        ui.rule()
        try:
            line = clean_input((await stdin.read_line(">> ")).strip())
        except asyncio.CancelledError:
            print("\nInterrupted. Bye.")
            break
        except (EOFError, KeyboardInterrupt):
            print()
            break
        except UnicodeDecodeError:
            # 输入行里有被拆断的 UTF-8 字节（旧终端/行编辑器退格半边造成的），丢掉这一行别崩
            print("\n[输入不是合法 UTF-8，已丢弃这一行]", file=sys.stderr)
            continue
        ui.rule()
        if not line:
            continue
        cmd, _, arg = line.partition(" ")
        cmd = cmd.strip()
        arg = arg.strip()
        low = cmd.lower()

        if cmd == "/exit":
            break
        if cmd == "/help":
            print(HELP)
            continue
        if cmd == "/history":
            entries = store.list_sessions()
            if not arg:
                show_history(entries)
                continue
            target = None
            if arg.isdigit() and 1 <= int(arg) <= len(entries):
                target = entries[int(arg) - 1]
            else:
                hits = [e for e in entries if e["id"].startswith(arg)]
                if len(hits) == 1:
                    target = hits[0]
                elif len(hits) > 1:
                    print(f"'{arg}' 匹配到多个会话，请写得更具体一点")
                    continue
            if target is None:
                print(f"没有匹配 '{arg}' 的存档会话，用 /history 看看列表")
                continue
            try:
                msgs = store.load(target["id"])
            except (OSError, ValueError) as e:
                print(f"载入失败: {e}", file=sys.stderr)
                continue
            agent.messages = msgs
            session_id = target["id"]
            print(f"已载入会话 {session_id}（{target['turns']} 轮 / {len(msgs)} 条）")
            print(f"  最后更新: {target['updated']}")
            print(f"  首条用户消息: {target['preview']}")
            continue
        if cmd == "/reset":
            persist()  # 先把当前会话存档，再开新的
            agent.messages = []
            session_id = store.new_unique_id()
            print(f"context cleared — new session {session_id}")
            continue
        if cmd == "/think":
            ui.show_reasoning = not ui.show_reasoning
            print(f"reasoning display: {'on' if ui.show_reasoning else 'off'}")
            continue
        if cmd == "/permissions":
            for out_line in permissions.describe():
                print(out_line)
            continue
        if low == "/api":
            _scrub_history(line)
            sub, _, rest = arg.partition(" ")
            sub = sub.strip().lower()
            rest = rest.strip()
            try:
                if not arg:
                    print("当前 API（仅本次运行；.env / 环境变量作为默认值）:")
                    for out_line in _api_status(provider, session_keys):
                        print("  " + out_line)
                    if is_unconfigured():
                        print("  提示     : 还没有接入 API，用 /API set 或 /API <key>")
                    print("  /API set            隐藏输入并设置当前 provider 的 key")
                    print("  /API <key>          直接设置 key（会从 readline 历史移除）")
                    print("  /API openai [key]   切换到 OpenAI（省略 key 用 OPENAI_API_KEY）")
                    print("  /API deepseek [key] 切换到 DeepSeek")
                    print("  /API model <name>   切换模型")
                    print("  /API base <url>     切换 base URL")
                    print("  /API reset          重新从 .env / 环境变量读取")
                elif sub == "set":
                    try:
                        key = getpass.getpass("API key（输入不回显）: ").strip()
                    except (EOFError, KeyboardInterrupt):
                        print("\n已取消")
                        continue
                    if not key:
                        print("key 为空，未修改")
                        continue
                    session_keys[current_api()["name"]] = key
                    cur = current_api()
                    cur["api_key"] = key
                    new = rebuild_provider(**cur)
                    print(f"API key 已更新：{_api_brief(new)}（仅本次运行生效）")
                elif sub == "reset":
                    name_now = current_api()["name"]
                    old_key = session_keys.pop(name_now, None)
                    try:
                        new = rebuild_provider()
                    except Exception:
                        # 环境里没有可用 key 时不要把本次运行的 key 也弄丢
                        if old_key:
                            session_keys[name_now] = old_key
                        raise
                    print(f"已重新读取配置：{_api_brief(new)}")
                elif sub in {"openai", "deepseek"}:
                    key = _unquote(rest) if rest else None
                    if key:
                        session_keys[sub] = key
                    else:
                        key = session_keys.get(sub)
                    cur = current_api()
                    if sub != cur["name"]:
                        changes = {
                            "name": sub,
                            "model": provider_default_model(sub),
                            "base_url": None,
                            "api_key": key,
                        }
                    else:
                        changes = cur
                        if key:
                            changes["api_key"] = key
                    new = rebuild_provider(**changes)
                    print(f"已切换：{_api_brief(new)}")
                elif sub == "provider":
                    target = rest.lower()
                    if not target:
                        print("用法: /API provider openai|deepseek")
                        continue
                    if target == current_api()["name"] and not is_unconfigured():
                        print(f"已经是 {target} 了")
                        continue
                    new = rebuild_provider(
                        name=target,
                        model=provider_default_model(target),
                        base_url=None,
                        api_key=session_keys.get(target),
                    )
                    print(f"已切换：{_api_brief(new)}")
                elif sub == "model":
                    if not rest:
                        print("用法: /API model <模型名>")
                        continue
                    if is_unconfigured():
                        print("还没有接入 API；先用 /API set 或 /API <key>")
                        continue
                    cur = current_api()
                    cur["model"] = _unquote(rest)
                    new = rebuild_provider(**cur)
                    print(f"模型已切换：{new.model}（provider={new.provider_name}）")
                elif sub == "base":
                    if not rest:
                        print("用法: /API base <url>")
                        continue
                    if is_unconfigured():
                        print("还没有接入 API；先用 /API set 或 /API <key>")
                        continue
                    cur = current_api()
                    cur["base_url"] = _unquote(rest)
                    new = rebuild_provider(**cur)
                    print(f"base_url 已切换：{new.base_url}")
                elif sub == "key":
                    if not rest:
                        print("用法: /API key <key>（隐藏输入请用 /API set）")
                        continue
                    key = _unquote(rest)
                    session_keys[current_api()["name"]] = key
                    cur = current_api()
                    cur["api_key"] = key
                    new = rebuild_provider(**cur)
                    print(f"API key 已更新：{_api_brief(new)}（仅本次运行生效）")
                else:
                    # /API <key> 或 /API "sk-..."
                    key = _unquote(arg)
                    if not key:
                        print("key 为空，未修改")
                        continue
                    session_keys[current_api()["name"]] = key
                    cur = current_api()
                    cur["api_key"] = key
                    new = rebuild_provider(**cur)
                    print(f"API key 已更新：{_api_brief(new)}（仅本次运行生效）")
            except Exception as e:
                msg = str(e)
                if "OPENAI_API_KEY" in msg:
                    print("还没有 OpenAI key。用 /API openai <key> 接入")
                elif "DEEPSEEK_API_KEY" in msg:
                    print("还没有 DeepSeek key。用 /API set 或 /API deepseek <key> 接入")
                else:
                    print(f"API 未切换: {e}")
            continue
        if low in {"/ai-name", "/name", "/rename", "/改名"}:
            if not arg:
                current = ui.display_name or f"AI({ui.model})"
                print(f"当前名字: {current}")
                print("用法: /AI-Name <名字>；/AI-Name reset 恢复默认（只改终端显示名）")
            elif arg.lower() in {"reset", "default", "-", "默认", "恢复"}:
                print(f"已恢复默认名字: {ui.set_name(None)}")
            else:
                try:
                    tag = ui.set_name(arg)
                except ValueError as e:
                    print(f"名字无效: {e}")
                    continue
                print(f"名字已改为: {tag}（仅本次运行生效）")
            continue
        if low in {"/role", "/角色", "/角色卡"}:
            sub, _, rest = arg.partition(" ")
            sub = sub.strip().lower()
            rest = rest.strip()
            if sub in {"", "show"}:
                if role_card:
                    print(f"当前角色卡（仅本次运行，{len(role_card)} 字符）:")
                    print(role_card)
                else:
                    print("当前没有角色卡。")
                    print("  /role set <一行文本>")
                    print("  /role edit [文件]   用 $EDITOR 编辑（默认临时文件）")
                    print("  /role load <文件>   从文件载入")
                    print("  /role reset         清空")
            elif sub == "set":
                text = clean_input(rest).strip()
                if not text:
                    print("用法: /role set <一行文本>；多行人设请用 /role edit")
                    continue
                if len(text.encode("utf-8")) > ROLE_MAX_BYTES:
                    print(f"角色卡太大（上限 {ROLE_MAX_BYTES} 字节）")
                    continue
                apply_role(text)
            elif sub == "reset":
                apply_role("")
            elif sub == "load":
                if not rest:
                    print("用法: /role load <文件>")
                    continue
                path = Path(rest).expanduser()
                try:
                    text = _read_role_file(path)
                except (OSError, ValueError) as e:
                    print(f"读取失败: {e}", file=sys.stderr)
                    continue
                apply_role(text, str(path))
            elif sub == "edit":
                if not (sys.stdin.isatty() and sys.stdout.isatty()):
                    print("当前不是交互终端；请用 /role set 或 /role load。")
                    continue
                target = Path(rest).expanduser() if rest else None
                tmp_path = None
                try:
                    if target is None:
                        fd, tmp_name = tempfile.mkstemp(prefix="minicode-role-", suffix=".md")
                        os.close(fd)
                        tmp_path = Path(tmp_name)
                        target = tmp_path
                        target.write_text(role_card, encoding="utf-8")
                    elif target.is_dir():
                        print(f"{target} 是目录")
                        continue
                    elif not target.exists():
                        if not target.parent.exists():
                            print(f"目录不存在: {target.parent}", file=sys.stderr)
                            continue
                        target.write_text(role_card, encoding="utf-8")
                    _state["suspend_sigint"] = True
                    try:
                        opened = _open_editor(target)
                    finally:
                        # 编辑器期间收到的 Ctrl-C 会被排到事件循环里；先让它跑完，
                        # 否则标志复位后它才执行，会把整个 REPL 一起结束。
                        try:
                            await asyncio.sleep(0)
                        finally:
                            _state["suspend_sigint"] = False
                    if not opened:
                        continue
                    try:
                        text = _read_role_file(target)
                    except (OSError, ValueError) as e:
                        print(f"读取失败: {e}", file=sys.stderr)
                        continue
                    apply_role(text, str(target) if rest else "")
                except OSError as e:
                    print(f"编辑失败: {e}", file=sys.stderr)
                finally:
                    if tmp_path is not None:
                        try:
                            tmp_path.unlink()
                        except OSError:
                            pass
            else:
                print("用法: /role [show|set|edit|load|reset]")
            continue

        if line.startswith("/"):
            suggestion = difflib.get_close_matches(cmd.lower(), _COMMANDS, n=1, cutoff=0.6)
            if suggestion:
                print(f"未知命令: {cmd}；你是不是想输入 {suggestion[0]}？")
            else:
                print(f"未知命令: {cmd}；输入 /help 查看命令")
            continue

        if is_unconfigured():
            print("你没有接入 API，请接入 API（/API）")
            print("  快速接入：/API set   或   /API openai <key>")
            bare = line.strip().lower()
            hint = difflib.get_close_matches(bare, _BARE_COMMANDS, n=1, cutoff=0.7)
            if hint:
                print(f"  想使用命令的话，请加上斜杠：/{hint[0]}")
            continue

        permissions.begin_turn()
        try:
            await agent.send(line)
        except asyncio.CancelledError:
            print("\n[interrupted — generation cancelled]")
        except Exception as e:
            text = str(e)
            low_text = text.lower()
            if "401" in text or "authentication" in low_text or "invalid api key" in low_text:
                print("\nAPI key 无效或未授权。用 /API set 重新输入。")
            else:
                print(f"\nerror: {e}", file=sys.stderr)
        finally:
            ui.end_turn()
            persist()  # 每轮落盘，中途崩了也不丢上下文

    ui.end_turn()
    persist()
    if readline is not None:
        try:
            _hist.parent.mkdir(parents=True, exist_ok=True)
            readline.write_history_file(_hist)
        except OSError:
            pass


def main() -> None:
    saved_term = _save_terminal()
    try:
        try:
            asyncio.run(repl())
        except (KeyboardInterrupt, asyncio.CancelledError):
            print("\nInterrupted. Bye.")
    finally:
        # 兜底：Ctrl-C 在 input() 阻塞时退出，readline 可能没来得及恢复 ECHO。
        _restore_terminal(saved_term)


if __name__ == "__main__":
    main()
