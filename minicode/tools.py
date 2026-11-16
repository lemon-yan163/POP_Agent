import asyncio
import os
import signal
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from .models import Tool
from .permissions import PermissionManager

ROOT = Path.cwd().resolve()
MAX_OUTPUT = 20_000


def _truncate(s: str) -> str:
    if len(s) <= MAX_OUTPUT:
        return s
    return s[:MAX_OUTPUT] + f"\n...[truncated {len(s) - MAX_OUTPUT} chars]"


def _safe(p: str) -> Path:
    abs_p = (ROOT / p).resolve()
    try:
        abs_p.relative_to(ROOT)
    except ValueError:
        raise ValueError(f"path outside project root: {p}")
    return abs_p


def _rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def create_tools(permissions: PermissionManager | None = None) -> list[Tool]:
    perms = permissions or PermissionManager()
    io_lock = asyncio.Lock()

    async def read_file(a: dict) -> str:
        p = _safe(a["path"])
        if p.is_dir():
            raise ValueError(f"{a['path']} is a directory; use run_shell with ls/find")
        await perms.require("read", _rel(p))
        try:
            text = p.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise ValueError(f"{a['path']} does not exist")
        except UnicodeDecodeError:
            raise ValueError(f"{a['path']} is not UTF-8 text (binary or other encoding); use run_shell to inspect")
        lines = text.split("\n")
        s = max(1, int(a.get("start_line", 1)))
        e = min(len(lines), int(a.get("end_line", len(lines))))
        out = "\n".join(f"{s + i:5d} | {line}" for i, line in enumerate(lines[s - 1:e]))
        return _truncate(out)

    async def write_file(a: dict) -> str:
        p = _safe(a["path"])
        await perms.require("edit", _rel(p))
        content = a.get("content", "")
        async with io_lock:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
        return f"wrote {len(content)} chars to {a['path']}"

    async def edit_file(a: dict) -> str:
        p = _safe(a["path"])
        await perms.require("edit", _rel(p))
        old, new = a["old_string"], a["new_string"]
        async with io_lock:
            try:
                raw = p.read_text(encoding="utf-8")
            except FileNotFoundError:
                raise ValueError(f"{a['path']} does not exist")
            count = raw.count(old)
            if count == 0:
                raise ValueError("old_string not found in file")
            if count > 1 and not a.get("replace_all"):
                raise ValueError(f"old_string appears {count} times; pass replace_all or add context")
            updated = raw.replace(old, new) if a.get("replace_all") else raw.replace(old, new, 1)
            p.write_text(updated, encoding="utf-8")
        n = count if a.get("replace_all") else 1
        return f"replaced {n} occurrence(s) in {a['path']}"

    async def web_search(a: dict) -> str:
        query = a["query"]
        await perms.require("websearch", query)

        def _sync_search() -> str:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
            results = []

            def _get(url: str):
                try:
                    return requests.get(url, headers=headers, timeout=5)
                except Exception:
                    return None

            urls = [
                f"https://www.bing.com/search?q={urllib.parse.quote(query)}",
                f"https://www.baidu.com/s?wd={urllib.parse.quote(query)}",
                f"https://www.so.com/s?q={urllib.parse.quote(query)}",
            ]
            with ThreadPoolExecutor(max_workers=3) as ex:
                pages = list(ex.map(_get, urls))

            # 1. 搜 Bing (国际版)
            try:
                resp = pages[0]
                soup = BeautifulSoup(resp.text, "html.parser")
                for li in soup.select("li.b_algo")[:3]:
                    title = li.select_one("h2").get_text(strip=True) if li.select_one("h2") else ""
                    link = li.select_one("a")["href"] if li.select_one("a") else ""
                    snippet = li.select_one(".b_caption p").get_text(strip=True) if li.select_one(".b_caption p") else ""
                    if title and link:
                        results.append(f"[Bing] {title}\n{link}\n{snippet}")
            except Exception:
                pass

            # 2. 搜 百度
            try:
                resp = pages[1]
                soup = BeautifulSoup(resp.text, "html.parser")
                for div in soup.select(".result, .c-container")[:3]:
                    title_div = div.select_one("h3, .t")
                    title = title_div.get_text(strip=True) if title_div else ""
                    a_tag = title_div.select_one("a") if title_div else div.select_one("a")
                    link = a_tag["href"] if a_tag else ""
                    snippet_div = div.select_one(".c-abstract, .content-right_2s-H4")
                    snippet = snippet_div.get_text(strip=True) if snippet_div else ""
                    if title and link:
                        results.append(f"[百度] {title}\n{link}\n{snippet}")
            except Exception:
                pass

            # 3. 搜 360
            try:
                resp = pages[2]
                soup = BeautifulSoup(resp.text, "html.parser")
                for li in soup.select(".res-list")[:3]:
                    title = li.select_one("h3").get_text(strip=True) if li.select_one("h3") else ""
                    link = li.select_one("h3 a")["href"] if li.select_one("h3 a") else ""
                    snippet = li.select_one(".res-desc").get_text(strip=True) if li.select_one(".res-desc") else ""
                    if title and link:
                        results.append(f"[360] {title}\n{link}\n{snippet}")
            except Exception:
                pass

            if not results:
                return "No results found from any engine."

            return "\n\n".join(results)

        raw = await asyncio.to_thread(_sync_search)
        return _truncate(raw)

    async def run_shell(a: dict) -> str:
        cmd = a["command"]
        # 模型给的 timeout_ms 不可信：没有上限时写个 999999999 就能让会话永久挂住，
        # 负值/0 又会让命令一启动就被杀。钳在 1s..10min。
        try:
            timeout = float(a.get("timeout_ms", 60_000)) / 1000
        except (TypeError, ValueError):
            timeout = 60.0
        timeout = min(max(timeout, 1.0), 600.0)

        # 权限检查会按子命令逐条匹配规则；结果写进 tool 消息，否则「人同意过」
        # 这件事对模型不可见（人在环路里但对模型隐形）。
        approval = await perms.require_shell(cmd)

        try:
            proc = await asyncio.create_subprocess_shell(
                cmd, cwd=ROOT,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
        except Exception as e:
            return f"{approval}\nError: failed to start command: {e}"

        def _kill_tree() -> None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass

        try:
            out_b, _ = await asyncio.wait_for(proc.communicate(), timeout)
        except asyncio.TimeoutError:
            _kill_tree()
            await proc.wait()
            return f"{approval}\n[killed: timeout after {timeout}s]"
        except asyncio.CancelledError:
            _kill_tree()
            raise
        out = out_b.decode("utf-8", errors="replace")
        return _truncate(f"{approval}\nexit_code: {proc.returncode}\n{out or '(no output)'}")

    return [
        Tool(
            "read_file", "Read a UTF-8 text file with line numbers.",
            {"type": "object", "properties": {"path": {"type": "string"}, "start_line": {"type": "number"}, "end_line": {"type": "number"}}, "required": ["path"]},
            read_file,
        ),
        Tool(
            "write_file", "Create or overwrite a file.",
            {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]},
            write_file,
        ),
        Tool(
            "edit_file", "Replace an exact substring. Must appear exactly once unless replace_all.",
            {"type": "object", "properties": {"path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"}, "replace_all": {"type": "boolean"}}, "required": ["path", "old_string", "new_string"]},
            edit_file,
        ),
        Tool(
            "run_shell",
            "Run a shell command in the project root. Every subcommand is checked against the "
            "project permission rules; when approval is needed it is requested on the user's terminal. "
            "The decision is reported as the first line of the result: "
            "'[Permission allowed: ...]', '[User approved ...]' or '[Permission denied ...]'. "
            "If denied, the command did not run — do not silently retry it.",
            {"type": "object", "properties": {"command": {"type": "string"}, "timeout_ms": {"type": "number"}}, "required": ["command"]},
            run_shell,
        ),
        Tool(
            "web_search", "Search the web using Bing, Baidu, and 360. Use this for real-time information, news, or documentation.",
            {"type": "object", "properties": {"query": {"type": "string", "description": "The search query"}}, "required": ["query"]},
            web_search,
        ),
    ]
