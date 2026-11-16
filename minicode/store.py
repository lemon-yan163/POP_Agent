"""会话存档：每个会话一个 JSON 文件，放在 .minicode/sessions/ 下。

设计要点：
- 启动时永远开新会话，旧的不动；要靠 /history 显式选择才载入，避免把上次的上下文
  悄悄灌进新任务里。
- 空会话不落盘，否则 /reset 几次就会攒一堆空文件。
- 写盘走临时文件 + os.replace，中途被打断也不会留下半个 JSON 把存档弄坏。
"""

import json
import os
import time
from pathlib import Path

from .models import Message, ToolCall

ROOT = Path.cwd().resolve()
SESSIONS_DIR = ROOT / ".minicode" / "sessions"
HISTORY_FILE = ROOT / ".minicode" / "input_history"
VERSION = 1


def new_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def new_unique_id() -> str:
    """会话 id 精确到秒；同一秒内连开两个会话时加序号，避免互相覆盖。"""
    base = new_id()
    sid, n = base, 1
    while (SESSIONS_DIR / f"{sid}.json").exists():
        n += 1
        sid = f"{base}-{n}"
    return sid


def _msg_to_dict(m: Message) -> dict:
    return {
        "role": m.role,
        "content": m.content,
        "tool_calls": [
            {"id": t.id, "name": t.name, "arguments": t.arguments} for t in m.tool_calls
        ],
        "tool_call_id": m.tool_call_id,
        "name": m.name,
        "reasoning_content": m.reasoning_content,
    }


def _dict_to_msg(d: dict) -> Message:
    return Message(
        role=d.get("role", "user"),
        content=d.get("content") or "",
        tool_calls=[
            ToolCall(
                id=t.get("id", ""),
                name=t.get("name", ""),
                arguments=t.get("arguments") or "{}",
            )
            for t in d.get("tool_calls") or []
        ],
        tool_call_id=d.get("tool_call_id") or "",
        name=d.get("name") or "",
        reasoning_content=d.get("reasoning_content") or "",
    )


def save(session_id: str, messages: list[Message], model: str) -> None:
    """把会话写盘；messages 为空时反而不留下空文件。"""
    path = SESSIONS_DIR / f"{session_id}.json"
    if not messages:
        # 空会话没有存档价值，顺手清掉可能已存在的同名文件
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        return

    now = time.strftime("%Y-%m-%d %H:%M:%S")
    created = now
    try:
        created = json.loads(path.read_text(encoding="utf-8")).get("created", now)
    except (OSError, json.JSONDecodeError):
        pass

    payload = {
        "version": VERSION,
        "id": session_id,
        "created": created,
        "updated": now,
        "model": model,
        "messages": [_msg_to_dict(m) for m in messages],
    }
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        os.chmod(SESSIONS_DIR, 0o700)  # 已存在的目录 mkdir 不会改权限，这里纠一次
    except OSError:
        pass
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    # 存档是完整对话明文（可能含密钥、私有代码），同机其他用户不该能读。
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        pass
    os.replace(tmp, path)


def list_sessions() -> list[dict]:
    """按最后更新时间倒序返回会话元信息。"""
    if not SESSIONS_DIR.is_dir():
        return []
    out: list[dict] = []
    for p in SESSIONS_DIR.glob("*.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        msgs = d.get("messages") or []
        if not msgs:
            continue
        first_user = next(
            (m.get("content", "") for m in msgs if m.get("role") == "user" and m.get("content")),
            "",
        )
        turns = sum(1 for m in msgs if m.get("role") == "user")
        out.append(
            {
                "id": d.get("id") or p.stem,
                "created": d.get("created", "?"),
                "updated": d.get("updated", "?"),
                "model": d.get("model", "?"),
                "count": len(msgs),
                "turns": turns,
                "preview": " ".join(first_user.split())[:70] or "(empty)",
                "path": p,
            }
        )
    out.sort(key=lambda e: (e["updated"], e["id"]), reverse=True)
    return out


def load(session_id: str) -> list[Message]:
    path = SESSIONS_DIR / f"{session_id}.json"
    d = json.loads(path.read_text(encoding="utf-8"))
    return [_dict_to_msg(m) for m in d.get("messages") or []]
