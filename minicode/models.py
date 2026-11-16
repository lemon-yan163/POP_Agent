from dataclasses import dataclass, field
from typing import Awaitable, Callable


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass
class Message:
    role: str
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str = ""
    name: str = ""
    # DeepSeek V4 思考模式：带 tools 的请求必须把历史 assistant 的 reasoning_content
    # 原样传回，否则后续 tool 轮次会被 400 拒绝。
    reasoning_content: str = ""


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    execute: Callable[[dict], Awaitable[str]]
