import asyncio
import json
import os

from .models import Message, Tool, ToolCall
from .permissions import PermissionDenied

MAX_STEPS = 40
# 压缩触发线是 budget * 0.7。老值 80k 是给 32k 窗口留的余量，而 deepseek-flash 的窗口是 1M，
# 那样等于用到 5% 就开始丢历史。真实预算由 provider 按模型名给出，这里只兜底。
COMPACT_BUDGET = int(os.getenv("MINICODE_COMPACT_BUDGET", "120000"))


class Agent:
    def __init__(self, provider, tools: list[Tool], on_event=None, compact_budget: int | None = None):
        self.provider = provider
        self.tools = {t.name: t for t in tools}
        self.tool_list = list(tools)
        self.on_event = on_event or (lambda e: None)
        self.messages: list[Message] = []
        self.compact_budget = compact_budget or getattr(provider, "compact_budget", COMPACT_BUDGET)
        # 同一轮里模型可能一次给出多个工具调用。串行执行让终端上的工具行、授权提示
        # 和输出严格按顺序出现；并行时授权提示可能被另一条命令的输出打断。
        self._tool_lock = asyncio.Lock()

    async def send(self, user_input: str) -> None:
        self.messages.append(Message(role="user", content=user_input))

        for _ in range(MAX_STEPS):
            self.messages = _maybe_compact(self.messages, self.compact_budget)
            self.on_event({"type": "step_start"})
            result = await self.provider.chat(
                self.messages, self.tool_list,
                on_text=self._on_text, on_reasoning=self._on_reasoning,
            )
            text, calls = result[0], result[1]
            reasoning = result[2] if len(result) > 2 else ""
            self.messages.append(
                Message(
                    role="assistant",
                    content=text,
                    tool_calls=calls,
                    reasoning_content=reasoning,
                )
            )
            if not calls:
                return

            tasks = [asyncio.create_task(self._run_tool(c)) for c in calls]
            try:
                outputs = list(await asyncio.gather(*tasks))
            except asyncio.CancelledError:
                # assistant(tool_calls) 已经进了 messages，tool 结果却还没写。此时收手，序列就永久缺损，
                # API 会一直回 400 "insufficient tool messages following tool_calls message"，
                # 而且 __main__ 的 finally 会把它落盘，/history 也救不回来。所以取消前必须补齐配对。
                #
                # 但要按子任务的真实结局补：已经跑完的工具（尤其 run_shell 已经真的执行了命令）
                # 必须保留真实输出，不能一律写成「被打断」——那会让模型以为命令没跑。
                for t in tasks:
                    t.cancel()
                try:
                    await asyncio.gather(*tasks, return_exceptions=True)
                except asyncio.CancelledError:
                    pass  # 用户又按了一次 Ctrl-C，不再等子任务收尾，直接按当前状态补齐
                for call, t in zip(calls, tasks):
                    self.messages.append(
                        Message(role="tool", tool_call_id=call.id, name=call.name, content=_settle(t))
                    )
                raise
            for call, output in zip(calls, outputs):
                self.messages.append(Message(role="tool", tool_call_id=call.id, name=call.name, content=output))

        raise RuntimeError(f"exceeded {MAX_STEPS} steps")

    def _on_text(self, delta: str) -> None:
        self.on_event({"type": "text_delta", "text": delta})

    def _on_reasoning(self, delta: str) -> None:
        self.on_event({"type": "reasoning_delta", "text": delta})

    async def _run_tool(self, call: ToolCall) -> str:
        async with self._tool_lock:
            args = _safe_parse(call.arguments)
            self.on_event({"type": "tool_start", "id": call.id, "name": call.name, "args": args})
            try:
                tool = self.tools.get(call.name)
                if tool is None:
                    raise ValueError(f"unknown tool: {call.name}")
                output = await tool.execute(args)
            except PermissionDenied as e:
                output = str(e)
            except Exception as e:
                output = f"Error: {e}"
            self.on_event({"type": "tool_end", "id": call.id, "name": call.name, "output": output})
            return output


def _settle(task: asyncio.Task) -> str:
    """取消后取子任务的真实结局：跑完的给真实输出，没跑完的才写「被打断」。"""
    if task.cancelled() or not task.done():
        return "[interrupted by user; result discarded]"
    exc = task.exception()
    if exc is not None:
        return f"Error: {exc}"
    return task.result()


def _safe_parse(s: str) -> dict:
    try:
        value = json.loads(s or "{}")
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _estimate(messages: list[Message]) -> int:
    """CJK-aware rough token estimate: ~4 ASCII chars or 1 CJK char per token."""
    total = 0
    for m in messages:
        s = json.dumps(m.__dict__, default=str, ensure_ascii=False)
        ascii_n = sum(1 for ch in s if ord(ch) < 128)
        total += ascii_n // 4 + (len(s) - ascii_n)
    return total


def _maybe_compact(messages: list[Message], budget: int) -> list[Message]:
    if _estimate(messages) < budget * 0.7 or len(messages) < 10:
        return messages

    first = messages[0]
    tail: list[Message] = []
    total = 0
    i = len(messages) - 1
    while i > 0:
        # 从后往前整体成组地取：一条 assistant(tool_calls) 及其全部 tool 结果必须同生同死，
        # 否则留下的 tool 消息会找不到它的 tool_calls 父消息，请求直接被 API 拒掉。
        j = i
        if messages[i].role == "tool":
            while j > 0 and messages[j - 1].role == "tool":
                j -= 1
            if j > 0 and messages[j - 1].role == "assistant" and messages[j - 1].tool_calls:
                j -= 1
            else:
                # 这段 tool 结果的父 assistant 已经被切走了，只能一起丢掉
                break
        block = messages[j:i + 1]
        size = _estimate(block)
        # tail 为空时也要至少保留最新的一轮，否则刚问的问题会被压掉，模型答的是上一个问题
        if tail and total + size > budget * 0.5:
            break
        tail[:0] = block
        total += size
        i = j - 1

    if tail and tail[0].role == "tool":
        # 兜底：尾部不能以 tool 消息开头（没有前置 assistant，序列非法）
        k = 0
        while k < len(tail) and tail[k].role == "tool":
            k += 1
        tail = tail[k:]

    dropped = len(messages) - 1 - len(tail)
    if dropped <= 0:
        return messages
    note = f"[{dropped} earlier messages omitted to save context]"
    if tail and tail[0].role == "user":
        tail[0] = Message(role="user", content=f"{note}\n\n{tail[0].content}")
        return [first, *tail]
    return [first, Message(role="user", content=note), *tail]
