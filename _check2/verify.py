import asyncio, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from minicode.models import Message, Tool, ToolCall
from minicode.agent import Agent, _estimate
from minicode.permissions import _norm_answer

print("== P1: 中断后序列是否合法 ==")

class FakeProvider:
    model = "deepseek-flash"
    compact_budget = 500_000
    async def chat(self, messages, tools, on_text=None, on_reasoning=None):
        return "", [ToolCall(id="call_1", name="slow", arguments="{}"),
                    ToolCall(id="call_2", name="slow", arguments="{}")]

async def slow(a):
    await asyncio.sleep(10)
    return "done"

async def p1():
    ag = Agent(FakeProvider(), [Tool("slow", "s", {}, slow)])
    asyncio.get_running_loop().call_later(0.2, asyncio.current_task().cancel)
    try:
        await ag.send("hi")
    except asyncio.CancelledError:
        pass
    for m in ag.messages:
        print(f"   {m.role:9} calls={[t.id for t in m.tool_calls]} id={m.tool_call_id!r} {m.content[:24]!r}")

    # 用真实 API 校验这个序列被接受
    from minicode.providers import DeepSeekProvider, _to_openai
    p = DeepSeekProvider()
    try:
        await p.client.chat.completions.create(
            model=p.model,
            messages=[
                {"role": "system", "content": "x"},
                *_to_openai(ag.messages, replay_reasoning=p.replay_reasoning),
            ],
            max_tokens=1,
        )
        print("   API: 接受 ✓")
    except Exception as e:
        print("   API ERROR:", str(e)[:200])

asyncio.run(p1())

print("\n== P2: 审批词表 ==")
cases = {"o": "once", "y": "once", "a": "always", "r": "reject", "n": "reject",
         "ａ": "always", "abort": "retry", "afk": "retry", "aha": "retry",
         "always": "always", "全部": "always", "取消": "reject", "y a": "retry",
         "y n": "retry", "": "reject", "yes": "once", "nah": "reject",
         "YES": "once", "No": "reject", "ok": "once", "deny": "reject",
         "save": "always", "永久允许": "always", "允许": "once",
         "不允许": "reject", "y  n": "retry", "save no": "retry",
         "all files gone": "retry", "please": "retry"}
bad = 0
for k, want in cases.items():
    got = _norm_answer(k)
    ok = "✓" if got == want else "✗"
    if got != want: bad += 1
    print(f"   {k!r:10} -> {got:7} (want {want}) {ok}")
print("   FAILED:", bad)

print("\n== P4: 预算 ==")
print("   flash budget:", __import__("minicode.providers", fromlist=["_budget_for"])._budget_for("deepseek-flash"))
print("   chat  budget:", __import__("minicode.providers", fromlist=["_budget_for"])._budget_for("deepseek-chat"))
print("   unknown     :", __import__("minicode.providers", fromlist=["_budget_for"])._budget_for("gpt-x"))
os.environ["MINICODE_COMPACT_BUDGET"] = "250000"
print("   env override:", __import__("minicode.providers", fromlist=["_budget_for"])._budget_for("deepseek-chat"))
del os.environ["MINICODE_COMPACT_BUDGET"]

print("\n== P6: timeout 钳制 & assistant content ==")
import re
src = Path("minicode/tools.py").read_text()
print("   clamp present:", "min(max(timeout, 1.0), 600.0)" in src)
from minicode.providers import _to_openai
print("   empty assistant ->", _to_openai([Message(role="assistant", content="")]))
print("   toolcall assistant ->", _to_openai([Message(role="assistant", content="", tool_calls=[ToolCall("i","n","{}")])]))

print("\n== P7: AI 名字 & 角色卡 ==")
from minicode.providers import build_system_prompt
from minicode.ui import ChatUI
ui = ChatUI("deepseek-flash")
print("   default indent:", len(ui.indent), "(want 21)")
ui.set_name("DemoBot")
print("   custom indent :", len(ui.indent), "(want 9, CJK width 2)")
ui.set_name(None)
try:
    ui.set_name("x" * 25)
except ValueError:
    print("   too long rejected: ✓")
else:
    print("   too long rejected: ✗")
base = build_system_prompt()
role = build_system_prompt("你是 DemoBot。")
print("   role absent by default:", "【用户角色卡】" not in base)
print("   role appended       :", "你是 DemoBot。" in role and "不能取消" in role)

print("\n== P8: provider / API ==")
from minicode.providers import OpenAIProvider, create_provider, provider_default_model
from minicode.__main__ import _mask_key, _unquote
o = create_provider("openai", api_key="sk-test-openai", base_url="http://example/v1", model="gpt-4o-mini")
print("   openai  :", o.provider_name, o.model, "replay", o.replay_reasoning, "budget", o.compact_budget)
d = create_provider("deepseek", api_key="sk-test-deepseek", model="deepseek-flash")
print("   deepseek:", d.provider_name, d.model, "replay", d.replay_reasoning, "budget", d.compact_budget)
print("   defaults:", provider_default_model("openai"), provider_default_model("deepseek"))
rm = Message(
    role="assistant", content="", tool_calls=[ToolCall("c", "n", "{}")], reasoning_content="why"
)
print("   reasoning off:", "reasoning_content" not in _to_openai([rm], replay_reasoning=False)[0])
print("   reasoning on :", _to_openai([rm], replay_reasoning=True)[0].get("reasoning_content") == "why")
print("   masked key   :", _mask_key("sk-1234567890abcdef"))
print("   quoted key   :", _unquote('"sk-test"'))

print("\n== P9: 无 key 占位 provider ==")
from minicode.providers import UnconfiguredProvider, default_provider_name
u = UnconfiguredProvider("missing key")
print("   stub          :", u.provider_name, u.model, u.compact_budget)
try:
    asyncio.run(u.chat([], []))
except RuntimeError as e:
    print("   chat message  :", e)
print("   default choice:", default_provider_name())
