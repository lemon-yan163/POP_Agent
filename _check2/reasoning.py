import asyncio, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from minicode.providers import DeepSeekProvider, _to_openai, SYSTEM
from minicode.models import Message

async def main():
    p = DeepSeekProvider()
    print("model:", p.model)

    # 第 1 轮：真实调用，看返回里有没有 reasoning
    raw = await p.client.chat.completions.create(
        model=p.model, messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": "1+1=?只回数字"}],
        max_tokens=64,
    )
    msg = raw.choices[0].message
    rc = (msg.model_extra or {}).get("reasoning_content")
    print("round1 content:", repr(msg.content))
    print("round1 reasoning:", repr((rc or "")[:80]))

    # 第 2 轮：把上一轮 assistant 消息（不带 reasoning_content）回传，模拟当前代码的行为
    hist = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "1+1=?只回数字"},
        {"role": "assistant", "content": msg.content or ""},
        {"role": "user", "content": "再加 1 呢"},
    ]
    try:
        r2 = await p.client.chat.completions.create(model=p.model, messages=hist, max_tokens=16)
        print("round2 (无 reasoning_content): OK ->", repr(r2.choices[0].message.content))
    except Exception as e:
        print("round2 (无 reasoning_content): FAIL ->", str(e)[:220])

    # 第 3 轮：带上 reasoning_content
    hist[2]["reasoning_content"] = rc or ""
    try:
        r3 = await p.client.chat.completions.create(model=p.model, messages=hist, max_tokens=16)
        print("round3 (带 reasoning_content): OK ->", repr(r3.choices[0].message.content))
    except Exception as e:
        print("round3 (带 reasoning_content): FAIL ->", str(e)[:220])

asyncio.run(main())
