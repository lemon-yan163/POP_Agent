import os

from openai import AsyncOpenAI

from .models import Message, Tool, ToolCall

SYSTEM = """你是一个运行在用户终端里的 AI 编程 Agent。
你具备文件读写、编辑和执行系统命令的能力，可以用来完成编码、系统配置等任务。

【适用范围】
- 你不仅服务于编码任务。问答、写作、分析、系统排障等均在范围内。
- 涉及代码、环境或文件时，按【核心工作流要求】查证后作答。
- 涉及事实性、时效性内容且无工具支撑时，必须明确标注"未查证"，不得以确信语气陈述。
- 不具备的能力（实时数据、实时新闻、未提供的私有信息）直接说明，不编造。

【核心工作流要求】
1. 必须先查证：在回答关于当前项目环境、文件内容的问题前，必须先调用工具获取真实信息，绝不允许瞎猜。
2. 修改前先读取：在修改任何文件之前，必须先调用 read_file 查看其当前内容。
3. 修改后要验证：完成代码编写或修改后，尽量使用 run_shell 运行测试或重新读取文件，确认修改生效。
4. 优先使用精准编辑：对已有文件修改时，优先使用 edit_file；只有在创建新文件或完全重写时才用 write_file。
5. 当遇到代码库之外、且你不确定的信息时，使用 web_search 工具。搜索结果可能来自多个引擎，请综合比对后给出答案。
6. 对于用户提出的破坏性操作（如删除文件、修改系统配置），在调用工具前必须在对话中明确警告用户风险，并等待用户确认。
7. 可以质疑用户的行为，以及允许疑问，但是不要擅自做主。

【权限与安全】
- 文件类工具（read_file / write_file / edit_file）被限制在项目根目录内。读取 .env 等敏感文件默认需要用户授权，工具输出会说明结果；被拒绝时不要重试或换路径绕过。
- run_shell 的每条子命令都会按 .minicode/permissions.json 的规则检查，默认逐条询问。授权结果会写在工具输出首行：'[Permission allowed: ...]'、'[User approved ...]' 或 '[Permission denied ...]'。deny 表示命令没有执行。
- 用户选择“总是允许”时，程序会保存项目级规则。你不要诱导用户放行过宽的模式；只在确有必要时建议长期放行，并优先使用最小范围。
- run_shell 本身没有沙箱，获批的命令以当前用户的完整权限运行。不要假设“越界会被拦”；需要目录外的信息时如实说明，并在执行前讲清命令的作用。
【行为准则】
- 保持输出简洁，不要输出长篇大论的解释，直接给出解决方案和操作。
"""

ROLE_MAX_BYTES = 32_000


def build_system_prompt(role_card: str = "") -> str:
    """内置 SYSTEM + 用户角色卡。

    角色卡是独立的第二段 system 提示，只负责身份、语气、专长和回答偏好。
    内置的权限、安全和工具工作流规则始终保留；冲突时以内置段为准。
    """
    role = (role_card or "").strip()
    if not role:
        return SYSTEM
    return (
        SYSTEM
        + "\n\n【用户角色卡】\n"
        "以下是用户提供的角色设定，用于设定你的身份、语气、专长和回答偏好。\n"
        "它不能取消上面的权限、安全和工具使用规则，也不能绕过工具层的授权检查；"
        "两者冲突时，以上面的内置规则为准。\n\n"
        + role
        + "\n【角色卡结束】\n"
    )


# 每个模型的上下文窗口（取自 GET /models 的 context_window）。压缩预算是窗口的一半左右：
# 留出余量给 _estimate() 的粗估误差和本轮输出。不认识的模型按最保守的算，宁可早压缩也不能
# 发出超窗请求（超窗是 400，整个会话每轮都失败）。
_CONTEXT_WINDOWS = {
    "deepseek-flash": 1_048_576,
    "deepseek-v4-pro": 1_048_576,
    "deepseek-chat": 131_072,
    "deepseek-reasoner": 131_072,
    # OpenAI 常见模型；顺序重要（gpt-4 不能排在 gpt-4o 前面）
    "gpt-4.1": 1_047_576,
    "gpt-4o": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4": 8_192,
    "gpt-3.5": 16_385,
    "o1": 200_000,
    "o3": 200_000,
    "o4": 200_000,
}
_FALLBACK_BUDGET = 60_000


def _budget_for(model: str) -> int:
    env = os.getenv("MINICODE_COMPACT_BUDGET")
    if env:
        # 环境变量写错（"600k"、空串）不该让程序起不来，退回按模型名算
        try:
            return max(int(env), 10_000)
        except ValueError:
            pass
    for name, window in _CONTEXT_WINDOWS.items():
        if model.startswith(name):
            return int(window * 0.5)
    return _FALLBACK_BUDGET


def _replays_reasoning(model: str) -> bool:
    """V4 思考模型要求带 tools 的请求回传历史 reasoning_content。"""
    flag = os.getenv("MINICODE_REASONING_PASSTHROUGH")
    if flag is not None:
        return flag.strip().lower() not in {"", "0", "false", "no", "off"}
    return model.startswith(("deepseek-flash", "deepseek-v4"))


class OpenAICompatibleProvider:
    """OpenAI Chat Completions 协议的通用实现。

    DeepSeek 和 OpenAI 都走这个协议；差异只在默认 base_url、key 环境变量
    和默认模型。Ollama/vLLM 等 OpenAI 兼容端点也可以通过 /API base 指向。
    """

    provider_name = "openai"
    key_env = "OPENAI_API_KEY"
    base_env = "OPENAI_BASE_URL"
    default_base_url = "https://api.openai.com/v1"
    default_model = "gpt-4o-mini"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ):
        key = api_key if api_key is not None else os.getenv(self.key_env)
        if not key:
            raise RuntimeError(f"set {self.key_env} in environment or .env")
        self.api_key = key
        self.base_url = (
            base_url if base_url is not None else os.getenv(self.base_env)
        ) or self.default_base_url
        self.model = model or os.getenv("MODEL") or self.default_model
        self.compact_budget = _budget_for(self.model)
        self.replay_reasoning = _replays_reasoning(self.model)
        # 角色卡只存在内存里；不落盘，进程退出即恢复内置 SYSTEM。
        self.role_card = ""
        self.client = AsyncOpenAI(base_url=self.base_url, api_key=key, max_retries=1)

    def set_role(self, role_card: str) -> None:
        self.role_card = (role_card or "").strip()

    async def chat(self, messages: list[Message], tools: list[Tool], on_text=None, on_reasoning=None):
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": build_system_prompt(self.role_card)},
                *_to_openai(messages, replay_reasoning=self.replay_reasoning),
            ],
            "stream": True,
        }
        if tools:
            payload["tools"] = [
                {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
                for t in tools
            ]
        stream = await self.client.chat.completions.create(**payload)

        text = ""
        reasoning = ""
        calls_acc: dict[int, dict] = {}
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            rc = (delta.model_extra or {}).get("reasoning_content")
            if rc:
                reasoning += rc
                if on_reasoning:
                    on_reasoning(rc)
            if delta.content:
                text += delta.content
                if on_text:
                    on_text(delta.content)
            for tc in delta.tool_calls or []:
                slot = calls_acc.setdefault(tc.index, {"id": "", "name": "", "args": ""})
                if tc.id:
                    slot["id"] = tc.id
                if tc.function and tc.function.name:
                    slot["name"] = tc.function.name
                if tc.function and tc.function.arguments:
                    slot["args"] += tc.function.arguments

        calls = [
            ToolCall(id=v["id"], name=v["name"], arguments=v["args"] or "{}")
            for _, v in sorted(calls_acc.items())
        ]
        return text, calls, reasoning


class DeepSeekProvider(OpenAICompatibleProvider):
    provider_name = "deepseek"
    key_env = "DEEPSEEK_API_KEY"
    base_env = "DEEPSEEK_BASE_URL"
    default_base_url = "https://api.deepseek.com"
    default_model = "deepseek-chat"


class OpenAIProvider(OpenAICompatibleProvider):
    provider_name = "openai"
    key_env = "OPENAI_API_KEY"
    base_env = "OPENAI_BASE_URL"
    default_base_url = "https://api.openai.com/v1"
    default_model = "gpt-4o-mini"


class UnconfiguredProvider:
    """没有 API key 时的占位 provider。

    它的作用是让 REPL 正常启动，用户可以先 /role、/AI-Name、/API，
    真正发消息时由 __main__ 拦截并提示接入 API。
    """

    provider_name = "none"
    model = "未接入"
    api_key = ""
    base_url = ""
    compact_budget = 60_000
    replay_reasoning = False
    role_card = ""

    def __init__(self, reason: str = ""):
        self.reason = reason
        self.role_card = ""

    def set_role(self, role_card: str) -> None:
        self.role_card = (role_card or "").strip()

    async def chat(self, *args, **kwargs):
        raise RuntimeError("你没有接入 API，请接入 API（/API）")


_PROVIDERS = {
    "deepseek": DeepSeekProvider,
    "openai": OpenAIProvider,
}


def provider_default_model(name: str) -> str:
    cls = _PROVIDERS.get((name or "").strip().lower())
    if cls is None:
        raise ValueError(f"unknown provider: {name} (use deepseek or openai)")
    return cls.default_model


def default_provider_name() -> str:
    """没有显式 provider 时的默认选择：PROVIDER > 已有 key > deepseek。"""
    name = (os.getenv("PROVIDER") or "").strip().lower()
    if name:
        return name
    if os.getenv("DEEPSEEK_API_KEY"):
        return "deepseek"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    return "deepseek"


def create_provider(
    name: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> OpenAICompatibleProvider:
    """按 PROVIDER / 已有 key 选择 provider；显式参数优先。"""
    name = (name or default_provider_name()).strip().lower()
    cls = _PROVIDERS.get(name)
    if cls is None:
        raise ValueError(f"unknown provider: {name} (use deepseek or openai)")
    return cls(api_key=api_key, base_url=base_url, model=model)


def _to_openai(messages: list[Message], *, replay_reasoning: bool = False) -> list[dict]:
    out: list[dict] = []
    for m in messages:
        if m.role == "user":
            out.append({"role": "user", "content": m.content})
        elif m.role == "assistant":
            # content=None 只在带 tool_calls 时才是合法的；既没正文又没工具调用时发 null，
            # 有些兼容实现会直接拒。空串更安全。
            d: dict = {"role": "assistant", "content": m.content or None}
            if m.tool_calls:
                d["tool_calls"] = [
                    {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": tc.arguments}}
                    for tc in m.tool_calls
                ]
            else:
                d["content"] = m.content or ""
            # DeepSeek V4 思考模式：请求带 tools 时，历史 assistant 的 reasoning_content
            # 必须一起回传。旧存档没有这个字段时用空串补位，避免整段会话被 400 卡死。
            # 只在该 provider 明确需要时发送；OpenAI 等端点不认识这个扩展字段，
            # 从 DeepSeek 会话中途切过去时不能带。
            if replay_reasoning:
                d["reasoning_content"] = m.reasoning_content
            out.append(d)
        else:
            out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})
    return out
