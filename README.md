# POP_Agent

终端里的 AI 编程 Agent（Python 包名 `minicode`）：文件读写/编辑、shell 执行、权限审批、会话存档。

## 运行

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env   # 填入对应的 API key
./run.sh
```

没有 API key 也能启动：会显示 `未接入`，进入 REPL 后直接发消息会提示
`你没有接入 API，请接入 API（/API）`，不会崩溃。此时用 `/API set` 或
`/API <key>` 接入即可。

## 配置

- `.env`：本地密钥，已被 `.gitignore` 忽略，绝对不要提交。
- `.minicode/permissions.json`：项目级权限规则和保存的审批。
- `.minicode/sessions/`：会话存档，明文，含完整工具输出，已被忽略。

### DeepSeek / OpenAI

`.env` 里用 `PROVIDER` 选择服务：

```dotenv
PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-your-deepseek-key
MODEL=deepseek-flash
```

OpenAI 只需要换成：

```dotenv
PROVIDER=openai
OPENAI_API_KEY=sk-your-openai-key
MODEL=gpt-4o-mini
```

也支持 OpenAI 兼容端点（Ollama / vLLM / 中转服务），把 `OPENAI_BASE_URL`
指向对应的 `/v1` 地址即可。

运行时也可以切换，不用重启：

```text
/API                查看当前 provider、模型和打码后的 key
/API set            隐藏输入并设置 key（推荐）
/API openai [key]   切换到 OpenAI
/API deepseek [key] 切换到 DeepSeek
/API model <name>   切换模型
/API base <url>     切换 base URL
/API reset          重新从 .env / 环境变量读取
```

`/API` 设置的 key 只存在当前进程里，不写入 `.env`，也不会进入会话存档；
inline 写法会尽量从 readline 历史中移除，但仍然会显示在终端回显里，
所以更推荐用 `/API set` 的隐藏输入。

## 安全提示

- 不要用 `git add -f .env` 强行提交密钥。
- 不要把 key 贴进对话，也不要让 Agent 读取 `.env`；工具输出会进入会话存档。
- 如果密钥曾经被提交、贴进聊天记录或截图，立即到服务商控制台轮换，并清理 git 历史。
- 提交前建议自检：

```bash
git status --short
git check-ignore -v .env .minicode/
git grep --cached -nE 'sk-(proj-)?[A-Za-z0-9_-]{20,}' -- ':!.env.example'
```

`.env.example` 只是占位模板，可以安全地提交。
