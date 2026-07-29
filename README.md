# VisionToolMCP

VisionToolMCP 是一个 MCP 服务器，为纯文本 Agent 提供视觉能力桥梁。它接受图像文件、base64 图像数据或图像 URL，将它们发送到多模态模型，并通过 MCP 返回文本内容和 `structuredContent` 结构化结果。

## 功能特性

- 📤 **图像上传** - 通过 base64 上传图像，获得 `imageId` 可重复使用，30 分钟自动清理
- 🔍 **图像描述** - 描述图像内容，支持可选的聚焦/指令引导
- 📝 **文字识别 (OCR)** - 从图像中提取可见文本
- ❓ **图像问答** - 回答关于单张图像的特定问题
- 🆚 **图像对比** - 比较两张图像并总结相关差异

## 环境要求

- Node.js 20+
- 设置以下任一 API 密钥（环境变量）：`VISIONTOOL_API_KEY`、`ANTHROPIC_API_KEY`、`OPENAI_API_KEY` 或 `GEMINI_API_KEY`

## 安装

```bash
npm install
npm run build
```

## 运行

```bash
VISIONTOOL_API_FORMAT=gemini VISIONTOOL_API_KEY=你的密钥 npm run dev
```

## MCP 客户端配置

在 MCP 客户端配置中，将命令指向编译后的服务器：

```json
{
  "mcpServers": {
    "visiontool": {
      "command": "node",
      "args": ["X:/MCP/VisionToolMCP/dist/index.js"],
      "env": {
        "VISIONTOOL_API_KEY": "你的Gemini密钥",
        "VISIONTOOL_API_FORMAT": "gemini",
        "VISIONTOOL_MODEL": "gemini-2.5-flash",
        "VISIONTOOL_BASE_URL": "https://generativelanguage.googleapis.com"
      }
    }
  }
}
```

## 配置选项

支持以下环境变量配置：

| 环境变量 | 说明 | 默认值 |
|---------|------|--------|
| `VISIONTOOL_API_FORMAT` | API 格式：`anthropic`、`openai` 或 `gemini` | `anthropic` |
| `VISIONTOOL_API_KEY` | 统一 API 密钥，也可使用提供商特定的密钥（如 `ANTHROPIC_API_KEY`、`OPENAI_API_KEY`、`GEMINI_API_KEY`） | - |
| `VISIONTOOL_MODEL` | 使用的模型 | `claude-opus-4-8` (Anthropic) / `gpt-4o-mini` (OpenAI) / `gemini-2.5-flash` (Gemini) |
| `VISIONTOOL_BASE_URL` | API 基础 URL | - |
| `VISIONTOOL_TIMEOUT_MS` | 请求超时（毫秒） | `60000` |
| `VISIONTOOL_MAX_IMAGE_BYTES` | 本地/base64 图像最大大小 | `5242880` |
| `VISIONTOOL_RETRIES` | 429/5xx 等临时 API 故障的重试次数 | `2` |
| `VISIONTOOL_RETRY_BASE_MS` | 指数退避重试的基础延迟 | `250` |
| `VISIONTOOL_ALLOWED_IMAGE_ROOTS` | 限制本地图像路径必须位于这些根目录下；多个目录用系统路径分隔符分隔 | 不限制 |
| `VISIONTOOL_DISABLE_URL_INPUTS` | 设为 `1`/`true`/`yes`/`on` 时禁用图像 URL 输入 | `false` |
| `VISIONTOOL_ALLOWED_URL_HOSTS` | 限制图像 URL host，逗号分隔；支持 `*.example.com` | 不限制 |
| `VISIONTOOL_ALLOW_PRIVATE_URLS` | 允许 localhost/private IP 图像 URL 被发送给上游视觉模型 | `false` |
| `VISIONTOOL_PROXY_URL` | 网络错误后重试使用的代理 URL | `HTTP_PROXY` / `HTTPS_PROXY` / `http://127.0.0.1:7890` |
| `VISIONTOOL_DISABLE_PROXY_FALLBACK` | 设为 `1`/`true`/`yes`/`on` 时关闭代理 fallback | `false` |
| `VISIONTOOL_BLOCK_CALLER_PREFIXES` | 调用者模型名黑名单，用逗号分隔；设置后 `_caller_model` 变为必填，名字**包含**任一关键字的模型被拒绝（子串匹配，如 `gpt` 可挡 `gpt-4o`、`opencode/gpt-5.5`、`azure-gpt-5`） | 空（默认放行所有调用者） |
| `VISIONTOOL_UPLOAD_TTL_MS` | 上传图像的自动过期时间（毫秒） | `1800000` (30 分钟) |
| `VISIONTOOL_ENABLE_OPENCODE` | 设为 `1`/`true`/`yes`/`on` 时启用 opencode 专属的 `opencode_pasted_image` 工具（自动检测 `~/.local/share/opencode/opencode.db`） | `false` |
| `VISIONTOOL_OPENCODE_DB` | 显式指定 opencode 数据库路径；设置后同样启用 `opencode_pasted_image` 工具 | - |
| `VISIONTOOL_ENABLE_CLAUDE` | 设为 `1`/`true`/`yes`/`on` 时启用 Claude Code 专属的 `claude_pasted_image` 工具；在 Claude Code 会话内检测到 `CLAUDE_CODE_SESSION_ID` 时也会自动启用 | `false` |
| `VISIONTOOL_CLAUDE_PROJECTS_DIR` | Claude Code transcript 目录，默认 `~/.claude/projects` | `~/.claude/projects` |
| `VISIONTOOL_CLAUDE_MAX_JSONL_BYTES` | 读取 transcript 的大小上限（字节），超过则报错 | `104857600` (100 MB) |

## 调用者护栏

此 MCP 服务器**默认对任何调用者开放**，不要求 `_caller_model`。理由：是否需要视觉能力是**上下文判断**而非身份判断--同一个多模态模型，能直接看到图片时不该调用，拿到 `[Unsupported Image]` 时就该调用。靠模型名白名单拦截会误伤正当用户（比如 Claude 因 harness 传输问题看不到图、却需要求助），且白名单本身可被调用方伪造，并非硬保证。

护栏放在**工具描述**里：每个视觉工具的描述都写明"仅当你无法直接看到图片时调用，能直接看到图的多模态模型请勿调用"，由调用方自判。

如果需要**硬保证**某个模型族永不路由到本服务器（例如防止某个多模态模型误调用浪费额度），可设置可选黑名单：

```bash
# 永不允许 claude / gpt 系列模型调用
VISIONTOOL_BLOCK_CALLER_PREFIXES=claude,gpt
```

设置黑名单后：
- `_caller_model` 变为**必填**（否则拒绝，因为无法判断是否在黑名单内）
- 匹配方式为**子串包含**：只要模型名（含 `provider/` 前缀部分）中出现任一关键字即被挡。例如 `gpt` 可挡 `gpt-4o`、`opencode/gpt-5.5`、`azure-gpt-5`

完整配置示例请参考 [`.env.example`](.env.example) 文件。

## opencode 专属：opencode_pasted_image

这是一个**仅 opencode 适用**的工具，默认不注册。当 MCP 服务运行在 opencode 会话内时，设置 `VISIONTOOL_ENABLE_OPENCODE=1`（或 `VISIONTOOL_OPENCODE_DB=<opencode.db 路径>`）即可启用。

**为什么需要它**：在 opencode 中粘贴的图片以 base64 data URL 内联存储在 `opencode.db` 的 `part` 表里，**不落盘**。纯文本调用模型（如 GLM/DeepSeek）接不到图片附件，会报 `Cannot read "image.png" (this model does not support image input)`，图片被丢弃。该工具直接从数据库里把"当前会话最新一张粘贴图"取出来，解码落盘并注册为上传，返回 `imageId`/`path` 供其它视觉工具使用。

**会话识别**：用"最新一条 `message` 所属的 session"作为当前会话（比 `session.time_updated` 更可靠，后者会被 UI 选中事件刷新）。只在当前会话内取最新图片，**不会跨会话兜底**——当前会话没有粘贴图时会直接报错，避免拿错别的对话的图。

**用法**：

```json
{
  "name": "opencode_pasted_image",
  "arguments": {}
}
```

响应：

```json
{
  "tool": "opencode_pasted_image",
  "imageId": "6b49d277-c27d-434c-8e52-8601c8f5a1fb",
  "path": "C:\\...\\visiontool-mcp-uploads\\image.png",
  "bytes": 356351,
  "mediaType": "image/png",
  "filename": "image.png",
  "sessionId": "ses_xxx",
  "timeCreated": 1785249912775,
  "expiresAt": "2026-07-28T15:31:08.235Z"
}
```

然后用 `imageId` 调用其它工具：

```json
{
  "name": "describe_image",
  "arguments": {
    "image": { "imageId": "6b49d277-c27d-434c-8e52-8601c8f5a1fb" }
  }
}
```

依赖 Node 22.5+ 内置的 `node:sqlite`（运行时动态导入，不可用时该工具会清晰报错，不影响其它工具）。其它 agent 平台（无 opencode 数据库）请勿启用。

## Claude Code 专属：claude_pasted_image

这是一个**仅 Claude Code 适用**的工具，在 Claude Code 会话内默认自动注册（Claude Code 会向 MCP server 注入 `CLAUDE_CODE_SESSION_ID` 环境变量）。也可手动设 `VISIONTOOL_ENABLE_CLAUDE=1` 启用。

**为什么需要它**：当用户在 Claude Code 对话里发了图片、但调用方模型不支持图像输入时，harness 会把图片块替换成 `[Unsupported Image]` 占位文本--调用方上下文里既无像素也无文件路径，唯一完整副本是以 base64 内联在会话 transcript JSONL（`~/.claude/projects/<转义后的cwd>/<sessionId>.jsonl`）里的。该工具直接从当前会话 transcript 里取"最新一张内联图"，解码落盘并注册为上传，返回 `imageId`/`path` 供其它视觉工具使用。

**会话识别**：用 Claude Code 注入的 `CLAUDE_CODE_SESSION_ID` 定位当前会话 transcript，目录转义规则为 `process.cwd()` 中每个非字母数字字符替换为 `-`。找不到时会在所有项目子目录下兜底查找该 sessionId 的 jsonl。只在当前会话内取最新图片，**不会跨会话兜底**--当前会话没有内联图时直接报错。

**用法**：

```json
{
  "name": "claude_pasted_image",
  "arguments": {}
}
```

响应：

```json
{
  "tool": "claude_pasted_image",
  "imageId": "6b49d277-c27d-434c-8e52-8601c8f5a1fb",
  "path": "C:\\...\\visiontool-mcp-uploads\\image.png",
  "bytes": 107005,
  "mediaType": "image/png",
  "sessionId": "fd71614b-ca6e-42e8-a55c-b990f0cb6247",
  "expiresAt": "2026-07-29T11:30:00.000Z"
}
```

然后用 `imageId` 调用其它工具：

```json
{
  "name": "describe_image",
  "arguments": {
    "image": { "imageId": "6b49d277-c27d-434c-8e52-8601c8f5a1fb" }
  }
}
```

无需任何额外依赖（仅用 Node 内置 `fs`）。其它 agent 平台（无 `CLAUDE_CODE_SESSION_ID`）请勿启用。

## 工具调用示例

所有工具默认无需 `_caller_model`（仅在设置了 `VISIONTOOL_BLOCK_CALLER_PREFIXES` 黑名单时才必填）：

### 图像上传（推荐）

对于对话中的截图或 Agent 自己生成的图像，先上传获得 `imageId` 可多次使用，30 分钟后自动清理：

```json
{
  "name": "upload_image",
  "arguments": {
    "base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
    "mediaType": "image/png",
    "filename": "screenshot"
  }
}
```

响应：
```json
{
  "tool": "upload_image",
  "imageId": "550e8400-e29b-41d4-a716-446655440000",
  "path": "/tmp/visiontool-mcp-uploads/screenshot.png",
  "bytes": 1234,
  "expiresAt": "2024-06-25T03:30:00.000Z"
}
```

然后用 `imageId` 调用其他工具：

```json
{
  "name": "describe_image",
  "arguments": {
    "image": {
      "imageId": "550e8400-e29b-41d4-a716-446655440000"
    }
  }
}
```

### 其他输入方式

也支持 `path`（本地文件）、`base64`（直接传数据）或 `url`（公开 URL）：

```json
{
  "name": "describe_image",
  "arguments": {
    "image": {
      "path": "X:/screenshots/current.png"
    },
    "detail": "medium",
    "maxTokens": 1024
  }
}
```

工具响应会同时包含可读文本和结构化对象。结构化对象字段为：

```json
{
  "tool": "describe_image",
  "model": "gemini-2.5-flash",
  "apiFormat": "gemini",
  "text": "model response text",
  "images": [
    {
      "source": "path",
      "mediaType": "image/png",
      "path": "X:/screenshots/current.png",
      "bytes": 12345
    }
  ]
}
```

## 图像输入安全边界

- `imageId` - 通过 `upload_image` 返回的 ID，30 分钟自动清理（推荐，最安全）
- `path` - 绝对或相对本地图像路径
- `base64` - 原始 base64 图像数据
- `url` - 可公开访问的图像 URL

支持的 MIME 类型：PNG、JPEG、WebP 和 GIF。

注意：`path` 会读取 MCP 服务器进程可访问的本地文件；`url` 会把 URL 交给上游视觉模型提供商读取。生产环境建议设置 `VISIONTOOL_ALLOWED_IMAGE_ROOTS`、`VISIONTOOL_ALLOWED_URL_HOSTS`，或用 `VISIONTOOL_DISABLE_URL_INPUTS=1` 禁用 URL 输入。默认会拒绝 localhost/private IP URL；确有需要时才设置 `VISIONTOOL_ALLOW_PRIVATE_URLS=1`。

## 网络与代理

遇到可重试的网络错误时，服务器会先直连，再通过代理 fallback 重试。代理选择顺序为 `VISIONTOOL_PROXY_URL`、`HTTPS_PROXY`、`HTTP_PROXY`、`http://127.0.0.1:7890`。如果当前环境不需要代理 fallback，可设置 `VISIONTOOL_DISABLE_PROXY_FALLBACK=1`。

## 开发

```bash
npm test
npm run build
```

## Agent 使用说明

此服务器与截图/捕获 MCP 配合使用。先用另一个工具截取屏幕截图，将返回的文件路径传递给 `describe_image` 或 `answer_about_image`，然后使用结构化的文本响应来决定下一步操作。
