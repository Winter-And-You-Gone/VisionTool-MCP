# VisionToolMCP

VisionToolMCP 是一个 MCP 服务器，为纯文本 Agent 提供视觉能力桥梁。它接受图像文件、base64 图像数据或图像 URL，将它们发送到多模态模型，并通过 MCP 返回结构化文本结果。

## 功能特性

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
        "VISIONTOOL_MODEL": "gemini-3.5-flash",
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
| `VISIONTOOL_MODEL` | 使用的模型 | `claude-opus-4-8` (Anthropic) / `gpt-4o-mini` (OpenAI) / `gemini-3.5-flash` (Gemini) |
| `VISIONTOOL_BASE_URL` | API 基础 URL | - |
| `VISIONTOOL_TIMEOUT_MS` | 请求超时（毫秒） | `60000` |
| `VISIONTOOL_MAX_IMAGE_BYTES` | 本地/base64 图像最大大小 | `5242880` |
| `VISIONTOOL_RETRIES` | 429/5xx 等临时 API 故障的重试次数 | `2` |
| `VISIONTOOL_RETRY_BASE_MS` | 指数退避重试的基础延迟 | `250` |
| `VISIONTOOL_ALLOWED_CALLER_PREFIXES` | 调用者模型前缀白名单，用逗号分隔 | `glm,deepseek` |

## 调用者模型白名单

默认情况下，此 MCP 服务器启用调用者模型白名单：

- **默认允许的前缀**：`glm`、`deepseek`
- **必需参数**：`_caller_model` - 调用模型必须标识自身

这可以防止昂贵的多模态模型（如 GPT-4o、Claude）意外调用此 MCP 并浪费 API 额度。自定义白名单示例：

```bash
# 仅允许 GLM 和 Qwen 系列
VISIONTOOL_ALLOWED_CALLER_PREFIXES=glm,qwen

# 允许所有调用者（不推荐用于生产环境）
VISIONTOOL_ALLOWED_CALLER_PREFIXES=*
```

完整配置示例请参考 [`.env.example`](.env.example) 文件。

- `path` - 绝对或相对本地图像路径
- `base64` - 原始 base64 图像数据
- `url` - 可公开访问的图像 URL

支持的 MIME 类型：PNG、JPEG、WebP 和 GIF。

## 开发

```bash
npm test
npm run build
```

## Agent 使用说明

此服务器与截图/捕获 MCP 配合使用。先用另一个工具截取屏幕截图，将返回的文件路径传递给 `describe_image` 或 `answer_about_image`，然后使用结构化的文本响应来决定下一步操作。
