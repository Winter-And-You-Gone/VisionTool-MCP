#!/usr/bin/env node
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  type CallToolResult,
  ErrorCode,
  type ListToolsResult,
  ListToolsRequestSchema,
  McpError,
  type Tool,
  type ToolAnnotations
} from '@modelcontextprotocol/sdk/types.js';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { z } from 'zod';

import {
  answerAboutImageSchema,
  claudePastedImageSchema,
  claudePastedImageResultOutputSchema,
  claudePastedImageToolInputSchema,
  compareImagesSchema,
  describeImageSchema,
  ocrImageSchema,
  opencodePastedImageResultOutputSchema,
  opencodePastedImageSchema,
  opencodePastedImageToolInputSchema,
  toolInputSchemas,
  uploadImageSchema,
  uploadImageToolInputSchema,
  uploadImageResultOutputSchema,
  visionResultOutputSchema
} from './schemas.js';
import {
  answerAboutImage,
  claudeEnabled,
  claudePastedImage,
  compareImages,
  describeImage,
  ocrImage,
  opencodeEnabled,
  opencodePastedImage,
  type VisionResult,
  type UploadResult,
  type ClaudePastedImageResult,
  type OpencodePastedImageResult,
  uploadImage
} from './vision.js';

const moduleRoot = path.dirname(fileURLToPath(import.meta.url));
const runtimeRoot = ['dist', 'src'].includes(path.basename(moduleRoot)) ? path.dirname(moduleRoot) : moduleRoot;
const packageVersion = readPackageVersion();

const visionToolAnnotations: ToolAnnotations = {
  readOnlyHint: true,
  destructiveHint: false,
  idempotentHint: true,
  openWorldHint: true
};

// Guidance prepended to every vision-tool description. The server is open by
// default (no caller identity required); this tells multimodal callers not to
// waste an upstream call when they can already see the image, while inviting
// text-only or unsupported-image callers to use the tools.
const usageGuard = '【护栏】仅当你无法直接看到图片时调用（例如你是纯文本模型，或图片在上下文中显示为 [Unsupported Image]）。能直接看到图片的多模态模型请勿调用，否则会浪费一次冗余的上游视觉请求。';

// Optional caller-model blocklist. By default no model is blocked: the server
// trusts the caller to decide whether it needs vision help (a multimodal model
// that can already see an image should not call these tools - that judgement is
// in the tool descriptions, not enforced by identity). Only when an operator
// wants a hard guarantee that a specific model family never routes here do they
// set VISIONTOOL_BLOCK_CALLER_PREFIXES; in that case _caller_model becomes
// mandatory and any caller whose model id contains a blocked substring is
// rejected. Matching is substring-based so "gpt" blocks "gpt-4o",
// "opencode/gpt-5.5", and "azure-gpt-5" alike.
export function assertCallerNotBlocked(callerModel: unknown): void {
  const rawBlocked = process.env.VISIONTOOL_BLOCK_CALLER_PREFIXES?.trim();
  if (!rawBlocked) {
    return; // No blocklist configured: open by default.
  }
  const substrings = rawBlocked.split(',').map((p) => p.trim().toLowerCase()).filter(Boolean);
  const caller = typeof callerModel === 'string' ? callerModel.trim() : '';
  if (!caller) {
    throw new McpError(
      ErrorCode.InvalidParams,
      '_caller_model is required because VISIONTOOL_BLOCK_CALLER_PREFIXES is set. Identify your model so it can be checked against the blocklist.'
    );
  }
  const callerLower = caller.toLowerCase();
  const blocked = substrings.some((sub) => callerLower.includes(sub));
  if (blocked) {
    throw new McpError(
      ErrorCode.InvalidParams,
      `Caller model "${caller}" is blocked by VISIONTOOL_BLOCK_CALLER_PREFIXES (${substrings.map((p) => `"${p}"`).join(', ')}).`
    );
  }
}

export function createVisionToolServer(): Server {
  const server = new Server(
    {
      name: 'visiontool-mcp',
      version: packageVersion
    },
    {
      capabilities: {
        tools: {}
      }
    }
  );

  server.setRequestHandler(ListToolsRequestSchema, async (): Promise<ListToolsResult> => {
    const tools: Tool[] = [
      {
        name: 'upload_image',
        description: `${usageGuard} Upload an image via base64 for later use with other vision tools. Auto-deleted after 30 minutes.`,
        inputSchema: asInputSchema(uploadImageToolInputSchema),
        outputSchema: asOutputSchema(uploadImageResultOutputSchema),
        annotations: { ...visionToolAnnotations, title: 'Upload Image' }
      },
      {
        name: 'describe_image',
        description: `${usageGuard} Describe an image for a text-only agent. Accepts a local path, base64 image data, URL, or imageId from upload_image.`,
        inputSchema: asInputSchema(toolInputSchemas.describe_image),
        outputSchema: asOutputSchema(visionResultOutputSchema),
        annotations: { ...visionToolAnnotations, title: 'Describe Image' }
      },
      {
        name: 'ocr_image',
        description: `${usageGuard} Extract visible text from an image with optional language and layout hints.`,
        inputSchema: asInputSchema(toolInputSchemas.ocr_image),
        outputSchema: asOutputSchema(visionResultOutputSchema),
        annotations: { ...visionToolAnnotations, title: 'OCR Image' }
      },
      {
        name: 'answer_about_image',
        description: `${usageGuard} Answer a specific question using visual evidence from one image.`,
        inputSchema: asInputSchema(toolInputSchemas.answer_about_image),
        outputSchema: asOutputSchema(visionResultOutputSchema),
        annotations: { ...visionToolAnnotations, title: 'Answer About Image' }
      },
      {
        name: 'compare_images',
        description: `${usageGuard} Compare two images and summarize relevant differences for a text-only agent.`,
        inputSchema: asInputSchema(toolInputSchemas.compare_images),
        outputSchema: asOutputSchema(visionResultOutputSchema),
        annotations: { ...visionToolAnnotations, title: 'Compare Images' }
      }
    ];

    if (opencodeEnabled()) {
      tools.push({
        name: 'opencode_pasted_image',
        description: '【opencode 专属】Extract the most recently pasted image from the current opencode session database and return an imageId/path for use with describe_image / ocr_image / answer_about_image / compare_images. opencode stores pasted images inline as base64 in opencode.db (never written to disk), and text-only caller models cannot receive the image attachment directly. This tool reads the active session (highest session.time_updated), finds the latest image part in that session only, decodes it, and registers it as an upload. It does NOT fall back to other sessions. Only registered when VISIONTOOL_ENABLE_OPENCODE=1 or VISIONTOOL_OPENCODE_DB is set.',
        inputSchema: asInputSchema(opencodePastedImageToolInputSchema),
        outputSchema: asOutputSchema(opencodePastedImageResultOutputSchema),
        annotations: { readOnlyHint: false, destructiveHint: false, idempotentHint: false, openWorldHint: false, title: 'opencode Pasted Image' }
      });
    }

    if (claudeEnabled()) {
      tools.push({
        name: 'claude_pasted_image',
        description: '【Claude Code 专属】Extract the most recently pasted image from the current Claude Code session transcript and return an imageId/path for use with describe_image / ocr_image / answer_about_image / compare_images. When a caller model cannot receive image attachments directly (e.g. the image shows up in context as "[Unsupported Image]"), call this tool — it reads the active Claude Code session transcript (CLAUDE_CODE_SESSION_ID), finds the latest inline base64 image block, decodes it, and registers it as an upload. It does NOT fall back to other sessions. Only registered when VISIONTOOL_ENABLE_CLAUDE=1 or CLAUDE_CODE_SESSION_ID is set (auto-detected inside Claude Code).',
        inputSchema: asInputSchema(claudePastedImageToolInputSchema),
        outputSchema: asOutputSchema(claudePastedImageResultOutputSchema),
        annotations: { readOnlyHint: false, destructiveHint: false, idempotentHint: false, openWorldHint: false, title: 'Claude Pasted Image' }
      });
    }

    return { tools };
  });

  server.setRequestHandler(CallToolRequestSchema, async (request): Promise<CallToolResult> => handleToolCall(request.params.name, request.params.arguments));
  return server;
}

export async function startVisionToolServer(): Promise<void> {
  const server = createVisionToolServer();
  const transport = new StdioServerTransport();
  await server.connect(transport);
}

async function handleToolCall(name: string, args: unknown): Promise<CallToolResult> {
  const callerModel = (args as Record<string, unknown>)?._caller_model;

  try {
    assertCallerNotBlocked(callerModel);

    switch (name) {
      case 'upload_image':
        return jsonUploadResult(await uploadImage(parseArgs(uploadImageSchema, args)));
      case 'describe_image':
        return jsonResult(await describeImage(parseArgs(describeImageSchema, args)));
      case 'ocr_image':
        return jsonResult(await ocrImage(parseArgs(ocrImageSchema, args)));
      case 'answer_about_image':
        return jsonResult(await answerAboutImage(parseArgs(answerAboutImageSchema, args)));
      case 'compare_images':
        return jsonResult(await compareImages(parseArgs(compareImagesSchema, args)));
      case 'opencode_pasted_image':
        if (!opencodeEnabled()) {
          throw new McpError(
            ErrorCode.MethodNotFound,
            'opencode_pasted_image is not enabled. Set VISIONTOOL_ENABLE_OPENCODE=1 or VISIONTOOL_OPENCODE_DB to the opencode.db path to enable it.'
          );
        }
        return jsonOpencodeResult(await opencodePastedImage(parseArgs(opencodePastedImageSchema, args)));
      case 'claude_pasted_image':
        if (!claudeEnabled()) {
          throw new McpError(
            ErrorCode.MethodNotFound,
            'claude_pasted_image is not enabled. Run inside a Claude Code session (which sets CLAUDE_CODE_SESSION_ID) or set VISIONTOOL_ENABLE_CLAUDE=1.'
          );
        }
        return jsonClaudeResult(await claudePastedImage(parseArgs(claudePastedImageSchema, args)));
      default:
        throw new McpError(ErrorCode.MethodNotFound, `Unknown tool: ${name}`);
    }
  } catch (error) {
    if (error instanceof McpError) {
      throw error;
    }

    return {
      isError: true,
      content: [
        {
          type: 'text',
          text: formatError(error)
        }
      ]
    };
  }
}

function parseArgs<T extends z.ZodTypeAny>(schema: T, args: unknown): z.infer<T> {
  const parsed = schema.safeParse(args ?? {});
  if (!parsed.success) {
    throw new McpError(ErrorCode.InvalidParams, z.prettifyError(parsed.error));
  }
  return parsed.data;
}

function asInputSchema(schema: unknown): Tool['inputSchema'] {
  return structuredClone(schema) as Tool['inputSchema'];
}

function asOutputSchema(schema: unknown): NonNullable<Tool['outputSchema']> {
  return structuredClone(schema) as NonNullable<Tool['outputSchema']>;
}

function jsonResult(value: VisionResult): CallToolResult {
  return {
    structuredContent: value as unknown as Record<string, unknown>,
    content: [
      {
        type: 'text',
        text: JSON.stringify(value, null, 2)
      }
    ]
  };
}

function jsonUploadResult(value: UploadResult): CallToolResult {
  return {
    structuredContent: value as unknown as Record<string, unknown>,
    content: [
      {
        type: 'text',
        text: JSON.stringify(value, null, 2)
      }
    ]
  };
}

function jsonOpencodeResult(value: OpencodePastedImageResult): CallToolResult {
  return {
    structuredContent: value as unknown as Record<string, unknown>,
    content: [
      {
        type: 'text',
        text: JSON.stringify(value, null, 2)
      }
    ]
  };
}

function jsonClaudeResult(value: ClaudePastedImageResult): CallToolResult {
  return {
    structuredContent: value as unknown as Record<string, unknown>,
    content: [
      {
        type: 'text',
        text: JSON.stringify(value, null, 2)
      }
    ]
  };
}

function formatError(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }
  return String(error);
}

function readPackageVersion(): string {
  const packageJsonPath = path.join(runtimeRoot, 'package.json');
  try {
    const content = readFileSync(packageJsonPath, 'utf8');
    const parsed = JSON.parse(content) as { version?: unknown };
    return typeof parsed.version === 'string' && parsed.version.length > 0 ? parsed.version : '0.0.0';
  } catch {
    return '0.0.0';
  }
}

function isDirectRun(): boolean {
  return process.argv[1] !== undefined && import.meta.url === pathToFileURL(process.argv[1]).href;
}

if (isDirectRun()) {
  await startVisionToolServer();
}
