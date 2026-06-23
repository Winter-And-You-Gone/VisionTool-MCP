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
  compareImagesSchema,
  describeImageSchema,
  ocrImageSchema,
  toolInputSchemas,
  visionResultOutputSchema
} from './schemas.js';
import {
  answerAboutImage,
  compareImages,
  describeImage,
  ocrImage,
  type VisionResult
} from './vision.js';

const moduleRoot = path.dirname(fileURLToPath(import.meta.url));
const runtimeRoot = ['dist', 'src'].includes(path.basename(moduleRoot)) ? path.dirname(moduleRoot) : moduleRoot;
const packageVersion = readPackageVersion();

// Caller model whitelist: only allow GLM / DeepSeek series by default
export const defaultAllowedCallerPrefixes = ['glm', 'deepseek'];

const visionToolAnnotations: ToolAnnotations = {
  readOnlyHint: true,
  destructiveHint: false,
  idempotentHint: true,
  openWorldHint: true
};

export function assertCallerAllowed(callerModel: unknown): void {
  const rawAllowed = process.env.VISIONTOOL_ALLOWED_CALLER_PREFIXES?.trim();
  const prefixes = rawAllowed
    ? rawAllowed.split(',').map((p) => p.trim().toLowerCase()).filter(Boolean)
    : defaultAllowedCallerPrefixes;
  const caller = typeof callerModel === 'string' ? callerModel.trim() : '';
  if (!caller) {
    throw new McpError(
      ErrorCode.InvalidParams,
      '_caller_model is required. The caller must identify its model before using VisionToolMCP.'
    );
  }
  // '*' means allow any non-empty caller.
  if (prefixes.includes('*')) {
    return;
  }
  const callerLower = caller.toLowerCase();
  const allowed = prefixes.some((prefix) => callerLower.startsWith(prefix));
  if (!allowed) {
    throw new McpError(
      ErrorCode.InvalidParams,
      `Caller model "${caller}" is not allowed. VISIONTOOL_ALLOWED_CALLER_PREFIXES only permits models starting with ${prefixes.map((p) => `"${p}"`).join(', ')}.`
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

  server.setRequestHandler(ListToolsRequestSchema, async (): Promise<ListToolsResult> => ({
    tools: [
      {
        name: 'describe_image',
        description: '【默认仅限 GLM/DeepSeek 系列模型调用】Describe an image for a text-only agent. Accepts a local path, base64 image data, or URL.',
        inputSchema: asInputSchema(toolInputSchemas.describe_image),
        outputSchema: asOutputSchema(visionResultOutputSchema),
        annotations: { ...visionToolAnnotations, title: 'Describe Image' }
      },
      {
        name: 'ocr_image',
        description: '【默认仅限 GLM/DeepSeek 系列模型调用】Extract visible text from an image with optional language and layout hints.',
        inputSchema: asInputSchema(toolInputSchemas.ocr_image),
        outputSchema: asOutputSchema(visionResultOutputSchema),
        annotations: { ...visionToolAnnotations, title: 'OCR Image' }
      },
      {
        name: 'answer_about_image',
        description: '【默认仅限 GLM/DeepSeek 系列模型调用】Answer a specific question using visual evidence from one image.',
        inputSchema: asInputSchema(toolInputSchemas.answer_about_image),
        outputSchema: asOutputSchema(visionResultOutputSchema),
        annotations: { ...visionToolAnnotations, title: 'Answer About Image' }
      },
      {
        name: 'compare_images',
        description: '【默认仅限 GLM/DeepSeek 系列模型调用】Compare two images and summarize relevant differences for a text-only agent.',
        inputSchema: asInputSchema(toolInputSchemas.compare_images),
        outputSchema: asOutputSchema(visionResultOutputSchema),
        annotations: { ...visionToolAnnotations, title: 'Compare Images' }
      }
    ]
  }));

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
    assertCallerAllowed(callerModel);

    switch (name) {
      case 'describe_image':
        return jsonResult(await describeImage(parseArgs(describeImageSchema, args)));
      case 'ocr_image':
        return jsonResult(await ocrImage(parseArgs(ocrImageSchema, args)));
      case 'answer_about_image':
        return jsonResult(await answerAboutImage(parseArgs(answerAboutImageSchema, args)));
      case 'compare_images':
        return jsonResult(await compareImages(parseArgs(compareImagesSchema, args)));
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
