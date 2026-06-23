#!/usr/bin/env node
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ErrorCode,
  ListToolsRequestSchema,
  McpError
} from '@modelcontextprotocol/sdk/types.js';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { z } from 'zod';

import {
  answerAboutImageSchema,
  compareImagesSchema,
  describeImageSchema,
  ocrImageSchema,
  toolInputSchemas
} from './schemas.js';
import {
  answerAboutImage,
  compareImages,
  describeImage,
  ocrImage
} from './vision.js';

const moduleRoot = path.dirname(fileURLToPath(import.meta.url));
const runtimeRoot = ['dist', 'src'].includes(path.basename(moduleRoot)) ? path.dirname(moduleRoot) : moduleRoot;
const packageVersion = readPackageVersion();

// Caller model whitelist: only allow GLM / DeepSeek series by default
const defaultAllowedCallerPrefixes = ['glm', 'deepseek'];

function assertCallerAllowed(callerModel?: string): void {
  const rawAllowed = process.env.VISIONTOOL_ALLOWED_CALLER_PREFIXES?.trim();
  const prefixes = rawAllowed
    ? rawAllowed.split(',').map((p) => p.trim().toLowerCase()).filter(Boolean)
    : defaultAllowedCallerPrefixes;
  // '*' means allow all callers
  if (prefixes.includes('*')) {
    return;
  }
  // If no caller model provided, just return (make it optional)
  if (!callerModel) {
    return;
  }
  const callerLower = callerModel.toLowerCase();
  const allowed = prefixes.some((prefix) => callerLower.startsWith(prefix));
  if (!allowed) {
    throw new McpError(
      ErrorCode.InvalidParams,
      `Caller model "${callerModel}" is not allowed. VISIONTOOL_ALLOWED_CALLER_PREFIXES only permits models starting with ${prefixes.map((p) => `"${p}"`).join(', ')}.`
    );
  }
}

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

server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [
    {
      name: 'describe_image',
      description: '【仅限 GLM/DeepSeek 系列模型调用】Describe an image for a text-only agent. Accepts a local path, base64 image data, or URL.',
      inputSchema: toolInputSchemas.describe_image
    },
    {
      name: 'ocr_image',
      description: '【仅限 GLM/DeepSeek 系列模型调用】Extract visible text from an image with optional language and layout hints.',
      inputSchema: toolInputSchemas.ocr_image
    },
    {
      name: 'answer_about_image',
      description: '【仅限 GLM/DeepSeek 系列模型调用】Answer a specific question using visual evidence from one image.',
      inputSchema: toolInputSchemas.answer_about_image
    },
    {
      name: 'compare_images',
      description: '【仅限 GLM/DeepSeek 系列模型调用】Compare two images and summarize relevant differences for a text-only agent.',
      inputSchema: toolInputSchemas.compare_images
    }
  ]
}));

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;
  const callerModel = (args as Record<string, unknown>)?._caller_model as string | undefined;

  try {
    // Validate caller model whitelist BEFORE processing
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
});

function parseArgs<T extends z.ZodTypeAny>(schema: T, args: unknown): z.infer<T> {
  const parsed = schema.safeParse(args ?? {});
  if (!parsed.success) {
    throw new McpError(ErrorCode.InvalidParams, z.prettifyError(parsed.error));
  }
  return parsed.data;
}

function jsonResult(value: unknown) {
  return {
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

const transport = new StdioServerTransport();
await server.connect(transport);
