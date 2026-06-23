import { readFile, stat } from 'node:fs/promises';
import path from 'node:path';
import https from 'node:https';
import { HttpsProxyAgent } from 'hpagent';

import type {
  AnswerAboutImageInput,
  CompareImagesInput,
  DescribeImageInput,
  ImageInput,
  OcrImageInput
} from './schemas.js';
import { supportedMimeTypes } from './schemas.js';

export type VisionResult = {
  tool: string;
  model: string;
  apiFormat: ApiFormat;
  text: string;
  images: Array<{
    source: 'path' | 'base64' | 'url';
    mediaType?: string;
    path?: string;
    url?: string;
    bytes?: number;
  }>;
};

type ApiFormat = 'anthropic' | 'openai' | 'gemini';

type ApiConfig = {
  apiFormat: ApiFormat;
  apiKey: string;
  model: string;
  url: string;
};

type PreparedImage = {
  source: 'path' | 'base64' | 'url';
  mediaType?: string;
  data?: string;
  url?: string;
  path?: string;
  bytes?: number;
};

type AnthropicContentBlock =
  | { type: 'text'; text: string }
  | {
      type: 'image';
      source:
        | { type: 'base64'; media_type: string; data: string }
        | { type: 'url'; url: string };
    };

type OpenAIContentBlock =
  | { type: 'text'; text: string }
  | { type: 'image_url'; image_url: { url: string } };

type GeminiInputBlock =
  | { text: string }
  | { inlineData: { mimeType: string; data: string } }
  | { fileData: { mimeType: string; fileUri: string } };

type AnthropicMessageResponse = {
  content?: Array<{ type?: string; text?: string }>;
  error?: { message?: string };
};

type OpenAIChatCompletionResponse = {
  choices?: Array<{
    message?: {
      content?: string | Array<{ type?: string; text?: string }>;
    };
  }>;
  error?: { message?: string };
};

type GeminiInteractionResponse = {
  output_text?: string;
  candidates?: Array<{
    content?: {
      parts?: Array<{ text?: string }>;
    };
  }>;
  error?: { message?: string };
};

type ApiResponseBody = AnthropicMessageResponse & OpenAIChatCompletionResponse & GeminiInteractionResponse;

const defaultAnthropicModel = 'claude-opus-4-8';
const defaultOpenAIModel = 'gpt-4o-mini';
const defaultGeminiModel = 'gemini-2.5-flash';
const defaultAnthropicBaseUrl = 'https://api.anthropic.com';
const defaultOpenAIBaseUrl = 'https://api.openai.com';
const defaultGeminiBaseUrl = 'https://generativelanguage.googleapis.com';
const defaultTimeoutMs = 60_000;
const defaultMaxImageBytes = 5 * 1024 * 1024;
const defaultRetryCount = 2;
const defaultRetryBaseMs = 250;
const maxRetryCount = 5;
const retryableStatuses = new Set([408, 409, 425, 429, 500, 502, 503, 504]);
const supportedMimeTypeSet = new Set<string>(supportedMimeTypes);
const defaultProxyUrl = 'http://127.0.0.1:7890';

// Cache proxy agent to avoid recreating it
let proxyAgent: HttpsProxyAgent | undefined;
const extensionMimeTypes = new Map<string, string>([
  ['.png', 'image/png'],
  ['.jpg', 'image/jpeg'],
  ['.jpeg', 'image/jpeg'],
  ['.webp', 'image/webp'],
  ['.gif', 'image/gif']
]);

export async function describeImage(input: DescribeImageInput): Promise<VisionResult> {
  const image = await prepareImage(input.image);
  const prompt = [
    'Describe the image for a text-only agent. Be concise, factual, and include visible UI/text/layout details when relevant.',
    `Detail level: ${input.detail}.`,
    input.focus ? `Focus: ${input.focus}` : undefined,
    input.instruction ? `Instruction: ${input.instruction}` : undefined
  ].filter(Boolean).join('\n');

  return runVisionTool('describe_image', [image], prompt, input.maxTokens, input.model);
}

export async function ocrImage(input: OcrImageInput): Promise<VisionResult> {
  const image = await prepareImage(input.image);
  const prompt = [
    'Extract all visible text from the image. Return the text only when possible, with brief notes for unreadable or uncertain areas.',
    input.language ? `Language hint: ${input.language}` : undefined,
    `Preserve layout: ${input.preserveLayout ? 'yes' : 'no'}.`,
    `Detail level: ${input.detail}.`
  ].filter(Boolean).join('\n');

  return runVisionTool('ocr_image', [image], prompt, input.maxTokens, input.model);
}

export async function answerAboutImage(input: AnswerAboutImageInput): Promise<VisionResult> {
  const image = await prepareImage(input.image);
  const prompt = [
    'Answer the question using only visual evidence from the image. If the image does not contain enough evidence, say so clearly.',
    `Detail level: ${input.detail}.`,
    `Question: ${input.question}`
  ].join('\n');

  return runVisionTool('answer_about_image', [image], prompt, input.maxTokens, input.model);
}

export async function compareImages(input: CompareImagesInput): Promise<VisionResult> {
  const firstImage = await prepareImage(input.firstImage);
  const secondImage = await prepareImage(input.secondImage);
  const prompt = [
    'Compare image 1 and image 2 for a text-only agent. Summarize important similarities, differences, and likely implications.',
    `Detail level: ${input.detail}.`,
    input.instruction ? `Instruction: ${input.instruction}` : undefined
  ].filter(Boolean).join('\n');

  return runVisionTool('compare_images', [firstImage, secondImage], prompt, input.maxTokens, input.model);
}

async function prepareImage(input: ImageInput): Promise<PreparedImage> {
  if (input.url !== undefined) {
    return { source: 'url', url: input.url };
  }

  if (input.base64 !== undefined) {
    const data = normalizeBase64(input.base64);
    const bytes = Buffer.byteLength(data, 'base64');
    assertImageSize(bytes);
    return {
      source: 'base64',
      mediaType: input.mediaType,
      data,
      bytes
    };
  }

  if (input.path === undefined) {
    throw new Error('Image input is missing path, base64, or url.');
  }

  const resolvedPath = path.resolve(input.path);
  const mediaType = input.mediaType ?? mediaTypeFromPath(resolvedPath);
  const fileStat = await stat(resolvedPath);
  if (!fileStat.isFile()) {
    throw new Error(`Image path is not a file: ${resolvedPath}`);
  }
  assertImageSize(fileStat.size);

  const data = await readFile(resolvedPath, 'base64');
  return {
    source: 'path',
    mediaType,
    data,
    path: resolvedPath,
    bytes: fileStat.size
  };
}

async function runVisionTool(
  tool: string,
  images: PreparedImage[],
  prompt: string,
  maxTokens: number,
  modelOverride?: string
): Promise<VisionResult> {
  const config = buildApiConfig(modelOverride);
  const text = await callVisionApi(config, images, prompt, maxTokens);
  return {
    tool,
    model: config.model,
    apiFormat: config.apiFormat,
    text,
    images: images.map(({ source, mediaType, path: imagePath, url, bytes }) => ({
      source,
      mediaType,
      path: imagePath,
      url,
      bytes
    }))
  };
}

async function callVisionApi(
  config: ApiConfig,
  images: PreparedImage[],
  prompt: string,
  maxTokens: number
): Promise<string> {
  const timeoutMs = getTimeoutMs();
  const maxAttempts = getRetryCount() + 1;
  let lastError: unknown;
  let useProxy = true; // Default to proxy first since direct connection often times out
  let proxyTried = true;

  for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
    try {
      const agent = useProxy ? getProxyAgent() : undefined;
      const url = new URL(config.url);
      const bodyStr = JSON.stringify(buildRequestBody(config, images, prompt, maxTokens));
      const headers = buildRequestHeaders(config);

      const response = await new Promise<{ ok: boolean; status: number; json: () => Promise<unknown> }>((resolve, reject) => {
        const req = https.request(
          {
            hostname: url.hostname,
            port: url.port,
            path: url.pathname + url.search,
            method: 'POST',
            headers: {
              ...headers,
              'Content-Length': Buffer.byteLength(bodyStr)
            },
            agent,
            timeout: timeoutMs
          },
          (res) => {
            let data = '';
            res.on('data', (chunk) => { data += chunk; });
            res.on('end', () => {
              resolve({
                ok: res.statusCode !== undefined && res.statusCode >= 200 && res.statusCode < 300,
                status: res.statusCode ?? 500,
                json: async () => JSON.parse(data)
              });
            });
          }
        );

        req.on('error', reject);
        req.write(bodyStr);
        req.end();
      });

      const body = await readApiResponse(response);
      if (!response.ok) {
        const message = body.error?.message ?? `${formatApiName(config.apiFormat)} API request failed with HTTP ${response.status}.`;
        if (retryableStatuses.has(response.status) && attempt < maxAttempts - 1) {
          lastError = new Error(message);
          await waitBeforeRetry(attempt, timeoutMs);
          continue;
        }
        throw new Error(message);
      }

      const text = extractResponseText(config.apiFormat, body);
      if (!text) {
        throw new Error(`${formatApiName(config.apiFormat)} API returned no text content.`);
      }

      return text;
    } catch (error) {
      // Network errors - retry
      if (isRetryableFetchError(error)) {
        if (attempt < maxAttempts - 1) {
          lastError = error;
          await waitBeforeRetry(attempt, timeoutMs);
          continue;
        }
      }

      throw error;
    }
  }

  throw lastError instanceof Error ? lastError : new Error(`${formatApiName(config.apiFormat)} API request failed after retries.`);
}

function buildApiConfig(modelOverride?: string): ApiConfig {
  const apiFormat = getApiFormat();
  const apiKey = getApiKey(apiFormat);
  const model = modelOverride ?? process.env.VISIONTOOL_MODEL ?? getDefaultModel(apiFormat);
  let url = buildApiUrl(apiFormat, model);
  
  // For Gemini, append API key to URL
  if (apiFormat === 'gemini') {
    url += (url.includes('?') ? '&' : '?') + `key=${apiKey}`;
  }
  
  return {
    apiFormat,
    apiKey,
    model,
    url
  };
}

function buildRequestHeaders(config: ApiConfig): Record<string, string> {
  if (config.apiFormat === 'openai') {
    return {
      'content-type': 'application/json',
      authorization: `Bearer ${config.apiKey}`
    };
  }

  if (config.apiFormat === 'gemini') {
    return {
      'content-type': 'application/json'
      // API key is in URL for Gemini
    };
  }

  return {
    'content-type': 'application/json',
    'x-api-key': config.apiKey,
    'anthropic-version': '2023-06-01'
  };
}

function buildRequestBody(config: ApiConfig, images: PreparedImage[], prompt: string, maxTokens: number): Record<string, unknown> {
  if (config.apiFormat === 'openai') {
    return {
      model: config.model,
      max_tokens: maxTokens,
      messages: [{ role: 'user', content: buildOpenAIContent(images, prompt) }]
    };
  }

  if (config.apiFormat === 'gemini') {
    const parts: Array<Record<string, unknown>> = [{ text: prompt }];
    images.forEach((image) => {
      parts.push(toGeminiImageBlock(image) as Record<string, unknown>);
    });
    return {
      contents: [{ parts }],
      generationConfig: {
        maxOutputTokens: maxTokens
      }
    };
  }

  return {
    model: config.model,
    max_tokens: maxTokens,
    messages: [{ role: 'user', content: buildAnthropicContent(images, prompt) }]
  };
}

function buildAnthropicContent(images: PreparedImage[], prompt: string): AnthropicContentBlock[] {
  const content: AnthropicContentBlock[] = [];
  images.forEach((image, index) => {
    content.push({ type: 'text', text: images.length > 1 ? `Image ${index + 1}:` : 'Image:' });
    content.push(toAnthropicImageBlock(image));
  });
  content.push({ type: 'text', text: prompt });
  return content;
}

function buildOpenAIContent(images: PreparedImage[], prompt: string): OpenAIContentBlock[] {
  const content: OpenAIContentBlock[] = [];
  images.forEach((image, index) => {
    content.push({ type: 'text', text: images.length > 1 ? `Image ${index + 1}:` : 'Image:' });
    content.push(toOpenAIImageBlock(image));
  });
  content.push({ type: 'text', text: prompt });
  return content;
}

type SimpleResponse = {
  ok: boolean;
  status: number;
  json(): Promise<unknown>;
};

async function readApiResponse(response: SimpleResponse): Promise<ApiResponseBody> {
  try {
    const body = await response.json() as unknown;
    return typeof body === 'object' && body !== null ? body as ApiResponseBody : {};
  } catch {
    return {};
  }
}

function extractResponseText(apiFormat: ApiFormat, body: ApiResponseBody): string | undefined {
  if (apiFormat === 'openai') {
    const content = body.choices?.[0]?.message?.content;
    if (typeof content === 'string') {
      return content.trim();
    }
    return content
      ?.filter((block) => block.type === 'text' && typeof block.text === 'string')
      .map((block) => block.text)
      .join('\n')
      .trim();
  }

  if (apiFormat === 'gemini') {
    return body.candidates?.[0]?.content?.parts
      ?.filter((part) => typeof part.text === 'string')
      .map((part) => part.text)
      .join('\n')
      .trim();
  }

  return body.content
    ?.filter((block) => block.type === 'text' && typeof block.text === 'string')
    .map((block) => block.text)
    .join('\n')
    .trim();
}

function toAnthropicImageBlock(image: PreparedImage): AnthropicContentBlock {
  if (image.source === 'url') {
    if (!image.url) {
      throw new Error('URL image input is missing url.');
    }
    return { type: 'image', source: { type: 'url', url: image.url } };
  }

  if (!image.data || !image.mediaType) {
    throw new Error('Base64 image input is missing data or media type.');
  }

  return {
    type: 'image',
    source: {
      type: 'base64',
      media_type: image.mediaType,
      data: image.data
    }
  };
}

function toOpenAIImageBlock(image: PreparedImage): OpenAIContentBlock {
  if (image.source === 'url') {
    if (!image.url) {
      throw new Error('URL image input is missing url.');
    }
    return { type: 'image_url', image_url: { url: image.url } };
  }

  if (!image.data || !image.mediaType) {
    throw new Error('Base64 image input is missing data or media type.');
  }

  return {
    type: 'image_url',
    image_url: {
      url: `data:${image.mediaType};base64,${image.data}`
    }
  };
}

function toGeminiImageBlock(image: PreparedImage): GeminiInputBlock {
  if (image.source === 'url') {
    if (!image.url) {
      throw new Error('URL image input is missing url.');
    }
    return { fileData: { mimeType: image.mediaType || 'image/png', fileUri: image.url } };
  }

  if (!image.data || !image.mediaType) {
    throw new Error('Base64 image input is missing data or media type.');
  }

  return {
    inlineData: {
      mimeType: image.mediaType,
      data: image.data
    }
  };
}

function mediaTypeFromPath(imagePath: string): string {
  const mediaType = extensionMimeTypes.get(path.extname(imagePath).toLowerCase());
  if (!mediaType || !supportedMimeTypeSet.has(mediaType)) {
    throw new Error(`Unsupported image extension for ${imagePath}. Provide mediaType explicitly or use PNG, JPEG, WebP, or GIF.`);
  }
  return mediaType;
}

function assertImageSize(bytes: number): void {
  const maxBytes = getMaxImageBytes();
  if (bytes <= 0) {
    throw new Error('Image input is empty.');
  }
  if (bytes > maxBytes) {
    throw new Error(`Image is ${bytes} bytes, which exceeds VISIONTOOL_MAX_IMAGE_BYTES (${maxBytes}).`);
  }
}

function normalizeBase64(value: string): string {
  return value.replace(/\s+/gu, '');
}

function getApiFormat(): ApiFormat {
  const rawFormat = process.env.VISIONTOOL_API_FORMAT?.trim().toLowerCase();
  if (rawFormat !== undefined && rawFormat !== '') {
    if (rawFormat === 'anthropic' || rawFormat === 'openai' || rawFormat === 'gemini') {
      return rawFormat;
    }
    throw new Error('VISIONTOOL_API_FORMAT must be "anthropic", "openai", or "gemini".');
  }

  const rawBaseUrl = process.env.VISIONTOOL_BASE_URL?.trim();
  if (rawBaseUrl) {
    const inferred = inferApiFormatFromBaseUrl(rawBaseUrl);
    if (inferred) {
      return inferred;
    }
  }

  if (process.env.GEMINI_API_KEY && !process.env.ANTHROPIC_API_KEY && !process.env.OPENAI_API_KEY && !process.env.VISIONTOOL_API_KEY) {
    return 'gemini';
  }
  if (process.env.OPENAI_API_KEY && !process.env.ANTHROPIC_API_KEY && !process.env.VISIONTOOL_API_KEY) {
    return 'openai';
  }

  return 'anthropic';
}

function inferApiFormatFromBaseUrl(rawBaseUrl: string): ApiFormat | undefined {
  try {
    const url = new URL(rawBaseUrl);
    const pathName = url.pathname.replace(/\/+$/u, '').toLowerCase();
    const hostName = url.hostname.toLowerCase();
    if (pathName.endsWith('/chat/completions') || hostName === 'api.openai.com' || hostName.endsWith('.openai.com')) {
      return 'openai';
    }
    if (pathName.endsWith('/messages') || hostName === 'api.anthropic.com' || hostName.endsWith('.anthropic.com')) {
      return 'anthropic';
    }
    if (pathName.endsWith('/interactions') || hostName === 'generativelanguage.googleapis.com' || hostName.endsWith('.googleapis.com')) {
      return 'gemini';
    }
    return undefined;
  } catch {
    return undefined;
  }
}

function getApiKey(apiFormat: ApiFormat): string {
  const apiKey = process.env.VISIONTOOL_API_KEY
    ?? (apiFormat === 'openai' ? process.env.OPENAI_API_KEY : undefined)
    ?? (apiFormat === 'gemini' ? process.env.GEMINI_API_KEY : undefined)
    ?? (apiFormat === 'anthropic' ? process.env.ANTHROPIC_API_KEY : undefined)
    ?? process.env.ANTHROPIC_API_KEY
    ?? process.env.OPENAI_API_KEY
    ?? process.env.GEMINI_API_KEY;
  if (!apiKey) {
    const providerKeyName = apiFormat === 'openai'
      ? 'OPENAI_API_KEY'
      : apiFormat === 'gemini'
        ? 'GEMINI_API_KEY'
        : 'ANTHROPIC_API_KEY';
    throw new Error(`VISIONTOOL_API_KEY or ${providerKeyName} is required to use VisionToolMCP.`);
  }
  return apiKey;
}

function getDefaultModel(apiFormat: ApiFormat): string {
  if (apiFormat === 'openai') {
    return defaultOpenAIModel;
  }
  if (apiFormat === 'gemini') {
    return defaultGeminiModel;
  }
  return defaultAnthropicModel;
}

function buildApiUrl(apiFormat: ApiFormat, model?: string): string {
  const rawBaseUrl = process.env.VISIONTOOL_BASE_URL?.trim() || getDefaultBaseUrl(apiFormat);
  let url: URL;
  try {
    url = new URL(rawBaseUrl);
  } catch {
    throw new Error(`VISIONTOOL_BASE_URL must be a valid URL. Received: ${rawBaseUrl}`);
  }

  if (url.protocol !== 'http:' && url.protocol !== 'https:') {
    throw new Error(`VISIONTOOL_BASE_URL must use http or https. Received: ${rawBaseUrl}`);
  }
  if (url.search || url.hash) {
    throw new Error('VISIONTOOL_BASE_URL must not include query strings or fragments.');
  }

  const baseUrl = url.toString().replace(/\/+$/u, '');
  const endpoint = getEndpointPath(apiFormat, model);
  if (baseUrl.endsWith(`/${endpoint}`)) {
    return baseUrl;
  }
  if (apiFormat === 'gemini' && baseUrl.endsWith('/v1beta')) {
    const modelName = model || 'gemini-2.5-flash';
    return `${baseUrl}/models/${modelName}:generateContent`;
  }
  if (baseUrl.endsWith('/v1')) {
    return `${baseUrl}/${endpoint.replace(/^v1\//u, '')}`;
  }
  return `${baseUrl}/${endpoint}`;
}

function getEndpointPath(apiFormat: ApiFormat, model?: string): string {
  if (apiFormat === 'openai') {
    return 'v1/chat/completions';
  }
  if (apiFormat === 'gemini') {
    const modelName = model || 'gemini-2.5-flash';
    return `v1beta/models/${modelName}:generateContent`;
  }
  return 'v1/messages';
}

function getDefaultBaseUrl(apiFormat: ApiFormat): string {
  if (apiFormat === 'openai') {
    return defaultOpenAIBaseUrl;
  }
  if (apiFormat === 'gemini') {
    return defaultGeminiBaseUrl;
  }
  return defaultAnthropicBaseUrl;
}

function formatApiName(apiFormat: ApiFormat): string {
  if (apiFormat === 'openai') {
    return 'OpenAI-compatible';
  }
  if (apiFormat === 'gemini') {
    return 'Gemini';
  }
  return 'Anthropic';
}

function getTimeoutMs(): number {
  return parsePositiveIntEnv('VISIONTOOL_TIMEOUT_MS', defaultTimeoutMs);
}

function getMaxImageBytes(): number {
  return parsePositiveIntEnv('VISIONTOOL_MAX_IMAGE_BYTES', defaultMaxImageBytes);
}

function getRetryCount(): number {
  return Math.min(parseNonNegativeIntEnv('VISIONTOOL_RETRIES', defaultRetryCount), maxRetryCount);
}

function getRetryBaseMs(): number {
  return parseNonNegativeIntEnv('VISIONTOOL_RETRY_BASE_MS', defaultRetryBaseMs);
}

function parsePositiveIntEnv(name: string, fallback: number): number {
  const raw = process.env[name];
  if (raw === undefined) {
    return fallback;
  }
  const value = Number.parseInt(raw, 10);
  return Number.isFinite(value) && value > 0 ? value : fallback;
}

function parseNonNegativeIntEnv(name: string, fallback: number): number {
  const raw = process.env[name];
  if (raw === undefined) {
    return fallback;
  }
  const value = Number.parseInt(raw, 10);
  return Number.isFinite(value) && value >= 0 ? value : fallback;
}

async function waitBeforeRetry(attempt: number, timeoutMs: number): Promise<void> {
  const delayMs = getRetryBaseMs() * (2 ** attempt);
  if (delayMs <= 0) {
    return;
  }

  await new Promise<void>((resolve) => {
    setTimeout(resolve, delayMs);
  });
}

function getProxyAgent(): HttpsProxyAgent {
  if (!proxyAgent) {
    proxyAgent = new HttpsProxyAgent({
      proxy: process.env.HTTPS_PROXY ?? process.env.HTTP_PROXY ?? defaultProxyUrl,
      keepAlive: true,
      keepAliveMsecs: 1000,
      maxSockets: 10,
      maxFreeSockets: 2,
      scheduling: 'lifo',
      timeout: getTimeoutMs()
    });
  }
  return proxyAgent;
}

function isRetryableFetchError(error: unknown): boolean {
  return error instanceof TypeError;
}

function isAbortError(error: unknown): boolean {
  return error instanceof Error && error.name === 'AbortError';
}

function createTimeoutError(timeoutMs: number): Error {
  return new Error(`Vision API request timed out after ${timeoutMs}ms.`);
}
