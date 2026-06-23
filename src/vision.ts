import { readFile, stat } from 'node:fs/promises';
import http from 'node:http';
import path from 'node:path';
import https from 'node:https';
import { HttpProxyAgent, HttpsProxyAgent } from 'hpagent';

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
type RequestAgent = http.Agent | https.Agent;

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

// Cache proxy agents to avoid recreating them while still honoring protocol differences.
let httpProxyAgent: HttpProxyAgent | undefined;
let httpsProxyAgent: HttpsProxyAgent | undefined;
let httpProxyAgentKey: string | undefined;
let httpsProxyAgentKey: string | undefined;
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
    const imageUrl = validateImageUrl(input.url);
    return { source: 'url', url: imageUrl };
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
  assertLocalPathAllowed(resolvedPath);
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
  const bodyStr = JSON.stringify(buildRequestBody(config, images, prompt, maxTokens));
  const headers = buildRequestHeaders(config);

  for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
    try {
      const response = await requestVisionApiWithFallback(config.url, bodyStr, headers, timeoutMs);

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
      if (isRetryableNetworkError(error) && attempt < maxAttempts - 1) {
        lastError = error;
        await waitBeforeRetry(attempt, timeoutMs);
        continue;
      }

      throw error;
    }
  }

  throw lastError instanceof Error ? lastError : new Error(`${formatApiName(config.apiFormat)} API request failed after retries.`);
}

async function requestVisionApiWithFallback(
  rawUrl: string,
  body: string,
  headers: Record<string, string>,
  timeoutMs: number
): Promise<SimpleResponse> {
  try {
    return await requestVisionApi(rawUrl, body, headers, timeoutMs);
  } catch (directError) {
    if (!isRetryableNetworkError(directError)) {
      throw directError;
    }

    const proxyUrl = getProxyUrl();
    if (!proxyUrl) {
      throw directError;
    }

    try {
      return await requestVisionApi(rawUrl, body, headers, timeoutMs, getProxyAgent(new URL(rawUrl).protocol));
    } catch (proxyError) {
      throw combineProxyFallbackError(directError, proxyError, proxyUrl);
    }
  }
}

async function requestVisionApi(
  rawUrl: string,
  body: string,
  headers: Record<string, string>,
  timeoutMs: number,
  agent?: RequestAgent
): Promise<SimpleResponse> {
  const url = new URL(rawUrl);
  const transport = url.protocol === 'http:' ? http : https;

  return await new Promise<SimpleResponse>((resolve, reject) => {
    const req = transport.request(
      {
        protocol: url.protocol,
        hostname: url.hostname,
        port: url.port,
        path: url.pathname + url.search,
        method: 'POST',
        headers: {
          ...headers,
          'Content-Length': Buffer.byteLength(body)
        },
        agent
      },
      (res) => {
        const chunks: Buffer[] = [];
        res.on('data', (chunk: Buffer | string) => {
          chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
        });
        res.on('end', () => {
          const responseBody = Buffer.concat(chunks).toString('utf8');
          resolve({
            ok: res.statusCode !== undefined && res.statusCode >= 200 && res.statusCode < 300,
            status: res.statusCode ?? 500,
            json: async () => responseBody.length > 0 ? JSON.parse(responseBody) : {}
          });
        });
      }
    );

    req.setTimeout(timeoutMs, () => {
      req.destroy(createTimeoutError(timeoutMs));
    });
    req.on('error', reject);
    req.write(body);
    req.end();
  });
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
    const candidateText = body.candidates?.[0]?.content?.parts
      ?.filter((part) => typeof part.text === 'string')
      .map((part) => part.text)
      .join('\n')
      .trim();
    return candidateText || body.output_text?.trim();
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
    if (pathName.endsWith(':generatecontent') || hostName === 'generativelanguage.googleapis.com' || hostName.endsWith('.googleapis.com')) {
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
  if (apiFormat === 'gemini' && url.pathname.replace(/\/+$/u, '').toLowerCase().endsWith('/interactions')) {
    throw new Error('Gemini VISIONTOOL_BASE_URL must use the generateContent API, for example https://generativelanguage.googleapis.com/v1beta or https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent.');
  }

  const baseUrl = url.toString().replace(/\/+$/u, '');
  const endpoint = getEndpointPath(apiFormat, model);
  if (apiFormat === 'gemini' && baseUrl.toLowerCase().endsWith(':generatecontent')) {
    return baseUrl;
  }
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

function getProxyUrl(): string | undefined {
  if (isTruthyEnv(process.env.VISIONTOOL_DISABLE_PROXY_FALLBACK)) {
    return undefined;
  }
  return process.env.VISIONTOOL_PROXY_URL?.trim()
    || process.env.HTTPS_PROXY?.trim()
    || process.env.HTTP_PROXY?.trim()
    || defaultProxyUrl;
}

function getProxyAgent(protocol: string): RequestAgent {
  const proxy = getProxyUrl();
  if (!proxy) {
    throw new Error('Proxy fallback is disabled.');
  }
  const options = {
    proxy,
    keepAlive: true,
    keepAliveMsecs: 1000,
    maxSockets: 10,
    maxFreeSockets: 2,
    scheduling: 'lifo' as const,
    timeout: getTimeoutMs()
  };

  const cacheKey = `${proxy}|${getTimeoutMs()}`;
  if (protocol === 'http:') {
    if (!httpProxyAgent || httpProxyAgentKey !== cacheKey) {
      httpProxyAgent = new HttpProxyAgent(options);
      httpProxyAgentKey = cacheKey;
    }
    return httpProxyAgent;
  }

  if (!httpsProxyAgent || httpsProxyAgentKey !== cacheKey) {
    httpsProxyAgent = new HttpsProxyAgent({
      proxy,
      keepAlive: true,
      keepAliveMsecs: 1000,
      maxSockets: 10,
      maxFreeSockets: 2,
      scheduling: 'lifo',
      timeout: getTimeoutMs()
    });
    httpsProxyAgentKey = cacheKey;
  }
  return httpsProxyAgent;
}

function assertLocalPathAllowed(resolvedPath: string): void {
  const rawRoots = process.env.VISIONTOOL_ALLOWED_IMAGE_ROOTS?.trim();
  if (!rawRoots) {
    return;
  }

  const roots = rawRoots
    .split(path.delimiter)
    .map((root) => root.trim())
    .filter(Boolean)
    .map((root) => path.resolve(root));
  if (roots.length === 0) {
    return;
  }

  const allowed = roots.some((root) => isPathWithinRoot(resolvedPath, root));
  if (!allowed) {
    throw new Error(`Image path is outside VISIONTOOL_ALLOWED_IMAGE_ROOTS: ${resolvedPath}`);
  }
}

function isPathWithinRoot(targetPath: string, rootPath: string): boolean {
  const normalizedTarget = normalizePathForCompare(targetPath);
  const normalizedRoot = normalizePathForCompare(rootPath);
  const relative = path.relative(normalizedRoot, normalizedTarget);
  return relative === '' || (!relative.startsWith('..') && !path.isAbsolute(relative));
}

function normalizePathForCompare(value: string): string {
  const normalized = path.resolve(value);
  return process.platform === 'win32' ? normalized.toLowerCase() : normalized;
}

function validateImageUrl(value: string): string {
  if (isTruthyEnv(process.env.VISIONTOOL_DISABLE_URL_INPUTS)) {
    throw new Error('URL image input is disabled by VISIONTOOL_DISABLE_URL_INPUTS.');
  }

  let url: URL;
  try {
    url = new URL(value);
  } catch {
    throw new Error(`Image URL must be a valid URL. Received: ${value}`);
  }

  if (url.protocol !== 'http:' && url.protocol !== 'https:') {
    throw new Error(`Image URL must use http or https. Received: ${value}`);
  }

  const allowedHosts = parseAllowedUrlHosts();
  if (allowedHosts.length > 0 && !allowedHosts.some((host) => hostMatches(url.hostname, host))) {
    throw new Error(`Image URL host "${url.hostname}" is not allowed by VISIONTOOL_ALLOWED_URL_HOSTS.`);
  }

  if (!isTruthyEnv(process.env.VISIONTOOL_ALLOW_PRIVATE_URLS) && isPrivateOrLocalHost(url.hostname)) {
    throw new Error(`Image URL host "${url.hostname}" looks private or local. Set VISIONTOOL_ALLOW_PRIVATE_URLS=1 only if sending this URL to the upstream vision provider is intentional.`);
  }

  return url.toString();
}

function parseAllowedUrlHosts(): string[] {
  return (process.env.VISIONTOOL_ALLOWED_URL_HOSTS ?? '')
    .split(',')
    .map((host) => host.trim().toLowerCase())
    .filter(Boolean);
}

function hostMatches(hostname: string, allowedHost: string): boolean {
  const host = stripIpv6Brackets(hostname).toLowerCase();
  const allowed = stripIpv6Brackets(allowedHost).toLowerCase();
  if (allowed.startsWith('*.')) {
    const suffix = allowed.slice(1);
    return host.endsWith(suffix) && host.length > suffix.length;
  }
  return host === allowed;
}

function isPrivateOrLocalHost(hostname: string): boolean {
  const host = stripIpv6Brackets(hostname).toLowerCase();
  if (host === 'localhost' || host.endsWith('.localhost') || host === '0.0.0.0' || host === '::' || host === '::1') {
    return true;
  }

  const ipv4Parts = host.split('.');
  if (ipv4Parts.length === 4 && ipv4Parts.every((part) => /^\d+$/u.test(part))) {
    const octets = ipv4Parts.map((part) => Number.parseInt(part, 10));
    if (octets.some((octet) => octet < 0 || octet > 255)) {
      return false;
    }
    const first = octets[0];
    const second = octets[1];
    if (first === undefined || second === undefined) {
      return false;
    }
    return first === 10
      || first === 127
      || first === 0
      || (first === 169 && second === 254)
      || (first === 172 && second >= 16 && second <= 31)
      || (first === 192 && second === 168);
  }

  return host.startsWith('fc') || host.startsWith('fd') || host.startsWith('fe80:');
}

function stripIpv6Brackets(hostname: string): string {
  return hostname.replace(/^\[/u, '').replace(/\]$/u, '');
}

function isTruthyEnv(value: string | undefined): boolean {
  if (value === undefined) {
    return false;
  }
  return ['1', 'true', 'yes', 'on'].includes(value.trim().toLowerCase());
}

function isRetryableNetworkError(error: unknown): boolean {
  if (!(error instanceof Error)) {
    return false;
  }

  const code = (error as NodeJS.ErrnoException).code;
  if (typeof code === 'string' && ['ECONNRESET', 'ETIMEDOUT', 'ECONNREFUSED', 'EAI_AGAIN', 'ENOTFOUND', 'EHOSTUNREACH', 'ENETUNREACH'].includes(code)) {
    return true;
  }

  return error.name === 'AbortError' || error.name === 'TimeoutError' || error.message.includes('timed out after');
}

function combineProxyFallbackError(directError: unknown, proxyError: unknown, proxyUrl: string): Error {
  const directMessage = directError instanceof Error ? directError.message : String(directError);
  const proxyMessage = proxyError instanceof Error ? proxyError.message : String(proxyError);
  const error = new Error(`${directMessage} Retried through proxy ${proxyUrl} but it also failed: ${proxyMessage}`) as NodeJS.ErrnoException;
  error.code = getErrorCode(proxyError) ?? getErrorCode(directError);
  if (proxyError instanceof Error) {
    error.name = proxyError.name;
  }
  return error;
}

function getErrorCode(error: unknown): string | undefined {
  const code = error instanceof Error ? (error as NodeJS.ErrnoException).code : undefined;
  return typeof code === 'string' ? code : undefined;
}

function createTimeoutError(timeoutMs: number): Error {
  const error = new Error(`Vision API request timed out after ${timeoutMs}ms.`) as NodeJS.ErrnoException;
  error.name = 'TimeoutError';
  error.code = 'ETIMEDOUT';
  return error;
}
