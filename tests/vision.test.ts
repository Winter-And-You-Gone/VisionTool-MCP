import assert from 'node:assert/strict';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { answerAboutImage, compareImages, describeImage, ocrImage } from '../src/vision.js';

type FetchCall = {
  url: string;
  init: RequestInit;
  body: {
    model: string;
    max_tokens: number;
    messages: Array<{
      role: string;
      content: Array<Record<string, unknown>>;
    }>;
  };
};

const originalFetch = globalThis.fetch;
const originalEnv = { ...process.env };

async function withMockedFetch<T>(
  handler: (url: string | URL | Request, init?: RequestInit) => Promise<Response> | Response,
  callback: (calls: FetchCall[]) => Promise<T>
): Promise<T> {
  const calls: FetchCall[] = [];
  globalThis.fetch = (async (url: string | URL | Request, init?: RequestInit): Promise<Response> => {
    const requestInit = init ?? {};
    if (typeof requestInit.body === 'string') {
      calls.push({
        url: String(url),
        init: requestInit,
        body: JSON.parse(requestInit.body) as FetchCall['body']
      });
    }
    return handler(url, init);
  }) as typeof fetch;

  try {
    return await callback(calls);
  } finally {
    globalThis.fetch = originalFetch;
  }
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' }
  });
}

function successResponse(text = 'mock vision result'): Response {
  return jsonResponse({ content: [{ type: 'text', text }] });
}

function openAISuccessResponse(text = 'mock vision result'): Response {
  return jsonResponse({ choices: [{ message: { content: text } }] });
}

function geminiSuccessResponse(text = 'mock vision result'): Response {
  return jsonResponse({ output_text: text });
}

function resetEnv(): void {
  process.env = { ...originalEnv };
  process.env.ANTHROPIC_API_KEY = 'test-key';
  process.env.VISIONTOOL_RETRIES = '0';
  process.env.VISIONTOOL_RETRY_BASE_MS = '0';
  delete process.env.VISIONTOOL_API_FORMAT;
  delete process.env.VISIONTOOL_API_KEY;
  delete process.env.VISIONTOOL_MODEL;
  delete process.env.VISIONTOOL_BASE_URL;
  delete process.env.VISIONTOOL_TIMEOUT_MS;
  delete process.env.VISIONTOOL_MAX_IMAGE_BYTES;
  delete process.env.OPENAI_API_KEY;
  delete process.env.GEMINI_API_KEY;
}

test('describeImage sends local files as base64 image blocks', async () => {
  resetEnv();
  const tempDir = await mkdtemp(path.join(os.tmpdir(), 'visiontool-'));
  const imagePath = path.join(tempDir, 'sample.png');
  await writeFile(imagePath, Buffer.from([0x89, 0x50, 0x4e, 0x47]));

  try {
    const result = await withMockedFetch(
      () => successResponse('a tiny png'),
      async (calls) => describeImage({
        image: { path: imagePath },
        detail: 'low',
        maxTokens: 256
      }).then((value) => ({ value, calls }))
    );

    assert.equal(result.value.tool, 'describe_image');
    assert.equal(result.value.text, 'a tiny png');
    assert.equal(result.value.images[0]?.source, 'path');
    assert.equal(result.value.images[0]?.mediaType, 'image/png');
    assert.equal(result.calls.length, 1);
    assert.equal(result.calls[0]?.body.max_tokens, 256);

    const content = result.calls[0]?.body.messages[0]?.content ?? [];
    const imageBlock = content.find((block) => block.type === 'image') as {
      source?: { type?: string; media_type?: string; data?: string };
    } | undefined;
    assert.equal(imageBlock?.source?.type, 'base64');
    assert.equal(imageBlock?.source?.media_type, 'image/png');
    assert.equal(imageBlock?.source?.data, Buffer.from([0x89, 0x50, 0x4e, 0x47]).toString('base64'));
  } finally {
    await rm(tempDir, { recursive: true, force: true });
  }
});

test('ocrImage normalizes whitespace in base64 payloads', async () => {
  resetEnv();
  const rawBase64 = Buffer.from('abc').toString('base64');

  const result = await withMockedFetch(
    () => successResponse('ABC'),
    async (calls) => ocrImage({
      image: { base64: `${rawBase64.slice(0, 2)}\n${rawBase64.slice(2)}`, mediaType: 'image/png' },
      detail: 'medium',
      maxTokens: 512,
      preserveLayout: true
    }).then((value) => ({ value, calls }))
  );

  assert.equal(result.value.text, 'ABC');
  assert.equal(result.value.images[0]?.source, 'base64');
  assert.equal(result.value.images[0]?.bytes, 3);

  const content = result.calls[0]?.body.messages[0]?.content ?? [];
  const imageBlock = content.find((block) => block.type === 'image') as {
    source?: { data?: string };
  } | undefined;
  assert.equal(imageBlock?.source?.data, rawBase64);
});

test('answerAboutImage sends URL image blocks without local file reads', async () => {
  resetEnv();

  const result = await withMockedFetch(
    () => successResponse('yes'),
    async (calls) => answerAboutImage({
      image: { url: 'https://example.com/image.png' },
      question: 'Is there an image?',
      detail: 'medium',
      maxTokens: 256
    }).then((value) => ({ value, calls }))
  );

  assert.equal(result.value.tool, 'answer_about_image');
  assert.equal(result.value.images[0]?.source, 'url');
  assert.equal(result.value.images[0]?.url, 'https://example.com/image.png');

  const content = result.calls[0]?.body.messages[0]?.content ?? [];
  const imageBlock = content.find((block) => block.type === 'image') as {
    source?: { type?: string; url?: string };
  } | undefined;
  assert.equal(imageBlock?.source?.type, 'url');
  assert.equal(imageBlock?.source?.url, 'https://example.com/image.png');
});

test('compareImages labels both images and honors model overrides', async () => {
  resetEnv();

  const result = await withMockedFetch(
    () => successResponse('different'),
    async (calls) => compareImages({
      firstImage: { base64: Buffer.from('one').toString('base64'), mediaType: 'image/png' },
      secondImage: { base64: Buffer.from('two').toString('base64'), mediaType: 'image/jpeg' },
      instruction: 'Only mention visible differences.',
      detail: 'high',
      maxTokens: 777,
      model: 'claude-test-model'
    }).then((value) => ({ value, calls }))
  );

  assert.equal(result.value.tool, 'compare_images');
  assert.equal(result.value.model, 'claude-test-model');
  assert.equal(result.calls[0]?.body.model, 'claude-test-model');
  assert.equal(result.calls[0]?.body.max_tokens, 777);

  const content = result.calls[0]?.body.messages[0]?.content ?? [];
  assert.equal(content.filter((block) => block.type === 'image').length, 2);
  assert.equal((content[0] as { text?: string }).text, 'Image 1:');
  assert.equal((content[2] as { text?: string }).text, 'Image 2:');
});

test('describeImage retries transient API statuses', async () => {
  resetEnv();
  process.env.VISIONTOOL_RETRIES = '2';
  process.env.VISIONTOOL_RETRY_BASE_MS = '0';
  let attempts = 0;

  const result = await withMockedFetch(
    () => {
      attempts += 1;
      if (attempts === 1) {
        return jsonResponse({ error: { message: 'rate limited' } }, 429);
      }
      return successResponse('retried result');
    },
    async (calls) => describeImage({
      image: { base64: Buffer.from('x').toString('base64'), mediaType: 'image/png' },
      detail: 'low',
      maxTokens: 256
    }).then((value) => ({ value, calls }))
  );

  assert.equal(result.value.text, 'retried result');
  assert.equal(result.calls.length, 2);
});

test('describeImage reports clear timeout errors', async () => {
  resetEnv();
  process.env.VISIONTOOL_TIMEOUT_MS = '1';

  await withMockedFetch(
    (_url, init) => new Promise<Response>((_resolve, reject) => {
      init?.signal?.addEventListener('abort', () => {
        reject(Object.assign(new Error('The operation was aborted.'), { name: 'AbortError' }));
      }, { once: true });
    }),
    async () => {
      await assert.rejects(
        () => describeImage({
          image: { base64: Buffer.from('x').toString('base64'), mediaType: 'image/png' },
          detail: 'low',
          maxTokens: 256
        }),
        /timed out after 1ms/
      );
    }
  );
});

test('describeImage validates local image size and extension before API calls', async () => {
  resetEnv();
  const tempDir = await mkdtemp(path.join(os.tmpdir(), 'visiontool-'));
  const unsupportedPath = path.join(tempDir, 'sample.bmp');
  const tooLargePath = path.join(tempDir, 'large.png');
  await writeFile(unsupportedPath, Buffer.from('x'));
  await writeFile(tooLargePath, Buffer.from('too-large'));
  process.env.VISIONTOOL_MAX_IMAGE_BYTES = '4';

  try {
    await withMockedFetch(
      () => successResponse('should not be called'),
      async (calls) => {
        await assert.rejects(
          () => describeImage({ image: { path: unsupportedPath }, detail: 'low', maxTokens: 256 }),
          /Unsupported image extension/
        );
        await assert.rejects(
          () => describeImage({ image: { path: tooLargePath }, detail: 'low', maxTokens: 256 }),
          /exceeds VISIONTOOL_MAX_IMAGE_BYTES/
        );
        assert.equal(calls.length, 0);
      }
    );
  } finally {
    await rm(tempDir, { recursive: true, force: true });
  }
});

test('describeImage uses custom VISIONTOOL_BASE_URL values', async () => {
  resetEnv();
  process.env.VISIONTOOL_BASE_URL = 'https://proxy.example.com/anthropic/v1';

  const result = await withMockedFetch(
    () => successResponse('custom base url'),
    async (calls) => describeImage({
      image: { base64: Buffer.from('x').toString('base64'), mediaType: 'image/png' },
      detail: 'low',
      maxTokens: 256
    }).then((value) => ({ value, calls }))
  );

  assert.equal(result.value.text, 'custom base url');
  assert.equal(result.calls[0]?.url, 'https://proxy.example.com/anthropic/v1/messages');
});

test('describeImage rejects invalid VISIONTOOL_BASE_URL values', async () => {
  resetEnv();
  process.env.VISIONTOOL_BASE_URL = 'not a url';

  await assert.rejects(
    () => describeImage({
      image: { base64: Buffer.from('x').toString('base64'), mediaType: 'image/png' },
      detail: 'low',
      maxTokens: 256
    }),
    /VISIONTOOL_BASE_URL must be a valid URL/
  );
});

test('describeImage supports OpenAI-compatible chat completions format', async () => {
  resetEnv();
  delete process.env.ANTHROPIC_API_KEY;
  process.env.OPENAI_API_KEY = 'openai-key';
  process.env.VISIONTOOL_API_FORMAT = 'openai';
  process.env.VISIONTOOL_BASE_URL = 'https://api.openai.com/v1';
  process.env.VISIONTOOL_MODEL = 'gpt-4o-mini';

  const rawBase64 = Buffer.from('openai-image').toString('base64');
  const result = await withMockedFetch(
    () => openAISuccessResponse('openai result'),
    async (calls) => describeImage({
      image: { base64: rawBase64, mediaType: 'image/png' },
      detail: 'low',
      maxTokens: 256
    }).then((value) => ({ value, calls }))
  );

  assert.equal(result.value.text, 'openai result');
  assert.equal(result.value.apiFormat, 'openai');
  assert.equal(result.value.model, 'gpt-4o-mini');
  assert.equal(result.calls[0]?.url, 'https://api.openai.com/v1/chat/completions');
  assert.equal((result.calls[0]?.init.headers as Record<string, string>).authorization, 'Bearer openai-key');

  const content = result.calls[0]?.body.messages[0]?.content ?? [];
  const imageBlock = content.find((block) => block.type === 'image_url') as {
    image_url?: { url?: string };
  } | undefined;
  assert.equal(imageBlock?.image_url?.url, `data:image/png;base64,${rawBase64}`);
});

test('describeImage infers OpenAI-compatible format from chat completions URL', async () => {
  resetEnv();
  delete process.env.ANTHROPIC_API_KEY;
  process.env.OPENAI_API_KEY = 'openai-key';
  process.env.VISIONTOOL_BASE_URL = 'https://proxy.example.com/openai/v1/chat/completions';

  const result = await withMockedFetch(
    () => openAISuccessResponse('inferred openai'),
    async (calls) => describeImage({
      image: { url: 'https://example.com/image.png' },
      detail: 'low',
      maxTokens: 256
    }).then((value) => ({ value, calls }))
  );

  assert.equal(result.value.apiFormat, 'openai');
  assert.equal(result.calls[0]?.url, 'https://proxy.example.com/openai/v1/chat/completions');
});

test('describeImage requires a provider key for the selected API format', async () => {
  resetEnv();
  process.env.VISIONTOOL_API_FORMAT = 'openai';
  delete process.env.VISIONTOOL_API_KEY;
  delete process.env.OPENAI_API_KEY;
  delete process.env.ANTHROPIC_API_KEY;

  await assert.rejects(
    () => describeImage({
      image: { base64: Buffer.from('x').toString('base64'), mediaType: 'image/png' },
      detail: 'low',
      maxTokens: 256
    }),
    /VISIONTOOL_API_KEY or OPENAI_API_KEY is required/
  );
});

test('describeImage supports Gemini interactions format', async () => {
  resetEnv();
  delete process.env.ANTHROPIC_API_KEY;
  process.env.GEMINI_API_KEY = 'gemini-key';
  process.env.VISIONTOOL_API_FORMAT = 'gemini';
  process.env.VISIONTOOL_BASE_URL = 'https://generativelanguage.googleapis.com/v1beta';
  process.env.VISIONTOOL_MODEL = 'gemini-3.5-flash';

  const rawBase64 = Buffer.from('gemini-image').toString('base64');
  const result = await withMockedFetch(
    () => geminiSuccessResponse('gemini result'),
    async (calls) => describeImage({
      image: { base64: rawBase64, mediaType: 'image/png' },
      detail: 'low',
      maxTokens: 256
    }).then((value) => ({ value, calls }))
  );

  assert.equal(result.value.text, 'gemini result');
  assert.equal(result.value.apiFormat, 'gemini');
  assert.equal(result.value.model, 'gemini-3.5-flash');
  assert.equal(result.calls[0]?.url, 'https://generativelanguage.googleapis.com/v1beta/interactions');
  assert.equal((result.calls[0]?.init.headers as Record<string, string>)['x-goog-api-key'], 'gemini-key');
  assert.equal(result.calls[0]?.body.model, 'gemini-3.5-flash');
  assert.equal((result.calls[0]?.body as { generation_config?: { max_output_tokens?: number } }).generation_config?.max_output_tokens, 256);

  const input = (result.calls[0]?.body as { input?: Array<Record<string, unknown>> }).input ?? [];
  const imageBlock = input.find((block) => block.type === 'image') as {
    data?: string;
    mime_type?: string;
  } | undefined;
  assert.equal(imageBlock?.data, rawBase64);
  assert.equal(imageBlock?.mime_type, 'image/png');
});

test('describeImage infers Gemini format from interactions URL', async () => {
  resetEnv();
  delete process.env.ANTHROPIC_API_KEY;
  process.env.GEMINI_API_KEY = 'gemini-key';
  process.env.VISIONTOOL_BASE_URL = 'https://generativelanguage.googleapis.com/v1beta/interactions';

  const result = await withMockedFetch(
    () => geminiSuccessResponse('inferred gemini'),
    async (calls) => describeImage({
      image: { url: 'https://example.com/image.png' },
      detail: 'low',
      maxTokens: 256
    }).then((value) => ({ value, calls }))
  );

  assert.equal(result.value.apiFormat, 'gemini');
  assert.equal(result.calls[0]?.url, 'https://generativelanguage.googleapis.com/v1beta/interactions');

  const input = (result.calls[0]?.body as { input?: Array<Record<string, unknown>> }).input ?? [];
  const imageBlock = input.find((block) => block.type === 'image') as {
    uri?: string;
  } | undefined;
  assert.equal(imageBlock?.uri, 'https://example.com/image.png');
});

test('describeImage requires a Gemini key for Gemini format', async () => {
  resetEnv();
  process.env.VISIONTOOL_API_FORMAT = 'gemini';
  delete process.env.VISIONTOOL_API_KEY;
  delete process.env.GEMINI_API_KEY;
  delete process.env.OPENAI_API_KEY;
  delete process.env.ANTHROPIC_API_KEY;

  await assert.rejects(
    () => describeImage({
      image: { base64: Buffer.from('x').toString('base64'), mediaType: 'image/png' },
      detail: 'low',
      maxTokens: 256
    }),
    /VISIONTOOL_API_KEY or GEMINI_API_KEY is required/
  );
});

test('describeImage rejects invalid API formats', async () => {
  resetEnv();
  process.env.VISIONTOOL_API_FORMAT = 'invalid';

  await assert.rejects(
    () => describeImage({
      image: { base64: Buffer.from('x').toString('base64'), mediaType: 'image/png' },
      detail: 'low',
      maxTokens: 256
    }),
    /VISIONTOOL_API_FORMAT must be "anthropic", "openai", or "gemini"/
  );
});

test('describeImage requires ANTHROPIC_API_KEY', async () => {
  resetEnv();
  delete process.env.ANTHROPIC_API_KEY;

  await assert.rejects(
    () => describeImage({
      image: { base64: Buffer.from('x').toString('base64'), mediaType: 'image/png' },
      detail: 'low',
      maxTokens: 256
    }),
    /ANTHROPIC_API_KEY is required/
  );
});
