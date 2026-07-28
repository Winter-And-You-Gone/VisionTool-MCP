import assert from 'node:assert/strict';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { answerAboutImage, compareImages, describeImage, ocrImage } from '../src/vision.js';

type HttpCall = {
  url: string;
  path: string;
  method?: string;
  headers: http.IncomingHttpHeaders;
  body: Record<string, unknown>;
};

type MockReply = {
  status?: number;
  body?: unknown;
  headers?: Record<string, string>;
  neverRespond?: boolean;
};

type ServerContext = {
  baseUrl: string;
  calls: HttpCall[];
};

const originalEnv = { ...process.env };

async function withMockVisionServer<T>(
  handler: (call: HttpCall) => MockReply | Promise<MockReply>,
  callback: (context: ServerContext) => Promise<T>
): Promise<T> {
  const calls: HttpCall[] = [];
  const server = http.createServer((req, res) => {
    const chunks: Buffer[] = [];
    req.on('data', (chunk: Buffer | string) => {
      chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
    });
    req.on('end', () => {
      void (async () => {
        const bodyText = Buffer.concat(chunks).toString('utf8');
        const call: HttpCall = {
          url: `http://${req.headers.host ?? '127.0.0.1'}${req.url ?? '/'}`,
          path: req.url ?? '/',
          method: req.method,
          headers: req.headers,
          body: bodyText.length > 0 ? JSON.parse(bodyText) as Record<string, unknown> : {}
        };
        calls.push(call);

        const reply = await handler(call);
        if (reply.neverRespond) {
          return;
        }

        res.statusCode = reply.status ?? 200;
        const headers = reply.headers ?? { 'content-type': 'application/json' };
        for (const [name, value] of Object.entries(headers)) {
          res.setHeader(name, value);
        }
        res.end(JSON.stringify(reply.body ?? anthropicSuccessBody()));
      })().catch((error: unknown) => {
        res.statusCode = 500;
        res.end(JSON.stringify({ error: { message: error instanceof Error ? error.message : String(error) } }));
      });
    });
  });

  await new Promise<void>((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', () => {
      server.off('error', reject);
      resolve();
    });
  });

  const address = server.address();
  assert.ok(address && typeof address === 'object');
  const baseUrl = `http://127.0.0.1:${address.port}`;

  try {
    return await callback({ baseUrl, calls });
  } finally {
    await new Promise<void>((resolve, reject) => {
      server.close((error) => {
        if (error) {
          reject(error);
          return;
        }
        resolve();
      });
    });
  }
}

function anthropicSuccessBody(text = 'mock vision result'): unknown {
  return { content: [{ type: 'text', text }] };
}

function openAISuccessBody(text = 'mock vision result'): unknown {
  return { choices: [{ message: { content: text } }] };
}

function geminiSuccessBody(text = 'mock vision result'): unknown {
  return { candidates: [{ content: { parts: [{ text }] } }] };
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
  delete process.env.VISIONTOOL_ALLOWED_IMAGE_ROOTS;
  delete process.env.VISIONTOOL_DISABLE_URL_INPUTS;
  delete process.env.VISIONTOOL_ALLOWED_URL_HOSTS;
  delete process.env.VISIONTOOL_ALLOW_PRIVATE_URLS;
  delete process.env.VISIONTOOL_PROXY_URL;
  delete process.env.VISIONTOOL_DISABLE_PROXY_FALLBACK;
  delete process.env.OPENAI_API_KEY;
  delete process.env.GEMINI_API_KEY;
}

test('describeImage sends local files as base64 image blocks', async () => {
  resetEnv();
  const tempDir = await mkdtemp(path.join(os.tmpdir(), 'visiontool-'));
  const imagePath = path.join(tempDir, 'sample.png');
  await writeFile(imagePath, Buffer.from([0x89, 0x50, 0x4e, 0x47]));

  try {
    await withMockVisionServer(
      () => ({ body: anthropicSuccessBody('a tiny png') }),
      async ({ baseUrl, calls }) => {
        process.env.VISIONTOOL_BASE_URL = `${baseUrl}/anthropic/v1`;

        const value = await describeImage({
          image: { path: imagePath },
          detail: 'low',
          maxTokens: 256
        });

        assert.equal(value.tool, 'describe_image');
        assert.equal(value.text, 'a tiny png');
        assert.equal(value.images[0]?.source, 'path');
        assert.equal(value.images[0]?.mediaType, 'image/png');
        assert.equal(calls.length, 1);
        assert.equal(calls[0]?.path, '/anthropic/v1/messages');
        assert.equal(calls[0]?.body.max_tokens, 256);

        const messages = calls[0]?.body.messages as Array<{ content?: Array<Record<string, unknown>> }> | undefined;
        const content = messages?.[0]?.content ?? [];
        const imageBlock = content.find((block) => block.type === 'image') as {
          source?: { type?: string; media_type?: string; data?: string };
        } | undefined;
        assert.equal(imageBlock?.source?.type, 'base64');
        assert.equal(imageBlock?.source?.media_type, 'image/png');
        assert.equal(imageBlock?.source?.data, Buffer.from([0x89, 0x50, 0x4e, 0x47]).toString('base64'));
      }
    );
  } finally {
    await rm(tempDir, { recursive: true, force: true });
  }
});

test('ocrImage normalizes whitespace in base64 payloads', async () => {
  resetEnv();
  const rawBase64 = Buffer.from('abc').toString('base64');

  await withMockVisionServer(
    () => ({ body: anthropicSuccessBody('ABC') }),
    async ({ baseUrl, calls }) => {
      process.env.VISIONTOOL_BASE_URL = `${baseUrl}/anthropic/v1`;

      const value = await ocrImage({
        image: { base64: `${rawBase64.slice(0, 2)}\n${rawBase64.slice(2)}`, mediaType: 'image/png' },
        detail: 'medium',
        maxTokens: 512,
        preserveLayout: true
      });

      assert.equal(value.text, 'ABC');
      assert.equal(value.images[0]?.source, 'base64');
      assert.equal(value.images[0]?.bytes, 3);

      const messages = calls[0]?.body.messages as Array<{ content?: Array<Record<string, unknown>> }> | undefined;
      const content = messages?.[0]?.content ?? [];
      const imageBlock = content.find((block) => block.type === 'image') as {
        source?: { data?: string };
      } | undefined;
      assert.equal(imageBlock?.source?.data, rawBase64);
    }
  );
});

test('answerAboutImage sends URL image blocks without local file reads', async () => {
  resetEnv();

  await withMockVisionServer(
    () => ({ body: anthropicSuccessBody('yes') }),
    async ({ baseUrl, calls }) => {
      process.env.VISIONTOOL_BASE_URL = `${baseUrl}/anthropic/v1`;

      const value = await answerAboutImage({
        image: { url: 'https://example.com/image.png' },
        question: 'Is there an image?',
        detail: 'medium',
        maxTokens: 256
      });

      assert.equal(value.tool, 'answer_about_image');
      assert.equal(value.images[0]?.source, 'url');
      assert.equal(value.images[0]?.url, 'https://example.com/image.png');

      const messages = calls[0]?.body.messages as Array<{ content?: Array<Record<string, unknown>> }> | undefined;
      const content = messages?.[0]?.content ?? [];
      const imageBlock = content.find((block) => block.type === 'image') as {
        source?: { type?: string; url?: string };
      } | undefined;
      assert.equal(imageBlock?.source?.type, 'url');
      assert.equal(imageBlock?.source?.url, 'https://example.com/image.png');
    }
  );
});

test('answerAboutImage rejects private or disabled URL image inputs before API calls', async () => {
  resetEnv();

  await withMockVisionServer(
    () => ({ body: anthropicSuccessBody('should not be called') }),
    async ({ baseUrl, calls }) => {
      process.env.VISIONTOOL_BASE_URL = `${baseUrl}/anthropic/v1`;

      await assert.rejects(
        () => answerAboutImage({
          image: { url: 'http://127.0.0.1/private.png' },
          question: 'What is visible?',
          detail: 'medium',
          maxTokens: 256
        }),
        /looks private or local/
      );

      process.env.VISIONTOOL_DISABLE_URL_INPUTS = '1';
      await assert.rejects(
        () => answerAboutImage({
          image: { url: 'https://example.com/image.png' },
          question: 'What is visible?',
          detail: 'medium',
          maxTokens: 256
        }),
        /URL image input is disabled/
      );

      assert.equal(calls.length, 0);
    }
  );
});

test('answerAboutImage allows private URLs only when explicitly enabled', async () => {
  resetEnv();

  await withMockVisionServer(
    () => ({ body: anthropicSuccessBody('private url accepted') }),
    async ({ baseUrl, calls }) => {
      process.env.VISIONTOOL_BASE_URL = `${baseUrl}/anthropic/v1`;
      process.env.VISIONTOOL_ALLOW_PRIVATE_URLS = '1';

      const value = await answerAboutImage({
        image: { url: 'http://127.0.0.1/image.png' },
        question: 'What is visible?',
        detail: 'medium',
        maxTokens: 256
      });

      assert.equal(value.text, 'private url accepted');
      assert.equal(value.images[0]?.url, 'http://127.0.0.1/image.png');
      assert.equal(calls.length, 1);
    }
  );
});

test('answerAboutImage enforces URL host allowlists', async () => {
  resetEnv();

  await withMockVisionServer(
    () => ({ body: anthropicSuccessBody('allowed host') }),
    async ({ baseUrl, calls }) => {
      process.env.VISIONTOOL_BASE_URL = `${baseUrl}/anthropic/v1`;
      process.env.VISIONTOOL_ALLOWED_URL_HOSTS = 'images.example.com,*.cdn.example.com';

      await assert.rejects(
        () => answerAboutImage({
          image: { url: 'https://example.com/image.png' },
          question: 'What is visible?',
          detail: 'medium',
          maxTokens: 256
        }),
        /not allowed by VISIONTOOL_ALLOWED_URL_HOSTS/
      );

      const value = await answerAboutImage({
        image: { url: 'https://assets.cdn.example.com/image.png' },
        question: 'What is visible?',
        detail: 'medium',
        maxTokens: 256
      });

      assert.equal(value.text, 'allowed host');
      assert.equal(calls.length, 1);
    }
  );
});

test('compareImages labels both images and honors model overrides', async () => {
  resetEnv();

  await withMockVisionServer(
    () => ({ body: anthropicSuccessBody('different') }),
    async ({ baseUrl, calls }) => {
      process.env.VISIONTOOL_BASE_URL = `${baseUrl}/anthropic/v1`;

      const value = await compareImages({
        firstImage: { base64: Buffer.from('one').toString('base64'), mediaType: 'image/png' },
        secondImage: { base64: Buffer.from('two').toString('base64'), mediaType: 'image/jpeg' },
        instruction: 'Only mention visible differences.',
        detail: 'high',
        maxTokens: 777,
        model: 'claude-test-model'
      });

      assert.equal(value.tool, 'compare_images');
      assert.equal(value.model, 'claude-test-model');
      assert.equal(calls[0]?.body.model, 'claude-test-model');
      assert.equal(calls[0]?.body.max_tokens, 777);

      const messages = calls[0]?.body.messages as Array<{ content?: Array<Record<string, unknown>> }> | undefined;
      const content = messages?.[0]?.content ?? [];
      assert.equal(content.filter((block) => block.type === 'image').length, 2);
      assert.equal((content[0] as { text?: string }).text, 'Image 1:');
      assert.equal((content[2] as { text?: string }).text, 'Image 2:');
    }
  );
});

test('describeImage retries transient API statuses', async () => {
  resetEnv();
  process.env.VISIONTOOL_RETRIES = '2';
  process.env.VISIONTOOL_RETRY_BASE_MS = '0';
  let attempts = 0;

  await withMockVisionServer(
    () => {
      attempts += 1;
      if (attempts === 1) {
        return { status: 429, body: { error: { message: 'rate limited' } } };
      }
      return { body: anthropicSuccessBody('retried result') };
    },
    async ({ baseUrl, calls }) => {
      process.env.VISIONTOOL_BASE_URL = `${baseUrl}/anthropic/v1`;

      const value = await describeImage({
        image: { base64: Buffer.from('x').toString('base64'), mediaType: 'image/png' },
        detail: 'low',
        maxTokens: 256
      });

      assert.equal(value.text, 'retried result');
      assert.equal(calls.length, 2);
    }
  );
});

test('describeImage reports clear timeout errors', async () => {
  resetEnv();
  process.env.VISIONTOOL_TIMEOUT_MS = '1';
  process.env.HTTP_PROXY = 'http://127.0.0.1:9';
  process.env.HTTPS_PROXY = 'http://127.0.0.1:9';

  await withMockVisionServer(
    () => ({ neverRespond: true }),
    async ({ baseUrl }) => {
      process.env.VISIONTOOL_BASE_URL = `${baseUrl}/anthropic/v1`;

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
    await withMockVisionServer(
      () => ({ body: anthropicSuccessBody('should not be called') }),
      async ({ baseUrl, calls }) => {
        process.env.VISIONTOOL_BASE_URL = `${baseUrl}/anthropic/v1`;

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

test('describeImage enforces allowed local image roots', async () => {
  resetEnv();
  const allowedDir = await mkdtemp(path.join(os.tmpdir(), 'visiontool-allowed-'));
  const deniedDir = await mkdtemp(path.join(os.tmpdir(), 'visiontool-denied-'));
  const allowedPath = path.join(allowedDir, 'allowed.png');
  const deniedPath = path.join(deniedDir, 'denied.png');
  await writeFile(allowedPath, Buffer.from([0x89, 0x50, 0x4e, 0x47]));
  await writeFile(deniedPath, Buffer.from([0x89, 0x50, 0x4e, 0x47]));

  try {
    await withMockVisionServer(
      () => ({ body: anthropicSuccessBody('allowed path') }),
      async ({ baseUrl, calls }) => {
        process.env.VISIONTOOL_BASE_URL = `${baseUrl}/anthropic/v1`;
        process.env.VISIONTOOL_ALLOWED_IMAGE_ROOTS = allowedDir;

        await assert.rejects(
          () => describeImage({ image: { path: deniedPath }, detail: 'low', maxTokens: 256 }),
          /outside VISIONTOOL_ALLOWED_IMAGE_ROOTS/
        );

        const value = await describeImage({ image: { path: allowedPath }, detail: 'low', maxTokens: 256 });
        assert.equal(value.text, 'allowed path');
        assert.equal(calls.length, 1);
      }
    );
  } finally {
    await rm(allowedDir, { recursive: true, force: true });
    await rm(deniedDir, { recursive: true, force: true });
  }
});

test('describeImage uses custom HTTP VISIONTOOL_BASE_URL values', async () => {
  resetEnv();

  await withMockVisionServer(
    () => ({ body: anthropicSuccessBody('custom base url') }),
    async ({ baseUrl, calls }) => {
      process.env.VISIONTOOL_BASE_URL = `${baseUrl}/anthropic/v1`;

      const value = await describeImage({
        image: { base64: Buffer.from('x').toString('base64'), mediaType: 'image/png' },
        detail: 'low',
        maxTokens: 256
      });

      assert.equal(value.text, 'custom base url');
      assert.equal(calls[0]?.url, `${baseUrl}/anthropic/v1/messages`);
      assert.equal(calls[0]?.headers['x-api-key'], 'test-key');
    }
  );
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
  process.env.VISIONTOOL_MODEL = 'gpt-4o-mini';
  const rawBase64 = Buffer.from('openai-image').toString('base64');

  await withMockVisionServer(
    () => ({ body: openAISuccessBody('openai result') }),
    async ({ baseUrl, calls }) => {
      process.env.VISIONTOOL_BASE_URL = `${baseUrl}/v1`;

      const value = await describeImage({
        image: { base64: rawBase64, mediaType: 'image/png' },
        detail: 'low',
        maxTokens: 256
      });

      assert.equal(value.text, 'openai result');
      assert.equal(value.apiFormat, 'openai');
      assert.equal(value.model, 'gpt-4o-mini');
      assert.equal(calls[0]?.path, '/v1/chat/completions');
      assert.equal(calls[0]?.headers.authorization, 'Bearer openai-key');

      const messages = calls[0]?.body.messages as Array<{ content?: Array<Record<string, unknown>> }> | undefined;
      const content = messages?.[0]?.content ?? [];
      const imageBlock = content.find((block) => block.type === 'image_url') as {
        image_url?: { url?: string };
      } | undefined;
      assert.equal(imageBlock?.image_url?.url, `data:image/png;base64,${rawBase64}`);
    }
  );
});

test('describeImage infers OpenAI-compatible format from chat completions URL', async () => {
  resetEnv();
  delete process.env.ANTHROPIC_API_KEY;
  process.env.OPENAI_API_KEY = 'openai-key';

  await withMockVisionServer(
    () => ({ body: openAISuccessBody('inferred openai') }),
    async ({ baseUrl, calls }) => {
      process.env.VISIONTOOL_BASE_URL = `${baseUrl}/openai/v1/chat/completions`;

      const value = await describeImage({
        image: { url: 'https://example.com/image.png' },
        detail: 'low',
        maxTokens: 256
      });

      assert.equal(value.apiFormat, 'openai');
      assert.equal(calls[0]?.path, '/openai/v1/chat/completions');
    }
  );
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

test('describeImage supports Gemini generateContent format', async () => {
  resetEnv();
  delete process.env.ANTHROPIC_API_KEY;
  process.env.GEMINI_API_KEY = 'gemini-key';
  process.env.VISIONTOOL_API_FORMAT = 'gemini';
  process.env.VISIONTOOL_MODEL = 'gemini-2.5-flash';
  const rawBase64 = Buffer.from('gemini-image').toString('base64');

  await withMockVisionServer(
    () => ({ body: geminiSuccessBody('gemini result') }),
    async ({ baseUrl, calls }) => {
      process.env.VISIONTOOL_BASE_URL = `${baseUrl}/v1beta`;

      const value = await describeImage({
        image: { base64: rawBase64, mediaType: 'image/png' },
        detail: 'low',
        maxTokens: 256
      });

      assert.equal(value.text, 'gemini result');
      assert.equal(value.apiFormat, 'gemini');
      assert.equal(value.model, 'gemini-2.5-flash');
      assert.equal(calls[0]?.path, '/v1beta/models/gemini-2.5-flash:generateContent?key=gemini-key');
      assert.equal(calls[0]?.headers['x-goog-api-key'], undefined);

      const body = calls[0]?.body as {
        contents?: Array<{ parts?: Array<{ inlineData?: { data?: string; mimeType?: string } }> }>;
        generationConfig?: { maxOutputTokens?: number };
      };
      assert.equal(body.generationConfig?.maxOutputTokens, 256);

      const parts = body.contents?.[0]?.parts ?? [];
      const imageBlock = parts.find((block) => block.inlineData !== undefined);
      assert.equal(imageBlock?.inlineData?.data, rawBase64);
      assert.equal(imageBlock?.inlineData?.mimeType, 'image/png');
    }
  );
});

test('describeImage infers Gemini format from generateContent URL', async () => {
  resetEnv();
  delete process.env.ANTHROPIC_API_KEY;
  process.env.GEMINI_API_KEY = 'gemini-key';

  await withMockVisionServer(
    () => ({ body: geminiSuccessBody('inferred gemini') }),
    async ({ baseUrl, calls }) => {
      process.env.VISIONTOOL_BASE_URL = `${baseUrl}/v1beta/models/gemini-custom:generateContent`;

      const value = await describeImage({
        image: { url: 'https://example.com/image.png' },
        detail: 'low',
        maxTokens: 256
      });

      assert.equal(value.apiFormat, 'gemini');
      assert.equal(calls[0]?.path, '/v1beta/models/gemini-custom:generateContent?key=gemini-key');

      const body = calls[0]?.body as {
        contents?: Array<{ parts?: Array<{ fileData?: { fileUri?: string } }> }>;
      };
      const parts = body.contents?.[0]?.parts ?? [];
      const imageBlock = parts.find((block) => block.fileData !== undefined);
      assert.equal(imageBlock?.fileData?.fileUri, 'https://example.com/image.png');
    }
  );
});

test('describeImage rejects obsolete Gemini interactions URLs', async () => {
  resetEnv();
  delete process.env.ANTHROPIC_API_KEY;
  process.env.GEMINI_API_KEY = 'gemini-key';
  process.env.VISIONTOOL_BASE_URL = 'https://generativelanguage.googleapis.com/v1beta/interactions';

  await assert.rejects(
    () => describeImage({
      image: { url: 'https://example.com/image.png' },
      detail: 'low',
      maxTokens: 256
    }),
    /generateContent API/
  );
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

test('describeImage falls back to reasoning_content when content is empty (OpenAI reasoning models)', async () => {
  resetEnv();
  delete process.env.ANTHROPIC_API_KEY;
  process.env.OPENAI_API_KEY = 'openai-key';
  process.env.VISIONTOOL_API_FORMAT = 'openai';
  process.env.VISIONTOOL_MODEL = 'minimax-m3';
  const rawBase64 = Buffer.from('openai-image').toString('base64');

  await withMockVisionServer(
    () => ({ body: { choices: [{ message: { content: null, reasoning_content: 'reasoned answer' }, finish_reason: 'stop' }] } }),
    async ({ baseUrl }) => {
      process.env.VISIONTOOL_BASE_URL = `${baseUrl}/v1`;

      const value = await describeImage({
        image: { base64: rawBase64, mediaType: 'image/png' },
        detail: 'low',
        maxTokens: 256
      });

      assert.equal(value.text, 'reasoned answer');
    }
  );
});

test('describeImage reports diagnostic details when OpenAI API returns no text content', async () => {
  resetEnv();
  delete process.env.ANTHROPIC_API_KEY;
  process.env.OPENAI_API_KEY = 'openai-key';
  process.env.VISIONTOOL_API_FORMAT = 'openai';
  process.env.VISIONTOOL_MODEL = 'minimax-m3';
  const rawBase64 = Buffer.from('openai-image').toString('base64');

  await withMockVisionServer(
    () => ({ body: { choices: [{ message: { content: null, reasoning_content: null }, finish_reason: 'stop' }] } }),
    async ({ baseUrl }) => {
      process.env.VISIONTOOL_BASE_URL = `${baseUrl}/v1`;

      let caught: Error | undefined;
      try {
        await describeImage({
          image: { base64: rawBase64, mediaType: 'image/png' },
          detail: 'low',
          maxTokens: 256
        });
      } catch (err) {
        caught = err as Error;
      }

      assert.ok(caught, 'expected describeImage to reject with a no-text-content error');
      assert.match(caught!.message, /OpenAI-compatible API returned no text content/);
      assert.match(caught!.message, /finish_reason: stop/);
      assert.match(caught!.message, /response body:/);
    }
  );
});
