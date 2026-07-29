import assert from 'node:assert/strict';
import { rm, writeFile, mkdir, readFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { InMemoryTransport } from '@modelcontextprotocol/sdk/inMemory.js';

import { createVisionToolServer } from '../src/index.js';

type HttpCall = {
  path: string;
  body: Record<string, unknown>;
};

const originalEnv = { ...process.env };

async function withMcpClient<T>(callback: (client: Client) => Promise<T>): Promise<T> {
  const server = createVisionToolServer();
  const client = new Client({ name: 'visiontool-test-client', version: '0.0.0' });
  const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();

  await Promise.all([
    client.connect(clientTransport),
    server.connect(serverTransport)
  ]);

  try {
    return await callback(client);
  } finally {
    await Promise.all([
      client.close(),
      server.close()
    ]);
  }
}

async function withMockVisionServer<T>(callback: (context: { baseUrl: string; calls: HttpCall[] }) => Promise<T>): Promise<T> {
  const calls: HttpCall[] = [];
  const server = http.createServer((req, res) => {
    const chunks: Buffer[] = [];
    req.on('data', (chunk: Buffer | string) => {
      chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
    });
    req.on('end', () => {
      const bodyText = Buffer.concat(chunks).toString('utf8');
      calls.push({
        path: req.url ?? '/',
        body: bodyText ? JSON.parse(bodyText) as Record<string, unknown> : {}
      });
      res.setHeader('content-type', 'application/json');
      res.end(JSON.stringify({ content: [{ type: 'text', text: 'server-level result' }] }));
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

  try {
    return await callback({ baseUrl: `http://127.0.0.1:${address.port}`, calls });
  } finally {
    await new Promise<void>((resolve, reject) => {
      server.close((error) => error ? reject(error) : resolve());
    });
  }
}

function resetEnv(): void {
  process.env = { ...originalEnv };
  process.env.ANTHROPIC_API_KEY = 'test-key';
  process.env.VISIONTOOL_ALLOWED_CALLER_PREFIXES = 'glm,deepseek';
  process.env.VISIONTOOL_RETRIES = '0';
  process.env.VISIONTOOL_RETRY_BASE_MS = '0';
  process.env.VISIONTOOL_DISABLE_PROXY_FALLBACK = '1';
  delete process.env.VISIONTOOL_ALLOW_PRIVATE_URLS;
  delete process.env.VISIONTOOL_API_FORMAT;
  delete process.env.VISIONTOOL_BASE_URL;
  delete process.env.VISIONTOOL_MODEL;
  delete process.env.CLAUDE_CODE_SESSION_ID;
  delete process.env.VISIONTOOL_ENABLE_CLAUDE;
  delete process.env.VISIONTOOL_CLAUDE_PROJECTS_DIR;
}

test('tools/list exposes output schemas and safety annotations', async () => {
  resetEnv();

  await withMcpClient(async (client) => {
    const result = await client.listTools();
    assert.equal(result.tools.length, 5);

    for (const tool of result.tools) {
      assert.equal(tool.outputSchema?.type, 'object');
      if (tool.name === 'upload_image') {
        assert.deepEqual(tool.outputSchema?.required, ['tool', 'imageId', 'path', 'bytes', 'expiresAt']);
      } else {
        assert.deepEqual(tool.outputSchema?.required, ['tool', 'model', 'apiFormat', 'text', 'images']);
      }
      assert.equal(tool.annotations?.readOnlyHint, true);
      assert.equal(tool.annotations?.destructiveHint, false);
      assert.equal(tool.annotations?.idempotentHint, true);
      assert.equal(tool.annotations?.openWorldHint, true);
    }
  });
});

test('tools/call returns structuredContent for successful calls', async () => {
  resetEnv();
  const tempPath = path.join(os.tmpdir(), `visiontool-server-${Date.now()}.png`);
  await writeFile(tempPath, Buffer.from([0x89, 0x50, 0x4e, 0x47]));

  try {
    await withMockVisionServer(async ({ baseUrl, calls }) => {
      process.env.VISIONTOOL_BASE_URL = `${baseUrl}/anthropic/v1`;

      await withMcpClient(async (client) => {
        const result = await client.callTool({
          name: 'describe_image',
          arguments: {
            image: { path: tempPath },
            detail: 'low',
            maxTokens: 256,
            _caller_model: 'glm-4.5'
          }
        });

        assert.equal(result.isError, undefined);
        assert.equal(result.structuredContent?.tool, 'describe_image');
        assert.equal(result.structuredContent?.text, 'server-level result');
        assert.equal(Array.isArray(result.structuredContent?.images), true);
        assert.equal(result.content[0]?.type, 'text');
        assert.equal(calls.length, 1);
      });
    });
  } finally {
    await rm(tempPath, { force: true });
  }
});

test('tools/call rejects missing or disallowed caller models before API calls', async () => {
  resetEnv();

  await withMockVisionServer(async ({ baseUrl, calls }) => {
    process.env.VISIONTOOL_BASE_URL = `${baseUrl}/anthropic/v1`;

    await withMcpClient(async (client) => {
      await assert.rejects(
        () => client.callTool({
          name: 'describe_image',
          arguments: {
            image: { base64: Buffer.from('x').toString('base64'), mediaType: 'image/png' },
            detail: 'low',
            maxTokens: 256
          }
        }),
        /_caller_model is required/
      );

      await assert.rejects(
        () => client.callTool({
          name: 'describe_image',
          arguments: {
            image: { base64: Buffer.from('x').toString('base64'), mediaType: 'image/png' },
            detail: 'low',
            maxTokens: 256,
            _caller_model: 'claude-opus-4-8'
          }
        }),
        /not allowed/
      );

      assert.equal(calls.length, 0);
    });
  });
});

test('tools/call accepts caller models with a provider prefix (e.g. winterapi/glm-5.2)', async () => {
  resetEnv();

  await withMockVisionServer(async ({ baseUrl, calls }) => {
    process.env.VISIONTOOL_BASE_URL = `${baseUrl}/anthropic/v1`;

    await withMcpClient(async (client) => {
      const result = await client.callTool({
        name: 'describe_image',
        arguments: {
          image: { base64: Buffer.from('x').toString('base64'), mediaType: 'image/png' },
          detail: 'low',
          maxTokens: 256,
          _caller_model: 'winterapi/glm-5.2'
        }
      });

      assert.equal(result.isError, undefined);
      assert.equal(result.structuredContent?.tool, 'describe_image');
      assert.equal(calls.length, 1);

      await assert.rejects(
        () => client.callTool({
          name: 'describe_image',
          arguments: {
            image: { base64: Buffer.from('x').toString('base64'), mediaType: 'image/png' },
            detail: 'low',
            maxTokens: 256,
            _caller_model: 'winterapi/claude-opus-4-8'
          }
        }),
        /not allowed/
      );
    });
  });
});

// Minimal 1x1 transparent PNG.
const tinyPngBase64 = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==';
const tinyPngBytes = Buffer.from(tinyPngBase64, 'base64');

function buildClaudeTranscriptLine(imageBase64: string, mediaType = 'image/png'): string {
  return JSON.stringify({
    type: 'user',
    message: {
      role: 'user',
      content: [
        { type: 'text', text: 'hi' },
        { type: 'image', source: { type: 'base64', media_type: mediaType, data: imageBase64 } }
      ]
    }
  });
}

test('claude_pasted_image extracts the latest inline image from the session transcript', async () => {
  resetEnv();
  process.env.VISIONTOOL_ALLOWED_CALLER_PREFIXES = 'glm,deepseek';

  const sessionId = 'test-session-uuid';
  const projectsDir = path.join(os.tmpdir(), `visiontool-claude-projects-${Date.now()}`);
  // Folder name must equal escapeClaudeCwd(process.cwd()) so the direct-path lookup hits.
  const escapedCwd = process.cwd().replace(/[^a-zA-Z0-9]/g, '-');
  const sessionDir = path.join(projectsDir, escapedCwd);
  await mkdir(sessionDir, { recursive: true });

  const transcriptPath = path.join(sessionDir, `${sessionId}.jsonl`);
  // Two images; the tool must pick the last one (tinyPng).
  await writeFile(
    transcriptPath,
    `${buildClaudeTranscriptLine(Buffer.from('AAAA').toString('base64'))}\n${buildClaudeTranscriptLine(tinyPngBase64)}\n`
  );

  try {
    process.env.CLAUDE_CODE_SESSION_ID = sessionId;
    process.env.VISIONTOOL_CLAUDE_PROJECTS_DIR = projectsDir;

    await withMcpClient(async (client) => {
      const list = await client.listTools();
      assert.ok(list.tools.some((t) => t.name === 'claude_pasted_image'));

      const result = await client.callTool({
        name: 'claude_pasted_image',
        arguments: { _caller_model: 'glm-4.5' }
      });

      assert.equal(result.isError, undefined);
      assert.equal(result.structuredContent?.tool, 'claude_pasted_image');
      assert.equal(result.structuredContent?.sessionId, sessionId);
      assert.equal(result.structuredContent?.mediaType, 'image/png');
      assert.equal(result.structuredContent?.bytes, tinyPngBytes.length);
      assert.equal(existsSync(result.structuredContent?.path), true);
      assert.deepEqual(await readFile(result.structuredContent?.path, null), tinyPngBytes);
    });
  } finally {
    await rm(projectsDir, { recursive: true, force: true });
  }
});

test('claude_pasted_image is not registered when CLAUDE_CODE_SESSION_ID is absent', async () => {
  resetEnv();

  await withMcpClient(async (client) => {
    const list = await client.listTools();
    assert.equal(list.tools.some((t) => t.name === 'claude_pasted_image'), false);

    await assert.rejects(
      () => client.callTool({
        name: 'claude_pasted_image',
        arguments: { _caller_model: 'glm-4.5' }
      }),
      /not enabled|MethodNotFound|claude_pasted_image/i
    );
  });
});

test('claude_pasted_image errors clearly when the transcript has no image', async () => {
  resetEnv();
  process.env.VISIONTOOL_ALLOWED_CALLER_PREFIXES = 'glm,deepseek';

  const sessionId = 'test-session-noimg';
  const projectsDir = path.join(os.tmpdir(), `visiontool-claude-projects-${Date.now()}`);
  const escapedCwd = process.cwd().replace(/[^a-zA-Z0-9]/g, '-');
  const sessionDir = path.join(projectsDir, escapedCwd);
  await mkdir(sessionDir, { recursive: true });
  const transcriptPath = path.join(sessionDir, `${sessionId}.jsonl`);
  await writeFile(transcriptPath, `${JSON.stringify({ type: 'user', message: { role: 'user', content: [{ type: 'text', text: 'no image here' }] } })}\n`);

  try {
    process.env.CLAUDE_CODE_SESSION_ID = sessionId;
    process.env.VISIONTOOL_CLAUDE_PROJECTS_DIR = projectsDir;

    await withMcpClient(async (client) => {
      const result = await client.callTool({
        name: 'claude_pasted_image',
        arguments: { _caller_model: 'glm-4.5' }
      });

      assert.equal(result.isError, true);
      assert.match(result.content[0]?.text ?? '', /No inline image/);
    });
  } finally {
    await rm(projectsDir, { recursive: true, force: true });
  }
});
