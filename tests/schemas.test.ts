import assert from 'node:assert/strict';
import test from 'node:test';

import {
  answerAboutImageSchema,
  compareImagesSchema,
  describeImageSchema,
  ocrImageSchema,
  toolInputSchemas
} from '../src/schemas.js';

test('describe_image accepts path input and defaults', () => {
  const parsed = describeImageSchema.parse({
    image: { path: 'sample.png' }
  });

  assert.equal(parsed.image.path, 'sample.png');
  assert.equal(parsed.detail, 'medium');
  assert.equal(parsed.maxTokens, 1024);
});

test('image input requires exactly one source selector', () => {
  assert.throws(() => describeImageSchema.parse({ image: {} }), /exactly one/);
  assert.throws(() => describeImageSchema.parse({
    image: { path: 'sample.png', url: 'https://example.com/sample.png' }
  }), /exactly one/);
});

test('base64 input requires mediaType', () => {
  assert.throws(() => describeImageSchema.parse({
    image: { base64: Buffer.from('x').toString('base64') }
  }), /mediaType is required/);

  const parsed = describeImageSchema.parse({
    image: {
      base64: Buffer.from('x').toString('base64'),
      mediaType: 'image/png'
    }
  });

  assert.equal(parsed.image.mediaType, 'image/png');
});

test('ocr_image accepts language and layout options', () => {
  const parsed = ocrImageSchema.parse({
    image: { url: 'https://example.com/sample.png' },
    language: 'Chinese',
    preserveLayout: false
  });

  assert.equal(parsed.language, 'Chinese');
  assert.equal(parsed.preserveLayout, false);
  assert.equal(parsed.maxTokens, 2048);
});

test('answer_about_image requires a question', () => {
  assert.throws(() => answerAboutImageSchema.parse({ image: { path: 'sample.png' } }));

  const parsed = answerAboutImageSchema.parse({
    image: { path: 'sample.png' },
    question: 'What is visible?'
  });

  assert.equal(parsed.question, 'What is visible?');
});

test('compare_images requires two images and accepts instruction', () => {
  const parsed = compareImagesSchema.parse({
    firstImage: { path: 'before.png' },
    secondImage: { path: 'after.png' },
    instruction: 'Focus on UI layout changes.'
  });

  assert.equal(parsed.firstImage.path, 'before.png');
  assert.equal(parsed.secondImage.path, 'after.png');
  assert.equal(parsed.maxTokens, 1536);
});

test('schemas reject unsupported mime types and invalid options', () => {
  assert.throws(() => describeImageSchema.parse({
    image: { base64: 'eA==', mediaType: 'image/bmp' }
  }));

  assert.throws(() => describeImageSchema.parse({
    image: { path: 'sample.png' },
    detail: 'extreme'
  }));

  assert.throws(() => describeImageSchema.parse({
    image: { path: 'sample.png' },
    maxTokens: 16
  }));
});

test('tool input schemas expose expected MCP tools', () => {
  assert.deepEqual(Object.keys(toolInputSchemas), [
    'describe_image',
    'ocr_image',
    'answer_about_image',
    'compare_images'
  ]);

  assert.equal(toolInputSchemas.describe_image.required.includes('image'), true);
  assert.equal(toolInputSchemas.answer_about_image.required.includes('question'), true);
});
