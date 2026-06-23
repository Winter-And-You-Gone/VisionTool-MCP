import assert from 'node:assert/strict';

import { describeImage } from '../src/vision.js';

if (process.env.MCP_VISION_LIVE !== '1') {
  console.log('Skipping live vision smoke test. Set MCP_VISION_LIVE=1 to run it.');
  process.exit(0);
}

const imagePath = process.env.VISIONTOOL_SMOKE_IMAGE;
assert.ok(imagePath, 'VISIONTOOL_SMOKE_IMAGE must point to a local PNG/JPEG/WebP/GIF image.');
assert.ok(process.env.ANTHROPIC_API_KEY, 'ANTHROPIC_API_KEY is required for live smoke testing.');

const result = await describeImage({
  image: { path: imagePath },
  instruction: 'Return one concise sentence describing the image.',
  detail: 'low',
  maxTokens: 256
});

assert.equal(result.tool, 'describe_image');
assert.ok(result.text.length > 0);
console.log(JSON.stringify(result, null, 2));
