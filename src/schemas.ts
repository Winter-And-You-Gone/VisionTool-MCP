import { z } from 'zod';

export const supportedMimeTypes = ['image/png', 'image/jpeg', 'image/webp', 'image/gif'] as const;
export const detailLevels = ['low', 'medium', 'high'] as const;

const maxPromptLength = 8000;
const maxBase64Length = 8_000_000;
const maxUrlLength = 4096;
const maxImageIdLength = 128;

const imagePathInputSchema = z.object({
  path: z.string().min(1).optional(),
  base64: z.string().min(1).max(maxBase64Length).optional(),
  url: z.string().url().max(maxUrlLength).optional(),
  imageId: z.string().min(1).max(maxImageIdLength).optional(),
  mediaType: z.enum(supportedMimeTypes).optional()
}).strict().refine(
  (value) => [value.path, value.base64, value.url, value.imageId].filter((item) => item !== undefined).length === 1,
  'Provide exactly one of path, base64, url, or imageId.'
).refine(
  (value) => value.base64 === undefined || value.mediaType !== undefined,
  'mediaType is required when base64 is provided.'
);

export const uploadImageSchema = z.object({
  base64: z.string().min(1).max(maxBase64Length),
  mediaType: z.enum(supportedMimeTypes),
  filename: z.string().min(1).max(255).optional(),
  _caller_model: z.string().min(1)
}).strict();

export const opencodePastedImageSchema = z.object({
  _caller_model: z.string().min(1)
}).strict();
export type OpencodePastedImageInput = z.infer<typeof opencodePastedImageSchema>;

const commonVisionOptionsSchema = z.object({
  detail: z.enum(detailLevels).optional().default('medium'),
  maxTokens: z.number().int().min(128).max(4096).optional().default(1024),
  model: z.string().min(1).optional(),
  _caller_model: z.string().min(1)
}).strict();

export const describeImageSchema = z.object({
  image: imagePathInputSchema,
  instruction: z.string().min(1).max(maxPromptLength).optional(),
  focus: z.string().min(1).max(1000).optional(),
  detail: commonVisionOptionsSchema.shape.detail,
  maxTokens: commonVisionOptionsSchema.shape.maxTokens,
  model: commonVisionOptionsSchema.shape.model,
  _caller_model: commonVisionOptionsSchema.shape._caller_model
}).strict();

export const ocrImageSchema = z.object({
  image: imagePathInputSchema,
  language: z.string().min(1).max(80).optional(),
  preserveLayout: z.boolean().optional().default(true),
  detail: commonVisionOptionsSchema.shape.detail,
  maxTokens: commonVisionOptionsSchema.shape.maxTokens.default(2048),
  model: commonVisionOptionsSchema.shape.model,
  _caller_model: commonVisionOptionsSchema.shape._caller_model
}).strict();

export const answerAboutImageSchema = z.object({
  image: imagePathInputSchema,
  question: z.string().min(1).max(maxPromptLength),
  detail: commonVisionOptionsSchema.shape.detail,
  maxTokens: commonVisionOptionsSchema.shape.maxTokens,
  model: commonVisionOptionsSchema.shape.model,
  _caller_model: commonVisionOptionsSchema.shape._caller_model
}).strict();

export const compareImagesSchema = z.object({
  firstImage: imagePathInputSchema,
  secondImage: imagePathInputSchema,
  instruction: z.string().min(1).max(maxPromptLength).optional(),
  detail: commonVisionOptionsSchema.shape.detail,
  maxTokens: commonVisionOptionsSchema.shape.maxTokens.default(1536),
  model: commonVisionOptionsSchema.shape.model,
  _caller_model: commonVisionOptionsSchema.shape._caller_model
}).strict();

export type ImageInput = z.infer<typeof imagePathInputSchema>;
export type DescribeImageInput = z.infer<typeof describeImageSchema>;
export type OcrImageInput = z.infer<typeof ocrImageSchema>;
export type AnswerAboutImageInput = z.infer<typeof answerAboutImageSchema>;
export type CompareImagesInput = z.infer<typeof compareImagesSchema>;
export type UploadImageInput = z.infer<typeof uploadImageSchema>;

const imageInputSchema = {
  type: 'object',
  properties: {
    path: { type: 'string', description: 'Local image path. Relative paths resolve from the MCP server working directory.' },
    base64: { type: 'string', description: 'Raw base64-encoded image data.' },
    url: { type: 'string', format: 'uri', description: 'Publicly reachable image URL.' },
    imageId: { type: 'string', description: 'Image ID returned by upload_image tool. Use this for images previously uploaded.' },
    mediaType: { type: 'string', enum: supportedMimeTypes, description: 'Required for base64 input. Optional for path input when the extension is ambiguous.' }
  },
  oneOf: [
    { required: ['path'] },
    { required: ['base64', 'mediaType'] },
    { required: ['url'] },
    { required: ['imageId'] }
  ],
  additionalProperties: false
} as const;

const commonProperties = {
  detail: { type: 'string', enum: detailLevels, default: 'medium', description: 'How much visual detail to request from the vision model.' },
  maxTokens: { type: 'integer', minimum: 128, maximum: 4096, default: 1024 },
  model: { type: 'string', description: 'Optional provider model override. Defaults to VISIONTOOL_MODEL or the selected provider default.' },
  _caller_model: { type: 'string', description: '【限制项】调用方模型名。默认仅允许 GLM / DeepSeek 系列；可用 VISIONTOOL_ALLOWED_CALLER_PREFIXES 覆盖，* 表示允许任意非空调用者。' }
} as const;

export const visionResultOutputSchema = {
  type: 'object',
  properties: {
    tool: { type: 'string', description: 'Tool that produced this result.' },
    model: { type: 'string', description: 'Vision model used for the upstream request.' },
    apiFormat: { type: 'string', enum: ['anthropic', 'openai', 'gemini'], description: 'Provider API format used for the upstream request.' },
    text: { type: 'string', description: 'Text answer returned by the vision model.' },
    images: {
      type: 'array',
      description: 'Normalized metadata for images submitted to the upstream model.',
      items: {
        type: 'object',
        properties: {
          source: { type: 'string', enum: ['path', 'base64', 'url'] },
          mediaType: { type: 'string' },
          path: { type: 'string' },
          url: { type: 'string' },
          bytes: { type: 'integer', minimum: 0 }
        },
        required: ['source'],
        additionalProperties: false
      }
    }
  },
  required: ['tool', 'model', 'apiFormat', 'text', 'images'],
  additionalProperties: false
} as const;

export const toolInputSchemas = {
  describe_image: {
    type: 'object',
    properties: {
      image: imageInputSchema,
      instruction: { type: 'string', description: 'Optional extra instruction for what to describe.' },
      focus: { type: 'string', description: 'Optional visual area or topic to focus on.' },
      ...commonProperties
    },
    required: ['image', '_caller_model'],
    additionalProperties: false
  },
  ocr_image: {
    type: 'object',
    properties: {
      image: imageInputSchema,
      language: { type: 'string', description: 'Optional language hint for visible text.' },
      preserveLayout: { type: 'boolean', default: true, description: 'Preserve rough reading order and layout where possible.' },
      ...commonProperties,
      maxTokens: { type: 'integer', minimum: 128, maximum: 4096, default: 2048 }
    },
    required: ['image', '_caller_model'],
    additionalProperties: false
  },
  answer_about_image: {
    type: 'object',
    properties: {
      image: imageInputSchema,
      question: { type: 'string', description: 'Question to answer using visual evidence from the image.' },
      ...commonProperties
    },
    required: ['image', 'question', '_caller_model'],
    additionalProperties: false
  },
  compare_images: {
    type: 'object',
    properties: {
      firstImage: imageInputSchema,
      secondImage: imageInputSchema,
      instruction: { type: 'string', description: 'Optional comparison instruction or focus.' },
      ...commonProperties,
      maxTokens: { type: 'integer', minimum: 128, maximum: 4096, default: 1536 }
    },
    required: ['firstImage', 'secondImage', '_caller_model'],
    additionalProperties: false
  }
} as const;

export const uploadImageToolInputSchema = {
  type: 'object',
  properties: {
    base64: { type: 'string', description: 'Raw base64-encoded image data.' },
    mediaType: { type: 'string', enum: supportedMimeTypes, description: 'Image MIME type.' },
    filename: { type: 'string', description: 'Optional filename for reference (optional).' },
    ...commonProperties
  },
  required: ['base64', 'mediaType', '_caller_model'],
  additionalProperties: false
} as const;

export const uploadImageResultOutputSchema = {
  type: 'object',
  properties: {
    tool: { type: 'string', description: 'Tool that produced this result.' },
    imageId: { type: 'string', description: 'Image ID to use in subsequent vision calls.' },
    path: { type: 'string', description: 'Local path where the image was saved.' },
    bytes: { type: 'integer', minimum: 0, description: 'Image size in bytes.' },
    expiresAt: { type: 'string', description: 'ISO timestamp when the image will be auto-deleted.' }
  },
  required: ['tool', 'imageId', 'path', 'bytes', 'expiresAt'],
  additionalProperties: false
} as const;

export const opencodePastedImageToolInputSchema = {
  type: 'object',
  properties: {
    _caller_model: commonProperties._caller_model
  },
  required: ['_caller_model'],
  additionalProperties: false
} as const;

export const opencodePastedImageResultOutputSchema = {
  type: 'object',
  properties: {
    tool: { type: 'string', description: 'Tool that produced this result.' },
    imageId: { type: 'string', description: 'Image ID to use in subsequent vision calls (describe_image, ocr_image, answer_about_image, compare_images).' },
    path: { type: 'string', description: 'Local path where the extracted image was saved.' },
    bytes: { type: 'integer', minimum: 0, description: 'Image size in bytes.' },
    mediaType: { type: 'string', description: 'Detected image MIME type.' },
    filename: { type: 'string', description: 'Original filename from the opencode part, if any.', nullable: true },
    sessionId: { type: 'string', description: 'opencode session ID the image was read from.' },
    timeCreated: { type: 'integer', description: 'Epoch milliseconds when the image part was created in opencode.' },
    expiresAt: { type: 'string', description: 'ISO timestamp when the extracted image will be auto-deleted.' }
  },
  required: ['tool', 'imageId', 'path', 'bytes', 'mediaType', 'filename', 'sessionId', 'timeCreated', 'expiresAt'],
  additionalProperties: false
} as const;
