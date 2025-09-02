import { z } from 'zod';

// Auth schemas
export const signInSchema = z.object({
  email: z.string().email('Please enter a valid email address'),
  callbackUrl: z.string().optional(),
});

export const totpSetupSchema = z.object({
  token: z
    .string()
    .length(6, 'TOTP token must be 6 digits')
    .regex(/^\d{6}$/, 'TOTP token must contain only digits'),
});

export const totpVerifySchema = z.object({
  token: z
    .string()
    .length(6, 'TOTP token must be 6 digits')
    .regex(/^\d{6}$/, 'TOTP token must contain only digits'),
  action: z.enum(['verify', 'disable']).optional(),
});

// User management schemas
export const updateUserSchema = z.object({
  name: z.string().min(1, 'Name is required').optional(),
  locale: z.enum(['hi-IN', 'en-IN']).optional(),
  phone: z
    .string()
    .regex(/^\+?[1-9]\d{1,14}$/, 'Please enter a valid phone number')
    .optional(),
});

export const updateUserRolesSchema = z.object({
  userId: z.string().cuid('Invalid user ID'),
  roles: z.array(
    z.enum(['ADMIN', 'APPROVER', 'EDITOR', 'DATA_ENTRY', 'VIEWER'])
  ),
});

// Content schemas
export const pageSchema = z.object({
  slug: z
    .string()
    .min(1, 'Slug is required')
    .regex(
      /^[a-z0-9-]+$/,
      'Slug can only contain lowercase letters, numbers, and hyphens'
    ),
  title: z
    .record(z.string(), z.string())
    .refine(
      (obj) => obj['hi-IN'] || obj['en-IN'],
      'Title must be provided in at least one language'
    ),
  blocks: z.array(z.any()).optional(),
  seo: z
    .object({
      title: z.record(z.string(), z.string()).optional(),
      description: z.record(z.string(), z.string()).optional(),
      keywords: z.array(z.string()).optional(),
    })
    .optional(),
  status: z.enum(['DRAFT', 'PUBLISHED', 'ARCHIVED']).default('DRAFT'),
});

// Media schema
export const mediaUploadSchema = z.object({
  alt: z.string().min(1, 'Alt text is required for accessibility'),
  caption: z.string().optional(),
});

// Form schemas
export const formSchema = z.object({
  name: z.string().min(1, 'Form name is required'),
  slug: z
    .string()
    .min(1, 'Slug is required')
    .regex(
      /^[a-z0-9-]+$/,
      'Slug can only contain lowercase letters, numbers, and hyphens'
    ),
  schemaJSON: z.any(), // JSON schema for the form
  slaDays: z.number().min(1).max(365),
  workflowJSON: z.any().optional(),
  active: z.boolean().default(true),
});

// Submission schema
export const submissionSchema = z.object({
  formId: z.string().cuid(),
  dataJSON: z.any(), // Form submission data
  files: z.array(z.string().url()).optional(),
  geo: z
    .object({
      lat: z.number().min(-90).max(90),
      lng: z.number().min(-180).max(180),
      address: z.string().optional(),
    })
    .optional(),
});

// Project schemas
export const projectSchema = z.object({
  title: z
    .record(z.string(), z.string())
    .refine(
      (obj) => obj['hi-IN'] || obj['en-IN'],
      'Title must be provided in at least one language'
    ),
  description: z.record(z.string(), z.string()).optional(),
  type: z.string().min(1, 'Project type is required'),
  ward: z.string().optional(),
  budget: z.number().positive('Budget must be positive'),
  startDate: z.string().datetime().optional(),
  endDate: z.string().datetime().optional(),
  geo: z.any().optional(), // GeoJSON
  tags: z.array(z.string()).optional(),
});

// Scheme schemas
export const schemeSchema = z.object({
  title: z
    .record(z.string(), z.string())
    .refine(
      (obj) => obj['hi-IN'] || obj['en-IN'],
      'Title must be provided in at least one language'
    ),
  description: z.record(z.string(), z.string()).optional(),
  category: z.string().min(1, 'Category is required'),
  criteriaJSON: z.any(), // Eligibility rules
  docsRequired: z.array(z.string()).optional(),
  processSteps: z.any().optional(),
  links: z.array(z.any()).optional(),
  active: z.boolean().default(true),
});

// Event schemas
export const eventSchema = z.object({
  title: z
    .record(z.string(), z.string())
    .refine(
      (obj) => obj['hi-IN'] || obj['en-IN'],
      'Title must be provided in at least one language'
    ),
  description: z.record(z.string(), z.string()).optional(),
  startDate: z.string().datetime(),
  endDate: z.string().datetime().optional(),
  location: z.string().optional(),
  geo: z
    .object({
      lat: z.number().min(-90).max(90),
      lng: z.number().min(-180).max(180),
    })
    .optional(),
  rsvpEnabled: z.boolean().default(false),
  maxAttendees: z.number().positive().optional(),
  tags: z.array(z.string()).optional(),
});

// Directory entry schemas
export const directoryEntrySchema = z.object({
  type: z.enum(['SHG', 'BUSINESS', 'ARTISAN']),
  name: z.string().min(1, 'Name is required'),
  contact: z.object({
    email: z.string().email().optional(),
    phone: z.string().optional(),
    address: z.string().optional(),
    website: z.string().url().optional(),
  }),
  description: z.record(z.string(), z.string()).optional(),
  products: z.array(z.any()).optional(),
  geo: z
    .object({
      lat: z.number().min(-90).max(90),
      lng: z.number().min(-180).max(180),
    })
    .optional(),
  tags: z.array(z.string()).optional(),
});

// Pledge schemas
export const pledgeSchema = z.object({
  pledgeType: z.enum(['TREE', 'SOLAR', 'WASTE']),
  title: z.string().optional(),
  amount: z.number().positive('Amount must be positive'),
  description: z.string().optional(),
  geo: z
    .object({
      lat: z.number().min(-90).max(90),
      lng: z.number().min(-180).max(180),
      address: z.string().optional(),
    })
    .optional(),
});

// Notification schemas
export const notificationSchema = z.object({
  title: z
    .record(z.string(), z.string())
    .refine(
      (obj) => obj['hi-IN'] || obj['en-IN'],
      'Title must be provided in at least one language'
    ),
  content: z
    .record(z.string(), z.string())
    .refine(
      (obj) => obj['hi-IN'] || obj['en-IN'],
      'Content must be provided in at least one language'
    ),
  channels: z.array(z.enum(['EMAIL', 'SMS', 'WHATSAPP', 'WEB_PUSH'])),
  audienceJSON: z.any(), // Targeting criteria
  scheduledAt: z.string().datetime().optional(),
});

// Translation schemas
export const translationKeySchema = z.object({
  key: z
    .string()
    .min(1, 'Key is required')
    .regex(
      /^[a-z0-9_.]+$/,
      'Key can only contain lowercase letters, numbers, dots, and underscores'
    ),
  defaultText: z.string().min(1, 'Default text is required'),
  module: z.string().optional(),
});

export const translationValueSchema = z.object({
  keyId: z.string().cuid(),
  locale: z.enum(['hi-IN', 'en-IN']),
  text: z.string().min(1, 'Translation text is required'),
});

export type SignInInput = z.infer<typeof signInSchema>;
export type TOTPSetupInput = z.infer<typeof totpSetupSchema>;
export type TOTPVerifyInput = z.infer<typeof totpVerifySchema>;
export type UpdateUserInput = z.infer<typeof updateUserSchema>;
export type UpdateUserRolesInput = z.infer<typeof updateUserRolesSchema>;
export type PageInput = z.infer<typeof pageSchema>;
export type MediaUploadInput = z.infer<typeof mediaUploadSchema>;
export type FormInput = z.infer<typeof formSchema>;
export type SubmissionInput = z.infer<typeof submissionSchema>;
export type ProjectInput = z.infer<typeof projectSchema>;
export type SchemeInput = z.infer<typeof schemeSchema>;
export type EventInput = z.infer<typeof eventSchema>;
export type DirectoryEntryInput = z.infer<typeof directoryEntrySchema>;
export type PledgeInput = z.infer<typeof pledgeSchema>;
export type NotificationInput = z.infer<typeof notificationSchema>;
export type TranslationKeyInput = z.infer<typeof translationKeySchema>;
export type TranslationValueInput = z.infer<typeof translationValueSchema>;
