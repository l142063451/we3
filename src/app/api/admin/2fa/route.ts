import { NextRequest, NextResponse } from 'next/server';
import { requireAuth } from '@/lib/auth';
import { generateTOTPSecret, enableTOTP, disableTOTP, verifyTOTP, getTOTPStatus } from '@/lib/auth/totp';
import { z } from 'zod';

// GET - Get current 2FA status
export async function GET() {
  try {
    const session = await requireAuth();
    const userId = session.user.id;

    const status = await getTOTPStatus(userId);

    return NextResponse.json({
      success: true,
      data: {
        enabled: status.enabled,
        backupCodesGenerated: status.backupCodesGenerated,
      },
    });
  } catch (error) {
    console.error('Error getting 2FA status:', error);
    return NextResponse.json(
      { success: false, error: 'Failed to get 2FA status' },
      { status: 500 }
    );
  }
}

// POST - Generate new TOTP secret (first step in enabling 2FA)
const generateSecretSchema = z.object({
  action: z.literal('generate'),
});

const enableSchema = z.object({
  action: z.literal('enable'),
  token: z.string().length(6),
  secret: z.string().min(1),
});

const disableSchema = z.object({
  action: z.literal('disable'),
  token: z.string().length(6),
});

const verifySchema = z.object({
  action: z.literal('verify'),
  token: z.string().length(6),
});

const actionSchema = z.discriminatedUnion('action', [
  generateSecretSchema,
  enableSchema,
  disableSchema,
  verifySchema,
]);

export async function POST(request: NextRequest) {
  try {
    const session = await requireAuth();
    const userId = session.user.id;
    
    const body = await request.json();
    const validatedData = actionSchema.parse(body);

    switch (validatedData.action) {
      case 'generate': {
        const setup = await generateTOTPSecret(userId);
        
        return NextResponse.json({
          success: true,
          data: {
            secret: setup.secret,
            qrCodeDataURL: setup.qrCodeDataURL,
            manualEntryKey: setup.manualEntryKey,
            backupCodes: setup.backupCodes,
          },
        });
      }

      case 'enable': {
        const success = await enableTOTP(userId, validatedData.token, validatedData.secret);
        
        if (!success) {
          return NextResponse.json(
            { success: false, error: 'Invalid verification code. Please try again.' },
            { status: 400 }
          );
        }

        return NextResponse.json({
          success: true,
          message: '2FA has been successfully enabled for your account.',
        });
      }

      case 'disable': {
        const success = await disableTOTP(userId, validatedData.token);
        
        if (!success) {
          return NextResponse.json(
            { success: false, error: 'Invalid verification code. Please try again.' },
            { status: 400 }
          );
        }

        return NextResponse.json({
          success: true,
          message: '2FA has been successfully disabled for your account.',
        });
      }

      case 'verify': {
        const isValid = await verifyTOTP(userId, validatedData.token);
        
        return NextResponse.json({
          success: true,
          data: { valid: isValid },
        });
      }

      default:
        return NextResponse.json(
          { success: false, error: 'Invalid action' },
          { status: 400 }
        );
    }
  } catch (error) {
    console.error('Error processing 2FA request:', error);
    
    if (error instanceof z.ZodError) {
      return NextResponse.json(
        { success: false, error: 'Invalid request data', details: error.issues },
        { status: 400 }
      );
    }

    return NextResponse.json(
      { success: false, error: 'Failed to process 2FA request' },
      { status: 500 }
    );
  }
}