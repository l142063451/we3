import speakeasy from 'speakeasy';
import QRCode from 'qrcode';
import { prisma } from '@/lib/db';

export interface TOTPSetup {
  secret: string;
  qrCodeDataURL: string;
  manualEntryKey: string;
  backupCodes: string[];
}

export async function generateTOTPSecret(
  userId: string,
  appName: string = 'Ummid Se Hari'
): Promise<TOTPSetup> {
  const user = await prisma.user.findUnique({
    where: { id: userId },
    select: { email: true, name: true }
  })

  if (!user) {
    throw new Error('User not found')
  }

  const secret = speakeasy.generateSecret({
    name: `${appName}:${user.email || user.name || 'User'}`,
    length: 32,
  });

  const qrCodeDataURL = await QRCode.toDataURL(secret.otpauth_url!);

  // Generate backup codes
  const backupCodes = await generateBackupCodes(userId);

  return {
    secret: secret.base32,
    qrCodeDataURL,
    manualEntryKey: secret.base32,
    backupCodes,
  };
}

export async function enableTOTP(
  userId: string,
  token: string,
  secret: string
): Promise<boolean> {
  const verified = speakeasy.totp.verify({
    secret,
    token,
    window: 2, // Allow 2 time steps before and after current time
  });

  if (!verified) {
    return false;
  }

  await prisma.user.update({
    where: { id: userId },
    data: {
      twoFAEnabled: true,
      twoFASecret: secret,
    },
  })

  // Log 2FA enablement
  await prisma.auditLog.create({
    data: {
      actorId: userId,
      action: 'enable_2fa',
      entity: 'User',
      entityId: userId,
      diffJSON: { event: '2fa_enabled' },
    },
  })

  return true;
}

export async function disableTOTP(
  userId: string,
  token: string
): Promise<boolean> {
  const user = await prisma.user.findUnique({
    where: { id: userId },
    select: { twoFASecret: true }
  })

  if (!user?.twoFASecret) {
    return false
  }

  const verified = speakeasy.totp.verify({
    secret: user.twoFASecret,
    token,
    window: 2,
  });

  if (!verified) {
    return false;
  }

  await prisma.user.update({
    where: { id: userId },
    data: {
      twoFAEnabled: false,
      twoFASecret: null,
    },
  })

  // Log 2FA disablement
  await prisma.auditLog.create({
    data: {
      actorId: userId,
      action: 'disable_2fa',
      entity: 'User',
      entityId: userId,
      diffJSON: { event: '2fa_disabled' },
    },
  })

  return true;
}

export async function verifyTOTP(
  userId: string,
  token: string
): Promise<boolean> {
  const user = await prisma.user.findUnique({
    where: { id: userId },
    select: { twoFASecret: true, twoFAEnabled: true }
  })

  if (!user?.twoFAEnabled || !user.twoFASecret) {
    return false
  }

  return speakeasy.totp.verify({
    secret: user.twoFASecret,
    token,
    window: 2,
  })
}

export async function verifyBackupCode(
  userId: string, 
  code: string
): Promise<boolean> {
  // This would need a backupCodes field in the User model
  // For now, we'll implement a simple version using a separate table or JSON field
  
  // Get user's backup codes (you'll need to add this field to User model)
  // const user = await prisma.user.findUnique({
  //   where: { id: userId },
  //   select: { backupCodes: true } // Add this field to schema
  // })

  // Mock implementation for now
  console.log('Backup code verification for user:', userId, 'code:', code);
  return code.length === 8; // Mock verification
}

export async function generateBackupCodes(userId: string): Promise<string[]> {
  const codes: string[] = [];

  // Generate 8 backup codes
  for (let i = 0; i < 8; i++) {
    codes.push(Math.random().toString(36).substr(2, 8).toUpperCase());
  }

  // Hash the backup codes before storing them
  // const hashedCodes = await Promise.all(
  //   codes.map(code => bcrypt.hash(code, 12))
  // );

  // Store hashed backup codes in database
  // Note: You might want to create a separate BackupCode model
  // or add a backupCodes JSON field to the User model
  
  await prisma.auditLog.create({
    data: {
      actorId: userId,
      action: 'generate_backup_codes',
      entity: 'User',
      entityId: userId,
      diffJSON: { 
        event: 'backup_codes_generated',
        codes_count: codes.length 
      },
    },
  })

  return codes;
}

export async function getTOTPStatus(userId: string): Promise<{
  enabled: boolean;
  secret?: string;
  backupCodesGenerated: boolean;
}> {
  const user = await prisma.user.findUnique({
    where: { id: userId },
    select: { 
      twoFAEnabled: true, 
      twoFASecret: true,
      // backupCodes: true // Add this field
    }
  })

  if (!user) {
    throw new Error('User not found')
  }

  return {
    enabled: user.twoFAEnabled || false,
    secret: user.twoFASecret || undefined,
    backupCodesGenerated: false, // Update this based on actual backup codes
  };
}
