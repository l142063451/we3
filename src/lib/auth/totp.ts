import speakeasy from 'speakeasy';
import QRCode from 'qrcode';
// import { prisma } from '@/lib/db'

export interface TOTPSetup {
  secret: string;
  qrCodeDataURL: string;
  manualEntryKey: string;
}

export async function generateTOTPSecret(
  userId: string,
  appName: string = 'Ummid Se Hari'
): Promise<TOTPSetup> {
  // Mock implementation until Prisma is working
  // const user = await prisma.user.findUnique({
  //   where: { id: userId },
  //   select: { email: true, name: true }
  // })

  // if (!user) {
  //   throw new Error('User not found')
  // }

  // Mock user data
  const user = { email: 'user@example.com', name: 'User' };

  const secret = speakeasy.generateSecret({
    name: `${appName}:${user.email || user.name || 'User'}`,
    length: 32,
  });

  const qrCodeDataURL = await QRCode.toDataURL(secret.otpauth_url!);

  return {
    secret: secret.base32,
    qrCodeDataURL,
    manualEntryKey: secret.base32,
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

  // Mock implementation until Prisma is working
  // await prisma.user.update({
  //   where: { id: userId },
  //   data: {
  //     twoFAEnabled: true,
  //     twoFASecret: secret,
  //   },
  // })

  // // Log 2FA enablement
  // await prisma.auditLog.create({
  //   data: {
  //     actorId: userId,
  //     action: 'enable_2fa',
  //     entity: 'User',
  //     entityId: userId,
  //     diffJSON: { event: '2fa_enabled' },
  //   },
  // })

  console.log('2FA enabled for user:', userId);

  return true;
}

export async function disableTOTP(
  userId: string,
  token: string
): Promise<boolean> {
  // Mock implementation until Prisma is working
  // const user = await prisma.user.findUnique({
  //   where: { id: userId },
  //   select: { twoFASecret: true }
  // })

  // if (!user?.twoFASecret) {
  //   return false
  // }

  const mockSecret = 'mock-secret'; // Mock secret

  const verified = speakeasy.totp.verify({
    secret: mockSecret,
    token,
    window: 2,
  });

  if (!verified) {
    return false;
  }

  // await prisma.user.update({
  //   where: { id: userId },
  //   data: {
  //     twoFAEnabled: false,
  //     twoFASecret: null,
  //   },
  // })

  // // Log 2FA disablement
  // await prisma.auditLog.create({
  //   data: {
  //     actorId: userId,
  //     action: 'disable_2fa',
  //     entity: 'User',
  //     entityId: userId,
  //     diffJSON: { event: '2fa_disabled' },
  //   },
  // })

  console.log('2FA disabled for user:', userId);

  return true;
}

export async function verifyTOTP(
  userId: string,
  token: string
): Promise<boolean> {
  // Mock implementation until Prisma is working
  // const user = await prisma.user.findUnique({
  //   where: { id: userId },
  //   select: { twoFASecret: true, twoFAEnabled: true }
  // })

  // if (!user?.twoFAEnabled || !user.twoFASecret) {
  //   return false
  // }

  // return speakeasy.totp.verify({
  //   secret: user.twoFASecret,
  //   token,
  //   window: 2,
  // })

  // Mock verification
  console.log('TOTP verification for user:', userId, 'token:', token);
  return token === '123456'; // Mock successful verification
}

export async function generateBackupCodes(userId: string): Promise<string[]> {
  const codes: string[] = [];

  // Generate 8 backup codes
  for (let i = 0; i < 8; i++) {
    codes.push(Math.random().toString(36).substr(2, 8).toUpperCase());
  }

  // Mock implementation until Prisma is working
  // await prisma.user.update({
  //   where: { id: userId },
  //   data: {
  //     // You might want to add a backupCodes field to the User model
  //     // backupCodes: codes.map(code => hashCode(code))
  //   },
  // })

  console.log('Generated backup codes for user:', userId);

  return codes;
}
