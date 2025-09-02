import { PrismaClient, UserRole } from '@prisma/client';

const prisma = new PrismaClient();

async function main() {
  console.log('🌱 Starting database seed...');

  // Create admin user
  const adminUser = await prisma.user.upsert({
    where: { email: 'admin@ummidsehari.in' },
    update: {},
    create: {
      email: 'admin@ummidsehari.in',
      name: 'System Administrator',
      locale: 'hi-IN',
      roles: [UserRole.ADMIN],
      emailVerified: new Date(),
    },
  });

  console.log('✅ Created admin user:', adminUser.email);

  // Create feature flags
  const featureFlags = [
    {
      key: 'enable_analytics',
      enabled: false,
      description: 'Enable analytics tracking',
    },
    {
      key: 'enable_pwa_install_prompt',
      enabled: true,
      description: 'Show PWA install prompt',
    },
    {
      key: 'enable_offline_mode',
      enabled: true,
      description: 'Enable offline functionality',
    },
  ];

  for (const flag of featureFlags) {
    await prisma.featureFlag.upsert({
      where: { key: flag.key },
      update: { enabled: flag.enabled, description: flag.description },
      create: flag,
    });
  }

  console.log('✅ Created feature flags');

  // Create basic translation keys
  const translationKeys = [
    {
      key: 'common.welcome',
      defaultText: 'स्वागत है | Welcome',
      module: 'common',
    },
    {
      key: 'nav.home',
      defaultText: 'मुख्य पृष्ठ | Home',
      module: 'navigation',
    },
    {
      key: 'auth.signin.title',
      defaultText: 'साइन इन करें | Sign In',
      module: 'auth',
    },
    {
      key: 'auth.signin.description',
      defaultText: 'अपने खाते में प्रवेश करें | Access your account',
      module: 'auth',
    },
  ];

  for (const key of translationKeys) {
    const translationKey = await prisma.translationKey.upsert({
      where: { key: key.key },
      update: { defaultText: key.defaultText, module: key.module },
      create: key,
    });

    // Create translations for Hindi and English
    await prisma.translationValue.upsert({
      where: { keyId_locale: { keyId: translationKey.id, locale: 'hi-IN' } },
      update: { text: key.defaultText.split(' | ')[0] },
      create: {
        keyId: translationKey.id,
        locale: 'hi-IN',
        text: key.defaultText.split(' | ')[0],
      },
    });

    await prisma.translationValue.upsert({
      where: { keyId_locale: { keyId: translationKey.id, locale: 'en-IN' } },
      update: { text: key.defaultText.split(' | ')[1] || key.defaultText },
      create: {
        keyId: translationKey.id,
        locale: 'en-IN',
        text: key.defaultText.split(' | ')[1] || key.defaultText,
      },
    });
  }

  console.log('✅ Created basic translation keys');

  // Log the seeding action
  await prisma.auditLog.create({
    data: {
      actorId: adminUser.id,
      action: 'seed',
      entity: 'Database',
      diffJSON: {
        event: 'database_seeded',
        timestamp: new Date().toISOString(),
      },
    },
  });

  console.log('🌱 Database seed completed successfully!');
}

main()
  .catch((e) => {
    console.error('❌ Error seeding database:', e);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });
