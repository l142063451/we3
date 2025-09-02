import { NextAuthOptions, getServerSession } from 'next-auth';
// import { PrismaAdapter } from '@next-auth/prisma-adapter'
import EmailProvider from 'next-auth/providers/email';
import GoogleProvider from 'next-auth/providers/google';
// import { prisma } from '@/lib/db'
import { UserRole } from '@/types/auth';
import { redirect } from 'next/navigation';

export const authOptions: NextAuthOptions = {
  // adapter: PrismaAdapter(prisma), // Commented out until Prisma is working
  session: {
    strategy: 'jwt',
    maxAge: 30 * 24 * 60 * 60, // 30 days
  },
  providers: [
    EmailProvider({
      server: {
        host: process.env.EMAIL_SERVER_HOST,
        port: Number(process.env.EMAIL_SERVER_PORT),
        auth: {
          user: process.env.EMAIL_SERVER_USER,
          pass: process.env.EMAIL_SERVER_PASSWORD,
        },
      },
      from: process.env.EMAIL_FROM,
      maxAge: 10 * 60, // 10 minutes
    }),
    ...(process.env.GOOGLE_CLIENT_ID && process.env.GOOGLE_CLIENT_SECRET
      ? [
          GoogleProvider({
            clientId: process.env.GOOGLE_CLIENT_ID,
            clientSecret: process.env.GOOGLE_CLIENT_SECRET,
          }),
        ]
      : []),
  ],
  callbacks: {
    async jwt({ token, user, trigger, session }) {
      if (trigger === 'update') {
        // Handle session updates
        if (session?.locale) token.locale = session.locale;
        if (session?.roles) token.roles = session.roles;
      }

      if (user) {
        // Mock implementation until Prisma is working
        // const dbUser = await prisma.user.findUnique({
        //   where: { id: user.id },
        //   select: {
        //     id: true,
        //     roles: true,
        //     locale: true,
        //     twoFAEnabled: true,
        //   },
        // })

        // if (dbUser) {
        //   token.id = dbUser.id
        //   token.roles = dbUser.roles
        //   token.locale = dbUser.locale
        //   token.twoFAEnabled = dbUser.twoFAEnabled
        // }

        // Mock values
        token.id = user.id;
        token.roles = [UserRole.VIEWER];
        token.locale = 'hi-IN';
        token.twoFAEnabled = false;
      }

      return token;
    },
    async session({ session, token }) {
      if (session?.user) {
        session.user.id = token.id;
        session.user.roles = token.roles;
        session.user.locale = token.locale;
        session.user.twoFAEnabled = token.twoFAEnabled;
      }
      return session;
    },
  },
  events: {
    async signIn({ user, isNewUser }) {
      // Mock implementation until Prisma is working
      // if (isNewUser) {
      //   // Set default role and locale for new users
      //   await prisma.user.update({
      //     where: { id: user.id },
      //     data: {
      //       roles: [UserRole.VIEWER],
      //       locale: process.env.NEXT_PUBLIC_DEFAULT_LOCALE || 'hi-IN',
      //     },
      //   })

      //   // Log user creation
      //   await prisma.auditLog.create({
      //     data: {
      //       actorId: user.id,
      //       action: 'signup',
      //       entity: 'User',
      //       entityId: user.id,
      //       diffJSON: { event: 'new_user_registered' },
      //     },
      //   })
      // }

      // // Log successful sign in
      // await prisma.auditLog.create({
      //   data: {
      //     actorId: user.id,
      //     action: 'signin',
      //     entity: 'User',
      //     entityId: user.id,
      //     diffJSON: { event: 'user_signin' },
      //   },
      // })

      console.log('User signed in:', user.id, isNewUser ? '(new user)' : '');
    },
  },
  pages: {
    signIn: '/auth/signin',
    verifyRequest: '/auth/verify-request',
    error: '/auth/error',
  },
};

export const getSession = () => getServerSession(authOptions);

export const requireAuth = async (redirectTo?: string) => {
  const session = await getSession();
  if (!session?.user) {
    redirect(redirectTo || '/auth/signin');
  }
  return session;
};

export const requireRole = async (
  roles: UserRole | UserRole[],
  redirectTo?: string
) => {
  const session = await requireAuth(redirectTo);
  const userRoles = session.user.roles;
  const requiredRoles = Array.isArray(roles) ? roles : [roles];

  if (!requiredRoles.some((role) => userRoles.includes(role))) {
    redirect(redirectTo || '/unauthorized');
  }

  return session;
};

export const requireAdmin = async (redirectTo?: string) => {
  return requireRole(UserRole.ADMIN, redirectTo);
};

export const hasRole = (
  userRoles: UserRole[],
  role: UserRole | UserRole[]
): boolean => {
  const requiredRoles = Array.isArray(role) ? role : [role];
  return requiredRoles.some((r) => userRoles.includes(r));
};

export const hasPermission = (
  userRoles: UserRole[],
  action: 'view' | 'create' | 'edit' | 'delete' | 'publish' | 'moderate',
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  entity?: string
): boolean => {
  // Admin can do everything
  if (userRoles.includes(UserRole.ADMIN)) return true;

  switch (action) {
    case 'view':
      return true; // Everyone can view public content

    case 'create':
    case 'edit':
      return userRoles.some((role) =>
        [
          UserRole.DATA_ENTRY,
          UserRole.EDITOR,
          UserRole.APPROVER,
          UserRole.ADMIN,
        ].includes(role)
      );

    case 'publish':
      return userRoles.some((role) =>
        [UserRole.APPROVER, UserRole.ADMIN].includes(role)
      );

    case 'delete':
    case 'moderate':
      return userRoles.some((role) =>
        [UserRole.EDITOR, UserRole.APPROVER, UserRole.ADMIN].includes(role)
      );

    default:
      return false;
  }
};
