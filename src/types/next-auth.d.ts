// import { UserRole } from '@prisma/client'
import { UserRole } from '@/types/auth';
import { DefaultSession, DefaultUser } from 'next-auth';
import { DefaultJWT } from 'next-auth/jwt';

// Temporary enum for development
// export enum UserRole {
//   ADMIN = 'ADMIN',
//   APPROVER = 'APPROVER',
//   EDITOR = 'EDITOR',
//   DATA_ENTRY = 'DATA_ENTRY',
//   VIEWER = 'VIEWER'
// }

declare module 'next-auth' {
  interface Session {
    user: {
      id: string;
      roles: UserRole[];
      locale: string;
      twoFAEnabled: boolean;
    } & DefaultSession['user'];
  }

  interface User extends DefaultUser {
    id: string;
    roles: UserRole[];
    locale: string;
    twoFAEnabled: boolean;
  }
}

declare module 'next-auth/jwt' {
  interface JWT extends DefaultJWT {
    id: string;
    roles: UserRole[];
    locale: string;
    twoFAEnabled: boolean;
  }
}
