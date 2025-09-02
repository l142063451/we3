// import { UserRole } from '@prisma/client'
import { UserRole } from '@/types/auth';

export interface Permission {
  action: string;
  resource: string;
  condition?: (context: { createdBy?: string; userId?: string }) => boolean;
}

export const PERMISSIONS: Record<UserRole, Permission[]> = {
  [UserRole.VIEWER]: [
    { action: 'read', resource: 'public_content' },
    { action: 'read', resource: 'own_profile' },
    { action: 'read', resource: 'own_submissions' },
  ],

  [UserRole.DATA_ENTRY]: [
    // Inherit viewer permissions
    { action: 'read', resource: 'public_content' },
    { action: 'read', resource: 'own_profile' },
    { action: 'read', resource: 'own_submissions' },
    // Additional permissions
    { action: 'create', resource: 'content_draft' },
    {
      action: 'update',
      resource: 'content_draft',
      condition: (ctx) => ctx.createdBy === ctx.userId,
    },
    { action: 'create', resource: 'form_submission' },
    { action: 'update', resource: 'form_submission' },
    { action: 'read', resource: 'form_submission' },
    { action: 'create', resource: 'project_data' },
    { action: 'update', resource: 'project_data' },
  ],

  [UserRole.EDITOR]: [
    // Inherit data entry permissions
    { action: 'read', resource: 'public_content' },
    { action: 'read', resource: 'own_profile' },
    { action: 'read', resource: 'own_submissions' },
    { action: 'create', resource: 'content_draft' },
    { action: 'update', resource: 'content_draft' },
    { action: 'create', resource: 'form_submission' },
    { action: 'update', resource: 'form_submission' },
    { action: 'read', resource: 'form_submission' },
    { action: 'create', resource: 'project_data' },
    { action: 'update', resource: 'project_data' },
    // Additional permissions
    { action: 'delete', resource: 'content_draft' },
    { action: 'moderate', resource: 'user_submissions' },
    { action: 'moderate', resource: 'comments' },
    { action: 'moderate', resource: 'directory_entries' },
    { action: 'create', resource: 'forms' },
    { action: 'update', resource: 'forms' },
    { action: 'read', resource: 'analytics' },
  ],

  [UserRole.APPROVER]: [
    // Inherit editor permissions
    { action: 'read', resource: 'public_content' },
    { action: 'read', resource: 'own_profile' },
    { action: 'read', resource: 'own_submissions' },
    { action: 'create', resource: 'content_draft' },
    { action: 'update', resource: 'content_draft' },
    { action: 'delete', resource: 'content_draft' },
    { action: 'create', resource: 'form_submission' },
    { action: 'update', resource: 'form_submission' },
    { action: 'read', resource: 'form_submission' },
    { action: 'moderate', resource: 'user_submissions' },
    { action: 'moderate', resource: 'comments' },
    { action: 'moderate', resource: 'directory_entries' },
    { action: 'create', resource: 'forms' },
    { action: 'update', resource: 'forms' },
    { action: 'create', resource: 'project_data' },
    { action: 'update', resource: 'project_data' },
    { action: 'read', resource: 'analytics' },
    // Additional permissions
    { action: 'publish', resource: 'content' },
    { action: 'approve', resource: 'directory_entries' },
    { action: 'approve', resource: 'pledges' },
    { action: 'create', resource: 'notifications' },
    { action: 'send', resource: 'notifications' },
  ],

  [UserRole.ADMIN]: [
    // Full access to everything
    { action: '*', resource: '*' },
  ],
};

export function hasPermission(
  userRoles: UserRole[],
  action: string,
  resource: string,
  context?: { createdBy?: string; userId?: string }
): boolean {
  // Admin has all permissions
  if (userRoles.includes(UserRole.ADMIN)) {
    return true;
  }

  // Check each role's permissions
  for (const role of userRoles) {
    const permissions = PERMISSIONS[role] || [];

    for (const permission of permissions) {
      // Check for wildcard permissions (admin only)
      if (permission.action === '*' && permission.resource === '*') {
        return true;
      }

      // Check for specific permission match
      if (permission.action === action && permission.resource === resource) {
        // If there's a condition, evaluate it
        if (permission.condition && context) {
          return permission.condition(context);
        } else if (permission.condition) {
          return false; // No context provided for condition
        }
        return true;
      }

      // Check for wildcard action
      if (permission.action === '*' && permission.resource === resource) {
        return true;
      }

      // Check for wildcard resource
      if (permission.action === action && permission.resource === '*') {
        return true;
      }
    }
  }

  return false;
}

export function requirePermission(
  userRoles: UserRole[],
  action: string,
  resource: string,
  context?: { createdBy?: string; userId?: string }
): void {
  if (!hasPermission(userRoles, action, resource, context)) {
    throw new Error(`Insufficient permissions: ${action} on ${resource}`);
  }
}

export function canModerate(userRoles: UserRole[]): boolean {
  return hasPermission(userRoles, 'moderate', 'user_submissions');
}

export function canPublish(userRoles: UserRole[]): boolean {
  return hasPermission(userRoles, 'publish', 'content');
}

export function canManageUsers(userRoles: UserRole[]): boolean {
  return userRoles.includes(UserRole.ADMIN);
}

export function canAccessAdmin(userRoles: UserRole[]): boolean {
  return userRoles.some((role) =>
    [
      UserRole.ADMIN,
      UserRole.APPROVER,
      UserRole.EDITOR,
      UserRole.DATA_ENTRY,
    ].includes(role)
  );
}

export function getHighestRole(userRoles: UserRole[]): UserRole {
  const roleHierarchy = [
    UserRole.VIEWER,
    UserRole.DATA_ENTRY,
    UserRole.EDITOR,
    UserRole.APPROVER,
    UserRole.ADMIN,
  ];

  for (let i = roleHierarchy.length - 1; i >= 0; i--) {
    if (userRoles.includes(roleHierarchy[i])) {
      return roleHierarchy[i];
    }
  }

  return UserRole.VIEWER;
}
