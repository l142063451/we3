import { NextRequest, NextResponse } from 'next/server';
import { prisma } from '@/lib/db';
import { requireAdmin, requireRole } from '@/lib/auth';
import { UserRole } from '@/types/auth';
import { z } from 'zod';

// Validation schemas
const userQuerySchema = z.object({
  page: z.string().optional().transform((val) => val ? parseInt(val, 10) : 1),
  limit: z.string().optional().transform((val) => val ? parseInt(val, 10) : 20),
  search: z.string().optional(),
  role: z.enum(['ADMIN', 'APPROVER', 'EDITOR', 'DATA_ENTRY', 'VIEWER']).optional(),
  sortBy: z.enum(['name', 'email', 'createdAt', 'lastActive']).optional().default('createdAt'),
  sortOrder: z.enum(['asc', 'desc']).optional().default('desc'),
  status: z.enum(['active', 'inactive', 'all']).optional().default('all'),
});

const createUserSchema = z.object({
  email: z.string().email(),
  name: z.string().min(1),
  phone: z.string().optional(),
  roles: z.array(z.enum(['ADMIN', 'APPROVER', 'EDITOR', 'DATA_ENTRY', 'VIEWER'])).min(1),
  locale: z.string().optional().default('hi-IN'),
  sendWelcomeEmail: z.boolean().optional().default(true),
});

const updateUserSchema = z.object({
  name: z.string().min(1).optional(),
  phone: z.string().optional(),
  roles: z.array(z.enum(['ADMIN', 'APPROVER', 'EDITOR', 'DATA_ENTRY', 'VIEWER'])).min(1).optional(),
  locale: z.string().optional(),
  emailVerified: z.boolean().optional(),
  twoFAEnabled: z.boolean().optional(),
});

// GET - List users with pagination and filtering
export async function GET(request: NextRequest) {
  try {
    // Require admin or approver role
    await requireRole([UserRole.ADMIN, UserRole.APPROVER]);

    const { searchParams } = new URL(request.url);
    const params = Object.fromEntries(searchParams);
    
    const validatedParams = userQuerySchema.parse(params);
    const { page, limit, search, role, sortBy, sortOrder, status } = validatedParams;

    // Build where clause
    const where: any = {};

    if (search) {
      where.OR = [
        { name: { contains: search, mode: 'insensitive' } },
        { email: { contains: search, mode: 'insensitive' } },
        { phone: { contains: search, mode: 'insensitive' } },
      ];
    }

    if (role) {
      where.roles = {
        has: role,
      };
    }

    if (status === 'active') {
      where.emailVerified = { not: null };
    } else if (status === 'inactive') {
      where.emailVerified = null;
    }

    // Calculate offset
    const skip = (page - 1) * limit;

    // Build orderBy clause
    const orderBy: any = {};
    orderBy[sortBy] = sortOrder;

    const [users, totalCount] = await Promise.all([
      prisma.user.findMany({
        where,
        select: {
          id: true,
          email: true,
          name: true,
          phone: true,
          roles: true,
          locale: true,
          twoFAEnabled: true,
          emailVerified: true,
          image: true,
          createdAt: true,
          updatedAt: true,
          // Add last activity info
          sessions: {
            select: {
              expires: true,
            },
            orderBy: {
              expires: 'desc',
            },
            take: 1,
          },
          // Count of audit logs for activity
          auditLogs: {
            select: {
              createdAt: true,
            },
            orderBy: {
              createdAt: 'desc',
            },
            take: 1,
          },
        },
        orderBy,
        skip,
        take: limit,
      }),
      prisma.user.count({ where }),
    ]);

    // Enhance user data with computed fields
    const enhancedUsers = users.map((user: any) => {
      const lastSession = user.sessions[0];
      const lastAuditLog = user.auditLogs[0];
      
      const lastActive = lastSession?.expires > new Date() 
        ? new Date() 
        : lastAuditLog?.createdAt || user.updatedAt;

      return {
        ...user,
        lastActive,
        isActive: user.emailVerified !== null,
        isOnline: lastSession?.expires > new Date(),
        // Remove sessions and auditLogs from response
        sessions: undefined,
        auditLogs: undefined,
      };
    });

    // Calculate pagination metadata
    const totalPages = Math.ceil(totalCount / limit);
    const hasNextPage = page < totalPages;
    const hasPrevPage = page > 1;

    return NextResponse.json({
      success: true,
      data: {
        users: enhancedUsers,
        pagination: {
          currentPage: page,
          totalPages,
          totalCount,
          limit,
          hasNextPage,
          hasPrevPage,
        },
        stats: {
          total: totalCount,
          active: enhancedUsers.filter((u: any) => u.isActive).length,
          online: enhancedUsers.filter((u: any) => u.isOnline).length,
          with2FA: enhancedUsers.filter((u: any) => u.twoFAEnabled).length,
        },
      },
    });
  } catch (error) {
    console.error('Error fetching users:', error);
    
    if (error instanceof z.ZodError) {
      return NextResponse.json(
        { success: false, error: 'Invalid query parameters', details: error.issues },
        { status: 400 }
      );
    }

    return NextResponse.json(
      { success: false, error: 'Failed to fetch users' },
      { status: 500 }
    );
  }
}

// POST - Create new user
export async function POST(request: NextRequest) {
  try {
    const session = await requireAdmin();
    const body = await request.json();
    const validatedData = createUserSchema.parse(body);

    // Check if user already exists
    const existingUser = await prisma.user.findUnique({
      where: { email: validatedData.email },
    });

    if (existingUser) {
      return NextResponse.json(
        { success: false, error: 'User with this email already exists' },
        { status: 400 }
      );
    }

    // Create the user
    const newUser = await prisma.user.create({
      data: {
        email: validatedData.email,
        name: validatedData.name,
        phone: validatedData.phone,
        roles: validatedData.roles,
        locale: validatedData.locale,
        // New users need to verify their email
        emailVerified: null,
      },
      select: {
        id: true,
        email: true,
        name: true,
        phone: true,
        roles: true,
        locale: true,
        twoFAEnabled: true,
        emailVerified: true,
        createdAt: true,
        updatedAt: true,
      },
    });

    // Log user creation
    await prisma.auditLog.create({
      data: {
        actorId: session.user.id,
        action: 'create_user',
        entity: 'User',
        entityId: newUser.id,
        diffJSON: {
          created: {
            email: newUser.email,
            name: newUser.name,
            roles: newUser.roles,
          },
        },
      },
    });

    // TODO: Send welcome email if requested
    if (validatedData.sendWelcomeEmail) {
      console.log('TODO: Send welcome email to:', newUser.email);
    }

    return NextResponse.json({
      success: true,
      data: newUser,
      message: 'User created successfully',
    });
  } catch (error) {
    console.error('Error creating user:', error);
    
    if (error instanceof z.ZodError) {
      return NextResponse.json(
        { success: false, error: 'Invalid request data', details: error.issues },
        { status: 400 }
      );
    }

    return NextResponse.json(
      { success: false, error: 'Failed to create user' },
      { status: 500 }
    );
  }
}

// PUT - Bulk update users (for batch operations)
const bulkUpdateSchema = z.object({
  userIds: z.array(z.string()).min(1),
  operation: z.enum(['activate', 'deactivate', 'delete', 'updateRoles']),
  data: z.record(z.string(), z.any()).optional(),
});

export async function PUT(request: NextRequest) {
  try {
    const session = await requireAdmin();
    const body = await request.json();
    const validatedData = bulkUpdateSchema.parse(body);

    const { userIds, operation, data } = validatedData;

    let result;
    const auditAction = operation;

    switch (operation) {
      case 'activate':
        result = await prisma.user.updateMany({
          where: { id: { in: userIds } },
          data: { emailVerified: new Date() },
        });
        break;

      case 'deactivate':
        result = await prisma.user.updateMany({
          where: { id: { in: userIds } },
          data: { emailVerified: null },
        });
        break;

      case 'updateRoles':
        if (!data?.roles || !Array.isArray(data.roles)) {
          throw new Error('Roles data is required for updateRoles operation');
        }
        result = await prisma.user.updateMany({
          where: { id: { in: userIds } },
          data: { roles: data.roles },
        });
        break;

      case 'delete':
        result = await prisma.user.deleteMany({
          where: { id: { in: userIds } },
        });
        break;

      default:
        throw new Error('Invalid operation');
    }

    // Log bulk operation
    await prisma.auditLog.create({
      data: {
        actorId: session.user.id,
        action: `bulk_${auditAction}`,
        entity: 'User',
        diffJSON: {
          operation,
          userIds,
          data,
          affected: result.count,
        },
      },
    });

    return NextResponse.json({
      success: true,
      data: {
        operation,
        affectedCount: result.count,
      },
      message: `${operation} operation completed for ${result.count} users`,
    });
  } catch (error) {
    console.error('Error performing bulk operation:', error);
    
    if (error instanceof z.ZodError) {
      return NextResponse.json(
        { success: false, error: 'Invalid request data', details: error.issues },
        { status: 400 }
      );
    }

    return NextResponse.json(
      { success: false, error: 'Failed to perform bulk operation' },
      { status: 500 }
    );
  }
}