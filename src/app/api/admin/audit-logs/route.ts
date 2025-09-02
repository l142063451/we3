import { NextRequest, NextResponse } from 'next/server';
import { prisma } from '@/lib/db';
import { requireAdmin } from '@/lib/auth';
import { z } from 'zod';

// Validation schema for audit log query parameters
const auditLogQuerySchema = z.object({
  page: z.string().optional().transform((val) => val ? parseInt(val, 10) : 1),
  limit: z.string().optional().transform((val) => val ? parseInt(val, 10) : 20),
  search: z.string().optional(),
  entity: z.string().optional(),
  action: z.string().optional(),
  actorId: z.string().optional(),
  startDate: z.string().optional().transform((val) => val ? new Date(val) : undefined),
  endDate: z.string().optional().transform((val) => val ? new Date(val) : undefined),
});

export async function GET(request: NextRequest) {
  try {
    // Ensure user has admin access
    await requireAdmin();

    const { searchParams } = new URL(request.url);
    const params = Object.fromEntries(searchParams);
    
    const validatedParams = auditLogQuerySchema.parse(params);
    const { page, limit, search, entity, action, actorId, startDate, endDate } = validatedParams;

    // Build where clause for filtering
    const where: any = {};

    if (search) {
      where.OR = [
        { action: { contains: search, mode: 'insensitive' } },
        { entity: { contains: search, mode: 'insensitive' } },
        { actor: { name: { contains: search, mode: 'insensitive' } } },
        { actor: { email: { contains: search, mode: 'insensitive' } } },
      ];
    }

    if (entity) {
      where.entity = entity;
    }

    if (action) {
      where.action = action;
    }

    if (actorId) {
      where.actorId = actorId;
    }

    if (startDate || endDate) {
      where.createdAt = {};
      if (startDate) where.createdAt.gte = startDate;
      if (endDate) where.createdAt.lte = endDate;
    }

    // Calculate offset for pagination
    const skip = (page - 1) * limit;

    // Get audit logs with pagination
    const [auditLogs, totalCount] = await Promise.all([
      prisma.auditLog.findMany({
        where,
        include: {
          actor: {
            select: {
              id: true,
              name: true,
              email: true,
              roles: true,
            },
          },
        },
        orderBy: {
          createdAt: 'desc',
        },
        skip,
        take: limit,
      }),
      prisma.auditLog.count({ where }),
    ]);

    // Calculate pagination metadata
    const totalPages = Math.ceil(totalCount / limit);
    const hasNextPage = page < totalPages;
    const hasPrevPage = page > 1;

    return NextResponse.json({
      success: true,
      data: {
        auditLogs,
        pagination: {
          currentPage: page,
          totalPages,
          totalCount,
          limit,
          hasNextPage,
          hasPrevPage,
        },
      },
    });
  } catch (error) {
    console.error('Error fetching audit logs:', error);
    
    if (error instanceof z.ZodError) {
      return NextResponse.json(
        { success: false, error: 'Invalid query parameters', details: error.issues },
        { status: 400 }
      );
    }

    return NextResponse.json(
      { success: false, error: 'Failed to fetch audit logs' },
      { status: 500 }
    );
  }
}

// POST endpoint to manually create audit log entries (for testing/admin purposes)
const createAuditLogSchema = z.object({
  action: z.string().min(1),
  entity: z.string().min(1),
  entityId: z.string().optional(),
  diffJSON: z.record(z.string(), z.any()).optional(),
  ipAddress: z.string().optional(),
  userAgent: z.string().optional(),
});

export async function POST(request: NextRequest) {
  try {
    // Ensure user has admin access
    const session = await requireAdmin();
    
    const body = await request.json();
    const validatedData = createAuditLogSchema.parse(body);

    // Get client IP and user agent
    const ipAddress = request.headers.get('x-forwarded-for') || 
                     request.headers.get('x-real-ip') || 
                     'unknown';
    const userAgent = request.headers.get('user-agent') || 'unknown';

    const auditLog = await prisma.auditLog.create({
      data: {
        actorId: session.user.id,
        action: validatedData.action,
        entity: validatedData.entity,
        entityId: validatedData.entityId,
        diffJSON: validatedData.diffJSON,
        ipAddress: validatedData.ipAddress || ipAddress,
        userAgent: validatedData.userAgent || userAgent,
      },
      include: {
        actor: {
          select: {
            id: true,
            name: true,
            email: true,
            roles: true,
          },
        },
      },
    });

    return NextResponse.json({
      success: true,
      data: auditLog,
    });
  } catch (error) {
    console.error('Error creating audit log:', error);
    
    if (error instanceof z.ZodError) {
      return NextResponse.json(
        { success: false, error: 'Invalid request data', details: error.issues },
        { status: 400 }
      );
    }

    return NextResponse.json(
      { success: false, error: 'Failed to create audit log' },
      { status: 500 }
    );
  }
}

// Helper function to get unique entities for filtering
export async function OPTIONS() {
  try {
    await requireAdmin();

    const [entities, actions, actors] = await Promise.all([
      prisma.auditLog.findMany({
        select: { entity: true },
        distinct: ['entity'],
        orderBy: { entity: 'asc' },
      }),
      prisma.auditLog.findMany({
        select: { action: true },
        distinct: ['action'],
        orderBy: { action: 'asc' },
      }),
      prisma.auditLog.findMany({
        select: {
          actor: {
            select: { id: true, name: true, email: true },
          },
        },
        distinct: ['actorId'],
        where: {
          actorId: { not: null },
        },
      }),
    ]);

    return NextResponse.json({
      success: true,
      data: {
        entities: entities.map((e: any) => e.entity),
        actions: actions.map((a: any) => a.action),
        actors: actors.map((a: any) => a.actor).filter(Boolean),
      },
    });
  } catch (error) {
    console.error('Error fetching audit log metadata:', error);
    return NextResponse.json(
      { success: false, error: 'Failed to fetch metadata' },
      { status: 500 }
    );
  }
}