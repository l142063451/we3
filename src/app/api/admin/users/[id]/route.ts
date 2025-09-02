import { NextRequest, NextResponse } from 'next/server';
import { prisma } from '@/lib/db';
import { requireAdmin, requireRole } from '@/lib/auth';
import { UserRole } from '@/types/auth';
import { z } from 'zod';

// Validation schema for updating a specific user
const updateUserSchema = z.object({
  name: z.string().min(1).optional(),
  phone: z.string().optional(),
  roles: z.array(z.enum(['ADMIN', 'APPROVER', 'EDITOR', 'DATA_ENTRY', 'VIEWER'])).min(1).optional(),
  locale: z.string().optional(),
  emailVerified: z.boolean().optional(),
  twoFAEnabled: z.boolean().optional(),
});

// GET - Get specific user by ID
export async function GET(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    await requireRole([UserRole.ADMIN, UserRole.APPROVER]);

    const userId = params.id;

    const user = await prisma.user.findUnique({
      where: { id: userId },
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
        // Recent sessions
        sessions: {
          select: {
            expires: true,
            sessionToken: true,
          },
          orderBy: {
            expires: 'desc',
          },
          take: 5,
        },
        // Recent activity
        auditLogs: {
          select: {
            action: true,
            entity: true,
            createdAt: true,
            ipAddress: true,
            userAgent: true,
          },
          orderBy: {
            createdAt: 'desc',
          },
          take: 20,
        },
        // User's content
        pages: {
          select: {
            id: true,
            title: true,
            status: true,
            createdAt: true,
          },
          orderBy: {
            createdAt: 'desc',
          },
          take: 10,
        },
        // User's submissions
        submissions: {
          select: {
            id: true,
            status: true,
            createdAt: true,
            form: {
              select: {
                name: true,
              },
            },
          },
          orderBy: {
            createdAt: 'desc',
          },
          take: 10,
        },
        // User's pledges
        pledges: {
          select: {
            id: true,
            pledgeType: true,
            amount: true,
            approved: true,
            createdAt: true,
          },
          orderBy: {
            createdAt: 'desc',
          },
          take: 10,
        },
      },
    });

    if (!user) {
      return NextResponse.json(
        { success: false, error: 'User not found' },
        { status: 404 }
      );
    }

    // Calculate user statistics
    const lastSession = user.sessions[0];
    const lastActivity = user.auditLogs[0];
    
    const stats = {
      isActive: user.emailVerified !== null,
      isOnline: lastSession?.expires > new Date(),
      lastActive: lastSession?.expires > new Date() 
        ? new Date() 
        : lastActivity?.createdAt || user.updatedAt,
      totalPages: user.pages.length,
      totalSubmissions: user.submissions.length,
      totalPledges: user.pledges.length,
      recentLogins: user.sessions.filter((s: any) => s.expires > new Date()).length,
    };

    return NextResponse.json({
      success: true,
      data: {
        ...user,
        stats,
      },
    });
  } catch (error) {
    console.error('Error fetching user:', error);
    return NextResponse.json(
      { success: false, error: 'Failed to fetch user' },
      { status: 500 }
    );
  }
}

// PUT - Update specific user
export async function PUT(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const session = await requireAdmin();
    const userId = params.id;
    
    const body = await request.json();
    const validatedData = updateUserSchema.parse(body);

    // Get current user data for audit log
    const currentUser = await prisma.user.findUnique({
      where: { id: userId },
      select: {
        name: true,
        phone: true,
        roles: true,
        locale: true,
        emailVerified: true,
        twoFAEnabled: true,
      },
    });

    if (!currentUser) {
      return NextResponse.json(
        { success: false, error: 'User not found' },
        { status: 404 }
      );
    }

    // Prepare update data
    const updateData: any = {};
    
    if (validatedData.name !== undefined) {
      updateData.name = validatedData.name;
    }
    
    if (validatedData.phone !== undefined) {
      updateData.phone = validatedData.phone;
    }
    
    if (validatedData.roles !== undefined) {
      updateData.roles = validatedData.roles;
    }
    
    if (validatedData.locale !== undefined) {
      updateData.locale = validatedData.locale;
    }
    
    if (validatedData.emailVerified !== undefined) {
      updateData.emailVerified = validatedData.emailVerified ? new Date() : null;
    }
    
    if (validatedData.twoFAEnabled !== undefined) {
      updateData.twoFAEnabled = validatedData.twoFAEnabled;
      // If disabling 2FA, also remove the secret
      if (!validatedData.twoFAEnabled) {
        updateData.twoFASecret = null;
      }
    }

    // Update user
    const updatedUser = await prisma.user.update({
      where: { id: userId },
      data: updateData,
      select: {
        id: true,
        email: true,
        name: true,
        phone: true,
        roles: true,
        locale: true,
        twoFAEnabled: true,
        emailVerified: true,
        updatedAt: true,
      },
    });

    // Create audit log
    const changes: any = {};
    Object.keys(validatedData).forEach(key => {
      const oldValue = (currentUser as any)[key];
      const newValue = (validatedData as any)[key];
      if (oldValue !== newValue) {
        changes[key] = { from: oldValue, to: newValue };
      }
    });

    if (Object.keys(changes).length > 0) {
      await prisma.auditLog.create({
        data: {
          actorId: session.user.id,
          action: 'update_user',
          entity: 'User',
          entityId: userId,
          diffJSON: {
            changes,
            updatedBy: session.user.email,
          },
        },
      });
    }

    return NextResponse.json({
      success: true,
      data: updatedUser,
      message: 'User updated successfully',
    });
  } catch (error) {
    console.error('Error updating user:', error);
    
    if (error instanceof z.ZodError) {
      return NextResponse.json(
        { success: false, error: 'Invalid request data', details: error.issues },
        { status: 400 }
      );
    }

    return NextResponse.json(
      { success: false, error: 'Failed to update user' },
      { status: 500 }
    );
  }
}

// DELETE - Delete specific user
export async function DELETE(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const session = await requireAdmin();
    const userId = params.id;

    // Prevent admin from deleting themselves
    if (userId === session.user.id) {
      return NextResponse.json(
        { success: false, error: 'You cannot delete your own account' },
        { status: 400 }
      );
    }

    // Get user data for audit log
    const user = await prisma.user.findUnique({
      where: { id: userId },
      select: {
        email: true,
        name: true,
        roles: true,
      },
    });

    if (!user) {
      return NextResponse.json(
        { success: false, error: 'User not found' },
        { status: 404 }
      );
    }

    // Delete user (CASCADE will handle related records)
    await prisma.user.delete({
      where: { id: userId },
    });

    // Create audit log
    await prisma.auditLog.create({
      data: {
        actorId: session.user.id,
        action: 'delete_user',
        entity: 'User',
        entityId: userId,
        diffJSON: {
          deleted: {
            email: user.email,
            name: user.name,
            roles: user.roles,
          },
          deletedBy: session.user.email,
        },
      },
    });

    return NextResponse.json({
      success: true,
      message: 'User deleted successfully',
    });
  } catch (error) {
    console.error('Error deleting user:', error);
    return NextResponse.json(
      { success: false, error: 'Failed to delete user' },
      { status: 500 }
    );
  }
}