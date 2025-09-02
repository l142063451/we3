'use client'

import React from 'react'
import { useSession } from 'next-auth/react'
import { redirect } from 'next/navigation'
import { canAccessAdmin } from '@/lib/auth/rbac'
import { Loader2 } from 'lucide-react'

export default function AdminLayout({
  children,
}: {
  children: React.ReactNode
}) {
  const { data: session, status } = useSession()

  if (status === 'loading') {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin" />
        <span className="ml-2">लोड हो रहा है...</span>
      </div>
    )
  }

  if (status === 'unauthenticated') {
    redirect('/auth/signin')
  }

  if (!session?.user?.roles || !canAccessAdmin(session.user.roles)) {
    redirect('/unauthorized')
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="flex">
        {/* Sidebar will be implemented as a separate component */}
        <div className="w-64 bg-white shadow-sm">
          {/* AdminSidebar component */}
        </div>
        
        <div className="flex-1">
          {/* Header will be implemented as a separate component */}
          <div className="h-16 bg-white border-b border-gray-200">
            {/* AdminHeader component */}
          </div>
          
          <main className="p-6">
            {children}
          </main>
        </div>
      </div>
    </div>
  )
}