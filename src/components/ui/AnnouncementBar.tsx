import React from 'react'
import { cn } from '@/lib/utils'

export interface AnnouncementBarProps {
  children: React.ReactNode
  variant?: 'info' | 'warning' | 'success' | 'error'
  dismissible?: boolean
  onDismiss?: () => void
  className?: string
}

export function AnnouncementBar({
  children,
  variant = 'info',
  dismissible = false,
  onDismiss,
  className
}: AnnouncementBarProps) {
  const variantStyles = {
    info: 'bg-info-50 border-info-200 text-info-800',
    warning: 'bg-warning-50 border-warning-200 text-warning-800',
    success: 'bg-success-50 border-success-200 text-success-800',
    error: 'bg-error-50 border-error-200 text-error-800',
  }

  const iconMap = {
    info: (
      <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
    ),
    warning: (
      <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.732-.833-2.464 0L3.34 16.5c-.77.833.192 2.5 1.732 2.5z" />
      </svg>
    ),
    success: (
      <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
    ),
    error: (
      <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
    ),
  }

  return (
    <div
      className={cn(
        'border-b px-4 py-3 transition-all duration-150',
        variantStyles[variant],
        className
      )}
      role="banner"
      aria-live="polite"
    >
      <div className="flex items-center justify-center max-w-7xl mx-auto">
        <div className="flex items-center space-x-2">
          <div className="flex-shrink-0">
            {iconMap[variant]}
          </div>
          <div className="flex-1 text-sm font-medium text-center">
            {children}
          </div>
          {dismissible && (
            <button
              onClick={onDismiss}
              className="flex-shrink-0 ml-4 rounded-md p-1 hover:bg-black/10 focus:outline-none focus:ring-2 focus:ring-current focus:ring-offset-2"
              aria-label="Dismiss announcement"
            >
              <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          )}
        </div>
      </div>
    </div>
  )
}