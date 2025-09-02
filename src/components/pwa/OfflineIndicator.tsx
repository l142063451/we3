'use client'

import { usePWA } from '@/hooks/pwa/usePWA'

export default function OfflineIndicator() {
  const { isOnline, pendingSync } = usePWA()

  if (isOnline && pendingSync === 0) {
    return null
  }

  return (
    <div className="fixed top-16 left-1/2 transform -translate-x-1/2 z-40">
      <div className={`px-4 py-2 rounded-full text-sm font-medium shadow-lg transition-all duration-300 ${
        !isOnline 
          ? 'bg-red-100 text-red-800 border border-red-200' 
          : 'bg-yellow-100 text-yellow-800 border border-yellow-200'
      }`}>
        <div className="flex items-center space-x-2">
          <div className="flex items-center">
            {!isOnline ? (
              <>
                <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M18.364 5.636l-12.728 12.728m0-12.728l12.728 12.728" />
                </svg>
                <span>Offline Mode</span>
              </>
            ) : (
              <>
                <div className="animate-spin w-4 h-4 mr-2 border-2 border-yellow-800 border-t-transparent rounded-full"></div>
                <span>Syncing {pendingSync} item{pendingSync !== 1 ? 's' : ''}...</span>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}