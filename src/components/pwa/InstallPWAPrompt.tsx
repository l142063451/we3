'use client'

import { useState, useEffect } from 'react'
import { usePWA } from '@/hooks/pwa/usePWA'

export default function InstallPWAPrompt() {
  const { showInstallPrompt, installPWA, dismissInstallPrompt } = usePWA()
  const [isVisible, setIsVisible] = useState(false)

  useEffect(() => {
    // Show prompt after a delay if conditions are met
    if (showInstallPrompt) {
      const timer = setTimeout(() => {
        setIsVisible(true)
      }, 5000) // Show after 5 seconds
      
      return () => clearTimeout(timer)
    }
  }, [showInstallPrompt])

  const handleInstall = () => {
    installPWA()
    setIsVisible(false)
  }

  const handleDismiss = () => {
    dismissInstallPrompt()
    setIsVisible(false)
  }

  if (!isVisible || !showInstallPrompt) {
    return null
  }

  return (
    <div className="fixed bottom-4 left-4 right-4 z-50 mx-auto max-w-sm">
      <div className="bg-white border border-gray-200 rounded-lg shadow-lg p-4">
        <div className="flex items-start space-x-3">
          <div className="flex-shrink-0">
            <div className="w-10 h-10 bg-primary-100 rounded-lg flex items-center justify-center">
              <svg className="w-5 h-5 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 18h.01M8 21h8a2 2 0 002-2V5a2 2 0 00-2-2H8a2 2 0 00-2 2v14a2 2 0 002 2z" />
              </svg>
            </div>
          </div>
          
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium text-gray-900">
              Install उम्मीद से हरी
            </p>
            <p className="text-sm text-gray-500 mt-1">
              Add to your home screen for quick access and offline use.
            </p>
            
            <div className="flex space-x-2 mt-3">
              <button
                onClick={handleInstall}
                className="flex-1 bg-primary-600 text-white text-xs font-medium py-2 px-3 rounded-md hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-1"
              >
                Install
              </button>
              <button
                onClick={handleDismiss}
                className="flex-1 bg-gray-100 text-gray-700 text-xs font-medium py-2 px-3 rounded-md hover:bg-gray-200 focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-1"
              >
                Not Now
              </button>
            </div>
          </div>
          
          <button
            onClick={handleDismiss}
            className="flex-shrink-0 text-gray-400 hover:text-gray-600"
            aria-label="Dismiss"
          >
            <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
            </svg>
          </button>
        </div>
      </div>
    </div>
  )
}