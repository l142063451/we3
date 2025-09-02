'use client'

import { useState, useEffect } from 'react'
import { Workbox } from 'workbox-window'

interface PWAContextValue {
  isOnline: boolean
  isPWAInstalled: boolean
  showInstallPrompt: boolean
  installPrompt: BeforeInstallPromptEvent | null
  pendingSync: number
  installPWA: () => void
  dismissInstallPrompt: () => void
  addToSyncQueue: (data: Record<string, unknown>) => void
  clearSyncQueue: () => void
}

export const usePWA = (): PWAContextValue => {
  const [isOnline, setIsOnline] = useState(true)
  const [isPWAInstalled, setIsPWAInstalled] = useState(false)
  const [showInstallPrompt, setShowInstallPrompt] = useState(false)
  const [installPrompt, setInstallPrompt] = useState<BeforeInstallPromptEvent | null>(null)
  const [pendingSync, setPendingSync] = useState(0)

  useEffect(() => {
    // Check online status
    const updateOnlineStatus = () => {
      setIsOnline(navigator.onLine)
    }

    window.addEventListener('online', updateOnlineStatus)
    window.addEventListener('offline', updateOnlineStatus)

    // Check if PWA is installed
    const checkPWAInstalled = () => {
      const isStandalone = window.matchMedia('(display-mode: standalone)').matches
      const isInAppBrowser = (window.navigator as { standalone?: boolean }).standalone === true
      setIsPWAInstalled(isStandalone || isInAppBrowser)
    }

    checkPWAInstalled()

    // Handle install prompt
    const handleBeforeInstallPrompt = (e: BeforeInstallPromptEvent) => {
      e.preventDefault()
      setInstallPrompt(e)
      setShowInstallPrompt(true)
    }

    window.addEventListener('beforeinstallprompt', handleBeforeInstallPrompt as EventListener)

    // Handle app installed
    const handleAppInstalled = () => {
      setIsPWAInstalled(true)
      setShowInstallPrompt(false)
      setInstallPrompt(null)
    }

    window.addEventListener('appinstalled', handleAppInstalled)

    // Initialize service worker
    if (typeof window !== 'undefined' && 'serviceWorker' in navigator) {
      const wb = new Workbox('/sw.js')
      
      wb.addEventListener('installed', (event) => {
        console.log('Service Worker installed:', event)
      })

      wb.addEventListener('controlling', (event) => {
        console.log('Service Worker controlling:', event)
      })

      wb.addEventListener('waiting', (event) => {
        console.log('Service Worker waiting:', event)
        // Show update available notification
      })

      wb.register()
    }

    // Load pending sync count from localStorage
    const loadPendingSync = () => {
      try {
        const queue = JSON.parse(localStorage.getItem('pwa-sync-queue') || '[]')
        setPendingSync(queue.length)
      } catch (error) {
        console.error('Error loading sync queue:', error)
        setPendingSync(0)
      }
    }

    loadPendingSync()

    return () => {
      window.removeEventListener('online', updateOnlineStatus)
      window.removeEventListener('offline', updateOnlineStatus)
      window.removeEventListener('beforeinstallprompt', handleBeforeInstallPrompt as EventListener)
      window.removeEventListener('appinstalled', handleAppInstalled)
    }
  }, [])

  const installPWA = async () => {
    if (installPrompt) {
      const result = await installPrompt.prompt()
      console.log('Install prompt result:', result)
      setInstallPrompt(null)
      setShowInstallPrompt(false)
    }
  }

  const dismissInstallPrompt = () => {
    setShowInstallPrompt(false)
    setInstallPrompt(null)
  }

  const addToSyncQueue = (data: Record<string, unknown>) => {
    try {
      const queue = JSON.parse(localStorage.getItem('pwa-sync-queue') || '[]')
      const syncItem = {
        id: Date.now().toString(),
        data,
        timestamp: Date.now(),
        retries: 0
      }
      queue.push(syncItem)
      localStorage.setItem('pwa-sync-queue', JSON.stringify(queue))
      setPendingSync(queue.length)
      
      // Try to sync immediately if online
      if (isOnline) {
        processSyncQueue()
      }
    } catch (error) {
      console.error('Error adding to sync queue:', error)
    }
  }

  const processSyncQueue = async () => {
    try {
      const queue = JSON.parse(localStorage.getItem('pwa-sync-queue') || '[]')
      if (queue.length === 0) return

      const successful: string[] = []
      
      for (const item of queue) {
        try {
          // Attempt to sync the data
          const response = await fetch('/api/sync', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(item.data)
          })

          if (response.ok) {
            successful.push(item.id)
          } else {
            // Increment retry count
            item.retries = (item.retries || 0) + 1
            if (item.retries >= 3) {
              // Remove after 3 failed attempts
              successful.push(item.id)
            }
          }
        } catch (error) {
          console.error('Sync error for item:', item.id, error)
          item.retries = (item.retries || 0) + 1
          if (item.retries >= 3) {
            successful.push(item.id)
          }
        }
      }

      // Remove successfully synced items
      if (successful.length > 0) {
        const updatedQueue = queue.filter((item: { id: string }) => !successful.includes(item.id))
        localStorage.setItem('pwa-sync-queue', JSON.stringify(updatedQueue))
        setPendingSync(updatedQueue.length)
      }
    } catch (error) {
      console.error('Error processing sync queue:', error)
    }
  }

  const clearSyncQueue = () => {
    localStorage.removeItem('pwa-sync-queue')
    setPendingSync(0)
  }

  // Periodically try to process sync queue when online
  useEffect(() => {
    if (isOnline && pendingSync > 0) {
      const interval = setInterval(processSyncQueue, 30000) // Every 30 seconds
      return () => clearInterval(interval)
    }
  }, [isOnline, pendingSync])

  return {
    isOnline,
    isPWAInstalled,
    showInstallPrompt,
    installPrompt,
    pendingSync,
    installPWA,
    dismissInstallPrompt,
    addToSyncQueue,
    clearSyncQueue
  }
}

declare global {
  interface Window {
    BeforeInstallPromptEvent: BeforeInstallPromptEvent
  }
  
  interface BeforeInstallPromptEvent extends Event {
    prompt(): Promise<{ outcome: 'accepted' | 'dismissed'; platform: string }>
    userChoice: Promise<{ outcome: 'accepted' | 'dismissed'; platform: string }>
  }
}