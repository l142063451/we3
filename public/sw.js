// Basic service worker for PWA functionality
const CACHE_NAME = 'ummid-se-hari-v1';
const STATIC_CACHE = 'static-cache-v1';
const DYNAMIC_CACHE = 'dynamic-cache-v1';

// Files to cache initially
const STATIC_ASSETS = [
  '/',
  '/manifest.json',
  '/favicon.svg',
  '/icon-192x192.png',
  '/icon-512x512.png',
  '/offline',
];

// Install event
self.addEventListener('install', (event) => {
  console.log('Service Worker: Install');
  event.waitUntil(
    caches.open(STATIC_CACHE).then((cache) => {
      console.log('Service Worker: Caching static assets');
      return cache.addAll(STATIC_ASSETS);
    })
  );
});

// Activate event
self.addEventListener('activate', (event) => {
  console.log('Service Worker: Activate');
  event.waitUntil(
    caches.keys().then((keyList) => {
      return Promise.all(
        keyList.map((key) => {
          if (key !== STATIC_CACHE && key !== DYNAMIC_CACHE) {
            console.log('Service Worker: Removing old cache', key);
            return caches.delete(key);
          }
        })
      );
    })
  );
});

// Fetch event
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);

  // Handle different types of requests
  if (request.method === 'GET') {
    // Static assets - cache first
    if (
      STATIC_ASSETS.includes(url.pathname) ||
      url.pathname.startsWith('/_next/static') ||
      url.pathname.includes('.css') ||
      url.pathname.includes('.js') ||
      url.pathname.includes('.png') ||
      url.pathname.includes('.svg')
    ) {
      event.respondWith(cacheFirst(request));
    }
    // API requests - network first
    else if (url.pathname.startsWith('/api/')) {
      event.respondWith(networkFirst(request));
    }
    // Pages - stale while revalidate
    else {
      event.respondWith(staleWhileRevalidate(request));
    }
  }
  // POST requests - network only with background sync
  else if (request.method === 'POST') {
    event.respondWith(networkOnly(request));
  }
});

// Cache first strategy
async function cacheFirst(request) {
  try {
    const cacheResponse = await caches.match(request);
    if (cacheResponse) {
      return cacheResponse;
    }

    const networkResponse = await fetch(request);
    if (networkResponse.ok) {
      const cache = await caches.open(STATIC_CACHE);
      cache.put(request, networkResponse.clone());
    }
    return networkResponse;
  } catch (error) {
    console.log('Cache first failed:', error);
    return new Response('Offline', { status: 503 });
  }
}

// Network first strategy
async function networkFirst(request) {
  try {
    const networkResponse = await fetch(request);
    if (networkResponse.ok) {
      const cache = await caches.open(DYNAMIC_CACHE);
      cache.put(request, networkResponse.clone());
    }
    return networkResponse;
  } catch (error) {
    const cacheResponse = await caches.match(request);
    return cacheResponse || new Response('Offline', { status: 503 });
  }
}

// Stale while revalidate strategy
async function staleWhileRevalidate(request) {
  try {
    const cache = await caches.open(DYNAMIC_CACHE);
    const cacheResponse = await cache.match(request);

    const networkPromise = fetch(request).then((networkResponse) => {
      if (networkResponse.ok) {
        cache.put(request, networkResponse.clone());
      }
      return networkResponse;
    });

    return cacheResponse || (await networkPromise);
  } catch (error) {
    const cacheResponse = await caches.match(request);
    return cacheResponse || new Response('Offline', { status: 503 });
  }
}

// Network only strategy
async function networkOnly(request) {
  try {
    return await fetch(request);
  } catch (error) {
    // For failed POST requests, we could implement background sync here
    return new Response('Network error', { status: 503 });
  }
}

// Background sync for form submissions (basic implementation)
self.addEventListener('sync', (event) => {
  if (event.tag === 'background-sync-form') {
    console.log('Service Worker: Background sync triggered');
    event.waitUntil(syncFormSubmissions());
  }
});

async function syncFormSubmissions() {
  // This would handle queued form submissions when back online
  console.log('Syncing queued form submissions...');
}
