self.addEventListener('install', (event) => {
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(clients.claim());
});

self.addEventListener('fetch', (event) => {
  event.respondWith(
    caches.open('static-v1').then(async (cache) => {
      try {
        const fresh = await fetch(event.request);
        cache.put(event.request, fresh.clone());
        return fresh;
      } catch (e) {
        const cached = await cache.match(event.request);
        if (cached) return cached;
        throw e;
      }
    })
  );
});
