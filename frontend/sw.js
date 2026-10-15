/*
 * CodeA4 service worker.
 *
 * Two jobs:
 *
 * 1. Cache the app shell so the interface loads instantly on a return visit
 *    and does not blank out on a flaky connection. Only the shell is cached —
 *    API responses never are, because a cached session or job status would
 *    show the user a stale view of work the server has since moved on from.
 *
 * 2. Receive push messages. Capture and render both take long enough that the
 *    user is told to go away and come back, so the notification is what brings
 *    them back at the right moment.
 */
const CACHE = 'codea4-shell-v1';
const SHELL = ['/', '/app.js', '/manifest.webmanifest', '/icons/icon-192.png'];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE).then((cache) => cache.addAll(SHELL)).then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);

  // Never cache the API, and never intercept uploads.
  if (url.pathname.startsWith('/api/') || event.request.method !== 'GET') return;
  if (url.origin !== self.location.origin) return;

  // Network-first, falling back to the cached shell. The other way round
  // would leave a user staring at a stale build after every deploy.
  event.respondWith(
    fetch(event.request)
      .then((response) => {
        const copy = response.clone();
        caches.open(CACHE).then((cache) => cache.put(event.request, copy));
        return response;
      })
      .catch(() => caches.match(event.request).then((hit) => hit || caches.match('/')))
  );
});

self.addEventListener('push', (event) => {
  let payload = { title: 'CodeA4', body: 'An update is ready.', url: '/' };
  try {
    if (event.data) payload = Object.assign(payload, event.data.json());
  } catch (e) { /* keep the default text rather than dropping the notification */ }

  event.waitUntil(
    self.registration.showNotification(payload.title, {
      body: payload.body,
      icon: '/icons/icon-192.png',
      badge: '/icons/icon-192.png',
      data: { url: payload.url }
    })
  );
});

self.addEventListener('notificationclick', (event) => {
  event.notification.close();
  const target = (event.notification.data && event.notification.data.url) || '/';
  // Focus an existing tab if there is one, rather than piling up duplicates.
  event.waitUntil(
    self.clients.matchAll({ type: 'window', includeUncontrolled: true }).then((clients) => {
      for (const client of clients) {
        if (client.url.includes(self.location.origin) && 'focus' in client) {
          client.navigate(target);
          return client.focus();
        }
      }
      return self.clients.openWindow(target);
    })
  );
});
