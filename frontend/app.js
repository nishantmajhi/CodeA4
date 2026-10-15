/*
 * CodeA4 front-end.
 *
 * Talks to the real API. Every long-running stage is a job the server owns:
 * the UI starts it, then polls `GET /api/jobs/<id>` until it finishes. The
 * page never blocks on a request, because capture and render both take long
 * enough that the user is expected to leave and come back — which is also why
 * the session id is persisted and why push notifications exist.
 */
(function () {
  'use strict';

  var API = '/api';
  var SESSION_KEY = 'codea4.session';
  var POLL_MS = 1200;

  var state = {
    sessionId: null,
    session: null,
    style: null,
    files: [],
    renderJobId: null,
    pages: [],
    currentPage: 0,
    activeRound: 1,
    polling: null
  };

  // ---------------------------------------------------------------- helpers

  function $(id) { return document.getElementById(id); }

  function setStatus(text) {
    var el = $('statusLine');
    if (el) el.textContent = 'status — ' + text;
  }

  function show(el, visible) {
    if (el) el.hidden = !visible;
  }

  /*
   * All API calls go through here so that a failure has one shape. The server
   * puts a human-readable sentence in `error`; that sentence is what the user
   * sees, because it is the one thing that can tell them what to do
   * differently (retake the photo in better light, and so on).
   */
  function api(path, options) {
    return fetch(API + path, options).then(function (res) {
      var isJson = (res.headers.get('content-type') || '').indexOf('json') >= 0;
      return (isJson ? res.json() : res.text()).then(function (body) {
        if (!res.ok) {
          var message = (body && body.error) || 'Request failed (' + res.status + ')';
          var err = new Error(message);
          err.status = res.status;
          throw err;
        }
        return body;
      });
    });
  }

  function escapeHtml(text) {
    return String(text).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }

  function formatBytes(n) {
    if (n < 1024) return n + ' B';
    if (n < 1024 * 1024) return (n / 1024).toFixed(1) + ' KB';
    return (n / (1024 * 1024)).toFixed(1) + ' MB';
  }

  /*
   * Poll a job to completion. Returns a promise resolving with the finished
   * job record, or rejecting with the server's error message.
   */
  function pollJob(jobId, onProgress) {
    return new Promise(function (resolve, reject) {
      function tick() {
        api('/jobs/' + jobId).then(function (job) {
          if (onProgress) onProgress(job);
          if (job.status === 'done') return resolve(job);
          if (job.status === 'failed') return reject(new Error(job.error || 'Job failed'));
          state.polling = setTimeout(tick, POLL_MS);
        }).catch(reject);
      }
      tick();
    });
  }

  // ---------------------------------------------------------------- stepper

  var STEPS = ['capture', 'style', 'code', 'preview', 'export'];

  function goTo(step) {
    STEPS.forEach(function (name) {
      show($('panel-' + name), name === step);
    });
    document.querySelectorAll('.stepper li').forEach(function (li) {
      li.classList.toggle('current', li.dataset.step === step);
    });
    document.querySelectorAll('.step-btn').forEach(function (btn) {
      if (btn.dataset.step === step) btn.setAttribute('aria-current', 'step');
      else btn.removeAttribute('aria-current');
    });
    var heading = document.querySelector('#panel-' + step + ' h2');
    if (heading) heading.focus();
    if (location.hash.slice(1) !== step) history.replaceState(null, '', '#' + step);
  }

  function unlockStep(step, unlocked) {
    document.querySelectorAll('.step-btn').forEach(function (btn) {
      if (btn.dataset.step === step) btn.disabled = !unlocked;
    });
  }

  // ---------------------------------------------------------------- session

  function startSession() {
    var existing = null;
    try { existing = localStorage.getItem(SESSION_KEY); } catch (e) { /* private mode */ }

    if (existing) {
      // Resume rather than start fresh: capture is spread over three sheets
      // and the user is explicitly invited to go away between them, so
      // reloading the tab must not throw away the sheets already accepted.
      return api('/sessions/' + existing)
        .then(function (session) {
          state.sessionId = session.session_id;
          state.session = session;
          return session;
        })
        .catch(function () { return createSession(); });
    }
    return createSession();
  }

  function createSession() {
    return api('/sessions', { method: 'POST' }).then(function (session) {
      state.sessionId = session.session_id;
      state.session = session;
      try { localStorage.setItem(SESSION_KEY, session.session_id); } catch (e) { /* ignore */ }
      return session;
    });
  }

  function loadExercise() {
    return api('/exercise').then(function (data) {
      $('exerciseText').textContent = data.text;
    }).catch(function () {
      $('exerciseText').textContent =
        'The exercise text could not be loaded. Please refresh the page.';
    });
  }

  // ---------------------------------------------------------------- capture

  var ROUND_LABELS = {
    waiting: 'Waiting',
    ready: 'Ready to capture',
    processing: 'Processing…',
    done: 'Captured',
    failed: 'Needs retaking'
  };

  function renderRounds() {
    var rounds = (state.session && state.session.rounds) || [];
    var next = rounds.find(function (r) { return r.status !== 'done'; });
    state.activeRound = next ? next.n : rounds.length;

    $('roundsTable').innerHTML = rounds.map(function (r) {
      var status = r.status;
      if (status === 'waiting' && r.n === state.activeRound) status = 'ready';
      var dot = status === 'done' ? 'done' : (r.n === state.activeRound ? 'active' : '');
      return '<tr><td>' + r.n + '</td><td class="status-cell">' +
        '<span class="dot ' + dot + '"></span>' +
        escapeHtml(ROUND_LABELS[status] || status) + '</td></tr>';
    }).join('');

    var allDone = rounds.length > 0 && rounds.every(function (r) { return r.status === 'done'; });
    show($('captureContinueBtn'), allDone && !!state.session.style_ready);
    $('captureBtn').textContent = allDone ? 'Retake a sheet' : 'Use camera — sheet ' + state.activeRound;
    show($('captureBtn').parentElement, !allDone || !state.session.style_ready);

    if (allDone && !state.session.style_ready) {
      $('captureHint').textContent =
        'All three sheets are in. Building your handwriting model…';
    } else if (allDone) {
      $('captureHint').textContent = 'Your handwriting model is ready.';
    } else {
      $('captureHint').textContent =
        'Sheet ' + state.activeRound + ' of 3. Take a short break between sheets — ' +
        'each is processed while you are away.';
    }
  }

  function uploadPhoto(file) {
    var round = state.activeRound;
    var form = new FormData();
    form.append('photo', file);

    show($('captureError'), false);
    show($('captureProgress'), true);
    $('captureProgress').removeAttribute('value');
    setStatus('processing sheet ' + round);
    $('vfLabel').textContent = 'Processing sheet ' + round + '…';

    api('/sessions/' + state.sessionId + '/rounds/' + round + '/photo', {
      method: 'POST', body: form
    }).then(function (res) {
      return pollJob(res.job_id, function (job) {
        if (typeof job.progress === 'number') {
          $('captureProgress').value = job.progress;
          $('captureProgress').max = 100;
        }
        if (job.stage) $('vfLabel').textContent = job.stage + '…';
      });
    }).then(function () {
      show($('captureProgress'), false);
      $('vfLabel').textContent = 'Sheet ' + round + ' accepted.';
      setStatus('sheet ' + round + ' accepted');
      return refreshSessionUntilStyle();
    }).catch(function (err) {
      show($('captureProgress'), false);
      $('captureErrorText').textContent = err.message;
      show($('captureError'), true);
      $('vfLabel').textContent = 'Sheet ' + round + ' could not be read.';
      setStatus('sheet ' + round + ' failed');
      return refreshSession();
    });
  }

  function refreshSession() {
    return api('/sessions/' + state.sessionId).then(function (session) {
      state.session = session;
      renderRounds();
      return session;
    });
  }

  /*
   * After the third sheet the server starts clustering on its own, so the UI
   * keeps refreshing until the style assets exist. Polling the session (rather
   * than the clustering job) means a user who reloads the page mid-way still
   * converges on the right state without needing to know a job id.
   */
  function refreshSessionUntilStyle() {
    return refreshSession().then(function (session) {
      var allDone = session.rounds.every(function (r) { return r.status === 'done'; });
      if (!allDone) return session;
      if (session.style_ready) {
        unlockStep('style', true);
        setStatus('handwriting model ready');
        show($('captureContinueBtn'), true);
        return session;
      }
      setStatus('building your handwriting model');
      return new Promise(function (resolve) {
        setTimeout(function () { resolve(refreshSessionUntilStyle()); }, POLL_MS);
      });
    });
  }

  // ------------------------------------------------------------------ style

  function loadStyle() {
    return api('/sessions/' + state.sessionId + '/style').then(function (data) {
      if (!data.style_ready) return null;
      state.style = data;
      renderStyleCards(data.style_sets);
      renderDrift(data.drift);
      unlockStep('code', true);
      return data;
    });
  }

  function renderStyleCards(styleSets) {
    var base = API + '/sessions/' + state.sessionId + '/style/sample.svg';
    $('styleCards').innerHTML = styleSets.map(function (set) {
      var share = Math.round((set.share || 0) * 100);
      var src = base + '?style=' + set.id + '&text=' + encodeURIComponent('handwriting 0123');
      return '' +
        '<div class="style-card">' +
        '  <div class="style-meta">' +
        '    <div class="style-name">Style ' + set.id + '</div>' +
        '    <div class="style-share">' + share + '% of samples</div>' +
        '  </div>' +
        '  <div class="style-instances">' +
        '    <img src="' + escapeHtml(src) + '" alt="A sample of style ' + set.id +
        '         written in your own hand" style="max-width:100%; height:auto;" />' +
        '  </div>' +
        '</div>';
    }).join('');
  }

  /*
   * Plot the measured residual series. The chart is drawn from the numbers the
   * fit was made from, not from an idealised curve, so what the user sees is
   * their own wander.
   */
  function renderDrift(drift) {
    var svg = $('driftChart');
    var W = 460, H = 120, MID = H / 2;
    var parts = ['<line x1="15" y1="' + MID + '" x2="' + (W - 15) + '" y2="' + MID +
                 '" stroke="#948e83" stroke-dasharray="2 3" />'];

    function polyline(points, dashed) {
      if (!points || points.length < 2) return '';
      var xs = points.map(function (p) { return p[0]; });
      var ys = points.map(function (p) { return p[1]; });
      var xMin = Math.min.apply(null, xs), xMax = Math.max.apply(null, xs);
      var spread = Math.max.apply(null, ys.map(Math.abs)) || 1;
      var d = points.map(function (p, i) {
        var x = 15 + ((p[0] - xMin) / (xMax - xMin || 1)) * (W - 30);
        var y = MID + (p[1] / spread) * (MID - 22);
        return (i === 0 ? 'M' : 'L') + x.toFixed(1) + ',' + y.toFixed(1);
      }).join(' ');
      return '<path d="' + d + '" fill="none" stroke="#17140f" stroke-width="1.6"' +
        (dashed ? ' stroke-dasharray="5 4"' : '') + ' />';
    }

    parts.push(polyline(drift.baseline && drift.baseline.points, false));
    parts.push(polyline(drift.margin && drift.margin.points, true));

    // Keep the accessible title/desc, replace only the plotted content.
    var keep = svg.querySelector('title').outerHTML + svg.querySelector('desc').outerHTML;
    svg.innerHTML = keep + parts.join('');

    function mm(value) { return (value * (state.style.meta.x_height_mm / 0.5)).toFixed(1); }
    $('driftSummary').textContent =
      'Solid line: baseline drift, vertical, within a line — typically ±' +
      mm(drift.baseline.std) + ' mm. Dashed line: margin drift, horizontal, down the page — ' +
      'typically ±' + mm(drift.margin.std) + ' mm. Both wander gradually and correct back, ' +
      'rather than snapping to the line.';
  }

  // ------------------------------------------------------------------- code

  function renderFileTable() {
    var body = $('fileTableBody');
    body.innerHTML = state.files.map(function (f, i) {
      return '<tr><td>' + escapeHtml(f.name) + '</td><td>' + formatBytes(f.size) +
        '</td><td><button class="btn btn-secondary" data-remove="' + i + '">Remove</button></td></tr>';
    }).join('');
    show($('fileTable'), state.files.length > 0);
    $('toPreviewBtn').disabled = state.files.length === 0;
    $('fileHint').textContent = state.files.length
      ? state.files.length + ' file(s) ready to render.'
      : 'Add at least one file to continue.';
  }

  function startRender() {
    var form = new FormData();
    state.files.forEach(function (f) { form.append('files', f, f.name); });

    show($('renderError'), false);
    show($('renderProgress'), true);
    $('renderProgress').removeAttribute('value');
    $('toPreviewBtn').disabled = true;
    setStatus('rendering pages');

    api('/sessions/' + state.sessionId + '/render', { method: 'POST', body: form })
      .then(function (res) {
        state.renderJobId = res.job_id;
        return pollJob(res.job_id, function (job) {
          if (typeof job.progress === 'number') {
            $('renderProgress').value = job.progress;
            $('renderProgress').max = 100;
          }
          if (job.stage) setStatus(job.stage.toLowerCase());
        });
      })
      .then(function (job) {
        show($('renderProgress'), false);
        $('toPreviewBtn').disabled = false;
        state.pages = job.result.pages || [];
        state.coverage = job.result.coverage;
        state.pdfAvailable = job.result.pdf_available;
        state.pdfError = job.result.pdf_error;
        renderPreview();
        unlockStep('preview', true);
        unlockStep('export', true);
        setStatus(state.pages.length + ' page(s) rendered');
        goTo('preview');
      })
      .catch(function (err) {
        show($('renderProgress'), false);
        $('toPreviewBtn').disabled = false;
        $('renderErrorText').textContent = err.message;
        show($('renderError'), true);
        setStatus('render failed');
      });
  }

  // ---------------------------------------------------------------- preview

  function renderPreview() {
    $('pageList').innerHTML = state.pages.map(function (p, i) {
      return '<button class="page-thumb' + (i === 0 ? ' current' : '') +
        '" data-page="' + i + '"' + (i === 0 ? ' aria-current="true"' : '') + '>' +
        '<span class="thumb-face"></span>' +
        '<span class="thumb-cap">' + escapeHtml(p.file) + '<br />page ' + p.page_num + '</span>' +
        '</button>';
    }).join('');
    showPage(0);

    if (state.coverage && state.coverage.missing && state.coverage.missing.length) {
      $('previewHint').textContent =
        'Note: these characters were not in your exercise sheets and have been substituted: ' +
        state.coverage.missing.join(' ');
    }

    $('sumFiles').textContent = new Set(state.pages.map(function (p) { return p.file; })).size;
    $('sumPages').textContent = state.pages.length;
    $('sumStyles').textContent = state.style ? state.style.style_sets.length : '—';
  }

  /*
   * Pages are fetched one at a time as the user selects them. A page of
   * handwriting is a few hundred KB of vector outlines, so loading a whole
   * render up front would stall the preview on a phone connection for pages
   * the user may never look at.
   */
  function showPage(index) {
    state.currentPage = index;
    document.querySelectorAll('.page-thumb').forEach(function (btn) {
      var isCurrent = Number(btn.dataset.page) === index;
      btn.classList.toggle('current', isCurrent);
      if (isCurrent) btn.setAttribute('aria-current', 'true');
      else btn.removeAttribute('aria-current');
    });

    var view = $('pageView');
    view.textContent = 'Loading page…';
    fetch(API + '/render/' + state.renderJobId + '/pages/' + index + '.svg')
      .then(function (res) {
        if (!res.ok) throw new Error('Page could not be loaded');
        return res.text();
      })
      .then(function (svg) {
        view.innerHTML = svg;
        var el = view.querySelector('svg');
        if (el) {
          el.removeAttribute('width');
          el.removeAttribute('height');
          el.setAttribute('style', 'width:100%; height:auto; display:block;');
          el.setAttribute('role', 'img');
          el.setAttribute('aria-label', 'Page ' + (index + 1) + ' rendered in your handwriting');
        }
      })
      .catch(function (err) { view.textContent = err.message; });
  }

  // ----------------------------------------------------------------- export

  function downloadPdf() {
    show($('exportError'), false);
    if (state.pdfAvailable === false) {
      $('exportErrorText').textContent = state.pdfError ||
        'This render has no PDF available.';
      show($('exportError'), true);
      return;
    }

    var url = API + '/render/' + state.renderJobId + '/pdf';
    fetch(url).then(function (res) {
      if (!res.ok) {
        return res.json().catch(function () { return {}; }).then(function (body) {
          throw new Error(body.error || 'The PDF could not be downloaded.');
        });
      }
      return res.blob();
    }).then(function (blob) {
      var link = document.createElement('a');
      link.href = URL.createObjectURL(blob);
      link.download = 'codea4-output.pdf';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(link.href);
      $('downloadStatusText').textContent =
        'codea4-output.pdf prepared — ' + state.pages.length + ' page(s), ready to print.';
      show($('downloadStatus'), true);
      setStatus('PDF downloaded');
    }).catch(function (err) {
      $('exportErrorText').textContent = err.message;
      show($('exportError'), true);
      setStatus('export failed');
    });
  }

  // ------------------------------------------------------------------- push

  function setupPush() {
    if (!('serviceWorker' in navigator) || !('PushManager' in window)) return;

    api('/push/vapid-key').then(function (info) {
      if (!info.enabled) return;   // Push is not configured; leave it hidden.
      show($('notifRow'), true);

      $('notifCheck').addEventListener('change', function () {
        if (!this.checked) return;
        var checkbox = this;
        Notification.requestPermission().then(function (permission) {
          if (permission !== 'granted') {
            checkbox.checked = false;
            return;
          }
          return navigator.serviceWorker.ready.then(function (reg) {
            return reg.pushManager.subscribe({
              userVisibleOnly: true,
              applicationServerKey: urlBase64ToUint8Array(info.public_key)
            });
          }).then(function (sub) {
            return api('/push/subscribe', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ session_id: state.sessionId, subscription: sub })
            });
          }).catch(function () { checkbox.checked = false; });
        });
      });
    }).catch(function () { /* push is optional */ });
  }

  function urlBase64ToUint8Array(base64String) {
    var padding = '='.repeat((4 - (base64String.length % 4)) % 4);
    var base64 = (base64String + padding).replace(/-/g, '+').replace(/_/g, '/');
    var raw = atob(base64);
    var output = new Uint8Array(raw.length);
    for (var i = 0; i < raw.length; i++) output[i] = raw.charCodeAt(i);
    return output;
  }

  // ------------------------------------------------------------------ wiring

  function bindEvents() {
    $('captureBtn').addEventListener('click', function () { $('photoInput').click(); });
    $('photoInput').addEventListener('change', function () {
      if (this.files && this.files[0]) uploadPhoto(this.files[0]);
      this.value = '';
    });
    $('retakeBtn').addEventListener('click', function () { $('photoInput').click(); });
    $('captureContinueBtn').addEventListener('click', function () {
      loadStyle().then(function () { goTo('style'); });
    });

    document.querySelectorAll('.back-btn, .continue-btn, .step-btn').forEach(function (btn) {
      btn.addEventListener('click', function () {
        var target = btn.dataset.target || btn.dataset.step;
        if (!target || btn.disabled) return;
        if (target === 'style' && !state.style) { loadStyle().then(function () { goTo('style'); }); return; }
        goTo(target);
      });
    });

    $('fileInput').addEventListener('change', function () {
      state.files = state.files.concat(Array.prototype.slice.call(this.files));
      renderFileTable();
      this.value = '';
    });

    $('fileTableBody').addEventListener('click', function (event) {
      var index = event.target.dataset.remove;
      if (index === undefined) return;
      state.files.splice(Number(index), 1);
      renderFileTable();
    });

    var dropzone = $('dropzone');
    ['dragover', 'dragenter'].forEach(function (name) {
      dropzone.addEventListener(name, function (e) { e.preventDefault(); dropzone.classList.add('is-over'); });
    });
    ['dragleave', 'drop'].forEach(function (name) {
      dropzone.addEventListener(name, function () { dropzone.classList.remove('is-over'); });
    });
    dropzone.addEventListener('drop', function (e) {
      e.preventDefault();
      state.files = state.files.concat(Array.prototype.slice.call(e.dataTransfer.files));
      renderFileTable();
    });

    $('toPreviewBtn').addEventListener('click', startRender);
    $('pageList').addEventListener('click', function (e) {
      var btn = e.target.closest('.page-thumb');
      if (btn) showPage(Number(btn.dataset.page));
    });

    $('downloadBtn').addEventListener('click', downloadPdf);
    $('retryDownloadBtn').addEventListener('click', downloadPdf);
    $('startOverBtn').addEventListener('click', function () {
      // Keep the handwriting model — it was three sheets of work — and clear
      // only the files and their render.
      state.files = [];
      state.pages = [];
      state.renderJobId = null;
      renderFileTable();
      show($('downloadStatus'), false);
      goTo('code');
    });
  }

  function registerServiceWorker() {
    if (!('serviceWorker' in navigator)) return;
    navigator.serviceWorker.register('/sw.js').catch(function () { /* offline shell is optional */ });
  }

  function init() {
    bindEvents();
    registerServiceWorker();
    setStatus('starting session');

    startSession()
      .then(function () {
        renderRounds();
        setStatus('ready');
        setupPush();
        if (state.session.style_ready) {
          unlockStep('style', true);
          return loadStyle();
        }
        var allDone = state.session.rounds.every(function (r) { return r.status === 'done'; });
        if (allDone) return refreshSessionUntilStyle().then(loadStyle);
      })
      .then(function () {
        var hash = location.hash.slice(1);
        if (STEPS.indexOf(hash) >= 0 && !document.querySelector('.step-btn[data-step="' + hash + '"]').disabled) {
          goTo(hash);
        }
      })
      .catch(function (err) {
        setStatus('could not start');
        $('captureErrorText').textContent =
          'Could not reach the server: ' + err.message;
        show($('captureError'), true);
      });

    loadExercise();
  }

  document.addEventListener('DOMContentLoaded', init);
})();
