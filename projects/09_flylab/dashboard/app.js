/* FlyLab live training dashboard. Static: reads only relative files, no build step. */
(function () {
  'use strict';

  var SKILLS = [
    ['walk_forward', 'Walk forward'],
    ['turn', 'Turn in place'],
    ['goto', 'Go to a location'],
    ['odor_seek', 'Find food by smell'],
    ['odor_avoid', 'Escape a bad smell'],
    ['light_seek', 'Walk toward light'],
    ['light_avoid', 'Hide from light']
  ];
  var NAME = {};
  SKILLS.forEach(function (s) { NAME[s[0]] = s[1]; });

  // 0-based levels, as written by the trainer
  var LEVELS = [
    { val: 6, test: 8, thr: 1.0 },
    { val: 12, test: 16, thr: 1.0 },
    { val: 24, test: 32, thr: 0.95 }
  ];
  var SURR_PASS = 0.95;
  var TEST_OPTIMAL = 0.90;   // optimal also needs >= 90% on the 32 held-out test arenas

  // 95% Wilson confidence interval for a rate p measured on n arenas
  function wilson(p, n) {
    if (!n) return [0, 1];
    var z = 1.96, c = (p + z * z / (2 * n)) / (1 + z * z / n);
    var h = z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n);
    return [Math.max(0, c - h), Math.min(1, c + h)];
  }
  function unitOf(id) {
    return id === 'turn' ? '°' : (id === 'odor_avoid' || id === 'light_avoid') ? ' mm gained' : ' mm';
  }
  var REFRESH_MS = 5 * 60 * 1000;
  var EVENTS_PAGE = 40;

  var state = { data: {}, videos: {}, selected: 'walk_forward', chart: null, events: [], eventLimit: EVENTS_PAGE, eventFilter: 'all' };

  // ---------- helpers ----------
  function $(id) { return document.getElementById(id); }
  function esc(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }
  function pct(x) { return (x == null || isNaN(x)) ? '–' : Math.round(x * 100) + '%'; }
  function num(x) { return (x == null) ? '–' : Number(x).toLocaleString('en-US'); }
  function bust(url) { return url + (url.indexOf('?') < 0 ? '?' : '&') + 't=' + Date.now(); }
  function fmtTime(t) { return t ? String(t).replace('T', ' ').slice(0, 16) : '–'; }
  function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

  function fetchJSON(url) {
    return fetch(bust(url), { cache: 'no-store' }).then(function (r) {
      if (!r.ok) throw new Error(url + ': HTTP ' + r.status);
      return r.json();
    });
  }
  function fetchText(url) {
    return fetch(bust(url), { cache: 'no-store' }).then(function (r) {
      if (!r.ok) throw new Error(url + ': HTTP ' + r.status);
      return r.text();
    });
  }
  function parseJSONL(text) {
    var rows = [];
    text.split(/\r?\n/).forEach(function (line) {
      line = line.trim();
      if (!line) return;
      try { rows.push(JSON.parse(line)); } catch (e) { /* partial line while being written */ }
    });
    rows.sort(function (a, b) { return (a.round || 0) - (b.round || 0); });
    return rows;
  }

  // 0-based level of a log row. Log rows store level 1-based (status.json stores it 0-based);
  // rows written before the field existed are inferred from their number of validation arenas.
  function rowLevel(r) {
    if (typeof r.level === 'number') return Math.max(0, r.level - 1);
    var n = r.n_val || (r.physics_errors ? r.physics_errors.length : 0);
    if (n >= 24) return 2;
    if (n >= 12) return 1;
    return 0;
  }

  // Level-ups: a jump between consecutive rows, plus any level-up after the last logged round
  // (the trainer re-scores the champion on the bigger arena sets without writing a log row).
  function levelUps(rows, st) {
    var out = [], prev = null;
    rows.forEach(function (r, i) {
      var lv = rowLevel(r);
      if (prev != null && lv > prev) out.push({ after: rows[i - 1].round, x: r.round - 0.5, from: prev, level: lv, time: rows[i - 1].time || '' });
      prev = lv;
    });
    if (st && typeof st.level === 'number' && rows.length) {
      var last = rows[rows.length - 1];
      if (st.level > rowLevel(last)) out.push({ after: last.round, x: last.round + 0.5, from: rowLevel(last), level: st.level, time: last.time || '', current: true });
    }
    return out;
  }

  function skillStatus(st) {
    var level = typeof st.level === 'number' ? st.level : 0;
    var L = LEVELS[Math.min(level, 2)];
    var b = st.best || {};
    var n = b.n || (b.physics_errors ? b.physics_errors.length : 0);
    var passed = b.physics_val != null && b.physics_val >= L.thr - 1e-9 &&
      b.surrogate != null && b.surrogate >= SURR_PASS && n >= L.val;
    var t = st.test || {};
    if (level >= 2 && passed && t.physics_test != null && t.physics_test >= TEST_OPTIMAL - 1e-9) return { cls: 'optimal', label: 'optimal', level: level, passed: true };
    if (passed) return { cls: 'passed', label: 'level ' + (level + 1) + ' of 3 · passed', level: level, passed: true };
    return { cls: 'training', label: 'level ' + (level + 1) + ' of 3', level: level, passed: false };
  }

  // ---------- inline markdown (tiny, safe: escape first) ----------
  function inlineMd(s) {
    var codes = [];
    s = esc(s);
    s = s.replace(/`([^`]+)`/g, function (_, c) { codes.push(c); return '\u0000' + (codes.length - 1) + '\u0000'; });
    s = s.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    s = s.replace(/(^|[^*\w])\*([^*\s][^*]*?)\*(?!\*)/g, '$1<em>$2</em>');
    s = s.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, function (_, t, u) {
      var href = u;
      if (!/^(https?:|#|\/)/.test(u)) href = '../' + u.replace(/^\.\//, '');
      if (/^javascript:/i.test(href)) href = '#';
      return '<a href="' + href + '">' + t + '</a>';
    });
    s = s.replace(/\u0000(\d+)\u0000/g, function (_, i) { return '<code>' + codes[+i] + '</code>'; });
    return s;
  }
  function splitRow(line) {
    var t = line.trim().replace(/^\|/, '').replace(/\|$/, '').replace(/\\\|/g, '\u0001');
    return t.split('|').map(function (c) { return c.replace(/\u0001/g, '|').trim(); });
  }

  function renderEngLog(md) {
    var lines = md.split(/\r?\n/);
    var start = -1;
    for (var i = 0; i < lines.length; i++) {
      if (/^##\s+Making skills survive the jump to physics/i.test(lines[i])) { start = i + 1; break; }
    }
    if (start < 0) { $('englog').innerHTML = '<p class="dim">Engineering log section not found in README.md.</p>'; return; }
    var section = [];
    for (var j = start; j < lines.length && !/^##\s/.test(lines[j]); j++) section.push(lines[j]);

    var html = '', para = [], tableLines = [];
    function flushPara() {
      if (!para.length) return;
      var txt = para.join(' ').replace(/<!--[\s\S]*?-->/g, '').trim();
      if (txt) html += '<p class="eng-extra">' + inlineMd(txt) + '</p>';
      para = [];
    }
    function flushTable() {
      if (!tableLines.length) return;
      var head = splitRow(tableLines[0]);
      var body = tableLines.slice(1).filter(function (l) { return !/^\s*\|?\s*:?-{2,}/.test(l); });
      var t = '<div class="table-scroll"><table class="eng"><thead><tr>';
      head.forEach(function (h) { t += '<th scope="col">' + inlineMd(h) + '</th>'; });
      t += '</tr></thead><tbody>';
      body.forEach(function (l) {
        var cells = splitRow(l);
        t += '<tr>';
        head.forEach(function (h, k) {
          t += '<td data-h="' + esc(h.replace(/[*`]/g, '')) + '">' + inlineMd(cells[k] || '') + '</td>';
        });
        t += '</tr>';
      });
      html += t + '</tbody></table></div>';
      tableLines = [];
    }
    section.forEach(function (l) {
      if (/^\s*\|/.test(l)) { flushPara(); tableLines.push(l); }
      else if (!l.trim()) { flushTable(); flushPara(); }
      else if (/^```/.test(l)) { /* ignore fences */ }
      else { flushTable(); para.push(l.trim()); }
    });
    flushTable(); flushPara();
    $('englog').innerHTML = html;
  }

  // ---------- cards ----------
  function renderCards() {
    var out = '';
    SKILLS.forEach(function (s) {
      var id = s[0], d = state.data[id];
      if (!d || !d.status) {
        out += '<article class="card"><div class="card-head"><div><h3>' + esc(s[1]) +
          '</h3><div class="skill-id">' + id + '</div></div></div><p class="dim">No data yet' +
          (d && d.error ? ' (' + esc(d.error) + ')' : '') + '.</p></article>';
        return;
      }
      var st = d.status, b = st.best || {}, t = st.test || {};
      var status = skillStatus(st);
      var valN = b.n || (b.physics_errors ? b.physics_errors.length : null);
      var testN = t.n || (t.physics_errors ? t.physics_errors.length : null);
      var testCount = (t.physics_test != null && testN) ? Math.round(t.physics_test * testN) + ' / ' + testN + ' arenas' : '';
      out += '<article class="card" id="card-' + id + '">' +
        '<div class="card-head"><div><h3>' + esc(s[1]) + '</h3><div class="skill-id">' + id + '</div></div>' +
        '<span class="badge ' + status.cls + '">' + esc(status.label) + '</span></div>' +
        '<div class="headline"><span class="big">' + pct(t.physics_test) + '</span>' +
        '<span class="lbl">held-out physics test' + (testCount ? ' · ' + testCount : '') +
        (t.physics_test != null && testN ? ' · 95% CI ' + wilson(t.physics_test, testN).map(pct).join('–') : '') +
        '</span></div>' +
        '<dl class="stats">' +
        '<dt>Physics validation</dt><dd>' + pct(b.physics_val) + (valN ? ' of ' + valN : '') + '</dd>' +
        '<dt>Surrogate</dt><dd>' + pct(b.surrogate) + '</dd>' +
        '<dt>Best brain from round</dt><dd>' + num(b.round) + '</dd>' +
        '<dt>Rounds</dt><dd>' + num(st.round) + '</dd>' +
        '<dt>ES generations</dt><dd>' + num(st.generations) + '</dd>' +
        '<dt>Physics runs</dt><dd>' + num(st.physics_runs) + '</dd>' +
        (st.rule_changes && st.rule_changes.length ? '<dt>Task rules</dt><dd>v' + esc(st.task_version) +
          ' since round ' + num(st.rule_changes[st.rule_changes.length - 1].after_round) + '</dd>' : '') +
        (st.reseed_round ? '<dt>Last reseed</dt><dd>round ' + num(st.reseed_round) + '</dd>' : '') +
        '</dl>' + videoHtml(id, st) + '</article>';
    });
    $('cards').innerHTML = out;
    // swap broken videos/posters for the placeholder
    Array.prototype.forEach.call(document.querySelectorAll('video.video'), function (v) {
      v.addEventListener('error', function () { v.replaceWith(placeholder('Video could not be loaded.')); }, true);
    });
  }
  function placeholder(text) {
    var d = document.createElement('div');
    d.className = 'video-ph';
    d.textContent = text;
    return d;
  }
  function videoHtml(id, st) {  // eslint-disable-line
    var v = state.videos[id];
    if (!v || !v.file) return '<div class="video-ph">No physics video rendered yet for this skill.</div>';
    var best = st.best && st.best.round;
    var cap = 'MuJoCo physics' + (v.arena ? ' · ' + esc(v.arena) : '') +
      (v.success != null ? ' · ' + (v.success ? 'success' : 'miss') : '') +
      (v.final_error != null ? ' · final error ' + esc(v.final_error) + unitOf(id) : '') +
      (v.brain_round != null ? ' · brain from round ' + esc(v.brain_round) : '') +
      (best != null && v.brain_round != null && v.brain_round !== best ? ' (current best is round ' + best + ')' : '');
    return '<video class="video" controls muted loop playsinline preload="none"' +
      (v.poster ? ' poster="videos/' + encodeURI(v.poster) + '"' : '') + '>' +
      '<source src="videos/' + encodeURI(v.file) + '" type="video/mp4"></video>' +
      '<p class="video-cap">' + cap + '</p>';
  }

  // ---------- chart ----------
  function markerEvents(rows, st) {
    var marks = [];
    levelUps(rows, st).forEach(function (u) { marks.push({ x: u.x, kind: 'level', label: 'level ' + (u.level + 1) }); });
    rows.forEach(function (r) {
      if (r.reseeded) marks.push({ x: r.round, kind: 'reseed', label: 'reseed' });
      (st.rule_changes || []).forEach(function (c) {
        if (c.after_round === r.round) marks.push({ x: r.round + 0.5, kind: 'level', label: 'new rules v' + c.version });
      });
      if (r.restarted_from_best) marks.push({ x: r.round, kind: 'restart', label: 'restart' });
    });
    return marks;
  }

  var markerPlugin = {
    id: 'flyMarkers',
    afterDatasetsDraw: function (chart, args, opts) {
      var marks = opts.marks || [], xs = chart.scales.x, area = chart.chartArea, ctx = chart.ctx;
      var used = {};
      ctx.save();
      marks.forEach(function (m) {
        var px = xs.getPixelForValue(m.x);
        if (px < area.left - 1 || px > area.right + 1) return;
        ctx.strokeStyle = m.kind === 'level' ? opts.levelColor : opts.eventColor;
        ctx.lineWidth = m.kind === 'level' ? 2 : 1.5;
        ctx.setLineDash(m.kind === 'level' ? [6, 3] : [2, 3]);
        ctx.beginPath(); ctx.moveTo(px, area.top); ctx.lineTo(px, area.bottom); ctx.stroke();
        var key = Math.round(px);
        var slot = used[key] = (used[key] || 0) + 1;
        ctx.setLineDash([]);
        ctx.fillStyle = opts.textColor;
        ctx.font = '11px system-ui, sans-serif';
        ctx.textAlign = px > area.right - 40 ? 'right' : 'left';
        ctx.fillText(m.label, px + (ctx.textAlign === 'left' ? 3 : -3), area.top + 10 + (slot - 1) * 12);
      });
      ctx.restore();
    }
  };

  function renderTabs() {
    var html = '';
    SKILLS.forEach(function (s) {
      html += '<button type="button" role="tab" data-skill="' + s[0] + '" aria-selected="' + (s[0] === state.selected) + '">' + esc(s[1]) + '</button>';
    });
    $('tabs').innerHTML = html;
    Array.prototype.forEach.call($('tabs').querySelectorAll('button'), function (b) {
      b.addEventListener('click', function () {
        state.selected = b.getAttribute('data-skill');
        try { localStorage.setItem('flylab-skill', state.selected); } catch (e) { /* ignore */ }
        renderTabs(); renderChart();
      });
    });
  }

  function renderChart() {
    var d = state.data[state.selected] || {}, rows = d.rows || [];
    var c1 = cssVar('--s1'), c2 = cssVar('--s2'), c3 = cssVar('--s3');
    var text2 = cssVar('--text-2'), grid = cssVar('--grid'), muted = cssVar('--muted'), surface = cssVar('--surface');
    $('legend').innerHTML =
      '<span><i class="sw" style="background:' + c1 + '"></i>Surrogate (training sim)</span>' +
      '<span><i class="sw" style="background:' + c2 + '"></i>Physics validation</span>' +
      '<span><i class="sw dot" style="background:' + c3 + '"></i>Held-out physics test (new bests only)</span>' +
      '<span><i class="sw vline" style="border-color:' + text2 + '"></i>Level-up</span>' +
      '<span><i class="sw vline" style="border-color:' + muted + '"></i>Reseed / restart</span>';

    if (state.chart) { state.chart.destroy(); state.chart = null; }
    if (typeof Chart === 'undefined') {
      $('chart-note').textContent = 'Chart library failed to load (offline?). Numbers above are still current.';
      return;
    }
    if (!rows.length) { $('chart-note').textContent = 'No training rounds logged yet for this skill.'; }

    var surr = rows.map(function (r) { return { x: r.round, y: r.surrogate != null ? r.surrogate * 100 : null }; });
    var val = rows.map(function (r) {
      return { x: r.round, y: r.physics_val != null ? r.physics_val * 100 : null, n: r.n_val || (r.physics_errors ? r.physics_errors.length : null), improved: r.improved };
    });
    var test = rows.filter(function (r) { return r.physics_test != null; })
      .map(function (r) { return { x: r.round, y: r.physics_test * 100 }; });
    var marks = markerEvents(rows, d.status);
    var lastRound = rows.length ? rows[rows.length - 1].round : 1;

    state.chart = new Chart($('chart'), {
      type: 'line',
      data: {
        datasets: [
          { label: 'Surrogate', data: surr, borderColor: c1, backgroundColor: c1, borderWidth: 2, pointRadius: 0, pointHoverRadius: 4, tension: 0, spanGaps: true },
          { label: 'Physics validation', data: val, borderColor: c2, backgroundColor: c2, borderWidth: 2, pointRadius: 2.5, pointHoverRadius: 5, tension: 0, spanGaps: true },
          { label: 'Held-out physics test', data: test, showLine: false, borderColor: surface, backgroundColor: c3, borderWidth: 2, pointRadius: 5, pointHoverRadius: 7, pointStyle: 'circle' }
        ]
      },
      options: {
        responsive: true, maintainAspectRatio: false, animation: false, parsing: true,
        interaction: { mode: 'x', intersect: false },
        scales: {
          x: { type: 'linear', min: rows.length ? Math.max(0, rows[0].round - 0.5) : 0, max: lastRound + 0.75, title: { display: true, text: 'training round', color: text2 }, ticks: { color: text2, precision: 0 }, grid: { color: grid } },
          y: { min: 0, max: 100, title: { display: true, text: 'success rate (%)', color: text2 }, ticks: { color: text2, stepSize: 20, callback: function (v) { return v + '%'; } }, grid: { color: grid } }
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              title: function (items) { return items.length ? 'Round ' + items[0].parsed.x : ''; },
              label: function (item) {
                var p = item.raw, s = item.dataset.label + ': ' + Math.round(item.parsed.y) + '%';
                if (item.datasetIndex === 1 && p.n) s += ' of ' + p.n + ' arenas' + (p.improved ? ' (new best)' : '');
                return s;
              }
            }
          },
          flyMarkers: { marks: marks, levelColor: text2, eventColor: muted, textColor: text2 }
        }
      },
      plugins: [markerPlugin]
    });

    var nb = rows.filter(function (r) { return r.improved; }).length;
    $('chart-note').textContent = rows.length
      ? rows.length + ' rounds logged, ' + nb + ' new bests, ' + test.length + ' held-out test measurements. ' +
        'Each point is scored on that level\'s arena set (6, 12 or 24 validation arenas), so a dip right after a level-up is expected. ' +
        'At a level-up the champion is re-scored on the larger sets without a new log row; the card above shows those current numbers.'
      : 'No training rounds logged yet for this skill.';
  }

  // ---------- events ----------
  function buildEvents() {
    var ev = [];
    SKILLS.forEach(function (s) {
      var d = state.data[s[0]];
      if (!d || !d.rows) return;
      var st = d.status || {};
      levelUps(d.rows, st).forEach(function (u) {
        var L = LEVELS[Math.min(u.level, 2)], text = (u.level - u.from > 1 ? 'passed levels ' + (u.from + 1) + '-' + u.level : 'passed level ' + (u.from + 1)) + ', moved to level ' + (u.level + 1) + ' of 3 (' + L.val + ' validation / ' + L.test + ' test arenas)';
        if (u.current && st.best && st.test) {
          text += '; champion re-scored: validation ' + pct(st.best.physics_val) + ' of ' + (st.best.n || '?') +
            ', held-out test ' + pct(st.test.physics_test) + ' of ' + (st.test.n || '?');
          var ss = skillStatus(st);
          if (ss.cls === 'optimal') text += ' (optimal)';
        }
        ev.push({ kind: 'level', skill: s[0], round: u.after, time: u.time, text: text, after: true });
      });
      (st.rule_changes || []).forEach(function (c) {
        ev.push({ kind: 'level', skill: s[0], round: c.after_round, time: c.time, after: true,
          text: 'task rules changed to v' + c.version + ' (stricter, more realistic success test); deployed brain re-scored from level 1' });
      });
      d.rows.forEach(function (r) {
        var base = { skill: s[0], round: r.round, time: r.time || '' };
        if (r.reseeded) ev.push(Object.assign({ kind: 'reseed', text: 'lineage reseeded from fresh weights (deployed brain kept until beaten in physics)' }, base));
        if (r.restarted_from_best) ev.push(Object.assign({ kind: 'restart', text: 'search restarted from the best brain' }, base));
        if (r.improved) {
          var n = r.n_val || (r.physics_errors ? r.physics_errors.length : null);
          ev.push(Object.assign({
            kind: 'best',
            text: 'new best: physics validation ' + pct(r.physics_val) + (n ? ' of ' + n : '') +
              ', surrogate ' + pct(r.surrogate) + (r.physics_test != null ? ', held-out test ' + pct(r.physics_test) : '')
          }, base));
        }
      });
    });
    var order = { best: 0, restart: 1, reseed: 2, level: 3 };
    ev.sort(function (a, b) { return a.time < b.time ? 1 : a.time > b.time ? -1 : (order[b.kind] - order[a.kind]); });
    state.events = ev;
  }
  function renderEvents() {
    var f = state.eventFilter;
    var list = state.events.filter(function (e) { return f === 'all' || e.kind === f || (f === 'reseed' && e.kind === 'restart'); });
    var tags = { best: 'new best', level: 'level-up', reseed: 'reseed', restart: 'restart' };
    var html = list.slice(0, state.eventLimit).map(function (e) {
      return '<li><time datetime="' + esc(e.time) + '">' + esc(fmtTime(e.time)) + '</time><div><span class="tag ' + e.kind + '">' +
        tags[e.kind] + '</span><strong>' + esc(NAME[e.skill]) + '</strong> · ' + (e.after ? 'after round ' : 'round ') + esc(e.round) + ': ' + esc(e.text) + '</div></li>';
    }).join('');
    $('events').innerHTML = html || '<li><span class="dim">No events yet.</span></li>';
    $('more-events').hidden = list.length <= state.eventLimit;
    $('more-events').textContent = 'Show more (' + (list.length - state.eventLimit) + ' older)';

    var filters = [['all', 'All'], ['best', 'New bests'], ['level', 'Level-ups'], ['reseed', 'Reseeds & restarts']];
    $('event-filters').innerHTML = filters.map(function (x) {
      return '<button type="button" data-f="' + x[0] + '" class="' + (x[0] === f ? 'on' : '') + '" aria-pressed="' + (x[0] === f) + '">' + x[1] + '</button>';
    }).join('');
    Array.prototype.forEach.call($('event-filters').querySelectorAll('button'), function (b) {
      b.addEventListener('click', function () { state.eventFilter = b.getAttribute('data-f'); state.eventLimit = EVENTS_PAGE; renderEvents(); });
    });
  }

  // ---------- load ----------
  function load() {
    var errors = [];
    var jobs = SKILLS.map(function (s) {
      var id = s[0];
      return Promise.all([
        fetchJSON('../training/' + id + '/status.json').catch(function (e) { errors.push(e.message); return null; }),
        fetchText('../training/' + id + '/log.jsonl').then(parseJSONL).catch(function (e) { errors.push(e.message); return []; })
      ]).then(function (res) { state.data[id] = { status: res[0], rows: res[1] }; });
    });
    var vid = fetchJSON('videos/index.json').then(function (j) {
      var m = {};
      ((j && j.videos) || []).forEach(function (v) { if (v && v.skill) m[v.skill] = v; });
      state.videos = m;
    }).catch(function () { state.videos = {}; });
    var readme = fetchText('../README.md').then(renderEngLog).catch(function (e) {
      $('englog').innerHTML = '<p class="dim">Could not load README.md (' + esc(e.message) + ').</p>';
    });

    return Promise.all(jobs.concat([vid, readme])).then(function () {
      var latest = '';
      SKILLS.forEach(function (s) {
        (state.data[s[0]].rows || []).forEach(function (r) { if (r.time && r.time > latest) latest = r.time; });
      });
      $('updated').textContent = latest ? fmtTime(latest) + ' (training machine time)' : 'no data yet';
      $('checked').textContent = 'checked ' + new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) + ', next check in 5 min';
      var err = $('error');
      if (errors.length) { err.hidden = false; err.textContent = 'Some files could not be loaded: ' + errors.join('; '); }
      else err.hidden = true;
      renderCards();
      renderTabs();
      renderChart();
      buildEvents();
      renderEvents();
    });
  }

  try { var saved = localStorage.getItem('flylab-skill'); if (saved && NAME[saved]) state.selected = saved; } catch (e) { /* ignore */ }
  $('refresh').addEventListener('click', load);
  $('more-events').addEventListener('click', function () { state.eventLimit += EVENTS_PAGE; renderEvents(); });
  if (window.matchMedia) {
    var mq = window.matchMedia('(prefers-color-scheme: dark)');
    var rerender = function () { renderChart(); };
    if (mq.addEventListener) mq.addEventListener('change', rerender); else if (mq.addListener) mq.addListener(rerender);
  }
  load();
  setInterval(load, REFRESH_MS);
})();
