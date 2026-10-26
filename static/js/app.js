/**
 * CampusPulse - Reactive Enterprise Application Engine
 * Pure Vanilla JavaScript with modular architecture, Chart.js integrations,
 * debounced real-time NLP triage, and zero-reload tab switching.
 * STRICTLY NO NEON / NO BLUE PALETTE. Full Dark & Light theme switching.
 */

// --------------------------------------------------------------------------
// Application State
// --------------------------------------------------------------------------
const state = {
  theme: localStorage.getItem('campuspulse-theme') || 'dark',
  currentTab: 'dashboard',
  explorerView: 'active',
  stats: null,
  complaints: [],
  selectedComplaint: null,
  charts: {},
  debounceTimer: null,
  simThreshold: 0.45,
  isAnalyzing: false,
};

// --------------------------------------------------------------------------
// Initialization on DOM Ready
// --------------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  initNavigation();
  initPresetChips();
  initTriageInputs();
  initExplorerFilters();
  initModalListeners();
  
  // Initial data load
  loadDashboardData();
  loadEvaluationData();
  loadConfigData();
});

// --------------------------------------------------------------------------
// Theme Engine (Dark & Light)
// --------------------------------------------------------------------------
function initTheme() {
  const saved = localStorage.getItem('campuspulse-theme') || 'dark';
  applyTheme(saved);

  const toggleBtn = document.getElementById('theme-toggle-btn');
  if (toggleBtn) {
    toggleBtn.addEventListener('click', () => {
      const nextTheme = state.theme === 'dark' ? 'light' : 'dark';
      applyTheme(nextTheme);
    });
  }
}

function applyTheme(theme) {
  state.theme = theme;
  document.documentElement.setAttribute('data-theme', theme);
  localStorage.setItem('campuspulse-theme', theme);

  const icon = document.getElementById('theme-icon');
  const label = document.getElementById('theme-label');
  if (icon) icon.textContent = theme === 'dark' ? '🌙' : '☀️';
  if (label) label.textContent = theme === 'dark' ? 'Dark Theme' : 'Light Theme';

  // Re-render active charts with matching theme palette
  if (state.stats) {
    renderDashboardCharts(state.stats);
  }
  loadEvaluationData();
}

// --------------------------------------------------------------------------
// Navigation & Tab Switching
// --------------------------------------------------------------------------
function initNavigation() {
  const tabButtons = document.querySelectorAll('.nav-tab-btn');
  tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetTab = btn.getAttribute('data-tab');
      switchTab(targetTab);
    });
  });
}

function switchTab(tabId) {
  state.currentTab = tabId;

  // Update navbar buttons
  document.querySelectorAll('.nav-tab-btn').forEach(btn => {
    btn.classList.toggle('active', btn.getAttribute('data-tab') === tabId);
  });

  // Update views
  document.querySelectorAll('.view-section').forEach(view => {
    view.classList.toggle('active', view.id === `view-${tabId}`);
  });

  // Trigger contextual refreshes
  if (tabId === 'dashboard') {
    loadDashboardData();
  } else if (tabId === 'explorer') {
    loadComplaintsQueue();
  } else if (tabId === 'evaluation') {
    loadEvaluationData();
  }
}

// --------------------------------------------------------------------------
// Dashboard Intelligence & Charts
// --------------------------------------------------------------------------
async function loadDashboardData() {
  try {
    const res = await fetch('/api/stats');
    if (!res.ok) throw new Error('Failed to load stats');
    const data = await res.json();
    state.stats = data;

    renderKPIs(data.kpis);
    renderDashboardCharts(data);
    renderRecentComplaints(data.recent_complaints || []);
    renderDuplicateClusters(data.duplicate_clusters || []);
  } catch (err) {
    console.error('Error loading dashboard data:', err);
    showToast('Failed to load live metrics', 'error');
  }
}

function renderKPIs(kpis) {
  if (!kpis) return;
  document.getElementById('kpi-total').textContent = kpis.total ?? 0;
  document.getElementById('kpi-genuine').textContent = `${kpis.genuine_total ?? 0} genuine`;
  document.getElementById('kpi-pending').textContent = kpis.pending ?? 0;
  document.getElementById('kpi-active').textContent = kpis.active_open ?? 0;
  document.getElementById('kpi-urgent').textContent = kpis.urgent_open ?? 0;
  document.getElementById('kpi-resolved').textContent = `${kpis.resolution_rate ?? 0}%`;
  document.getElementById('kpi-resolved-sub').textContent = `${kpis.resolved ?? 0} tickets completed`;
  document.getElementById('kpi-spam').textContent = kpis.spam_blocked ?? 0;
}

function renderDashboardCharts(data) {
  const isLight = state.theme === 'light';

  // Strict Warm Earth & Metallic Palette (Zero Neon Blue)
  const categoryPalette = ['#D97706', '#C2410C', '#059669', '#9A3412', '#9F1239'];
  const priorityPalette = {
    'Critical': '#DC2626',
    'High': '#D97706',
    'Medium': '#CA8A04',
    'Low': '#78716C'
  };

  const chartBorder = isLight ? '#FFFFFF' : '#141720';
  const labelColor = isLight ? '#57534E' : '#A8A29E';
  const gridColor = isLight ? 'rgba(41, 37, 36, 0.08)' : 'rgba(255, 255, 255, 0.06)';
  const tooltipBg = isLight ? '#FFFFFF' : '#181C26';
  const tooltipTitle = isLight ? '#1C1917' : '#FAF8F5';
  const tooltipBody = isLight ? '#44403C' : '#E7E5E4';
  const tooltipBorder = isLight ? 'rgba(41, 37, 36, 0.12)' : 'rgba(255, 255, 255, 0.1)';

  // 1. Category Distribution Donut Chart
  const catCanvas = document.getElementById('chart-category');
  if (catCanvas && window.Chart) {
    if (state.charts.category) state.charts.category.destroy();

    const catLabels = Object.keys(data.category_distribution || {});
    const catValues = Object.values(data.category_distribution || {});

    state.charts.category = new Chart(catCanvas, {
      type: 'doughnut',
      data: {
        labels: catLabels,
        datasets: [{
          data: catValues,
          backgroundColor: categoryPalette.slice(0, catLabels.length),
          borderColor: chartBorder,
          borderWidth: 2,
          hoverOffset: 6
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'right',
            labels: {
              color: labelColor,
              font: { family: 'Plus Jakarta Sans', size: 12 },
              padding: 14
            }
          },
          tooltip: {
            backgroundColor: tooltipBg,
            titleColor: tooltipTitle,
            bodyColor: tooltipBody,
            borderColor: tooltipBorder,
            borderWidth: 1,
            padding: 10
          }
        },
        cutout: '72%'
      }
    });
  }

  // 2. Priority Queue Bar Chart
  const prioCanvas = document.getElementById('chart-priority');
  if (prioCanvas && window.Chart) {
    if (state.charts.priority) state.charts.priority.destroy();

    const order = ['Critical', 'High', 'Medium', 'Low'];
    const pValues = order.map(k => (data.priority_distribution || {})[k] || 0);

    state.charts.priority = new Chart(prioCanvas, {
      type: 'bar',
      data: {
        labels: order,
        datasets: [{
          label: 'Active Tickets',
          data: pValues,
          backgroundColor: order.map(k => priorityPalette[k]),
          borderRadius: 6,
          barPercentage: 0.65
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: tooltipBg,
            titleColor: tooltipTitle,
            bodyColor: tooltipBody,
            borderColor: tooltipBorder,
            borderWidth: 1,
            padding: 10
          }
        },
        scales: {
          x: {
            grid: { display: false },
            ticks: { color: labelColor, font: { family: 'Plus Jakarta Sans' } }
          },
          y: {
            grid: { color: gridColor },
            ticks: { color: labelColor, precision: 0 }
          }
        }
      }
    });
  }

  // 3. Sentiment Breakdown Bar Chart
  const sentCanvas = document.getElementById('chart-sentiment');
  if (sentCanvas && window.Chart) {
    if (state.charts.sentiment) state.charts.sentiment.destroy();

    const sentLabels = ['Negative', 'Neutral', 'Positive'];
    const sentValues = sentLabels.map(s => (data.sentiment_distribution || {})[s] || 0);
    const sentColors = ['#DC2626', '#78716C', '#15803D'];

    state.charts.sentiment = new Chart(sentCanvas, {
      type: 'bar',
      data: {
        labels: sentLabels,
        datasets: [{
          label: 'Student Tone',
          data: sentValues,
          backgroundColor: sentColors,
          borderRadius: 6,
          barPercentage: 0.6
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { grid: { color: gridColor }, ticks: { color: labelColor } },
          y: { grid: { display: false }, ticks: { color: labelColor, font: { family: 'Plus Jakarta Sans' } } }
        }
      }
    });
  }
}

function renderRecentComplaints(complaints) {
  const tbody = document.getElementById('recent-complaints-tbody');
  if (!tbody) return;

  if (!complaints.length) {
    tbody.innerHTML = `<tr><td colspan="6" style="text-align: center; color: var(--text-muted); padding: 24px;">No complaints recorded yet.</td></tr>`;
    return;
  }

  tbody.innerHTML = complaints.map(c => `
    <tr>
      <td class="font-mono" style="color: var(--accent-gold-light); font-weight: 600;">${c.complaint_id}</td>
      <td style="font-weight: 600; color: var(--text-pure); max-width: 280px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
        ${escapeHtml(c.title)}
      </td>
      <td><span class="card-badge">${c.category}</span></td>
      <td><span class="badge badge-${c.priority_level.toLowerCase()}">${c.priority_level} (${c.priority_score})</span></td>
      <td><span class="badge badge-${formatStatusClass(c.status)}">${c.status}</span></td>
      <td style="color: var(--text-muted); font-size: 0.76rem;">${c.submitted_at ? c.submitted_at.substring(0, 16) : 'N/A'}</td>
    </tr>
  `).join('');
}

function renderDuplicateClusters(clusters) {
  const container = document.getElementById('duplicate-clusters-list');
  if (!container) return;

  if (!clusters || !clusters.length) {
    container.innerHTML = `
      <div style="padding: 24px; text-align: center; color: var(--text-muted); font-size: 0.85rem;">
        No multi-ticket duplicate symptom clusters detected in current queue.
      </div>
    `;
    return;
  }

  container.innerHTML = clusters.map(c => `
    <div class="similar-match-item" style="border-left: 3px solid var(--accent-gold);">
      <div class="match-header">
        <span class="match-id">${c.cluster_id || 'Cluster'} • ${c.category}</span>
        <span class="match-score-pill">${c.cluster_size} Repeated Tickets</span>
      </div>
      <div class="match-desc" style="font-weight: 500; color: var(--text-pure);">
        "${escapeHtml(c.lead_title || c.representative_text || '')}"
      </div>
      <div class="match-tags">
        ${(c.ticket_ids || []).map(id => `<span class="term-tag">${id}</span>`).join('')}
      </div>
    </div>
  `).join('');
}

// --------------------------------------------------------------------------
// Real-Time NLP Triage Studio
// --------------------------------------------------------------------------
function initTriageInputs() {
  const descInput = document.getElementById('input-desc');
  const locInput = document.getElementById('input-loc');
  const thresholdSlider = document.getElementById('input-threshold');
  const thresholdVal = document.getElementById('threshold-display-val');

  if (thresholdSlider && thresholdVal) {
    thresholdSlider.addEventListener('input', (e) => {
      state.simThreshold = parseFloat(e.target.value);
      thresholdVal.textContent = state.simThreshold.toFixed(2);
      triggerRealtimeAnalysis();
    });
  }

  if (descInput) {
    descInput.addEventListener('input', () => {
      clearTimeout(state.debounceTimer);
      state.debounceTimer = setTimeout(() => {
        triggerRealtimeAnalysis();
      }, 260);
    });
  }

  const submitBtn = document.getElementById('btn-submit-complaint');
  if (submitBtn) {
    submitBtn.addEventListener('click', handleComplaintSubmission);
  }
}

function initPresetChips() {
  document.querySelectorAll('.preset-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      const title = chip.getAttribute('data-title') || '';
      const desc = chip.getAttribute('data-desc') || '';
      const loc = chip.getAttribute('data-loc') || 'Campus';

      const titleInput = document.getElementById('input-title');
      const descInput = document.getElementById('input-desc');
      const locInput = document.getElementById('input-loc');

      if (titleInput) titleInput.value = title;
      if (descInput) descInput.value = desc;
      if (locInput) locInput.value = loc;

      triggerRealtimeAnalysis();
    });
  });
}

async function triggerRealtimeAnalysis() {
  const desc = document.getElementById('input-desc')?.value.trim();
  const loc = document.getElementById('input-loc')?.value.trim() || 'General Campus';

  if (!desc || desc.length < 3) {
    resetLiveHUD();
    return;
  }

  const liveIndicator = document.getElementById('hud-status-indicator');
  if (liveIndicator) liveIndicator.innerHTML = '<span class="live-pulse" style="background:var(--accent-gold);box-shadow:0 0 6px var(--accent-gold)"></span> Computing...';

  try {
    const res = await fetch('/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text: desc,
        location: loc,
        similarity_threshold: state.simThreshold
      })
    });

    if (!res.ok) throw new Error('Analysis failed');
    const data = await res.json();
    updateLiveHUD(data);

    if (liveIndicator) liveIndicator.innerHTML = '<span class="live-pulse"></span> Active Intelligence';
  } catch (err) {
    console.error('Real-time analysis error:', err);
    if (liveIndicator) liveIndicator.innerHTML = '● Inference Ready';
  }
}

function updateLiveHUD(data) {
  // 1. Authenticity Banner
  const banner = document.getElementById('hud-auth-banner');
  const bannerTitle = document.getElementById('hud-auth-title');
  const bannerDesc = document.getElementById('hud-auth-desc');
  const bannerScore = document.getElementById('hud-auth-score');

  if (banner && data.genuineness) {
    const g = data.genuineness;
    banner.className = 'authenticity-banner';

    if (g.verdict === 'GENUINE') {
      banner.classList.add('banner-genuine');
      bannerTitle.textContent = '🛡️ Authenticity Verified: Genuine Student Complaint';
    } else if (g.verdict === 'NEEDS_REVIEW') {
      banner.classList.add('banner-review');
      bannerTitle.textContent = '⚠️ Authenticity Alert: Staff Review Required';
    } else {
      banner.classList.add('banner-spam');
      bannerTitle.textContent = '🚫 Mischief / Threat / Spam Filter Triggered';
    }

    const flagsTxt = g.flags && g.flags.length ? `Flags: ${g.flags.join(', ')}` : '';
    bannerDesc.textContent = `${g.summary_reason} ${flagsTxt}`;
    bannerScore.innerHTML = `${g.score}<span>/100</span>`;
  }

  // 2. Metric Tiles
  const predCatElem = document.getElementById('hud-category-val');
  const predConfElem = document.getElementById('hud-category-conf');
  if (predCatElem) predCatElem.textContent = data.predicted_category || 'N/A';
  if (predConfElem) predConfElem.textContent = `Confidence: ${(data.confidence * 100).toFixed(1)}%`;

  const sentValElem = document.getElementById('hud-sentiment-val');
  const sentSubElem = document.getElementById('hud-sentiment-sub');
  if (sentValElem && data.sentiment) {
    sentValElem.textContent = data.sentiment.label;
    sentValElem.style.color = data.sentiment.label === 'Negative' ? '#DC2626' : (data.sentiment.label === 'Positive' ? '#15803D' : '#78716C');
    sentSubElem.textContent = `Polarity: ${data.sentiment.compound.toFixed(2)}`;
  }

  const prioValElem = document.getElementById('hud-priority-val');
  const prioSubElem = document.getElementById('hud-priority-sub');
  if (prioValElem && data.priority) {
    prioValElem.textContent = `${data.priority.priority_level} (${data.priority.priority_score})`;
    prioValElem.className = `hud-tile-val val-${getPriorityColorClass(data.priority.priority_level)}`;
    prioSubElem.textContent = `${data.priority.factors.length} rule factors active`;
  }

  // 3. Category Probabilities Breakdown
  const probaContainer = document.getElementById('hud-proba-bars');
  if (probaContainer && data.probabilities) {
    probaContainer.innerHTML = Object.entries(data.probabilities).map(([cat, p]) => {
      const pct = (p * 100).toFixed(1);
      return `
        <div class="proba-bar-row">
          <div class="proba-bar-header">
            <span>${cat}</span>
            <span class="font-mono">${pct}%</span>
          </div>
          <div class="proba-bar-track">
            <div class="proba-bar-fill" style="width: ${pct}%;"></div>
          </div>
        </div>
      `;
    }).join('');
  }

  // 4. Duplicate Matches
  const matchesContainer = document.getElementById('hud-similar-matches');
  if (matchesContainer) {
    if (!data.similar_matches || !data.similar_matches.length) {
      matchesContainer.innerHTML = `
        <div style="font-size: 0.78rem; color: var(--text-muted); text-align: center; padding: 12px;">
          No matching duplicates above ${(state.simThreshold * 100).toFixed(0)}% cosine similarity.
        </div>
      `;
    } else {
      matchesContainer.innerHTML = data.similar_matches.map(m => `
        <div class="similar-match-item">
          <div class="match-header">
            <span class="match-id">${m.complaint_id} • ${m.category}</span>
            <span class="match-score-pill">${m.similarity_percentage}% Match</span>
          </div>
          <div class="match-desc">${escapeHtml(m.title || m.description)}</div>
          <div class="match-tags">
            ${(m.shared_terms || []).map(t => `<span class="term-tag">${escapeHtml(t)}</span>`).join('')}
          </div>
        </div>
      `).join('');
    }
  }

  // 5. Preprocessing Inspector Drawer
  const prepCleaned = document.getElementById('hud-prep-cleaned');
  const prepTokens = document.getElementById('hud-prep-tokens');
  if (prepCleaned && data.preprocessing) {
    prepCleaned.textContent = data.preprocessing.final_cleaned || '(No alphanumeric tokens)';
    prepTokens.textContent = `${data.preprocessing.token_count} tokens retained, ${data.preprocessing.stopwords_removed.length} stopwords filtered`;
  }
}

function resetLiveHUD() {
  const banner = document.getElementById('hud-auth-banner');
  if (banner) {
    banner.className = 'authenticity-banner banner-genuine';
    document.getElementById('hud-auth-title').textContent = '🛡️ Authenticity Shield: Ready for Input';
    document.getElementById('hud-auth-desc').textContent = 'Type a campus issue description or pick a 1-click preset on the left.';
    document.getElementById('hud-auth-score').innerHTML = '--<span>/100</span>';
  }
  document.getElementById('hud-category-val').textContent = '--';
  document.getElementById('hud-category-conf').textContent = 'Confidence: --%';
  document.getElementById('hud-sentiment-val').textContent = '--';
  document.getElementById('hud-sentiment-sub').textContent = 'Polarity: --';
  document.getElementById('hud-priority-val').textContent = '--';
  document.getElementById('hud-priority-sub').textContent = 'Factors: --';
  document.getElementById('hud-proba-bars').innerHTML = '<div style="color:var(--text-muted);font-size:0.78rem;">Awaiting input...</div>';
  document.getElementById('hud-similar-matches').innerHTML = '<div style="color:var(--text-muted);font-size:0.78rem;text-align:center;">Cosine duplicate matcher standing by...</div>';
}

async function handleComplaintSubmission() {
  const title = document.getElementById('input-title')?.value.trim();
  const desc = document.getElementById('input-desc')?.value.trim();
  const loc = document.getElementById('input-loc')?.value.trim() || 'General Campus';

  if (!title) {
    showToast('Please provide a complaint title', 'error');
    document.getElementById('input-title')?.focus();
    return;
  }
  if (!desc || desc.length < 5) {
    showToast('Please describe the issue in detail (at least 5 characters)', 'error');
    document.getElementById('input-desc')?.focus();
    return;
  }

  const submitBtn = document.getElementById('btn-submit-complaint');
  if (submitBtn) {
    submitBtn.disabled = true;
    submitBtn.textContent = 'Verifying & Submitting...';
  }

  try {
    const res = await fetch('/api/complaints', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        title,
        description: desc,
        location: loc,
        similarity_threshold: state.simThreshold
      })
    });

    if (!res.ok) throw new Error('Submission failed');
    const result = await res.json();

    showToast(`Ticket ${result.complaint_id} logged successfully!`, 'success');

    // Reset inputs
    document.getElementById('input-title').value = '';
    document.getElementById('input-desc').value = '';
    resetLiveHUD();

    // Reload background data
    loadDashboardData();

    // Switch to queue explorer to show the new ticket
    setTimeout(() => {
      switchTab('explorer');
    }, 800);
  } catch (err) {
    console.error('Submission error:', err);
    showToast('Failed to submit complaint. Check console.', 'error');
  } finally {
    if (submitBtn) {
      submitBtn.disabled = false;
      submitBtn.textContent = 'Submit Verified Complaint';
    }
  }
}

// --------------------------------------------------------------------------
// Complaint Explorer & Queue Table
// --------------------------------------------------------------------------
function initExplorerFilters() {
  const searchInput = document.getElementById('filter-search');
  const catSelect = document.getElementById('filter-category');
  const prioSelect = document.getElementById('filter-priority');
  const statusSelect = document.getElementById('filter-status');
  const queueTabs = document.querySelectorAll('.queue-sub-tab');

  if (searchInput) {
    searchInput.addEventListener('input', () => {
      clearTimeout(state.debounceTimer);
      state.debounceTimer = setTimeout(loadComplaintsQueue, 250);
    });
  }

  [catSelect, prioSelect, statusSelect].forEach(sel => {
    if (sel) sel.addEventListener('change', loadComplaintsQueue);
  });

  queueTabs.forEach(tab => {
    tab.addEventListener('click', () => {
      queueTabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      state.explorerView = tab.getAttribute('data-view');
      loadComplaintsQueue();
    });
  });

  const exportBtn = document.getElementById('btn-export-csv');
  if (exportBtn) {
    exportBtn.addEventListener('click', exportQueueCSV);
  }
}

async function loadComplaintsQueue() {
  const search = document.getElementById('filter-search')?.value.trim() || '';
  const category = document.getElementById('filter-category')?.value || 'All';
  const priority = document.getElementById('filter-priority')?.value || 'All';
  const statusFilter = document.getElementById('filter-status')?.value || 'All';

  const params = new URLSearchParams({
    view: state.explorerView,
    limit: '100'
  });
  if (search) params.append('search', search);
  if (category !== 'All') params.append('category', category);
  if (priority !== 'All') params.append('priority', priority);
  if (statusFilter !== 'All') params.append('status', statusFilter);

  const tbody = document.getElementById('queue-tbody');
  if (tbody) tbody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: var(--text-muted); padding: 32px;">Loading queue records...</td></tr>`;

  try {
    const res = await fetch(`/api/complaints?${params.toString()}`);
    if (!res.ok) throw new Error('Failed to fetch queue');
    const data = await res.json();
    state.complaints = data.complaints || [];

    renderQueueTable(state.complaints);
  } catch (err) {
    console.error('Queue load error:', err);
    if (tbody) tbody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: var(--status-critical); padding: 32px;">Failed to load records.</td></tr>`;
  }
}

function renderQueueTable(complaints) {
  const tbody = document.getElementById('queue-tbody');
  const countDisplay = document.getElementById('queue-count-display');
  if (countDisplay) countDisplay.textContent = `${complaints.length} records shown`;

  if (!tbody) return;
  if (!complaints.length) {
    tbody.innerHTML = `
      <tr>
        <td colspan="7" style="text-align: center; color: var(--text-muted); padding: 40px;">
          No matching complaints found for the current filters.
        </td>
      </tr>
    `;
    return;
  }

  tbody.innerHTML = complaints.map(c => `
    <tr>
      <td class="font-mono" style="font-weight: 700; color: var(--accent-gold-light);">
        ${c.complaint_id}
      </td>
      <td>
        <div style="font-weight: 600; color: var(--text-pure);">${escapeHtml(c.title)}</div>
        <div style="font-size: 0.76rem; color: var(--text-muted); max-width: 320px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
          ${escapeHtml(c.description)}
        </div>
      </td>
      <td>
        <span class="card-badge">${c.category}</span>
        <div style="font-size: 0.72rem; color: var(--text-muted); margin-top: 2px;">${escapeHtml(c.location || 'General')}</div>
      </td>
      <td>
        <span class="badge badge-${c.priority_level.toLowerCase()}">${c.priority_level} (${c.priority_score})</span>
      </td>
      <td>
        <span class="badge badge-${formatStatusClass(c.status)}">${c.status}</span>
      </td>
      <td>
        <span class="badge ${c.genuineness_verdict === 'GENUINE' ? 'badge-resolved' : (c.genuineness_verdict === 'NEEDS_REVIEW' ? 'badge-high' : 'badge-spam')}">
          ${c.genuineness_score}/100
        </span>
      </td>
      <td>
        <button class="btn btn-secondary btn-sm" onclick="openComplaintModal('${c.complaint_id}')">
          View & Triage
        </button>
      </td>
    </tr>
  `).join('');
}

// --------------------------------------------------------------------------
// Complaint Details Modal & Status Updater
// --------------------------------------------------------------------------
function initModalListeners() {
  const overlay = document.getElementById('complaint-modal');
  const closeBtn = document.getElementById('modal-close-btn');
  const modalCancel = document.getElementById('modal-cancel-btn');
  const modalSave = document.getElementById('modal-save-btn');

  if (closeBtn) closeBtn.addEventListener('click', closeModal);
  if (modalCancel) modalCancel.addEventListener('click', closeModal);

  if (overlay) {
    overlay.addEventListener('click', (e) => {
      if (e.target === overlay) closeModal();
    });
  }

  if (modalSave) {
    modalSave.addEventListener('click', handleStatusSave);
  }
}

async function openComplaintModal(complaintId) {
  try {
    const res = await fetch(`/api/complaints/${complaintId}`);
    if (!res.ok) throw new Error('Failed to load complaint details');
    const c = await res.json();
    state.selectedComplaint = c;

    document.getElementById('modal-title').textContent = `${c.complaint_id}: ${c.title}`;
    document.getElementById('modal-desc').textContent = c.description;
    document.getElementById('modal-loc').textContent = c.location || 'General Campus';
    document.getElementById('modal-category').textContent = c.category;
    document.getElementById('modal-status-select').value = c.status;
    document.getElementById('modal-priority').textContent = `${c.priority_level} (${c.priority_score}/100)`;
    document.getElementById('modal-sentiment').textContent = `${c.sentiment_label} (${c.sentiment_score})`;
    document.getElementById('modal-auth').textContent = `${c.genuineness_verdict} (${c.genuineness_score}/100)`;

    // Priority factors list
    const factorsList = document.getElementById('modal-factors-list');
    if (factorsList) {
      const factors = c.priority_factors || [];
      if (factors.length) {
        factorsList.innerHTML = factors.map(f => `<li style="margin-bottom: 4px;">${escapeHtml(typeof f === 'string' ? f : JSON.stringify(f))}</li>`).join('');
      } else {
        factorsList.innerHTML = '<li>Standard category baseline triage</li>';
      }
    }

    document.getElementById('complaint-modal').classList.add('open');
  } catch (err) {
    console.error('Error opening modal:', err);
    showToast('Failed to load ticket details', 'error');
  }
}

function closeModal() {
  const modal = document.getElementById('complaint-modal');
  if (modal) modal.classList.remove('open');
  state.selectedComplaint = null;
}

async function handleStatusSave() {
  if (!state.selectedComplaint) return;
  const newStatus = document.getElementById('modal-status-select')?.value;
  const id = state.selectedComplaint.complaint_id;

  try {
    const res = await fetch(`/api/complaints/${id}/status`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ status: newStatus })
    });

    if (!res.ok) throw new Error('Status update failed');
    showToast(`Status updated to "${newStatus}"`, 'success');
    closeModal();
    loadComplaintsQueue();
    loadDashboardData();
  } catch (err) {
    console.error('Status update error:', err);
    showToast('Failed to update ticket status', 'error');
  }
}

function exportQueueCSV() {
  if (!state.complaints.length) {
    showToast('No records to export', 'error');
    return;
  }

  const headers = ['Complaint ID', 'Title', 'Category', 'Location', 'Priority Level', 'Priority Score', 'Status', 'Genuineness Score', 'Submitted At'];
  const rows = state.complaints.map(c => [
    c.complaint_id,
    `"${(c.title || '').replace(/"/g, '""')}"`,
    c.category,
    `"${(c.location || '').replace(/"/g, '""')}"`,
    c.priority_level,
    c.priority_score,
    c.status,
    c.genuineness_score,
    c.submitted_at
  ]);

  const csvContent = 'data:text/csv;charset=utf-8,' + [headers.join(','), ...rows.map(r => r.join(','))].join('\n');
  const encodedUri = encodeURI(csvContent);
  const link = document.createElement('a');
  link.setAttribute('href', encodedUri);
  link.setAttribute('download', `CampusPulse_Tickets_${new Date().toISOString().substring(0, 10)}.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  showToast('Exported CSV successfully', 'success');
}

// --------------------------------------------------------------------------
// Model Intelligence & Evaluation View
// --------------------------------------------------------------------------
async function loadEvaluationData() {
  try {
    const res = await fetch('/api/evaluation');
    if (!res.ok) return;
    const m = await res.json();

    document.getElementById('eval-acc').textContent = `${(m.test_accuracy * 100).toFixed(1)}%`;
    document.getElementById('eval-f1').textContent = m.test_macro_f1.toFixed(3);
    document.getElementById('eval-gain').textContent = `+${(m.accuracy_gain_over_baseline * 100).toFixed(1)}%`;
    document.getElementById('eval-support').textContent = `${m.test_samples} test cases`;

    // Render Confusion Matrix Heatmap
    renderConfusionMatrix(m.categories, m.confusion_matrix);
    // Render Per-Class Bar Chart
    renderPerClassChart(m.categories, m.classification_report);
  } catch (err) {
    console.error('Failed to load evaluation data:', err);
  }
}

function renderConfusionMatrix(categories, matrix) {
  const container = document.getElementById('eval-cm-table');
  if (!container || !categories || !matrix) return;

  let tableHtml = `<table class="data-table" style="text-align: center;"><thead><tr><th>True \\ Pred</th>`;
  categories.forEach(cat => {
    tableHtml += `<th>${cat}</th>`;
  });
  tableHtml += `</tr></thead><tbody>`;

  matrix.forEach((row, rIdx) => {
    tableHtml += `<tr><td style="font-weight: 700; color: var(--accent-gold); text-align: left;">${categories[rIdx]}</td>`;
    row.forEach((val, cIdx) => {
      const isDiagonal = rIdx === cIdx;
      const bg = isDiagonal && val > 0 ? 'rgba(21, 128, 61, 0.22)' : (val > 0 ? 'rgba(185, 28, 28, 0.18)' : 'transparent');
      const color = isDiagonal && val > 0 ? 'var(--status-genuine)' : (val > 0 ? 'var(--status-critical)' : 'var(--text-muted)');
      tableHtml += `<td style="background: ${bg}; color: ${color}; font-weight: 700; font-family: 'JetBrains Mono', monospace;">${val}</td>`;
    });
    tableHtml += `</tr>`;
  });
  tableHtml += `</tbody></table>`;
  container.innerHTML = tableHtml;
}

function renderPerClassChart(categories, report) {
  const canvas = document.getElementById('chart-eval-classes');
  if (!canvas || !window.Chart || !report) return;

  if (state.charts.evalClass) state.charts.evalClass.destroy();

  const isLight = state.theme === 'light';
  const labelColor = isLight ? '#57534E' : '#A8A29E';
  const gridColor = isLight ? 'rgba(41, 37, 36, 0.08)' : 'rgba(255, 255, 255, 0.06)';

  const labels = categories || Object.keys(report).filter(k => !k.includes('avg') && k !== 'accuracy');
  const precisions = labels.map(c => ((report[c]?.precision || 0) * 100).toFixed(1));
  const recalls = labels.map(c => ((report[c]?.recall || 0) * 100).toFixed(1));
  const f1s = labels.map(c => ((report[c]?.['f1-score'] || 0) * 100).toFixed(1));

  state.charts.evalClass = new Chart(canvas, {
    type: 'bar',
    data: {
      labels,
      datasets: [
        { label: 'Precision (%)', data: precisions, backgroundColor: '#C2410C', borderRadius: 4 },
        { label: 'Recall (%)', data: recalls, backgroundColor: '#15803D', borderRadius: 4 },
        { label: 'F1-Score (%)', data: f1s, backgroundColor: '#D97706', borderRadius: 4 }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'top',
          labels: { color: labelColor, font: { family: 'Plus Jakarta Sans', size: 11 } }
        }
      },
      scales: {
        x: { grid: { display: false }, ticks: { color: labelColor } },
        y: { max: 100, grid: { color: gridColor }, ticks: { color: labelColor } }
      }
    }
  });
}

// --------------------------------------------------------------------------
// Admin Config View
// --------------------------------------------------------------------------
async function loadConfigData() {
  try {
    const res = await fetch('/api/config');
    if (!res.ok) return;
    const cfg = await res.json();

    const abbrevContainer = document.getElementById('config-abbrevs-list');
    if (abbrevContainer && cfg.campus_abbreviations) {
      abbrevContainer.innerHTML = Object.entries(cfg.campus_abbreviations).map(([abbr, full]) => `
        <div style="background: var(--bg-surface); border: 1px solid var(--border-subtle); padding: 8px 12px; border-radius: var(--radius-sm); font-size: 0.8rem; display: flex; justify-content: space-between;">
          <strong style="color: var(--accent-gold); font-family: 'JetBrains Mono', monospace;">${abbr.toUpperCase()}</strong>
          <span style="color: var(--text-secondary);">${full}</span>
        </div>
      `).join('');
    }
  } catch (err) {
    console.error('Failed to load config:', err);
  }
}

// --------------------------------------------------------------------------
// Toast Notification Engine
// --------------------------------------------------------------------------
function showToast(message, type = 'success') {
  const container = document.getElementById('toast-container');
  if (!container) return;

  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.innerHTML = `
    <span>${type === 'success' ? '✓' : '⚠'}</span>
    <span>${escapeHtml(message)}</span>
  `;

  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(10px)';
    setTimeout(() => toast.remove(), 250);
  }, 3200);
}

// --------------------------------------------------------------------------
// Utility Helpers
// --------------------------------------------------------------------------
function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

function formatStatusClass(status) {
  switch (status) {
    case 'Pending': return 'pending';
    case 'In Progress': return 'in-progress';
    case 'Under Review': return 'review';
    case 'Resolved': return 'resolved';
    case 'Spam/Blocked': return 'spam';
    default: return 'low';
  }
}

function getPriorityColorClass(level) {
  switch (level) {
    case 'Critical': return 'rust';
    case 'High': return 'amber';
    case 'Medium': return 'gold';
    default: return 'stone';
  }
}
