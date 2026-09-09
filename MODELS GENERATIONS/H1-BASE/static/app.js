/**
 * ========================================================================================
 * EURUSD.0 H1/H2 HIGH-PIP SWING ENSEMBLE CLIENT
 * ========================================================================================
 */

let allStrategies = [];
let selectedStrategyId = 1;
let activeCategory = 'ALL';
let searchQuery = '';
let isPolling = true;
let isInspectorCompact = false;

document.addEventListener('DOMContentLoaded', () => {
    initUIEventListeners();
    fetchStatus();
    setInterval(fetchStatus, 1500);
});

function initUIEventListeners() {
    // Category pills
    const pills = document.querySelectorAll('.cat-pill');
    pills.forEach(pill => {
        pill.addEventListener('click', () => {
            pills.forEach(p => p.classList.remove('active'));
            pill.classList.add('active');
            activeCategory = pill.getAttribute('data-cat');
            renderCardsGrid();
        });
    });

    // Search input
    const searchInput = document.getElementById('strategy-search');
    const clearBtn = document.getElementById('clear-search');
    searchInput.addEventListener('input', (e) => {
        searchQuery = e.target.value.toLowerCase().trim();
        renderCardsGrid();
    });
    clearBtn.addEventListener('click', () => {
        searchInput.value = '';
        searchQuery = '';
        renderCardsGrid();
    });

    // Mode Toggle (Live / Mock)
    const modeBtn = document.getElementById('mode-btn');
    modeBtn.addEventListener('click', () => {
        const sourcePill = document.getElementById('source-pill');
        const isLive = sourcePill.textContent.includes('LIVE');
        fetch('/api/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mock_mode: isLive })
        });
    });

    // Navigation buttons
    document.getElementById('prev-script-btn').addEventListener('click', () => {
        navigateStrategy(-1);
    });
    document.getElementById('next-script-btn').addEventListener('click', () => {
        navigateStrategy(1);
    });

    // Keyboard navigation (Arrow keys)
    window.addEventListener('keydown', (e) => {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
        if (e.key === 'ArrowLeft') navigateStrategy(-1);
        if (e.key === 'ArrowRight') navigateStrategy(1);
    });

    // Inspector toggle
    const toggleBtn = document.getElementById('toggle-insp-btn');
    const inspector = document.getElementById('active-script-inspector');
    toggleBtn.addEventListener('click', () => {
        isInspectorCompact = !isInspectorCompact;
        inspector.classList.toggle('compact', isInspectorCompact);
    });
}

function navigateStrategy(direction) {
    if (!allStrategies || allStrategies.length === 0) return;
    const ids = allStrategies.map(s => s.num);
    const currIdx = ids.indexOf(selectedStrategyId);
    let nextIdx = currIdx + direction;
    if (nextIdx < 0) nextIdx = ids.length - 1;
    if (nextIdx >= ids.length) nextIdx = 0;
    selectedStrategyId = ids[nextIdx];
    renderCardsGrid();
    renderInspector();
}

async function fetchStatus() {
    if (!isPolling) return;
    try {
        const res = await fetch('/api/status');
        if (!res.ok) return;
        const data = await res.json();
        updateDashboard(data);
    } catch (err) {
        console.warn('Status fetch failed:', err);
    }
}

function updateDashboard(data) {
    if (!data) return;

    // 1. Top Bar Ticker
    if (data.tick) {
        document.getElementById('bid-price').textContent = data.tick.bid.toFixed(5);
        document.getElementById('ask-price').textContent = data.tick.ask.toFixed(5);
        document.getElementById('spread-pips').textContent = `${data.tick.spread.toFixed(1)}p`;
        document.getElementById('source-pill').textContent = data.tick.source || 'LIVE MT5';
    }

    if (data.server_time) {
        document.getElementById('server-clock').textContent = data.server_time.split(' ')[1] || data.server_time;
    }

    // 2. Master Actionable Swing Decision Banner
    if (data.master_decision) {
        const m = data.master_decision;
        const masterCard = document.getElementById('master-card');
        const actionTitle = document.getElementById('master-action');

        actionTitle.textContent = m.action;
        masterCard.className = 'master-card';
        if (m.direction === 'BUY') masterCard.classList.add('buy');
        if (m.direction === 'SELL') masterCard.classList.add('sell');

        document.getElementById('master-regime').textContent = m.regime;
        document.getElementById('master-holding').textContent = m.holding_period;
        document.getElementById('master-conf-score').textContent = `${m.confluence_score}%`;
        document.getElementById('master-conf-bar').style.width = `${m.confluence_score}%`;

        // Order Grid
        document.getElementById('master-entry').textContent = m.entry > 0 ? m.entry.toFixed(5) : '--';
        document.getElementById('master-sl').textContent = m.sl > 0 ? m.sl.toFixed(5) : '--';
        document.getElementById('master-risk').textContent = m.risk_pips > 0 ? `-${m.risk_pips.toFixed(1)} pips` : '--';

        document.getElementById('master-tp1').textContent = m.tp1 > 0 ? m.tp1.toFixed(5) : '--';
        document.getElementById('master-reward1').textContent = m.reward_pips_tp1 > 0 ? `+${m.reward_pips_tp1.toFixed(1)} pips` : '--';

        document.getElementById('master-tp2').textContent = m.tp2 > 0 ? m.tp2.toFixed(5) : '--';
        document.getElementById('master-reward2').textContent = m.reward_pips_tp2 > 0 ? `+${m.reward_pips_tp2.toFixed(1)} pips (${m.rr_ratio})` : '--';

        // Driver Pills
        const driverContainer = document.getElementById('master-driver-pills');
        if (m.key_drivers && m.key_drivers.length > 0) {
            driverContainer.innerHTML = m.key_drivers.map(d => `<span class="driver-pill">${d}</span>`).join('');
        } else {
            driverContainer.innerHTML = `<span style="color: var(--text-dim); font-size: 10px;">Awaiting multi-timeframe convergence...</span>`;
        }
    }

    // 3. Update Strategies Data
    if (data.strategies) {
        allStrategies = data.strategies;
        const activeCount = allStrategies.filter(s => s.overall_status === 'READY' || s.overall_status === 'TRIGGERED').length;
        document.getElementById('active-sig-badge').textContent = activeCount;

        renderCardsGrid();
        renderInspector();
    }

    // 4. Update Footer Info
    if (data.account) {
        document.getElementById('mt5-account').textContent = `${data.account.login} (${data.account.server})`;
    }
    if (data.symbol) {
        document.getElementById('active-symbol').textContent = data.symbol;
    }
}

function renderCardsGrid() {
    const grid = document.getElementById('cards-grid');
    if (!allStrategies || allStrategies.length === 0) return;

    const filtered = allStrategies.filter(s => {
        if (activeCategory === 'SIGNALS') {
            if (s.overall_status !== 'READY' && s.overall_status !== 'TRIGGERED') return false;
        } else if (activeCategory !== 'ALL') {
            if (s.category_code !== activeCategory) return false;
        }

        if (searchQuery) {
            const matchName = s.name.toLowerCase().includes(searchQuery);
            const matchShort = s.short_name.toLowerCase().includes(searchQuery);
            const matchId = s.id.toString() === searchQuery || `#${s.id}` === searchQuery;
            const matchCat = s.category.toLowerCase().includes(searchQuery);
            if (!matchName && !matchShort && !matchId && !matchCat) return false;
        }

        return true;
    });

    if (filtered.length === 0) {
        grid.innerHTML = `<div class="loading-state">No curated swing strategies matching filter "${activeCategory}" / query "${searchQuery}"</div>`;
        return;
    }

    grid.innerHTML = filtered.map(s => {
        const isSelected = s.num === selectedStrategyId;
        const dirClass = s.direction.toLowerCase();
        const statusClass = s.overall_status.toLowerCase();

        const dotsHtml = (s.steps || []).map(step => {
            const dotClass = step.status.toLowerCase();
            return `<span class="dot ${dotClass}" title="${step.title}: ${step.status}"></span>`;
        }).join('');

        return `
            <div class="strat-card ${isSelected ? 'selected' : ''} ${statusClass === 'triggered' ? 'status-triggered ' + dirClass : ''}" 
                 onclick="selectStrategy(${s.num})" 
                 id="card-${s.num}">
                <div class="card-head">
                    <div class="card-num-title">
                        <span class="card-id-tag">#${s.num}</span>
                        <span class="card-title" title="${s.name}">${s.short_name}</span>
                    </div>
                    <span class="card-tf-badge">${s.timeframe}</span>
                </div>
                <div class="card-body">
                    <span class="dir-badge ${dirClass}">${s.direction}</span>
                    <div class="progress-summary">
                        <div class="step-dots">${dotsHtml}</div>
                        <span class="progress-text">[${s.true_count}/${s.total_steps}]</span>
                    </div>
                </div>
                <div class="card-foot">
                    <span class="card-cat" title="${s.category}">[${s.category_code}] ${s.category}</span>
                    <span class="card-state-pill ${statusClass}">${s.overall_status}</span>
                </div>
            </div>
        `;
    }).join('');
}

function selectStrategy(id) {
    selectedStrategyId = id;
    renderCardsGrid();
    renderInspector();
}

function renderInspector() {
    if (!allStrategies || allStrategies.length === 0) return;
    const strat = allStrategies.find(s => s.num === selectedStrategyId) || allStrategies[0];
    if (!strat) return;

    document.getElementById('insp-script-id').textContent = `STRATEGY #${strat.num}`;
    document.getElementById('insp-script-name').textContent = strat.name;
    document.getElementById('insp-timeframe').textContent = strat.timeframe;
    document.getElementById('insp-category').textContent = `[${strat.category_code}] ${strat.category}`;

    const dirPill = document.getElementById('insp-direction');
    dirPill.textContent = strat.direction;
    dirPill.className = `insp-dir-pill ${strat.direction.toLowerCase()}`;

    document.getElementById('insp-true-count').textContent = `${strat.true_count} / ${strat.total_steps} Conditions TRUE`;
    document.getElementById('insp-true-pct').textContent = `(H2: ${strat.h2_status} | H1: ${strat.h1_status})`;

    // Comparison Table
    const tbody = document.getElementById('comparison-tbody');
    if (strat.steps && strat.steps.length > 0) {
        tbody.innerHTML = strat.steps.map(step => {
            const stClass = (step.status || 'PENDING').toLowerCase();
            return `
                <tr>
                    <td class="cell-step-num">${step.step_num}</td>
                    <td class="cell-title">${step.title}</td>
                    <td class="cell-target">${step.target}</td>
                    <td class="cell-current">${step.current}</td>
                    <td class="cell-delta">${step.delta}</td>
                    <td><span class="status-badge ${stClass}">${step.status}</span></td>
                </tr>
            `;
        }).join('');
    } else {
        tbody.innerHTML = `<tr><td colspan="6" style="text-align: center; padding: 20px; color: var(--text-dim);">No steps defined for this strategy.</td></tr>`;
    }

    // Sidebar: Metrics
    const metricsGrid = document.getElementById('insp-metrics-grid');
    if (strat.metrics && Object.keys(strat.metrics).length > 0) {
        metricsGrid.innerHTML = Object.entries(strat.metrics).map(([k, v]) => `
            <div class="metric-item">
                <span class="metric-key">${k}</span>
                <span class="metric-val">${v}</span>
            </div>
        `).join('');
    } else {
        metricsGrid.innerHTML = `<div style="grid-column: 1/-1; color: var(--text-dim); font-size: 11px;">Standard technical indicators active.</div>`;
    }

    // Sidebar: Execution Envelope
    const sigDetails = document.getElementById('insp-signal-details');
    if (strat.active_signal) {
        const sig = strat.active_signal;
        sigDetails.innerHTML = `
            <div class="sig-box">
                <div class="sig-row"><span>Direction:</span> <strong style="color: ${strat.direction==='BUY' ? 'var(--green)' : 'var(--red)'}">${sig.direction}</strong></div>
                <div class="sig-row"><span>Entry Price:</span> <strong>${sig.entry.toFixed(5)}</strong></div>
                <div class="sig-row"><span>Stop Loss:</span> <strong>${sig.sl.toFixed(5)} (-${sig.risk_pips}p)</strong></div>
                <div class="sig-row"><span>Target 1:</span> <strong style="color: var(--green);">${sig.tp1.toFixed(5)} (+${sig.tp1_pips}p)</strong></div>
                <div class="sig-row"><span>Target 2:</span> <strong style="color: var(--cyan);">${sig.tp2.toFixed(5)} (+${sig.tp2_pips}p)</strong></div>
                <div class="sig-row"><span>Risk/Reward:</span> <strong style="color: var(--cyan);">${sig.rr}</strong></div>
            </div>
        `;
    } else {
        sigDetails.innerHTML = `<div class="empty-signal">Waiting for H2 and H1 conditions to align...</div>`;
    }
}
