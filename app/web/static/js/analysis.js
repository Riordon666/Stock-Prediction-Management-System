(function () {
    let stockData = [];
    let analysisResult = null;

    let activeChartTab = 'price';
    let renderedTabs = { price: false, indicators: false, volume: false };
    let chartTabsInitialized = false;

    (function () {
        function markCopied(el) {
            try {
                el.classList.add('copied');
                setTimeout(function () {
                    try { el.classList.remove('copied'); } catch (e) {}
                }, 700);
            } catch (e) {}
        }

        function copyText(text) {
            if (!text) return Promise.reject(new Error('empty'));
            if (navigator.clipboard && typeof navigator.clipboard.writeText === 'function') {
                return navigator.clipboard.writeText(text);
            }
            return new Promise(function (resolve, reject) {
                try {
                    var ta = document.createElement('textarea');
                    ta.value = text;
                    ta.setAttribute('readonly', '');
                    ta.style.position = 'fixed';
                    ta.style.opacity = '0';
                    ta.style.left = '-9999px';
                    document.body.appendChild(ta);
                    ta.select();
                    var ok = document.execCommand('copy');
                    document.body.removeChild(ta);
                    ok ? resolve() : reject(new Error('copy failed'));
                } catch (e) {
                    reject(e);
                }
            });
        }

        document.addEventListener('click', function (e) {
            var target = e && e.target;
            if (!target) return;
            var codeEl = target.closest ? target.closest('.ticker-code') : null;
            if (!codeEl) return;
            var code = (codeEl.getAttribute('data-code') || codeEl.textContent || '').trim();
            if (!code) return;
            copyText(code).then(function () {
                markCopied(codeEl);
            }).catch(function () {
            });
        });
    })();

    // === THEME-AWARE CHART LOGIC START ===
    let charts = {}; // Store chart instances

    function getApexChartThemeOptions() {
        const isDarkMode = $('html').attr('data-theme') === 'dark';
        const textColor = isDarkMode ? '#e5e7eb' : '#111827';
        const gridColor = isDarkMode ? 'rgba(255,255,255,0.12)' : 'rgba(17,24,39,0.12)';
        const options = {
            theme: {
                mode: isDarkMode ? 'dark' : 'light'
            },
            chart: {
                background: 'transparent',
                foreColor: textColor
            },
            grid: {
                borderColor: gridColor
            },
            xaxis: {
                axisBorder: {
                    color: gridColor
                },
                axisTicks: {
                    color: gridColor
                }
            },
            tooltip: {
                theme: isDarkMode ? 'dark' : 'light'
            }
        };

        if (isDarkMode) {
            options.plotOptions = {
                radar: {
                    polygons: {
                        strokeColors: '#555',
                        fill: { colors: ['#393939', '#444444'] }
                    }
                }
            };
        } else {
            options.plotOptions = {
                radar: {
                    polygons: {
                        strokeColors: '#e9e9e9',
                        fill: { colors: ['#f8f8f8', '#fff'] }
                    }
                }
            };
        }

        return options;
    }

    function destroyAllCharts() {
        Object.values(charts).forEach(chart => {
            if (chart && typeof chart.destroy === 'function') {
                chart.destroy();
            }
        });
        charts = {};
    }

    function rerenderAllCharts() {
        if (!analysisResult) return;
        if (typeof ApexCharts === 'undefined') return;
        destroyAllCharts();
        renderedTabs = { price: false, indicators: false, volume: false };
        renderScoreGauge();
        renderRadarChart();
        renderActiveTabChart();
    }
    // === THEME-AWARE CHART LOGIC END ===


let lastChartTheme = $('html').attr('data-theme') || '';
let chartThemeRerenderTimer = null;

(function () {
    try {
        const el = document.documentElement;
        const observer = new MutationObserver(function (mutations) {
            for (let i = 0; i < mutations.length; i++) {
                const m = mutations[i];
                if (m.type === 'attributes' && m.attributeName === 'data-theme') {
                    const nextTheme = $('html').attr('data-theme') || '';
                    if (nextTheme === lastChartTheme) return;
                    lastChartTheme = nextTheme;

                    if (chartThemeRerenderTimer) {
                        clearTimeout(chartThemeRerenderTimer);
                        chartThemeRerenderTimer = null;
                    }
                    chartThemeRerenderTimer = setTimeout(function () {
                        try {
                            rerenderAllCharts();
                        } catch (e) {
                            console.error(e);
                        }
                    }, 650);
                    return;
                }
            }
        });
        observer.observe(el, { attributes: true, attributeFilter: ['data-theme'] });
    } catch (e) {
        console.error(e);
    }
})();

(function () {
    try {
        document.addEventListener('theme-changed', function () {
            try {
                rerenderAllCharts();
            } catch (e) {
                console.error(e);
            }
        });
    } catch (e) {
        console.error(e);
    }
})();

// 提交表单进行分析
$('#analysis-form').submit(function(e) {
    e.preventDefault();
    const stockCode = $('#stock-code').val().trim();
    const marketType = $('#market-type').val();
    const period = $('#analysis-period').val();

    if (!stockCode) {
        showError('请输入股票代码！');
        return;
    }

    fetchStockData(stockCode, marketType, period);
});

function showLoading() {
    $('#analysis-error').hide().text('');
    $('.analysis-submit').prop('disabled', true);
    $('#analysis-result').hide();
    $('#analysis-loading').show();
    setLoadingProgress(0);
    setLoadingTarget(0);
    startLoadingProgress();
}

function hideLoading() {
    $('.analysis-submit').prop('disabled', false);
    stopProgressPolling();
    stopLoadingProgress();
    $('#analysis-loading').hide();
}

let loadingProgress = 0;
let loadingTarget = 0;
let loadingTimer = null;

let currentRequestId = '';
let progressPollTimer = null;
let lastProgressStage = '';

function createRequestId() {
    try {
        if (window.crypto && typeof window.crypto.randomUUID === 'function') {
            return window.crypto.randomUUID();
        }
    } catch (e) {}
    return 'r_' + Date.now().toString(36) + '_' + Math.random().toString(36).slice(2, 10);
}

function stageLabel(stage) {
    const s = String(stage || '').trim();
    const map = {
        stock_data_start: '获取行情数据…',
        stock_data_history_ok: '行情数据处理…',
        stock_data_indicators_ok: '计算均线指标…',
        stock_data_done: '行情数据完成',
        analysis_start: '开始分析…',
        analysis_history_ok: '准备分析数据…',
        analysis_indicators_ok: '计算技术指标…',
        analysis_scoring_ok: '生成评分建议…',
        analysis_stock_info_ok: '获取股票信息…',
        done: '完成',
        done_no_data: '无数据',
        error: '出错',
        unknown: '加载中…'
    };
    return map[s] || '加载中…';
}

function setProgressText(percent, stage) {
    const p = Math.max(0, Math.min(100, Math.round(Number(percent) || 0)));
    const s = stage ? stageLabel(stage) : (lastProgressStage ? stageLabel(lastProgressStage) : '');
    const text = s ? (p + '% · ' + s) : (p + '%');
    $('#analysis-progress-text').text(text);
}

function startProgressPolling(requestId) {
    stopProgressPolling();
    if (!requestId) return;

    const poll = function () {
        $.ajax({
            url: `/api/progress?request_id=${encodeURIComponent(requestId)}`,
            type: 'GET',
            dataType: 'json',
            success: function (resp) {
                const percent = Number(resp && resp.percent);
                const stage = resp && resp.stage;
                if (stage) {
                    lastProgressStage = stage;
                }
                if (Number.isFinite(percent)) {
                    setLoadingTarget(percent);
                    setProgressText(percent, stage);
                    if (percent >= 100) {
                        stopProgressPolling();
                    }
                }
            }
        });
    };

    poll();
    progressPollTimer = setInterval(poll, 1200);
}

function stopProgressPolling() {
    if (progressPollTimer) {
        clearInterval(progressPollTimer);
        progressPollTimer = null;
    }
}

function setLoadingProgress(value) {
    const v = Math.max(0, Math.min(100, Math.round(Number(value) || 0)));
    loadingProgress = v;
    $('#analysis-progress-bar').css('width', v + '%');
    setProgressText(v);
}

function setLoadingTarget(value) {
    loadingTarget = Math.max(0, Math.min(100, Math.round(Number(value) || 0)));
}

function startLoadingProgress() {
    stopLoadingProgress();
    loadingTimer = setInterval(function () {
        if (loadingProgress >= 100) {
            return;
        }
        if (loadingProgress < loadingTarget) {
            const step = Math.max(1, Math.ceil((loadingTarget - loadingProgress) / 10));
            setLoadingProgress(loadingProgress + step);
        }
    }, 200);
}

function stopLoadingProgress() {
    if (loadingTimer) {
        clearInterval(loadingTimer);
        loadingTimer = null;
    }
}

function showError(msg) {
    $('#analysis-error').show().text(msg);
}

function formatNumber(value, decimals) {
    const n = Number(value);
    if (!Number.isFinite(n)) return '--';
    return n.toFixed(decimals == null ? 2 : decimals);
}

function formatDateLabel(value, timestamp) {
    const d = timestamp != null ? new Date(timestamp) : new Date(value);
    if (!d || Number.isNaN(d.getTime())) return String(value);
    const y = d.getFullYear();
    const m = String(d.getMonth() + 1).padStart(2, '0');
    const day = String(d.getDate()).padStart(2, '0');
    return `${y}-${m}-${day}`;
}

function formatPercent(value, decimals) {
    const n = Number(value);
    if (!Number.isFinite(n)) return '--';
    return (n * 100).toFixed(decimals == null ? 2 : decimals) + '%';
}

function formatAIAnalysis(text) {
    if (!text) return '';

    const safeText = String(text)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');

    let formatted = safeText
        .replace(/\*\*(.*?)\*\*/g, '<strong class="keyword">$1</strong>')
        .replace(/__(.*?)__/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/_(.*?)_/g, '<em>$1</em>')
        .replace(/^# (.*?)$/gm, '<h4 class="mt-3 mb-2">$1</h4>')
        .replace(/^## (.*?)$/gm, '<h5 class="mt-2 mb-2">$1</h5>')
        .replace(/支撑位/g, '<span class="keyword">支撑位</span>')
        .replace(/压力位/g, '<span class="keyword">压力位</span>')
        .replace(/趋势/g, '<span class="keyword">趋势</span>')
        .replace(/均线/g, '<span class="keyword">均线</span>')
        .replace(/MACD/g, '<span class="term">MACD</span>')
        .replace(/RSI/g, '<span class="term">RSI</span>')
        .replace(/KDJ/g, '<span class="term">KDJ</span>')
        .replace(/([上涨升])/g, '<span class="trend-up">$1</span>')
        .replace(/([下跌降])/g, '<span class="trend-down">$1</span>')
        .replace(/(买入|做多|多头|突破)/g, '<span class="trend-up">$1</span>')
        .replace(/(卖出|做空|空头|跌破)/g, '<span class="trend-down">$1</span>')
        .replace(/(\d+\.\d{2})/g, '<span class="price">$1</span>')
        .replace(/\n\n+/g, '</p><p class="analysis-para">')
        .replace(/\n/g, '<br>');

    return '<p class="analysis-para">' + formatted + '</p>';
}

function getScoreColorClass(score) {
    const s = Number(score);
    if (!Number.isFinite(s)) return 'score-mid';
    if (s >= 8) return 'score-good';
    if (s >= 5) return 'score-mid';
    return 'score-bad';
}

function getTrendColorClass(trend) {
    if (trend === 'UP') return 'trend-up';
    if (trend === 'DOWN') return 'trend-down';
    return 'text-muted';
}

function getTrendIcon(trend) {
    if (trend === 'UP') return '<i class="fas fa-arrow-up"></i>';
    if (trend === 'DOWN') return '<i class="fas fa-arrow-down"></i>';
    return '<i class="fas fa-minus"></i>';
}

// 获取股票数据
function fetchStockData(stockCode, marketType, period) {
    showLoading();
    currentRequestId = createRequestId();
    lastProgressStage = '';
    setLoadingTarget(0);
    startProgressPolling(currentRequestId);

    $.ajax({
        url: `/api/stock_data?stock_code=${stockCode}&market_type=${marketType}&period=${period}&request_id=${encodeURIComponent(currentRequestId)}`,
        type: 'GET',
        dataType: 'json',
        success: function(response) {
            if (!response.data) {
                hideLoading();
                showError('响应格式不正确: 缺少data字段');
                return;
            }

            if (response.data.length === 0) {
                hideLoading();
                showError('未找到股票数据');
                return;
            }

            stockData = response.data;
            fetchEnhancedAnalysis(stockCode, marketType, period);
        },
        error: function(xhr, status, error) {
            hideLoading();

            let errorMsg = '获取股票数据失败';
            if (xhr.responseJSON && xhr.responseJSON.error) {
                errorMsg += ': ' + xhr.responseJSON.error;
            } else if (error) {
                errorMsg += ': ' + error;
            }
            showError(errorMsg);
        }
    });
}

// 获取增强分析数据
function fetchEnhancedAnalysis(stockCode, marketType, period) {
    $.ajax({
        url: '/api/enhanced_analysis?_=' + new Date().getTime(),
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({
            stock_code: stockCode,
            market_type: marketType,
            period: period,
            request_id: currentRequestId,
        }),
        success: function(response) {
            if (!response.result) {
                hideLoading();
                showError('增强分析响应格式不正确');
                return;
            }

            try {
                analysisResult = response.result;
                renderAnalysisResult();
                $('#analysis-result').show();
            } catch (e) {
                console.error(e);
                const msg = (e && (e.message || e.toString())) ? (e.message || e.toString()) : '';
                showError('渲染分析结果失败，请重试' + (msg ? ('：' + msg) : ''));
            } finally {
                setLoadingProgress(100);
                hideLoading();
            }
        },
        error: function(xhr, status, error) {
            hideLoading();

            let errorMsg = '获取分析数据失败';
            if (xhr.responseJSON && xhr.responseJSON.error) {
                errorMsg += ': ' + xhr.responseJSON.error;
            } else if (error) {
                errorMsg += ': ' + error;
            }
            showError(errorMsg);
        }
    });
}

// 渲染分析结果
function renderAnalysisResult() {
    if (!analysisResult) return;

    const basic = analysisResult.basic_info || {};
    const priceData = analysisResult.price_data || {};
    const scores = analysisResult.scores || {};
    const reco = analysisResult.recommendation || {};
    const tech = analysisResult.technical_analysis || {};
    const indicators = tech.indicators || {};
    const trend = tech.trend || {};
    const srAll = tech.support_resistance || {};
    const srSupport = srAll.support || {};
    const srResistance = srAll.resistance || {};

    $('#total-score').removeClass('score-good score-mid score-bad');

    // 渲染股票基本信息
    const stockName = (basic.stock_name || basic.stock_code || '--');
    const stockCode = (basic.stock_code || '--');
    $('#stock-name').text(stockName + ' (' + stockCode + ')');
    $('#stock-info').text((basic.industry || '未知行业') + ' | ' + (basic.analysis_date || ''));

    // 渲染价格信息
    const mt = ($('#market-type').val() || 'A').toUpperCase();
    const currencyPrefix = mt === 'US' ? '$' : (mt === 'HK' ? 'HK$' : '¥');
    const currentPrice = Number(priceData.current_price);
    const priceChange = Number(priceData.price_change);
    const priceChangeValue = Number(priceData.price_change_value);
    $('#stock-price').text(currencyPrefix + formatNumber(currentPrice, 2));
    const priceChangeClass = (Number.isFinite(priceChange) ? priceChange : 0) >= 0 ? 'trend-up' : 'trend-down';
    const priceChangeIcon = (Number.isFinite(priceChange) ? priceChange : 0) >= 0 ? '<i class="fas fa-caret-up"></i>' : '<i class="fas fa-caret-down"></i>';
    $('#price-change').html(`<span class="${priceChangeClass}">${priceChangeIcon} ${formatNumber(priceChangeValue, 2)} (${formatPercent(priceChange, 2)})</span>`);

    // 渲染评分和建议
    const totalScore = Number(scores.total_score);
    const scoreClass = getScoreColorClass(totalScore);
    $('#total-score').text(Number.isFinite(totalScore) ? totalScore : '--').addClass(scoreClass);
    $('#recommendation').text(reco.action || '--');

    // 渲染技术指标
    $('#rsi-value').text(formatNumber(indicators.rsi, 2));

    const maTrendClass = getTrendColorClass(trend.ma_trend);
    const maTrendIcon = getTrendIcon(trend.ma_trend);
    $('#ma-trend').html(`<span class="${maTrendClass}">${maTrendIcon} ${(trend.ma_status || '--')}</span>`);

    const macdSignal = Number(indicators.macd) > Number(indicators.macd_signal) ? 'BUY' : 'SELL';
    const macdClass = macdSignal === 'BUY' ? 'trend-up' : 'trend-down';
    const macdIcon = macdSignal === 'BUY' ? '<i class="fas fa-arrow-up"></i>' : '<i class="fas fa-arrow-down"></i>';
    $('#macd-signal').html(`<span class="${macdClass}">${macdIcon} ${macdSignal}</span>`);

    $('#volatility').text(formatPercent(indicators.volatility, 2));

    // 渲染支撑压力位
    let supportResistanceHtml = '';

    // 渲染压力位
    if (srResistance && srResistance.short_term && srResistance.short_term.length > 0 && Number.isFinite(currentPrice) && currentPrice) {
        const resistance = srResistance.short_term[0];
        const distance = ((Number(resistance) - currentPrice) / currentPrice * 100).toFixed(2);
        supportResistanceHtml += `
            <tr>
                <td><span class="badge bg-danger">短期压力</span></td>
                <td>${formatNumber(resistance, 2)}</td>
                <td>+${distance}%</td>
            </tr>
        `;
    }

    if (srResistance && srResistance.medium_term && srResistance.medium_term.length > 0 && Number.isFinite(currentPrice) && currentPrice) {
        const resistance = srResistance.medium_term[0];
        const distance = ((Number(resistance) - currentPrice) / currentPrice * 100).toFixed(2);
        supportResistanceHtml += `
            <tr>
                <td><span class="badge bg-warning text-dark">中期压力</span></td>
                <td>${formatNumber(resistance, 2)}</td>
                <td>+${distance}%</td>
            </tr>
        `;
    }

    // 渲染支撑位
    if (srSupport && srSupport.short_term && srSupport.short_term.length > 0 && Number.isFinite(currentPrice) && currentPrice) {
        const support = srSupport.short_term[0];
        const distance = ((Number(support) - currentPrice) / currentPrice * 100).toFixed(2);
        supportResistanceHtml += `
            <tr>
                <td><span class="badge bg-success">短期支撑</span></td>
                <td>${formatNumber(support, 2)}</td>
                <td>${distance}%</td>
            </tr>
        `;
    }

    if (srSupport && srSupport.medium_term && srSupport.medium_term.length > 0 && Number.isFinite(currentPrice) && currentPrice) {
        const support = srSupport.medium_term[0];
        const distance = ((Number(support) - currentPrice) / currentPrice * 100).toFixed(2);
        supportResistanceHtml += `
            <tr>
                <td><span class="badge bg-info">中期支撑</span></td>
                <td>${formatNumber(support, 2)}</td>
                <td>${distance}%</td>
            </tr>
        `;
    }

    $('#support-resistance-table').html(supportResistanceHtml);

    // 渲染AI分析
    $('#ai-analysis').html(formatAIAnalysis(analysisResult.ai_analysis || ''));

    try {
        if (typeof ApexCharts === 'undefined') {
            showError('图表库加载失败（ApexCharts 未加载），已跳过图表渲染');
            return;
        }
        activeChartTab = 'price';
        renderedTabs = { price: false, indicators: false, volume: false };
        initChartTabs();
        setActiveTab(activeChartTab);
        renderScoreGauge();
        renderRadarChart();
        renderActiveTabChart();
    } catch (e) {
        console.error(e);
        const msg = (e && (e.message || e.toString())) ? (e.message || e.toString()) : '';
        showError('图表渲染失败，已跳过图表' + (msg ? ('：' + msg) : ''));
    }
}

function initChartTabs() {
    if (chartTabsInitialized) return;
    chartTabsInitialized = true;
    $(document).on('click', '.chart-tab', function () {
        const tab = $(this).data('tab');
        if (!tab) return;
        setActiveTab(tab);
    });
}

function setActiveTab(tab) {
    activeChartTab = tab;
    $('.chart-tab').removeClass('active');
    $(`.chart-tab[data-tab="${tab}"]`).addClass('active');
    $('.chart-panel').removeClass('active');
    $(`#chart-panel-${tab}`).addClass('active');

    renderActiveTabChart();
}

function renderActiveTabChart() {
    if (!analysisResult) return;
    if (activeChartTab === 'price') {
        if (!renderedTabs.price) {
            renderPriceChart();
            renderedTabs.price = true;
        }
        return;
    }
    if (activeChartTab === 'indicators') {
        if (!renderedTabs.indicators) {
            renderIndicatorsChart();
            renderedTabs.indicators = true;
        }
        return;
    }
    if (activeChartTab === 'volume') {
        if (!renderedTabs.volume) {
            renderVolumeChart();
            renderedTabs.volume = true;
        }
    }
}

// 绘制雷达图
function renderRadarChart() {
    if (!analysisResult) return;

    const options = {
        series: [{
            name: '评分',
            data: [
                analysisResult.scores.trend_score || 0,
                analysisResult.scores.indicators_score || 0,
                analysisResult.scores.support_resistance_score || 0,
                analysisResult.scores.volatility_volume_score || 0
            ]
        }],
        chart: {
            height: 220,
            type: 'radar',
            toolbar: {
                show: false
            }
        },
        grid: {
            padding: {
                left: 0,
                right: 0,
                top: 0,
                bottom: 0
            }
        },
        plotOptions: {
            radar: {
                size: 75,
                offsetX: 0,
                offsetY: 0
            }
        },
        xaxis: {
            categories: ['趋势分析', '技术指标', '支撑压力位', '波动与成交量'],
            labels: {
                style: {
                    fontSize: '12px'
                }
            }
        },
        yaxis: {
            max: 10,
            min: 0
        },
        fill: {
            opacity: 0.5,
            colors: ['#4e73df']
        },
        markers: {
            size: 5
        }
    };

    // 清除旧图表
    $('#radar-chart').empty();

    const finalOptions = $.extend(true, {}, options, getApexChartThemeOptions());
    const chart = new ApexCharts(document.querySelector("#radar-chart"), finalOptions);
    charts.radar = chart;
    chart.render();
}

function renderScoreGauge() {
    if (!analysisResult) return;

    const score = Number(analysisResult.scores.total_score || 0);
    const percent = Math.max(0, Math.min(100, score * 10));
    const options = {
        series: [percent],
        chart: {
            height: 180,
            type: 'radialBar',
            toolbar: {
                show: false
            }
        },
        plotOptions: {
            radialBar: {
                startAngle: -120,
                endAngle: 120,
                hollow: {
                    size: '62%'
                },
                track: {
                    background: 'rgba(110, 168, 255, 0.12)'
                },
                dataLabels: {
                    name: {
                        show: true,
                        fontSize: '12px'
                    },
                    value: {
                        show: true,
                        fontSize: '28px',
                        fontWeight: 900,
                        formatter: function () {
                            return String(score);
                        }
                    }
                }
            }
        },
        labels: ['综合评分']
    };

    $('#score-gauge').empty();
    const finalOptions = $.extend(true, {}, options, getApexChartThemeOptions());
    const chart = new ApexCharts(document.querySelector("#score-gauge"), finalOptions);
    charts.gauge = chart;
    chart.render();
}

function computeEMA(values, period) {
    const k = 2 / (period + 1);
    const ema = new Array(values.length).fill(null);
    if (values.length < period) return ema;
    let sum = 0;
    for (let i = 0; i < period; i++) {
        sum += values[i];
    }
    let prev = sum / period;
    ema[period - 1] = prev;
    for (let i = period; i < values.length; i++) {
        prev = values[i] * k + prev * (1 - k);
        ema[i] = prev;
    }
    return ema;
}

function computeRSISeries(closes, period) {
    const p = period || 14;
    const rsi = new Array(closes.length).fill(null);
    if (closes.length <= p) return rsi;
    let gains = 0;
    let losses = 0;
    for (let i = 1; i <= p; i++) {
        const diff = closes[i] - closes[i - 1];
        if (diff >= 0) gains += diff;
        else losses -= diff;
    }
    let avgGain = gains / p;
    let avgLoss = losses / p;
    rsi[p] = avgLoss === 0 ? 100 : 100 - (100 / (1 + (avgGain / avgLoss)));
    for (let i = p + 1; i < closes.length; i++) {
        const diff = closes[i] - closes[i - 1];
        const gain = diff > 0 ? diff : 0;
        const loss = diff < 0 ? -diff : 0;
        avgGain = (avgGain * (p - 1) + gain) / p;
        avgLoss = (avgLoss * (p - 1) + loss) / p;
        rsi[i] = avgLoss === 0 ? 100 : 100 - (100 / (1 + (avgGain / avgLoss)));
    }
    return rsi;
}

function computeMACDSeries(closes) {
    const ema12 = computeEMA(closes, 12);
    const ema26 = computeEMA(closes, 26);
    const macd = closes.map((_, i) => {
        if (ema12[i] == null || ema26[i] == null) return null;
        return ema12[i] - ema26[i];
    });
    const macdForSignal = macd.map(v => (v == null ? 0 : v));
    const signal = computeEMA(macdForSignal, 9).map((v, i) => (macd[i] == null ? null : v));
    const hist = macd.map((v, i) => {
        if (v == null || signal[i] == null) return null;
        return v - signal[i];
    });
    return { macd, signal, hist };
}

function computeEMAProgressive(values, period) {
    const k = 2 / (period + 1);
    const ema = new Array(values.length).fill(null);
    let prev = null;
    for (let i = 0; i < values.length; i++) {
        const v = values[i];
        if (!Number.isFinite(v)) {
            ema[i] = prev;
            continue;
        }
        if (prev == null) {
            prev = v;
        } else {
            prev = v * k + prev * (1 - k);
        }
        ema[i] = prev;
    }
    return ema;
}

function computeRSISeriesProgressive(closes, period) {
    const p = period || 14;
    const rsi = new Array(closes.length).fill(null);
    if (closes.length === 0) return rsi;
    rsi[0] = 50;

    let sumG = 0;
    let sumL = 0;
    for (let i = 1; i < closes.length; i++) {
        const prev = closes[i - 1];
        const cur = closes[i];
        if (!Number.isFinite(prev) || !Number.isFinite(cur)) {
            rsi[i] = rsi[i - 1];
            continue;
        }

        const diff = cur - prev;
        const gain = diff > 0 ? diff : 0;
        const loss = diff < 0 ? -diff : 0;
        sumG += gain;
        sumL += loss;

        if (i > p) {
            const dOld = closes[i - p] - closes[i - p - 1];
            const gOld = dOld > 0 ? dOld : 0;
            const lOld = dOld < 0 ? -dOld : 0;
            sumG -= gOld;
            sumL -= lOld;
        }

        const win = Math.min(i, p);
        const avgG = sumG / win;
        const avgL = sumL / win;
        rsi[i] = avgL === 0 ? 100 : 100 - (100 / (1 + (avgG / avgL)));
    }
    return rsi;
}

function computeMACDSeriesProgressive(closes) {
    const ema12 = computeEMAProgressive(closes, 12);
    const ema26 = computeEMAProgressive(closes, 26);
    const macd = closes.map((_, i) => {
        if (ema12[i] == null || ema26[i] == null) return null;
        return ema12[i] - ema26[i];
    });
    const signal = computeEMAProgressive(macd.map(v => (v == null ? 0 : v)), 9);
    const hist = macd.map((v, i) => {
        if (v == null || signal[i] == null) return null;
        return v - signal[i];
    });
    return { macd, signal, hist };
}

function computeMovingAverageValues(data, period) {
    const result = new Array(data.length).fill(null);
    const window = [];
    let sum = 0;

    for (let i = 0; i < data.length; i++) {
        const close = Number(data[i] && data[i].close);
        if (!Number.isFinite(close)) {
            result[i] = null;
            continue;
        }
        window.push(close);
        sum += close;
        if (window.length > period) {
            sum -= window.shift();
        }
        // 数据不足 period 时用已有窗口的平均值，使均线从开头连续显示
        result[i] = sum / window.length;
    }
    return result;
}

function renderIndicatorsChart() {
    if (!stockData || stockData.length === 0) return;
    const closes = stockData.map(item => {
        const v = Number(item.close);
        return Number.isFinite(v) ? v : null;
    });
    if (!closes.some(v => Number.isFinite(v))) return;
    const dates = stockData.map(item => new Date(item.date));

    const rsi = computeRSISeriesProgressive(closes, 14);
    const macd = computeMACDSeriesProgressive(closes);

    const macdVals = [
        ...macd.macd,
        ...macd.signal,
        ...macd.hist
    ].filter(v => Number.isFinite(v));
    let macdMin = -1;
    let macdMax = 1;
    if (macdVals.length > 0) {
        macdMin = Math.min(...macdVals, 0);
        macdMax = Math.max(...macdVals, 0);
        const span = macdMax - macdMin;
        const pad = span > 0 ? span * 0.18 : (Math.abs(macdMax || 0) * 0.25 + 0.001);
        macdMin -= pad;
        macdMax += pad;
    }

    const series = [
        {
            name: 'RSI(14)',
            type: 'line',
            yAxisIndex: 0,
            data: dates.map((d, i) => ({ x: d, y: rsi[i] }))
        },
        {
            name: 'MACD',
            type: 'line',
            yAxisIndex: 1,
            data: dates.map((d, i) => ({ x: d, y: macd.macd[i] }))
        },
        {
            name: 'Signal',
            type: 'line',
            yAxisIndex: 1,
            data: dates.map((d, i) => ({ x: d, y: macd.signal[i] }))
        },
        {
            name: 'Histogram',
            type: 'column',
            yAxisIndex: 1,
            data: dates.map((d, i) => ({ x: d, y: macd.hist[i] }))
        }
    ];

    const options = {
        series,
        chart: {
            height: 420,
            type: 'line',
            stacked: false,
            toolbar: {
                show: true
            }
        },
        legend: {
            position: 'bottom',
            horizontalAlign: 'left',
            fontSize: '12px',
            itemMargin: {
                horizontal: 10,
                vertical: 0
            }
        },
        stroke: {
            width: [2, 2, 2, 0],
            curve: 'smooth'
        },
        markers: {
            size: 0
        },
        dataLabels: {
            enabled: false
        },
        title: {
            text: `${analysisResult.basic_info.stock_name} (${analysisResult.basic_info.stock_code}) 技术指标（RSI / MACD）`,
            align: 'left',
            style: {
                fontSize: '14px'
            }
        },
        xaxis: {
            type: 'datetime',
            labels: {
                datetimeUTC: false,
                formatter: formatDateLabel
            },
            datetimeFormatter: {
                year: 'yyyy-MM-dd',
                month: 'yyyy-MM-dd',
                day: 'yyyy-MM-dd'
            }
        },
        yaxis: [
            {
                min: 0,
                max: 100,
                tickAmount: 5,
                labels: {
                    formatter: function (value) {
                        return formatNumber(value, 0);
                    }
                },
                title: {
                    text: 'RSI'
                }
            },
            {
                opposite: true,
                min: macdMin,
                max: macdMax,
                tickAmount: 6,
                labels: {
                    formatter: function (value) {
                        return formatNumber(value, 4);
                    }
                },
                title: {
                    text: 'MACD'
                }
            }
        ],
        tooltip: {
            shared: true,
            intersect: false,
            x: {
                format: 'yyyy-MM-dd'
            },
            y: {
                formatter: function (value, { seriesIndex }) {
                    if (value == null || !Number.isFinite(value)) return '-';
                    // RSI 显示整数，MACD/Signal/Histogram 保留小数，避免被四舍五入成 0
                    if (seriesIndex === 0) return formatNumber(value, 0);
                    return formatNumber(value, 4);
                }
            }
        },
        plotOptions: {
            bar: {
                columnWidth: '60%'
            }
        }
    };

    $('#indicators-chart').empty();
    const finalOptions = $.extend(true, {}, options, getApexChartThemeOptions());
    const chart = new ApexCharts(document.querySelector("#indicators-chart"), finalOptions);
    charts.indicators = chart;
    chart.render();
}

function renderVolumeChart() {
    if (!stockData || stockData.length === 0) return;
    const volumeSeries = stockData.map(item => {
        const v = item.volume == null ? null : Number(item.volume);
        return {
            x: new Date(item.date),
            y: Number.isFinite(v) ? v : null
        };
    });

    const maPeriod = 20;
    const volWindow = [];
    let volSum = 0;
    const avgSeries = volumeSeries.map(p => {
        const v = p.y;
        if (!Number.isFinite(v)) {
            return { x: p.x, y: null };
        }
        volWindow.push(v);
        volSum += v;
        if (volWindow.length > maPeriod) {
            volSum -= volWindow.shift();
        }
        return { x: p.x, y: volSum / volWindow.length };
    });

    const options = {
        series: [
            {
                name: '成交量',
                type: 'column',
                data: volumeSeries
            },
            {
                name: `均量线(MA${maPeriod})`,
                type: 'line',
                data: avgSeries
            }
        ],
        chart: {
            height: 420,
            type: 'line',
            toolbar: {
                show: true
            }
        },
        legend: {
            position: 'bottom',
            horizontalAlign: 'center',
            fontSize: '12px',
            itemMargin: {
                horizontal: 10,
                vertical: 0
            }
        },
        stroke: {
            width: [0, 2],
            curve: 'smooth'
        },
        markers: {
            size: 0
        },
        plotOptions: {
            bar: {
                columnWidth: '70%'
            }
        },
        dataLabels: {
            enabled: false
        },
        title: {
            text: `${analysisResult.basic_info.stock_name} (${analysisResult.basic_info.stock_code}) 成交量`,
            align: 'left',
            style: {
                fontSize: '14px'
            }
        },
        xaxis: {
            type: 'datetime'
            ,
            labels: {
                datetimeUTC: false,
                formatter: formatDateLabel
            },
            datetimeFormatter: {
                year: 'yyyy-MM-dd',
                month: 'yyyy-MM-dd',
                day: 'yyyy-MM-dd'
            }
        },
        yaxis: {
            labels: {
                formatter: function (value) {
                    return formatNumber(value, 0);
                }
            }
        },
        tooltip: {
            shared: true,
            intersect: false,
            x: {
                format: 'yyyy-MM-dd'
            },
            y: {
                formatter: function (value, { seriesIndex }) {
                    if (value == null || !Number.isFinite(value)) return '-';
                    if (seriesIndex === 0) return formatNumber(value, 0);
                    return formatNumber(value, 0);
                }
            }
        }
    };

    $('#volume-chart').empty();
    const finalOptions = $.extend(true, {}, options, getApexChartThemeOptions());
    const chart = new ApexCharts(document.querySelector("#volume-chart"), finalOptions);
    charts.volume = chart;
    chart.render();
}

// 绘制价格图表
function renderPriceChart() {
    if (!stockData || stockData.length === 0) return;

    // 准备价格数据
    const seriesData = [];

    const dates = stockData.map(item => new Date(item.date));
    const ma5Computed = computeMovingAverageValues(stockData, 5);
    const ma20Computed = computeMovingAverageValues(stockData, 20);
    const ma60Computed = computeMovingAverageValues(stockData, 60);

    // 添加收盘价折线（更平滑美观）
    const closeData = stockData.map(item => ({
        x: new Date(item.date),
        y: item.close
    }));
    seriesData.push({
        name: '收盘价',
        type: 'line',
        data: closeData
    });

    // 添加均线数据
    const ma5Data = dates.map((d, i) => {
        const v = Number(stockData[i] && stockData[i].MA5);
        return { x: d, y: Number.isFinite(v) && v > 0 ? v : ma5Computed[i] };
    });
    seriesData.push({
        name: 'MA5',
        type: 'line',
        data: ma5Data
    });

    const ma20Data = dates.map((d, i) => {
        const v = Number(stockData[i] && stockData[i].MA20);
        return { x: d, y: Number.isFinite(v) && v > 0 ? v : ma20Computed[i] };
    });
    seriesData.push({
        name: 'MA20',
        type: 'line',
        data: ma20Data
    });

    const ma60Data = dates.map((d, i) => {
        const v = Number(stockData[i] && stockData[i].MA60);
        return { x: d, y: Number.isFinite(v) && v > 0 ? v : ma60Computed[i] };
    });
    seriesData.push({
        name: 'MA60',
        type: 'line',
        data: ma60Data
    });

    // 创建图表
    const options = {
        series: seriesData,
        chart: {
            height: 420,
            type: 'line',
            toolbar: {
                show: true,
                tools: {
                    download: true,
                    selection: true,
                    zoom: true,
                    zoomin: true,
                    zoomout: true,
                    pan: true,
                    reset: true
                }
            }
        },
        stroke: {
            width: [2, 1.5, 1.5, 1.5],
            curve: 'smooth'
        },
        markers: {
            size: 0
        },
        title: {
            text: `${analysisResult.basic_info.stock_name} (${analysisResult.basic_info.stock_code}) 价格走势`,
            align: 'left',
            style: {
                fontSize: '14px'
            }
        },
        xaxis: {
            type: 'datetime'
            ,
            labels: {
                datetimeUTC: false,
                formatter: formatDateLabel
            },
            datetimeFormatter: {
                year: 'yyyy-MM-dd',
                month: 'yyyy-MM-dd',
                day: 'yyyy-MM-dd'
            }
        },
        yaxis: {
            tooltip: {
                enabled: true
            },
            labels: {
                formatter: function(value) {
                    return formatNumber(value, 2);  // 统一使用2位小数
                }
            }
        },
        tooltip: {
            shared: true,
            intersect: false,
            x: {
                format: 'yyyy-MM-dd'
            }
        }
    };

    // 清除旧图表
    $('#price-trend-chart').empty();

    const finalOptions = $.extend(true, {}, options, getApexChartThemeOptions());
    const chart = new ApexCharts(document.querySelector("#price-trend-chart"), finalOptions);
    charts.priceTrend = chart;
    chart.render();
}

function getQueryParam(name) {
    const params = new URLSearchParams(window.location.search);
    return params.get(name);
}

$(function () {
    const code = getQueryParam('stock_code') || getQueryParam('stockCode') || getQueryParam('code');
    const market = getQueryParam('market_type') || getQueryParam('marketType');
    const period = getQueryParam('period');
    if (code) {
        $('#stock-code').val(code);
    }
    if (market) {
        $('#market-type').val(market);
    }
    if (period) {
        $('#analysis-period').val(period);
    }
    if (code) {
        fetchStockData((code || '').trim(), $('#market-type').val(), $('#analysis-period').val());
    }
});

})();