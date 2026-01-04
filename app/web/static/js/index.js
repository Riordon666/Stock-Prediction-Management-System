$(document).ready(function() {

    // 快速分析按钮点击事件
    $('#quick-analysis-btn').click(function() {
        const stockCode = $('#quick-stock-code').val().trim();
        const marketType = $('#quick-market-type').val();

        if (!stockCode) {
            alert('请输入股票代码');
            return;
        }

        // 跳转到股票详情页
        window.location.href = `/stock_detail/${stockCode}?market_type=${marketType}`;
    });

    // 回车键提交表单
    $('#quick-stock-code').keypress(function(e) {
        if (e.which === 13) {
            $('#quick-analysis-btn').click();
            return false;
        }
    });

    // 加载最新新闻
    loadLatestNews();

    // 加载舆情热点
    loadHotspots();

    // 更新市场状态
    updateMarketStatus();

    // 启动滚动新闻
    startTickerNews();

    // 刷新按钮点击事件
    $('.refresh-news-btn').click(function() {
        isNewsExpanded = false;
        resetAutoFetchTimer();
        triggerNewsFetch(function () {
            loadLatestNews();
            loadHotspots();
            startTickerNews();
        });
    });

    // 只看重要切换事件
    $('#only-important').change(function() {
        isNewsExpanded = false;
        loadLatestNews();
        refreshCountdown = 300;
        if ($('#refresh-time').length) {
            $('#refresh-time').text('刷新倒计时: 5:00');
        }
    });

    $(document).on('click', '#news-expand-toggle', function() {
        isNewsExpanded = !isNewsExpanded;
        if (Array.isArray(lastNewsList) && lastNewsList.length) {
            displayNewsTimeline(lastNewsList);
        } else {
            loadLatestNews();
        }
    });

    // 启动自动获取任务（可被手动刷新重置到5分钟）
    resetAutoFetchTimer();

    setInterval(function() {
        updateCurrentTime();
        updateRefreshCountdown();
    }, 1000);
});

var refreshCountdown = 300;

var tickerAnimation = null;
var tickerSignature = '';
var tickerRafId = 0;

var autoFetchTimerId = 0;

function triggerNewsFetch(done) {
    $.ajax({
        url: '/api/fetch_news',
        method: 'POST',
        complete: function () {
            if (typeof done === 'function') {
                done();
            }
        }
    });
}

function resetAutoFetchTimer() {
    try {
        if (autoFetchTimerId) {
            clearTimeout(autoFetchTimerId);
        }
    } catch (e) {}

    refreshCountdown = 300;
    if ($('#refresh-time').length) {
        $('#refresh-time').text('刷新倒计时: 5:00');
    }

    autoFetchTimerId = setTimeout(function () {
        updateMarketStatus();
        triggerNewsFetch(function () {
            loadLatestNews(true);
            loadHotspots();
            startTickerNews();
        });
        resetAutoFetchTimer();
    }, 300000);
}

function updateMarketStatus() {
    const now = new Date();
    const hours = now.getHours();
    const minutes = now.getMinutes();
    const weekday = now.getDay();

    const isWeekend = weekday === 0 || weekday === 6;

    let chinaStatus = { open: false, text: '未开市' };
    if (!isWeekend && ((hours === 9 && minutes >= 30) || hours === 10 || (hours === 11 && minutes <= 30) || (hours >= 13 && hours < 15))) {
        chinaStatus = { open: true, text: '交易中' };
    }

    let hkStatus = { open: false, text: '未开市' };
    if (!isWeekend && ((hours === 9 && minutes >= 30) || hours === 10 || hours === 11 || (hours >= 13 && hours < 16))) {
        hkStatus = { open: true, text: '交易中' };
    }

    let taiwanStatus = { open: false, text: '未开市' };
    if (!isWeekend && (hours === 9 || hours === 10 || hours === 11 || hours === 12 || (hours === 13 && minutes <= 30))) {
        taiwanStatus = { open: true, text: '交易中' };
    }

    let japanStatus = { open: false, text: '未开市' };
    if (!isWeekend && (hours === 9 || hours === 10 || (hours === 11 && minutes <= 30) || (hours === 12 && minutes >= 30) || hours === 13 || hours === 14)) {
        japanStatus = { open: true, text: '交易中' };
    }

    let ukStatus = { open: false, text: '未开市' };
    if (!isWeekend && ((hours >= 15 && hours < 23) || (hours === 23 && minutes <= 30))) {
        ukStatus = { open: true, text: '交易中' };
    }

    let germanStatus = { open: false, text: '未开市' };
    if (!isWeekend && ((hours >= 15 && hours < 23) || (hours === 23 && minutes <= 30))) {
        germanStatus = { open: true, text: '交易中' };
    }

    let franceStatus = { open: false, text: '未开市' };
    if (!isWeekend && ((hours >= 15 && hours < 23) || (hours === 23 && minutes <= 30))) {
        franceStatus = { open: true, text: '交易中' };
    }

    let usStatus = { open: false, text: '未开市' };
    if ((hours >= 21 && minutes >= 30) || hours >= 22 || hours < 4) {
        const usDay = hours < 12 ? (weekday === 6 ? 5 : weekday - 1) : weekday;
        if (usDay !== 0 && usDay !== 6) {
            usStatus = { open: true, text: '交易中' };
        }
    }

    let nasdaqStatus = usStatus;

    let brazilStatus = { open: false, text: '未开市' };
    if ((hours >= 20 && minutes >= 30) || hours >= 21 || hours < 3) {
        const brazilDay = hours < 12 ? (weekday === 6 ? 5 : weekday - 1) : weekday;
        if (brazilDay !== 0 && brazilDay !== 6) {
            brazilStatus = { open: true, text: '交易中' };
        }
    }

    updateMarketStatusUI('china-market', chinaStatus);
    updateMarketStatusUI('hk-market', hkStatus);
    updateMarketStatusUI('taiwan-market', taiwanStatus);
    updateMarketStatusUI('japan-market', japanStatus);

    updateMarketStatusUI('uk-market', ukStatus);
    updateMarketStatusUI('german-market', germanStatus);
    updateMarketStatusUI('france-market', franceStatus);

    updateMarketStatusUI('us-market', usStatus);
    updateMarketStatusUI('nasdaq-market', nasdaqStatus);
    updateMarketStatusUI('brazil-market', brazilStatus);
}

function updateMarketStatusUI(elementId, status) {
    const element = $(`#${elementId}`);
    if (!element.length) {
        return;
    }
    const iconElement = element.find('i');
    const textElement = element.find('.status-text');

    if (status.open) {
        iconElement.removeClass('status-closed').addClass('status-open');
    } else {
        iconElement.removeClass('status-open').addClass('status-closed');
    }

    textElement.text(status.text);
}

function updateRefreshCountdown() {
    if ($('#refresh-time').length === 0) {
        return;
    }

    refreshCountdown--;
    if (refreshCountdown <= 0) {
        refreshCountdown = 300;
    }

    const minutes = Math.floor(refreshCountdown / 60);
    const seconds = refreshCountdown % 60;
    $('#refresh-time').text(`刷新倒计时: ${minutes}:${seconds < 10 ? '0' + seconds : seconds}`);
}

function updateCurrentTime() {
    if ($('#current-time-value').length === 0) {
        return;
    }

    const now = new Date();
    const timeString = now.toLocaleTimeString('zh-CN', { hour12: false });
    $('#current-time-value').text(timeString);
}

// HTML转义辅助函数
function escapeHtml(input) {
    const str = (input == null) ? '' : String(input);
    return str
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

// 加载最新新闻函数
var lastNewsList = [];
var isNewsExpanded = false;

function isMobileView() {
    try {
        return window.matchMedia && window.matchMedia('(max-width: 520px)').matches;
    } catch (e) {
        return false;
    }
}

function loadLatestNews(silent = false) {
    if (!silent) {
        $('#news-timeline').html('<div class="portal-loading">加载新闻中...</div>');
    }

    const onlyImportant = $('#only-important').is(':checked');

    $.ajax({
        url: '/api/latest_news',
        method: 'GET',
        data: {
            days: 2,
            limit: 500,
            important: onlyImportant ? 1 : 0
        },
        success: function(response) {
            if (response.success && response.news && response.news.length > 0) {
                lastNewsList = response.news;
                displayNewsTimeline(response.news);
            } else {
                lastNewsList = [];
                if (!silent) {
                    $('#news-timeline').html('<div class="portal-empty">暂无最新新闻</div>');
                }
            }
        },
        error: function(err) {
            console.error('获取新闻失败:', err);
            lastNewsList = [];
            if (!silent) {
                $('#news-timeline').html('<div class="portal-empty">获取新闻失败，请稍后重试</div>');
            }
        }
    });
}

// 加载舆情热点函数
function loadHotspots() {
    $('#hotspot-list').html('<div class="portal-loading">加载热点中...</div>');

    $.ajax({
        url: '/api/hotspots',
        method: 'GET',
        data: {
            limit: 10
        },
        success: function(response) {
            if (response.success && response.items && response.items.length > 0) {
                displayHotspots(response.items);
            } else {
                $('#hotspot-list').html('<div class="portal-empty">暂无热点</div>');
            }
        },
        error: function(err) {
            console.error('获取热点失败:', err);
            $('#hotspot-list').html('<div class="portal-empty">获取热点失败，请稍后重试</div>');
        }
    });
}

// 显示热点列表
function displayHotspots(hotspots) {
    if (hotspots.length === 0) {
        $('#hotspot-list').html('<div class="portal-empty">暂无热点</div>');
        return;
    }

    let hotspotsHtml = '<div class="hotspot-list">';

    hotspots.forEach((item, index) => {
        const rankClass = index < 3 ? 'rank-top' : '';
        const title = escapeHtml(item.title || item.content || '');
        const url = escapeHtml(item.url || '');
        const extra = escapeHtml(item.extra || '');
        hotspotsHtml += `
            <div class="hotspot-item">
                <span class="hotspot-rank ${rankClass}">${index + 1}</span>
                <span class="hotspot-title">${url ? `<a href="${url}" target="_blank" rel="noopener noreferrer">${title}</a>` : title}${extra ? `<span class="hotspot-extra">${extra}</span>` : ''}</span>
            </div>
        `;
    });

    hotspotsHtml += '</div>';

    $('#hotspot-list').html(hotspotsHtml);
}

// 显示新闻时间线
function displayNewsTimeline(newsList) {
    const list = Array.isArray(newsList) ? newsList.slice() : [];
    if (list.length === 0) {
        $('#news-timeline').html('<div class="portal-empty">暂无新闻</div>');
        return;
    }

    let timelineHtml = '<div class="news-timeline-container">';

    // 首先按完整的日期时间排序，确保最新消息在最前面
    list.sort((a, b) => {
        // 构建完整的日期时间字符串 (YYYY-MM-DD HH:MM)
        const dateTimeA = (a.date || '') + ' ' + (a.time || '00:00');
        const dateTimeB = (b.date || '') + ' ' + (b.time || '00:00');

        // 转换为Date对象进行比较
        const timeA = new Date(dateTimeA);
        const timeB = new Date(dateTimeB);

        // 返回降序结果（最新的在前）
        return timeB - timeA;
    });

    const limitForMobile = 6;
    const shouldCollapse = isMobileView() && !isNewsExpanded && list.length > limitForMobile;
    const renderList = shouldCollapse ? list.slice(0, limitForMobile) : list;

    // 按天和时间点分组
    const newsGroups = {};
    renderList.forEach(news => {
        // 创建格式为"YYYY-MM-DD HH:MM"的键
        const date = news.date || '';
        const time = news.time || '00:00';

        // 显示用的键：日期+时间
        const displayKey = `${date} ${time.substring(0, 5)}`;

        if (!newsGroups[displayKey]) {
            newsGroups[displayKey] = [];
        }
        newsGroups[displayKey].push(news);
    });

    // 获取并按时间降序排列所有组键
    const sortedKeys = Object.keys(newsGroups).sort((a, b) => {
        const timeA = new Date(a);
        const timeB = new Date(b);
        return timeB - timeA;
    });

    // 生成时间线HTML
    sortedKeys.forEach(displayKey => {
        const newsItems = newsGroups[displayKey];
        const parts = displayKey.split(' ');
        const date = parts[0];
        const time = parts[1];

        // 格式化显示日期（只在新的一天开始时显示）
        const formattedDate = formatDate(date);

        timelineHtml += `
            <div class="time-point">
                <div class="time-label">${time}</div>
                <div class="time-date">${formattedDate}</div>
                <div class="news-items">
        `;

        newsItems.forEach(news => {
            let contentClass = '';
            // 根据内容中是否含有特定关键词添加样式
            if (news.content && (news.content.includes('增长') || news.content.includes('上涨') || news.content.includes('利好'))) {
                contentClass = 'text-success';
            } else if (news.content && (news.content.includes('下跌') || news.content.includes('下降') || news.content.includes('利空'))) {
                contentClass = 'text-danger';
            }

            const content = escapeHtml(news.content || '');

            timelineHtml += `
                <div class="news-item">
                    <div class="news-content ${contentClass}">${content}</div>
                </div>
            `;
        });

        timelineHtml += `
                </div>
            </div>
        `;
    });

    timelineHtml += '</div>';

    if (isMobileView() && list.length > limitForMobile) {
        timelineHtml += `<div class="news-expand-wrap"><button type="button" class="news-expand-btn" id="news-expand-toggle">${isNewsExpanded ? '收起' : '展开查看全部新闻'}</button></div>`;
    }

    $('#news-timeline').html(timelineHtml);
}

function formatDate(dateStr) {
    const today = new Date();
    const todayStr = today.toISOString().split('T')[0];

    if (dateStr === todayStr) {
        return '';
    }

    const yesterday = new Date(today);
    yesterday.setDate(yesterday.getDate() - 1);
    const yesterdayStr = yesterday.toISOString().split('T')[0];

    if (dateStr === yesterdayStr) {
        return '昨天';
    }

    const date = new Date(dateStr);
    return `${date.getMonth() + 1}月${date.getDate()}日`;
}

// 显示滚动新闻
function displayTickerNews(newsList) {
    const wrapper = $('#ticker-container .ticker-wrapper');
    const list = Array.isArray(newsList) ? newsList : [];

    if (list.length === 0) {
        wrapper.html('<div class="ticker-item">暂无最新消息</div>');
        tickerSignature = '';
        try {
            if (tickerAnimation) {
                tickerAnimation.cancel();
            }
        } catch (e) {}
        tickerAnimation = null;
        try {
            if (tickerRafId) {
                cancelAnimationFrame(tickerRafId);
            }
        } catch (e) {}
        tickerRafId = 0;
        return;
    }

    const baseItems = list.slice(0, 3).map(function (news) {
        return escapeHtml((news && (news.content || news.title)) || '');
    }).filter(Boolean);

    if (baseItems.length === 0) {
        wrapper.html('<div class="ticker-item">暂无最新消息</div>');
        return;
    }

    const nextSignature = baseItems.join('||');
    if (nextSignature === tickerSignature && (tickerAnimation || tickerRafId)) {
        return;
    }
    tickerSignature = nextSignature;

    try {
        if (tickerAnimation) {
            tickerAnimation.cancel();
        }
    } catch (e) {}
    tickerAnimation = null;
    try {
        if (tickerRafId) {
            cancelAnimationFrame(tickerRafId);
        }
    } catch (e) {}
    tickerRafId = 0;

    let tickerItems = '';
    baseItems.forEach(function (text) {
        tickerItems += `<div class="ticker-item" data-cycle="1">${text}</div>`;
    });
    // Large gap between 3rd item and restart
    tickerItems += `<div class="ticker-loop-gap" aria-hidden="true"></div>`;
    baseItems.forEach(function (text) {
        tickerItems += `<div class="ticker-item" data-cycle="2">${text}</div>`;
    });

    wrapper.html(tickerItems);
    wrapper.css('animation', 'none');

    const el = wrapper[0];
    const container = document.getElementById('ticker-container');
    if (!el || !container) {
        return;
    }

    requestAnimationFrame(function () {
        const startX = 0;

        const first1 = el.querySelector('.ticker-item[data-cycle="1"]');
        const first2 = el.querySelector('.ticker-item[data-cycle="2"]');
        let cycleWidth = 0;
        if (first1 && first2) {
            cycleWidth = Math.max(0, first2.offsetLeft - first1.offsetLeft);
        }
        if (!cycleWidth) {
            cycleWidth = Math.floor((el.scrollWidth || 0) / 2);
        }

        const pxPerSecond = isMobileView() ? 130 : 75;
        const durationMs = Math.max(6000, Math.round((cycleWidth / pxPerSecond) * 1000));

        el.style.transform = `translateX(${startX}px)`;

        if (typeof el.animate === 'function') {
            try {
                tickerAnimation = el.animate(
                    [
                        { transform: `translateX(${startX}px)` },
                        { transform: `translateX(${startX - cycleWidth}px)` },
                    ],
                    {
                        duration: durationMs,
                        iterations: Infinity,
                        easing: 'linear',
                    }
                );
                return;
            } catch (e) {
                tickerAnimation = null;
            }
        }

        const startTs = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
        const tick = function (nowTs) {
            const now = (typeof nowTs === 'number') ? nowTs : ((typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now());
            const elapsed = Math.max(0, now - startTs);
            const t = durationMs > 0 ? (elapsed % durationMs) : 0;
            const p = durationMs > 0 ? (t / durationMs) : 0;
            const x = startX - (p * cycleWidth);
            el.style.transform = `translateX(${x}px)`;
            tickerRafId = requestAnimationFrame(tick);
        };
        tickerRafId = requestAnimationFrame(tick);
    });
}

// 启动滚动新闻
function startTickerNews() {
    $.ajax({
        url: '/api/latest_news',
        method: 'GET',
        data: {
            days: 1,
            limit: 3
        },
        success: function(response) {
            if (response.success && response.news && response.news.length > 0) {
                displayTickerNews(response.news);
            } else {
                $('#ticker-container .ticker-wrapper').html('<div class="ticker-item">暂无最新消息</div>');
            }
        },
        error: function(err) {
            console.error('获取滚动新闻失败:', err);
            $('#ticker-container .ticker-wrapper').html('<div class="ticker-item">获取最新消息失败</div>');
        }
    });
}