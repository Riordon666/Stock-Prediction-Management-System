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
        loadLatestNews();
        loadHotspots();
        startTickerNews();
        refreshCountdown = 300;
        if ($('#refresh-time').length) {
            $('#refresh-time').text('刷新倒计时: 5:00');
        }
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

    // 定时刷新
    setInterval(function() {
        updateMarketStatus();
        loadLatestNews(true); // 静默刷新
        loadHotspots();
        startTickerNews();
        refreshCountdown = 300;
        if ($('#refresh-time').length) {
            $('#refresh-time').text('刷新倒计时: 5:00');
        }
    }, 300000); // 5分钟

    setInterval(function() {
        updateCurrentTime();
        updateRefreshCountdown();
    }, 1000);
});

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
let lastNewsList = [];
let isNewsExpanded = false;

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

    // 更新DOM
    $('#news-timeline').html(timelineHtml);
}

// 日期格式化辅助函数
function formatDate(dateStr) {
    // 检查是否与当天日期相同
    const today = new Date();
    const todayStr = today.toISOString().split('T')[0];

    if (dateStr === todayStr) {
        return '';
    }

    // 昨天
    const yesterday = new Date(today);
    yesterday.setDate(yesterday.getDate() - 1);
    const yesterdayStr = yesterday.toISOString().split('T')[0];

    if (dateStr === yesterdayStr) {
        return '昨天';
    }

    // 其他日期用中文格式
    const date = new Date(dateStr);
    return `${date.getMonth() + 1}月${date.getDate()}日`;
}

// 添加页面自动刷新功能
let refreshCountdown = 300; // 5分钟倒计时（秒）

// 更新市场状态
function updateMarketStatus() {
    const now = new Date();
    const hours = now.getHours();
    const minutes = now.getMinutes();
    const weekday = now.getDay(); // 0为周日，6为周六

    // 检查是否为工作日
    const isWeekend = weekday === 0 || weekday === 6;

    // 亚太市场时区
    // A股状态 (9:30-11:30, 13:00-15:00)
    let chinaStatus = { open: false, text: '未开市' };
    if (!isWeekend && ((hours === 9 && minutes >= 30) || hours === 10 || (hours === 11 && minutes <= 30) ||
        (hours >= 13 && hours < 15))) {
        chinaStatus = { open: true, text: '交易中' };
    }

    // 港股状态 (9:30-12:00, 13:00-16:00)
    let hkStatus = { open: false, text: '未开市' };
    if (!isWeekend && ((hours === 9 && minutes >= 30) || hours === 10 || hours === 11 ||
        (hours >= 13 && hours < 16))) {
        hkStatus = { open: true, text: '交易中' };
    }

    // 台股状态 (9:00-13:30)
    let taiwanStatus = { open: false, text: '未开市' };
    if (!isWeekend && ((hours === 9) || hours === 10 || hours === 11 || hours === 12 ||
        (hours === 13 && minutes <= 30))) {
        taiwanStatus = { open: true, text: '交易中' };
    }

    // 日本股市 (9:00-11:30, 12:30-15:00)
    let japanStatus = { open: false, text: '未开市' };
    if (!isWeekend && ((hours === 9) || hours === 10 || (hours === 11 && minutes <= 30) ||
        (hours === 12 && minutes >= 30) || hours === 13 || hours === 14)) {
        japanStatus = { open: true, text: '交易中' };
    }

    // 欧洲市场 - 需要调整时区，这里是基于欧洲夏令时(UTC+2)与北京时间(UTC+8)相差6小时计算
    // 英国股市 (伦敦，北京时间15:00-23:30)
    let ukStatus = { open: false, text: '未开市' };
    if (!isWeekend && ((hours >= 15 && hours < 23) || (hours === 23 && minutes <= 30))) {
        ukStatus = { open: true, text: '交易中' };
    }

    // 德国股市 (法兰克福，北京时间15:00-23:30)
    let germanStatus = { open: false, text: '未开市' };
    if (!isWeekend && ((hours >= 15 && hours < 23) || (hours === 23 && minutes <= 30))) {
        germanStatus = { open: true, text: '交易中' };
    }

    // 法国股市 (巴黎，北京时间15:00-23:30)
    let franceStatus = { open: false, text: '未开市' };
    if (!isWeekend && ((hours >= 15 && hours < 23) || (hours === 23 && minutes <= 30))) {
        franceStatus = { open: true, text: '交易中' };
    }

    // 美洲市场
    // 美股状态 (纽约，北京时间21:30-4:00)
    let usStatus = { open: false, text: '未开市' };
    if ((hours >= 21 && minutes >= 30) || hours >= 22 || hours < 4) {
        // 检查美股的工作日 (当北京时间是周六早上，美国还是周五)
        const usDay = hours < 12 ? (weekday === 6 ? 5 : weekday - 1) : weekday;
        if (usDay !== 0 && usDay !== 6) {
            usStatus = { open: true, text: '交易中' };
        }
    }

    // 纳斯达克与美股相同
    let nasdaqStatus = usStatus;

    // 巴西股市 (圣保罗，北京时间20:30-3:00)
    let brazilStatus = { open: false, text: '未开市' };
    if ((hours >= 20 && minutes >= 30) || hours >= 21 || hours < 3) {
        const brazilDay = hours < 12 ? (weekday === 6 ? 5 : weekday - 1) : weekday;
        if (brazilDay !== 0 && brazilDay !== 6) {
            brazilStatus = { open: true, text: '交易中' };
        }
    }

    // 更新DOM
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

// 更新市场状态UI
function updateMarketStatusUI(elementId, status) {
    const element = $(`#${elementId}`);
    const iconElement = element.find('i');
    const textElement = element.find('.status-text');

    if (status.open) {
        iconElement.removeClass('status-closed').addClass('status-open');
    } else {
        iconElement.removeClass('status-open').addClass('status-closed');
    }

    textElement.text(status.text);
}

// 更新倒计时
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

// 更新当前时间
function updateCurrentTime() {
    if ($('#current-time-value').length === 0) {
        return;
    }

    const now = new Date();
    const timeString = now.toLocaleTimeString('zh-CN', { hour12: false });
    $('#current-time-value').text(timeString);
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

// 显示滚动新闻
function displayTickerNews(newsList) {
    if (newsList.length === 0) {
        $('#ticker-container .ticker-wrapper').html('<div class="ticker-item">暂无最新消息</div>');
        return;
    }

    const wrapper = $('#ticker-container .ticker-wrapper');
    const baseItems = (newsList || []).slice(0, 12).map(function (news) {
        return escapeHtml((news && (news.content || news.title)) || '');
    }).filter(Boolean);

    if (baseItems.length === 0) {
        wrapper.html('<div class="ticker-item">暂无最新消息</div>');
        return;
    }

    const items = baseItems.concat(baseItems);
    let tickerItems = '';
    items.forEach(function (text) {
        tickerItems += `<div class="ticker-item">${text}</div>`;
    });

    wrapper.html(tickerItems);

    // 触发重新启动动画（避免连续刷新时动画不重置）
    wrapper.css('animation', 'none');
    void wrapper[0].offsetHeight;
    const duration = isMobileView() ? 20 : 36;
    wrapper.css('animation', 'ticker ' + duration + 's linear infinite');
}