(function () {
    let chart = null;
    let lastRenderData = null;

    function isDarkMode() {
        try {
            return (document.documentElement.getAttribute('data-theme') || '') !== 'light';
        } catch (e) {
            return true;
        }
    }

    function getThemeOptions() {
        const dark = isDarkMode();
        const textColor = dark ? '#e5e7eb' : '#111827';
        const gridColor = dark ? 'rgba(255,255,255,0.12)' : 'rgba(17,24,39,0.12)';
        return {
            theme: { mode: dark ? 'dark' : 'light' },
            chart: { background: 'transparent', foreColor: textColor },
            grid: { borderColor: gridColor },
            xaxis: {
                axisBorder: { color: gridColor },
                axisTicks: { color: gridColor },
                labels: { style: { colors: textColor } }
            },
            yaxis: {
                labels: { style: { colors: textColor } }
            },
            tooltip: { theme: dark ? 'dark' : 'light' }
        };
    }

    function destroyChart() {
        try {
            if (chart && typeof chart.destroy === 'function') {
                chart.destroy();
            }
        } catch (e) {
        }
        chart = null;
    }

    function showError(msg) {
        $('#predict-error').show().text(msg || '');
    }

    function clearError() {
        $('#predict-error').hide().text('');
    }

    function setLoading(loading) {
        $('.predict-submit').prop('disabled', !!loading);
        if (loading) {
            $('#predict-result').hide();
            $('#predict-loading').css('display', 'flex');
        } else {
            $('#predict-loading').css('display', 'none');
        }
    }

    function formatNumber(value, decimals) {
        const n = Number(value);
        if (!Number.isFinite(n)) return '--';
        return n.toFixed(decimals == null ? 2 : decimals);
    }

    function currencyPrefix(marketType) {
        const mt = String(marketType || 'A').toUpperCase();
        if (mt === 'US') return '$';
        if (mt === 'HK') return 'HK$';
        return '¥';
    }

    function buildSeries(history, forecast) {
        const histDates = (history || []).map(x => x.date);
        const futureDates = (forecast || []).map(x => x.date);
        const categories = histDates.concat(futureDates);

        const histClose = (history || []).map(x => Number(x.close));
        const futureClose = (forecast || []).map(x => Number(x.close));

        const historySeries = histClose.concat(new Array(futureClose.length).fill(null));

        const lastClose = histClose.length ? histClose[histClose.length - 1] : null;
        const forecastSeries = new Array(Math.max(0, histClose.length - 1)).fill(null)
            .concat(lastClose == null ? [null] : [lastClose])
            .concat(futureClose);

        return { categories, historySeries, forecastSeries };
    }

    function renderChart(data) {
        if (!data) return;
        lastRenderData = data;

        const history = data.history || [];
        const forecast = data.forecast || [];
        if (!history.length || !forecast.length) {
            showError('预测数据为空');
            return;
        }

        const boundaryDate = data.boundary_date || (history[history.length - 1] && history[history.length - 1].date);
        const seriesPack = buildSeries(history, forecast);

        destroyChart();
        $('#predict-chart').empty();

        const mt = String(data.market_type || 'A').toUpperCase();
        const prefix = currencyPrefix(mt);

        const baseOptions = {
            series: [
                { name: '历史(最近30天)', data: seriesPack.historySeries },
                { name: '预测(未来)', data: seriesPack.forecastSeries }
            ],
            chart: {
                type: 'line',
                height: 420,
                toolbar: { show: true }
            },
            stroke: {
                width: [3, 3],
                curve: 'smooth',
                dashArray: [0, 6]
            },
            markers: {
                size: [0, 0],
                hover: { size: 4 }
            },
            dataLabels: { enabled: false },
            xaxis: {
                categories: seriesPack.categories,
                tickPlacement: 'on',
                labels: { rotate: -45 }
            },
            yaxis: {
                labels: {
                    formatter: function (v) {
                        return prefix + formatNumber(v, 2);
                    }
                }
            },
            legend: {
                position: 'top',
                horizontalAlign: 'center'
            },
            tooltip: {
                shared: true,
                intersect: false,
                y: {
                    formatter: function (v) {
                        return prefix + formatNumber(v, 2);
                    }
                }
            },
            annotations: {
                xaxis: [
                    {
                        x: boundaryDate,
                        borderColor: 'rgba(110, 168, 255, 0.95)',
                        strokeDashArray: 0,
                        label: {
                            borderColor: 'rgba(110, 168, 255, 0.95)',
                            style: {
                                color: isDarkMode() ? '#0b1220' : '#ffffff',
                                background: 'rgba(110, 168, 255, 0.95)'
                            },
                            text: '历史/预测分界'
                        }
                    }
                ]
            }
        };

        const finalOptions = $.extend(true, {}, baseOptions, getThemeOptions());
        chart = new ApexCharts(document.querySelector('#predict-chart'), finalOptions);
        chart.render();

        try {
            const lastReal = history[history.length - 1];
            const firstPred = forecast[0];
            $('#predict-summary').text(
                '最近收盘：' + prefix + formatNumber(lastReal && lastReal.close, 2) +
                '；预测第1天：' + prefix + formatNumber(firstPred && firstPred.close, 2)
            );
        } catch (e) {
            $('#predict-summary').text('');
        }
    }

    function rerenderIfNeeded() {
        if (!lastRenderData) return;
        if (typeof ApexCharts === 'undefined') return;
        try {
            renderChart(lastRenderData);
        } catch (e) {
        }
    }

    function bindThemeObserver() {
        try {
            const el = document.documentElement;
            const observer = new MutationObserver(function (mutations) {
                for (let i = 0; i < mutations.length; i++) {
                    const m = mutations[i];
                    if (m.type === 'attributes' && m.attributeName === 'data-theme') {
                        setTimeout(function () {
                            rerenderIfNeeded();
                        }, 650);
                        return;
                    }
                }
            });
            observer.observe(el, { attributes: true, attributeFilter: ['data-theme'] });
        } catch (e) {
        }

        try {
            document.addEventListener('theme-changed', function () {
                setTimeout(function () {
                    rerenderIfNeeded();
                }, 650);
            });
        } catch (e) {
        }
    }

    function requestPredict(stockCode, marketType, days) {
        clearError();
        setLoading(true);

        $.ajax({
            url: '/api/predict_gru',
            type: 'GET',
            dataType: 'json',
            data: {
                stock_code: stockCode,
                market_type: marketType,
                days: days
            },
            success: function (resp) {
                setLoading(false);
                if (!resp || !resp.success) {
                    showError((resp && resp.error) ? resp.error : '预测失败');
                    return;
                }
                $('#predict-result').show();
                renderChart(resp);
            },
            error: function (xhr, status, error) {
                setLoading(false);
                let msg = '预测失败';
                if (xhr && xhr.responseJSON && xhr.responseJSON.error) {
                    msg += ': ' + xhr.responseJSON.error;
                } else if (error) {
                    msg += ': ' + error;
                }
                showError(msg);
            }
        });
    }

    $(function () {
        bindThemeObserver();

        $('#predict-form').on('submit', function (e) {
            e.preventDefault();
            const stockCode = ($('#predict-stock-code').val() || '').trim();
            const marketType = ($('#predict-market-type').val() || 'A').trim();
            const days = Number($('#predict-days').val() || 10);

            if (!stockCode) {
                showError('请输入股票代码！');
                return;
            }
            if (![10, 20, 30].includes(days)) {
                showError('预测天数只支持10/20/30');
                return;
            }
            if (typeof ApexCharts === 'undefined') {
                showError('图表库加载失败（ApexCharts 未加载）');
                return;
            }
            requestPredict(stockCode, marketType, days);
        });
    });
})();
