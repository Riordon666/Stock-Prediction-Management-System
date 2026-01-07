(function () {
    'use strict';

    function $(sel, root) {
        return (root || document).querySelector(sel);
    }

    function $all(sel, root) {
        return Array.prototype.slice.call((root || document).querySelectorAll(sel));
    }

    function safeText(v) {
        if (v == null) return '';
        return String(v);
    }

    function formatTime(item) {
        var d = safeText(item.date);
        var t = safeText(item.time);
        if (t && t.length >= 5) t = t.slice(0, 5);
        if (d && t) return d + ' ' + t;
        return d || t || '';
    }

    function renderIntoPlaceholders(placeholders, news) {
        if (!placeholders || placeholders.length === 0) return;

        var count = Math.min(placeholders.length, news.length);
        for (var i = 0; i < count; i++) {
            var el = placeholders[i];
            el.classList.remove('placeholder');
            el.classList.add('news-item');
            el.innerHTML = '';

            var meta = document.createElement('div');
            meta.className = 'news-meta';

            var time = document.createElement('span');
            time.className = 'news-time';
            time.textContent = formatTime(news[i]);

            var source = document.createElement('span');
            source.className = 'news-source';
            source.textContent = '财联社';

            meta.appendChild(time);
            meta.appendChild(source);

            var content = document.createElement('div');
            content.className = 'news-text';
            content.textContent = safeText(news[i].content || news[i].title || '');

            el.appendChild(meta);
            el.appendChild(content);
        }

        for (var j = count; j < placeholders.length; j++) {
            placeholders[j].textContent = '暂无更多快讯';
        }
    }

    function setLoading(placeholders) {
        placeholders.forEach(function (p) {
            p.textContent = '加载中...';
        });
    }

    function setError(placeholders) {
        placeholders.forEach(function (p) {
            p.textContent = '获取快讯失败，请稍后重试';
        });
    }

    function fetchLatestNews(limit) {
        var url = '/api/latest_news?days=1&limit=' + encodeURIComponent(String(limit || 10));
        return fetch(url, { credentials: 'same-origin' })
            .then(function (r) { return r.json(); })
            .then(function (data) {
                if (!data || !data.success) {
                    throw new Error((data && data.error) ? data.error : 'api_failed');
                }
                return data.news || [];
            });
    }

    function initHomeNews() {
        var homeCard = $('#home .card');
        if (!homeCard) return;

        var placeholders = $all('#home .card .placeholder');
        if (placeholders.length === 0) return;

        setLoading(placeholders);

        fetchLatestNews(placeholders.length)
            .then(function (news) {
                if (!news || news.length === 0) {
                    placeholders.forEach(function (p) {
                        p.textContent = '暂无最新快讯';
                    });
                    return;
                }
                renderIntoPlaceholders(placeholders, news);
            })
            .catch(function (e) {
                console.error(e);
                setError(placeholders);
            });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initHomeNews);
    } else {
        initHomeNews();
    }
})();
