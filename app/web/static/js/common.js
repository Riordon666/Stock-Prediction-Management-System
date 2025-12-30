(function () {
    var THEME_KEY = 'theme';
    var toggle = document.getElementById('input');
    if (!toggle) return;

    var lastPointer = null;

    function recordPointer(e) {
        try {
            if (e && typeof e.clientX === 'number' && typeof e.clientY === 'number') {
                lastPointer = { x: e.clientX, y: e.clientY };
                return;
            }

            if (e && e.touches && e.touches[0]) {
                lastPointer = { x: e.touches[0].clientX, y: e.touches[0].clientY };
            }
        } catch (err) {}
    }

    (function bindPointerCapture() {
        var el = toggle.closest('.switch') || toggle;
        try { el.addEventListener('pointerdown', recordPointer, { passive: true }); } catch (e) {}
        try { el.addEventListener('touchstart', recordPointer, { passive: true }); } catch (e) {}
        try { el.addEventListener('mousedown', recordPointer, { passive: true }); } catch (e) {}
    })();

    function getInitialTheme() {
        try {
            var saved = localStorage.getItem(THEME_KEY);
            if (saved === 'light' || saved === 'dark') return saved;
        } catch (e) {}
        return 'dark';
    }

    function applyTheme(theme) {
        var isLight = theme === 'light';
        document.documentElement.classList.toggle('light', isLight);
        document.body.classList.toggle('light', isLight);
        try {
            document.documentElement.setAttribute('data-theme', isLight ? 'light' : 'dark');
        } catch (e) {}
        toggle.checked = !isLight;
        try {
            localStorage.setItem(THEME_KEY, theme);
        } catch (e) {}
        try { ensureMatrixBackground(); } catch (e) {}
    }

    function ensureMatrixBackground() {
        try {
            if (!document.body) return;

            var root = document.querySelector('.matrix-container');
            if (!root) {
                root = document.createElement('div');
                root.className = 'matrix-container';
                root.setAttribute('aria-hidden', 'true');

                var patterns = 5;
                var cols = 40;
                for (var p = 0; p < patterns; p++) {
                    var pattern = document.createElement('div');
                    pattern.className = 'matrix-pattern';
                    for (var i = 0; i < cols; i++) {
                        var col = document.createElement('div');
                        col.className = 'matrix-column';
                        col.style.left = (i * 25) + 'px';
                        col.style.animationDelay = (-(1.5 + Math.random() * 2.6)).toFixed(2) + 's';
                        col.style.animationDuration = (2.3 + Math.random() * 2.2).toFixed(2) + 's';

                        var pick = (i % 5) + 1;
                        col.classList.add('v' + pick);
                        pattern.appendChild(col);
                    }
                    root.appendChild(pattern);
                }

                document.body.insertBefore(root, document.body.firstChild);
            }

            try { root.style.removeProperty('display'); } catch (e) {}
        } catch (e) {}
    }

    var currentTheme = getInitialTheme();
    var isAnimating = false;
    applyTheme(currentTheme);
    ensureMatrixBackground();

    function prefersReducedMotion() {
        return !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
    }

    function getSwitchCenter() {
        var el = toggle.closest('.switch') || toggle;
        var rect = el.getBoundingClientRect();
        return {
            x: rect.left + rect.width / 2,
            y: rect.top + rect.height / 2
        };
    }

    function fallbackThemeTransition(nextTheme, done) {
        if (prefersReducedMotion()) {
            applyTheme(nextTheme);
            currentTheme = nextTheme;
            done();
            return;
        }

        var p;
        try {
            p = lastPointer || getSwitchCenter();
        } catch (e) {
            p = { x: window.innerWidth / 2, y: 24 };
        }

        var oldTheme = currentTheme;

        applyTheme(nextTheme);
        currentTheme = nextTheme;

        var overlay = document.createElement('div');
        overlay.className = 'theme-transition-overlay';
        overlay.setAttribute('aria-hidden', 'true');

        if (oldTheme === 'light') {
            overlay.classList.add('is-light');

            var grid = document.createElement('div');
            grid.className = 'theme-transition-overlay-grid';
            overlay.appendChild(grid);
        } else {
            overlay.classList.add('is-dark');
        }

        try {
            document.body.appendChild(overlay);
        } catch (e) {
            done();
            return;
        }

        var dx = Math.max(p.x, window.innerWidth - p.x);
        var dy = Math.max(p.y, window.innerHeight - p.y);
        var endRadius = Math.ceil(Math.sqrt(dx * dx + dy * dy)) + 30;

        var from = 'circle(' + endRadius + 'px at ' + p.x + 'px ' + p.y + 'px)';
        var to = 'circle(0px at ' + p.x + 'px ' + p.y + 'px)';

        var anim;
        try {
            anim = overlay.animate(
                { clipPath: [from, to] },
                {
                    duration: 520,
                    easing: 'cubic-bezier(0.2, 0.9, 0.2, 1)',
                    fill: 'both'
                }
            );
        } catch (e) {
            try { overlay.remove(); } catch (e2) {}
            done();
            return;
        }

        anim.finished.then(function () {
            try { overlay.remove(); } catch (e) {}
            done();
        }, function () {
            try { overlay.remove(); } catch (e) {}
            done();
        });
    }

    function transitionTheme(nextTheme, done) {
        if (prefersReducedMotion()) {
            applyTheme(nextTheme);
            currentTheme = nextTheme;
            done();
            return;
        }

        if (!document.startViewTransition) {
            fallbackThemeTransition(nextTheme, done);
            return;
        }

        var pOld = lastPointer || getSwitchCenter();
        var pNew;

        var expand = currentTheme === 'dark' && nextTheme === 'light';
        var pseudo = expand ? '::view-transition-new(root)' : '::view-transition-old(root)';

        var zVarsApplied = false;
        if (!expand) {
            zVarsApplied = true;
            document.documentElement.style.setProperty('--vt-old-z', '2');
            document.documentElement.style.setProperty('--vt-new-z', '1');
        }

        var vt;
        try {
            vt = document.startViewTransition(function () {
                applyTheme(nextTheme);
                try {
                    pNew = getSwitchCenter();
                } catch (e) {}
            });
            currentTheme = nextTheme;
        } catch (e) {
            fallbackThemeTransition(nextTheme, done);
            return;
        }

        vt.ready.then(function () {
            var p = (expand ? (lastPointer || pNew || pOld) : (lastPointer || pOld));
            lastPointer = null;

            var dx = Math.max(p.x, window.innerWidth - p.x);
            var dy = Math.max(p.y, window.innerHeight - p.y);
            var endRadius = Math.ceil(Math.sqrt(dx * dx + dy * dy)) + 30;

            var clipFrom = expand
                ? 'circle(0px at ' + p.x + 'px ' + p.y + 'px)'
                : 'circle(' + endRadius + 'px at ' + p.x + 'px ' + p.y + 'px)';
            var clipTo = expand
                ? 'circle(' + endRadius + 'px at ' + p.x + 'px ' + p.y + 'px)'
                : 'circle(0px at ' + p.x + 'px ' + p.y + 'px)';

            document.documentElement.animate(
                { clipPath: [clipFrom, clipTo] },
                {
                    duration: 520,
                    easing: 'cubic-bezier(0.2, 0.9, 0.2, 1)',
                    pseudoElement: pseudo,
                    fill: 'both'
                }
            );
        });

        vt.finished.then(function () {
            if (zVarsApplied) {
                document.documentElement.style.removeProperty('--vt-old-z');
                document.documentElement.style.removeProperty('--vt-new-z');
            }
            done();
        }, function () {
            if (zVarsApplied) {
                document.documentElement.style.removeProperty('--vt-old-z');
                document.documentElement.style.removeProperty('--vt-new-z');
            }
            done();
        });
    }

    toggle.addEventListener('change', function () {
        if (isAnimating) {
            applyTheme(currentTheme);
            return;
        }

        var nextTheme = toggle.checked ? 'dark' : 'light';
        if (nextTheme === currentTheme) return;

        isAnimating = true;
        toggle.disabled = true;

        transitionTheme(nextTheme, function () {
            toggle.disabled = false;
            isAnimating = false;
            try { ensureMatrixBackground(); } catch (e) {}
            try {
                document.dispatchEvent(new CustomEvent('theme-changed', { detail: { theme: nextTheme } }));
            } catch (e) {}
        });
    });
})();
