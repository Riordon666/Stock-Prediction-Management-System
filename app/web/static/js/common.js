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
    }

    var currentTheme = getInitialTheme();
    var isAnimating = false;
    applyTheme(currentTheme);

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

    function transitionTheme(nextTheme, done) {
        if (prefersReducedMotion() || !document.startViewTransition) {
            applyTheme(nextTheme);
            currentTheme = nextTheme;
            done();
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
            applyTheme(nextTheme);
            currentTheme = nextTheme;
            done();
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
            try {
                document.dispatchEvent(new CustomEvent('theme-changed', { detail: { theme: nextTheme } }));
            } catch (e) {}
        });
    });
})();
