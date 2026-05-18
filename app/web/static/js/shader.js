/**
 * Shader Background Controller
 * 
 * Responsibilities:
 *   - Theme monitoring (listen for theme-changed events)
 *   - Lazy load darkShader.js ONLY when theme === 'dark'
 *   - Coordinate activate/deactivate of light & dark shader systems
 *   - Handle resize & visibility events
 *   - Script loading (Three.js, sunset.js, darkShader.js)
 * 
 * Does NOT contain any shader code, render logic, or WebGL management.
 * Each system (SunsetShader, DarkShader) is fully self-contained.
 */
;(function () {
    'use strict';

    var _activeTheme = null;    // 'light' | 'dark' | null
    var _darkLoaded = false;    // whether darkShader.js has been loaded
    var _lightLoaded = false;   // whether sunset.js has been loaded
    var _threeLoaded = false;   // whether Three.js is available

    /* ══════════════════════════════════════════════════════
       SCRIPT LOADING
       ══════════════════════════════════════════════════════ */

    function hasThreeJS() {
        return typeof THREE !== 'undefined' && THREE.WebGLRenderer;
    }

    var _cacheBust = '?v=' + Date.now();

    function loadScript(src) {
        return new Promise(function (resolve, reject) {
            var s = document.createElement('script');
            s.src = src + _cacheBust;
            s.onload = function () { resolve(); };
            s.onerror = function () { reject(new Error('Failed to load: ' + src)); };
            document.head.appendChild(s);
        });
    }

    function ensureThreeJS() {
        if (hasThreeJS()) {
            _threeLoaded = true;
            return Promise.resolve();
        }
        return loadScript('/static/js/three.min.js').then(function () {
            _threeLoaded = true;
            console.log('[ShaderBG] Three.js loaded, v' + THREE.REVISION);
        });
    }

    function ensureLightShader() {
        if (_lightLoaded && typeof SunsetShader !== 'undefined') return Promise.resolve();
        return ensureThreeJS().then(function () {
            return loadScript('/static/js/sunset.js');
        }).then(function () {
            _lightLoaded = true;
            console.log('[ShaderBG] SunsetShader loaded');
        });
    }

    /**
     * Lazy load darkShader.js - ONLY called when theme === 'dark'.
     * This is the key optimization: dark shader is never loaded
     * during initial page load for light theme.
     */
    function ensureDarkShader() {
        if (_darkLoaded && typeof DarkShader !== 'undefined') return Promise.resolve();
        return ensureThreeJS().then(function () {
            return loadScript('/static/js/darkShader.js');
        }).then(function () {
            _darkLoaded = true;
            console.log('[ShaderBG] DarkShader lazy-loaded');
        });
    }

    /* ══════════════════════════════════════════════════════
       THEME DETECTION
       ══════════════════════════════════════════════════════ */

    function detectTheme() {
        return document.documentElement.classList.contains('light') ||
            document.documentElement.getAttribute('data-theme') === 'light'
            ? 'light' : 'dark';
    }

    /* ══════════════════════════════════════════════════════
       ACTIVATE / DEACTIVATE
       ══════════════════════════════════════════════════════ */

    function activateLight() {
        ensureLightShader().then(function () {
            if (typeof SunsetShader === 'undefined') {
                console.error('[ShaderBG] SunsetShader not available');
                return;
            }
            if (!SunsetShader.isInitialized()) SunsetShader.init();
            SunsetShader.activate();
            _activeTheme = 'light';
            console.log('[ShaderBG] Light theme active');
        }).catch(function (e) {
            console.error('[ShaderBG] Failed to activate light:', e);
        });
    }

    function activateDark() {
        ensureDarkShader().then(function () {
            if (typeof DarkShader === 'undefined') {
                console.error('[ShaderBG] DarkShader not available');
                return;
            }
            if (!DarkShader.isInitialized()) {
                DarkShader.init();
            }
            DarkShader.activate();
            _activeTheme = 'dark';
            console.log('[ShaderBG] Dark theme active');
        }).catch(function (e) {
            console.error('[ShaderBG] Failed to activate dark:', e);
        });
    }

    function deactivateAll() {
        if (typeof SunsetShader !== 'undefined' && SunsetShader.isActive()) {
            SunsetShader.deactivate();
        }
        if (typeof DarkShader !== 'undefined' && DarkShader.isActive()) {
            DarkShader.deactivate();
        }
        // Fully dispose the dark shader to free WebGL context
        if (_darkLoaded && typeof DarkShader !== 'undefined' && DarkShader.isInitialized()) {
            setTimeout(function () {
                DarkShader.dispose();
                console.log('[ShaderBG] DarkShader disposed');
            }, 1200); // Wait for fade-out animation
        }
        _activeTheme = null;
    }

    /* ══════════════════════════════════════════════════════
       THEME SWITCHING
       ══════════════════════════════════════════════════════ */

    function switchTheme(newTheme) {
        if (_activeTheme === newTheme) return;

        console.log('[ShaderBG] Switching: ' + (_activeTheme || 'none') + ' → ' + newTheme);

        // Deactivate and dispose current system IMMEDIATELY to free WebGL context
        var disposeDelay = 0;
        if (_activeTheme === 'light' && typeof SunsetShader !== 'undefined') {
            SunsetShader.deactivate();
            SunsetShader.dispose();
            _lightLoaded = false;
            disposeDelay = 1000; // Give browser time to fully release WebGL context
            console.log('[ShaderBG] SunsetShader disposed');
        }
        if (_activeTheme === 'dark' && typeof DarkShader !== 'undefined') {
            DarkShader.deactivate();
            DarkShader.dispose();
            _darkLoaded = false;
            disposeDelay = 1000;
            console.log('[ShaderBG] DarkShader disposed');
        }

        // Activate new system after WebGL context is fully released
        setTimeout(function () {
            if (newTheme === 'light') {
                activateLight();
            } else {
                activateDark();
            }
        }, disposeDelay);
    }

    /* ══════════════════════════════════════════════════════
       EVENT HANDLERS
       ══════════════════════════════════════════════════════ */

    // Theme change event
    document.addEventListener('theme-changed', function (e) {
        var newTheme = (e && e.detail && e.detail.theme) || detectTheme();
        // Show performance warning toast when switching to light theme
        if (newTheme === 'light' && _activeTheme !== 'light') {
            showThemeToast('浅色主题对设备性能要求较高');
        }
        switchTheme(newTheme);
    });

    function showThemeToast(message) {
        var toast = document.getElementById('theme-toast');
        if (!toast) {
            toast = document.createElement('div');
            toast.id = 'theme-toast';
            toast.className = 'theme-toast';
            toast.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg><span></span>';
            document.body.appendChild(toast);
        }
        toast.querySelector('span').textContent = message;
        toast.classList.add('show');
        setTimeout(function () {
            toast.classList.remove('show');
        }, 3000);
    }

    // Resize
    window.addEventListener('resize', function () {
        var w = window.innerWidth, h = window.innerHeight;
        if (typeof SunsetShader !== 'undefined' && SunsetShader.isActive()) {
            SunsetShader.resize(w, h);
        }
        if (typeof DarkShader !== 'undefined' && DarkShader.isActive()) {
            DarkShader.resize(w, h);
        }
    });

    // Visibility change - only pause/resume, don't deactivate/activate
    // (deactivate disposes resources which is wrong for tab switching)
    document.addEventListener('visibilitychange', function () {
        if (document.hidden) {
            // Just cancel animation frames to save CPU, don't dispose
            if (typeof SunsetShader !== 'undefined' && SunsetShader.isActive()) {
                SunsetShader.setOpacity(SunsetShader.getOpacity());
            }
            if (typeof DarkShader !== 'undefined' && DarkShader.isActive()) {
                DarkShader.setOpacity(DarkShader.getOpacity());
            }
        }
        // No need to re-activate on visibility restore - render loops
        // will naturally resume when opacity target > 0
    });

    /* ══════════════════════════════════════════════════════
       INITIAL STATE
       ══════════════════════════════════════════════════════ */

    function initIfReady() {
        var theme = detectTheme();
        console.log('[ShaderBG] Initial theme: ' + theme);

        if (theme === 'light') {
            // Light theme: load sunset.js and activate
            setTimeout(function () { activateLight(); }, 300);
        } else {
            // Dark theme: lazy load darkShader.js and activate
            setTimeout(function () { activateDark(); }, 300);
        }
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initIfReady);
    } else {
        initIfReady();
    }

    /* ══════════════════════════════════════════════════════
       DEBUG API
       ══════════════════════════════════════════════════════ */
    window.__shaderBG = {
        switchTheme: switchTheme,
        getState: function () {
            return {
                activeTheme: _activeTheme,
                darkLoaded: _darkLoaded,
                lightLoaded: _lightLoaded,
                threeLoaded: _threeLoaded,
                sunsetActive: typeof SunsetShader !== 'undefined' ? SunsetShader.isActive() : false,
                darkActive: typeof DarkShader !== 'undefined' ? DarkShader.isActive() : false
            };
        }
    };
})();