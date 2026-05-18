/**
 * Dark Shader System - Black Hole Single-Pass
 * THREE.ShaderMaterial + glslVersion: THREE.GLSL3
 * Single Image pass, no Buffer targets
 */
;(function () {
    'use strict';

    var _inited = false, _active = false, _renderer = null, _canvas = null;
    var _scene = null, _camera = null, _material = null;
    var _currentOpacity = 0, _targetOpacity = 0;
    var _animId = null;

    var QUAD_VS = [
        'out vec2 vUv;',
        'void main() {',
        '    vUv = uv;',
        '    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);',
        '}'
    ].join('\n');

    var BLACKHOLE_FS = [
        'uniform vec3 iResolution;',
        'uniform float iTime;',
        'uniform float opacity;',
        'in vec2 vUv;',
        'out vec4 fragColor;',
        '',
        'void main() {',
        '    vec2 F = vUv * iResolution.xy;',
        '    float i = 0.2, a;',
        '    vec2 r = iResolution.xy,',
        '         p = (F+F - r) / r.y / 0.7,',
        '         d = vec2(-1.0, 1.0),',
        '         b = p - i*d;',
        '    float bv = dot(b, b);',
        '    vec2 dv = d / (0.1 + i / bv);',
        '    vec2 c = p * mat2(1.0, dv.x, 1.0, dv.y);',
        '    a = dot(c, c);',
        '    vec2 v = c * mat2(cos(0.5*log(a) + iTime*i + vec4(0.0, 33.0, 11.0, 0.0))) / i;',
        '    vec2 w = vec2(0.0);',
        '    for(; i++ < 9.0; w += 1.0 + sin(v))',
        '        v += 0.7 * sin(v.yx*i + iTime) / i + 0.5;',
        '    i = length(sin(v/0.3)*0.4 + c*(3.0 + d));',
        '    vec4 O = 1.0 - exp(-exp(c.x * vec4(0.6, -0.4, -1.0, 0.0))',
        '                   / w.xyyx',
        '                   / (2.0 + i*i/4.0 - i)',
        '                   / (0.5 + 1.0/a)',
        '                   / (0.03 + abs(length(p) - 0.7)));',
        '    O.a = opacity;',
        '    fragColor = O;',
        '}'
    ].join('\n');

    function init() {
        if (_inited) return;
        try {
            _canvas = document.createElement('canvas');
            _canvas.className = 'shader-bg-canvas';
            _canvas.setAttribute('aria-hidden', 'true');

            _renderer = new THREE.WebGLRenderer({ canvas: _canvas, alpha: true, antialias: false, powerPreference: 'low-power', failIfMajorPerformanceCaveat: false });
            var gl = _renderer.getContext();
            if (!gl || !(gl instanceof WebGL2RenderingContext)) { console.error('[DarkShader] Requires WebGL2'); if (_renderer) _renderer.dispose(); _renderer = null; return; }

            _renderer.setPixelRatio(Math.min(window.devicePixelRatio, 0.5));
            _renderer.setSize(window.innerWidth, window.innerHeight);
            _renderer.setClearColor(0x000000, 0);

            _camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

            _material = new THREE.ShaderMaterial({
                vertexShader: QUAD_VS,
                fragmentShader: BLACKHOLE_FS,
                glslVersion: THREE.GLSL3,
                uniforms: {
                    iResolution: { value: new THREE.Vector3(window.innerWidth, window.innerHeight, 0) },
                    iTime: { value: 0.0 },
                    opacity: { value: 0.0 }
                },
                transparent: true,
                depthTest: false,
                depthWrite: false
            });

            _scene = new THREE.Scene();
            _scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), _material));
            _inited = true;
            console.log('[DarkShader] Init OK');
        } catch (e) { console.error('[DarkShader] Init failed:', e); dispose(); }
    }

    function renderLoop() {
        if (!_active || !_renderer) return;
        _animId = requestAnimationFrame(renderLoop);
        try {
            var diff = _targetOpacity - _currentOpacity;
            if (Math.abs(diff) > 0.005) _currentOpacity += diff * 0.08;
            else _currentOpacity = _targetOpacity;

            _material.uniforms.iTime.value = performance.now() / 1000.0;
            _material.uniforms.opacity.value = _currentOpacity;
            _renderer.setRenderTarget(null);
            _renderer.render(_scene, _camera);

            if (_targetOpacity <= 0.01 && _currentOpacity <= 0.01) pauseLoop();
        } catch (e) { console.error('[DarkShader] Render err:', e); pauseLoop(); }
    }

    function pauseLoop() { if (_animId) { cancelAnimationFrame(_animId); _animId = null; } }
    function resumeLoop() { if (!_animId && _active && _renderer) renderLoop(); }

    function activate() {
        if (!_inited || _active) return;
        _active = true; _targetOpacity = 1.0; _currentOpacity = 0;
        if (!_canvas.parentNode) document.body.insertBefore(_canvas, document.body.firstChild);
        _canvas.classList.add('shader-bg-visible');
        document.documentElement.classList.add('shader-active');
        resumeLoop();
    }

    function deactivate() {
        if (!_active) return;
        _active = false; _targetOpacity = 0;
        if (_canvas) _canvas.classList.remove('shader-bg-visible');
    }

    function resize(w, h) {
        if (!_renderer) return;
        _renderer.setSize(w, h);
        if (_material) _material.uniforms.iResolution.value.set(w, h, 0);
    }

    function dispose() {
        pauseLoop();
        if (_material) { _material.dispose(); _material = null; }
        if (_renderer) { _renderer.dispose(); _renderer = null; }
        if (_canvas && _canvas.parentNode) _canvas.parentNode.removeChild(_canvas);
        _canvas = null; _scene = null; _camera = null;
        _currentOpacity = 0; _targetOpacity = 0;
        _inited = false; _active = false;
        console.log('[DarkShader] Disposed');
    }

    window.DarkShader = {
        init: init,
        activate: activate,
        deactivate: deactivate,
        dispose: dispose,
        resize: resize,
        isActive: function () { return _active; },
        isInitialized: function () { return _inited; },
        getOpacity: function () { return _currentOpacity; }
    };
})();