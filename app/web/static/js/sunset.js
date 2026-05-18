/**
 * Light Shader System - Sunset Cloud Scene (WebGL Ping-Pong)
 * 
 * Self-contained module with own canvas, WebGL context, render loop.
 * Extracted verbatim from original shader.js - NO changes to shader logic.
 * 
 * Public API:
 *   SunsetShader.init()           → initialize renderer
 *   SunsetShader.activate()       → attach canvas, start render loop
 *   SunsetShader.deactivate()     → stop render loop, cleanup
 *   SunsetShader.resize(w, h)     → update resolution
 *   SunsetShader.isActive()       → boolean
 *   SunsetShader.setOpacity(v)    → set target opacity (0..1)
 */
;(function () {
    'use strict';

    var PIXEL_RATIO_CAP = 1.5;

    /* ── Internal State ── */
    var _inited = false;
    var _active = false;
    var _renderer = null;
    var _canvas = null;
    var _scene = null;
    var _camera = null;
    var _bufScene = null;
    var _bufCamera = null;
    var _rtA = null;
    var _rtB = null;
    var _pingPong = false;
    var _finalMat = null;
    var _bufMat = null;
    var _noiseTex = null;
    var _animId = null;
    var _startTime = 0;
    var _frameCount = 0;
    var _currentOpacity = 0;
    var _targetOpacity = 0;

    /* ══════════════════════════════════════════════════════
       SHADER SOURCES (verbatim from original)
       ══════════════════════════════════════════════════════ */

    var LIGHT_BUF_VS = [
        'varying vec2 vUv;',
        'void main() {',
        '    vUv = uv;',
        '    gl_Position = vec4(position, 1.0);',
        '}'
    ].join('\n');

    var LIGHT_BUF_FS = [
        'precision highp float;',
        'uniform sampler2D iChannel0;',
        'uniform sampler2D iChannel1;',
        'uniform float iTime;',
        'uniform vec2 iResolution;',
        'varying vec2 vUv;',

        'float noise(vec2 x) {',
        '    vec2 f = fract(x);',
        '    vec2 u = f*f*(f*(f*6.0-15.0)+10.0);',
        '    vec2 p = floor(x);',
        '    float a = texture2D(iChannel0, (p+vec2(0.0,0.0))/1024.0).x;',
        '    float b = texture2D(iChannel0, (p+vec2(1.0,0.0))/1024.0).x;',
        '    float c = texture2D(iChannel0, (p+vec2(0.0,1.0))/1024.0).x;',
        '    float d = texture2D(iChannel0, (p+vec2(1.0,1.0))/1024.0).x;',
        '    return a+(b-a)*u.x+(c-a)*u.y+(a-b-c+d)*u.x*u.y;',
        '}',

        'float fbm8(vec2 x, float decay) {',
        '    float a = 0.0;',
        '    float b = 1.0;',
        '    float t = 0.0;',
        '    for(int i = 0; i < 8; i++) {',
        '        float n = noise(x);',
        '        a += b*n;',
        '        t += b;',
        '        b *= decay;',
        '        x *= 2.0;',
        '    }',
        '    return a/t;',
        '}',

        'float boxFunc(vec2 uv, float x1, float x2, float y1, float y2) {',
        '    float v = 1.0;',
        '    v *= step(x1, uv.x) * (1.0 - step(x2, uv.x));',
        '    v *= step(y1, uv.y) * (1.0 - step(y2, uv.y));',
        '    return v;',
        '}',

        'vec4 foreground(vec2 uv, float t) {',
        '    float midlevel, h, disp, dist;',
        '    vec2 uv2;',
        '    uv.y -= 0.2;',

        '    midlevel=-0.1; disp=1.7; dist=1.0;',
        '    uv2 = uv + vec2(t/dist + 40.0, 0.0);',
        '    h = (fbm8(uv2, 0.7) - 0.5)*disp;',
        '    if(uv.y < h+midlevel-0.12) return vec4(0.43,0.32,0.31,1.);',
        '    if(uv.y < h+midlevel-0.08) return vec4(0.55,0.42,0.41,1.);',
        '    if(uv.y < h+midlevel-0.04) return vec4(0.66,0.42,0.40,1.);',
        '    if(uv.y < h+midlevel)     return vec4(0.77,0.48,0.46,1.);',

        '    midlevel=0.05; disp=1.7; dist=2.0;',
        '    uv2 = uv + vec2(t/dist + 38.0, 0.0);',
        '    h = (fbm8(uv2, 0.7) - 0.5)*disp;',
        '    if(uv.y < h+midlevel-0.1)  return vec4(0.95,0.66,0.48,1.);',
        '    if(uv.y < h+midlevel-0.04) return vec4(0.98,0.76,0.64,1.);',
        '    if(uv.y < h+midlevel)     return vec4(0.95,0.80,0.77,1.);',

        '    return vec4(0.95, 0.80, 0.77, 0.);',
        '}',

        'vec4 background(vec2 uv, float t) {',
        '    float midlevel, h, disp, dist;',
        '    vec2 uv2;',

        '    midlevel=0.3; disp=0.9; dist=10.0;',
        '    uv2 = uv + vec2(t/dist+32.5, 0.0);',
        '    h = (fbm8(uv2,0.7)-0.5)*disp;',
        '    if(uv.y < h+midlevel-0.14) return vec4(0.48,0.19,0.20,1.);',
        '    if(uv.y < h+midlevel-0.1)  return vec4(0.68,0.28,0.19,1.);',
        '    if(uv.y < h+midlevel-0.07) return vec4(0.88,0.38,0.24,1.);',
        '    if(uv.y < h+midlevel)     return vec4(0.95,0.45,0.30,1.);',

        '    midlevel=0.35; disp=1.0; dist=15.0;',
        '    uv2 = uv + vec2(t/dist+30.0, 0.0);',
        '    h = (fbm8(uv2,0.7)-0.5)*disp;',
        '    if(uv.y < h+midlevel-0.04) return vec4(0.98,0.76,0.64,1.);',
        '    if(uv.y < h+midlevel)     return vec4(0.95,0.80,0.77,1.);',

        '    midlevel=0.35; disp=3.5; dist=20.0;',
        '    uv2 = uv + vec2(t/dist+27.5, 0.0);',
        '    h = (fbm8(uv2,0.7)-0.5)*disp;',
        '    if(uv.y < h+midlevel-0.12) return vec4(0.43,0.32,0.31,1.);',
        '    if(uv.y < h+midlevel-0.08) return vec4(0.55,0.42,0.41,1.);',
        '    if(uv.y < h+midlevel-0.04) return vec4(0.66,0.42,0.40,1.);',
        '    if(uv.y < h+midlevel)     return vec4(0.77,0.48,0.46,1.);',

        '    midlevel=0.45; disp=2.0; dist=25.0;',
        '    uv2 = uv + vec2(t/dist+23.0, 0.0);',
        '    h = (fbm8(uv2,0.7)-0.5)*disp;',
        '    if(uv.y < h+midlevel-0.04) return vec4(0.98,0.57,0.36,1.);',
        '    if(uv.y < h+midlevel)     return vec4(1.0,0.62,0.44,1.);',

        '    midlevel=0.5; disp=2.3; dist=30.0;',
        '    uv2 = uv + vec2(t/dist+20.5, 0.0);',
        '    h = (fbm8(uv2,0.7)-0.5)*disp;',
        '    if(uv.y < h+midlevel-0.12) return vec4(0.41,0.27,0.27,1.);',
        '    if(uv.y < h+midlevel-0.08) return vec4(0.53,0.35,0.32,1.);',
        '    if(uv.y < h+midlevel-0.04) return vec4(0.80,0.24,0.17,1.);',
        '    if(uv.y < h+midlevel)     return vec4(0.99,0.29,0.20,1.);',

        '    midlevel=0.5; disp=2.5; dist=35.0;',
        '    uv2 = uv + vec2(t/dist+18.0, 0.0);',
        '    h = (fbm8(uv2,0.7)-0.5)*disp;',
        '    if(uv.y < h+midlevel-0.1)  return vec4(0.88,0.38,0.24,1.);',
        '    if(uv.y < h+midlevel-0.05) return vec4(0.98,0.42,0.28,1.);',
        '    if(uv.y < h+midlevel)     return vec4(1.0,0.48,0.35,1.);',

        '    midlevel=0.6; disp=2.0; dist=40.0;',
        '    uv2 = uv + vec2(t/dist+18.0, 0.0);',
        '    h = (fbm8(uv2,0.7)-0.5)*disp;',
        '    if(uv.y < h+midlevel-0.1)  return vec4(0.95,0.66,0.48,1.);',
        '    if(uv.y < h+midlevel)     return vec4(1.0,0.76,0.60,1.);',

        '    midlevel=0.75; disp=3.5; dist=45.0;',
        '    uv2 = uv + vec2(t/dist+15.5, 0.0);',
        '    h = (fbm8(uv2,0.7)-0.5)*disp;',
        '    if(uv.y < h+midlevel-0.2)  return vec4(1.0,0.55,0.33,1.);',
        '    if(uv.y < h+midlevel-0.15) return vec4(0.98,0.50,0.24,1.);',
        '    if(uv.y < h+midlevel-0.1)  return vec4(0.90,0.55,0.40,1.);',
        '    if(uv.y < h+midlevel)     return vec4(1.0,0.62,0.44,1.);',

        '    midlevel=0.7; disp=2.7; dist=50.0;',
        '    uv2 = uv + vec2(t/dist+12.0, 0.0);',
        '    h = (fbm8(uv2,0.7)-0.5)*disp;',
        '    if(uv.y < h+midlevel-0.04) return vec4(0.73,0.36,0.30,1.);',
        '    if(uv.y < h+midlevel)     return vec4(0.80,0.40,0.34,1.);',

        '    midlevel=0.8; disp=2.7; dist=60.0;',
        '    uv2 = uv + vec2(t/dist+9.5, 0.0);',
        '    h = (fbm8(uv2,0.7)-0.5)*disp;',
        '    if(uv.y < h+midlevel-0.1)  return vec4(0.93,0.58,0.35,1.);',
        '    if(uv.y < h+midlevel)     return vec4(1.0,0.76,0.60,1.);',

        '    midlevel=0.9; disp=3.0; dist=70.0;',
        '    uv2 = uv + vec2(t/dist+7.0, 0.0);',
        '    h = (fbm8(uv2,0.7)-0.5)*disp;',
        '    if(uv.y < h+midlevel-0.1)  return vec4(0.56,0.25,0.22,1.);',
        '    if(uv.y < h+midlevel-0.05) return vec4(0.60,0.30,0.27,1.);',
        '    if(uv.y < h+midlevel)     return vec4(0.74,0.35,0.30,1.);',

        '    midlevel=1.0; disp=5.0; dist=100.0;',
        '    uv2 = uv + vec2(t/dist+3.5, 0.0);',
        '    h = (fbm8(uv2,0.7)-0.5)*disp;',
        '    if(uv.y < h+midlevel-0.1)  return vec4(0.92,0.85,0.82,1.);',
        '    if(uv.y < h+midlevel)     return vec4(1.0,0.94,0.91,1.);',

        '    return vec4(0.58, 0.7, 1.0, 1.);',
        '}',

        'void main() {',
        '    vec2 fragCoord = vUv * iResolution;',
        '    vec2 uv = fragCoord / iResolution.y;',
        '    float t = iTime * 4.0;',
        '    vec4 bg = background(uv, t);',
        '',
        '    vec4 fg = vec4(0.);',
        '    if (uv.y < 0.5) {',
        '        for (int i = 0; i < 5; i++) {',
        '            fg += foreground(uv, t + 4.0*float(i)/5.0/60.0) / 5.0;',
        '        }',
        '    }',
        '',
        '    vec3 col = bg.rgb;',

        '    float k;',
        '    vec2 uv2;',
        '    uv.y -= 0.2;',
        '    uv2 = fract(uv*9.0);',
        '    float wagon = 1.0;',
        '    wagon *= 1.0 - step(0.45, uv.x);',
        '    wagon *= 1.0 - step(0.115, uv.y);',
        '    wagon *= step(0.103, uv.y);',
        '    wagon *= step(0.05, 1.0 - abs(uv2.x*2.0 - 1.0));',

        '    float join = 1.0;',
        '    join *= 1.0 - step(0.45, uv.x);',
        '    join *= 1.0 - step(0.11, uv.y);',
        '    join *= step(0.107, uv.y);',

        '    float roof = 1.0;',
        '    roof *= 1.0 - step(0.45, uv.x);',
        '    roof *= 1.0 - step(0.117, uv.y);',
        '    roof *= step(0.11, uv.y);',
        '    roof *= step(0.15, 1.0 - abs(uv2.x*2.0 - 1.0));',

        '    float loco = boxFunc(uv, 0.45, 0.5, 0.103, 0.112);',
        '    float chem1 = boxFunc(uv, 0.49, 0.495, 0.103, 0.12);',
        '    float chem2 = boxFunc(uv, 0.488, 0.496, 0.12, 0.123);',
        '    float locoRoof = boxFunc(uv, 0.443, 0.47, 0.11, 0.117);',

        '    float wheel = 1.0 - step(0.00004, dot(uv-vec2(0.457,0.106), uv-vec2(0.457,0.106)));',
        '    wheel += 1.0 - step(0.00002, dot(uv-vec2(0.487,0.105), uv-vec2(0.487,0.105)));',
        '    wheel += 1.0 - step(0.00002, dot(uv-vec2(0.497,0.105), uv-vec2(0.497,0.105)));',
        '',
        '    if (uv.x < 0.45 && uv.y > 0.025 && uv.y < 0.2) {',
        '        wheel += 1.0 - step(0.002, dot(uv2-vec2(0.2,0.95), uv2-vec2(0.2,0.95)));',
        '        wheel += 1.0 - step(0.002, dot(uv2-vec2(0.8,0.95), uv2-vec2(0.8,0.95)));',
        '    }',
        '    col = mix(col, vec3(0.18,0.12,0.15), join);',
        '    col = mix(col, vec3(0.48,0.19,0.20), wagon);',
        '    col = mix(col, vec3(0.18,0.12,0.15), roof);',
        '    col = mix(col, vec3(0.38,0.19,0.20), loco);',
        '    col = mix(col, vec3(0.38,0.19,0.20), chem1);',
        '    col = mix(col, vec3(0.18,0.12,0.15), locoRoof);',
        '    col = mix(col, vec3(0.18,0.12,0.15), chem2 + wheel);',

        '    uv2 = uv + vec2(t/5.0 + 3.5, 0.0);',
        '    uv2.x -= t/5.0*0.2;',
        '    float hs = fbm8(uv2, 0.9) - 0.55;',
        '    if(uv.x < 0.49) {',
        '        float x = -uv.x + 0.49;',
        '        float y = abs(uv.y + hs*0.4 - 0.16*sqrt(x) - 0.12) - 0.8*x*exp(-x*10.0);',
        '        if(y < 0.0) col = vec3(1.0, 0.94, 0.91);',
        '        if(y < -0.02) col = vec3(0.92, 0.85, 0.82);',
        '    }',

        '    uv2 = uv + vec2(t/5.0 + 32.5, 0.0);',
        '    uv2.x = fract(uv2.x*3.0);',
        '    k = 1.0;',
        '    k *= smoothstep(0.001, 0.003, abs(uv2.y - pow(uv2.x-0.5,2.0)*0.15 - 0.12));',
        '    k *= min(step(0.05, 1.0-abs(uv2.x*2.0-1.0)) + step(0.17, uv2.y), 1.0);',
        '    k *= min(smoothstep(0.02, 0.05, 1.0-abs(uv2.x*2.0-1.0)) + step(0.177, uv2.y), 1.0);',
        '    k *= min(step(0.1, uv2.y) + smoothstep(-0.09, -0.085, -uv2.y - 0.001/(1.0-abs(uv2.x*2.0-1.0))), 1.0);',
        '    float pr = fract(uv2.x*16.0);',
        '    k *= min(smoothstep(0.05, 0.2, 1.0-abs(pr*2.0-1.0)) + step(0.12, uv2.y-pow(uv2.x-0.5,2.0)*0.15) + step(-0.1, -uv2.y), 1.0);',
        '    col = mix(vec3(0.29,0.09,0.08)*smoothstep(-0.08, 0.08, uv.y), col, k);',

        '    col = mix(col, fg.rgb, fg.a);',

        '    uv = fragCoord / iResolution.xy;',
        '    col = mix(col, texture2D(iChannel1, uv).rgb, 0.3);',

        '    gl_FragColor = vec4(col, 1.0);',
        '}'
    ].join('\n');

    var LIGHT_FINAL_VS = [
        'varying vec2 vUv;',
        'void main() {',
        '    vUv = uv;',
        '    gl_Position = vec4(position, 1.0);',
        '}'
    ].join('\n');

    var LIGHT_FINAL_FS = [
        'precision highp float;',
        'uniform sampler2D iChannel0;',
        'uniform float uOpacity;',
        'varying vec2 vUv;',

        'void main() {',
        '    vec2 uv = vUv;',
        '    vec3 col = texture2D(iChannel0, uv).rgb;',
        '    col *= 0.5 + 0.5*pow(16.0*uv.x*uv.y*(1.0-uv.x)*(1.0-uv.y), 0.2);',
        '    gl_FragColor = vec4(col, uOpacity);',
        '}'
    ].join('\n');

    /* ── Noise Texture ── */
    function createNoiseTexture(size) {
        size = size || 1024;
        var data = new Uint8Array(size * size * 4);
        for (var i = 0; i < size * size; i++) {
            var v = Math.floor(Math.random() * 256);
            data[i * 4] = v; data[i * 4 + 1] = v; data[i * 4 + 2] = v; data[i * 4 + 3] = 255;
        }
        var tex = new THREE.DataTexture(data, size, size, THREE.RGBAFormat);
        tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
        tex.magFilter = tex.minFilter = THREE.LinearFilter;
        tex.needsUpdate = true;
        return tex;
    }

    /* ══════════════════════════════════════════════════════
       INIT
       ══════════════════════════════════════════════════════ */
    function init() {
        if (_inited) return;

        _canvas = document.createElement('canvas');
        _canvas.className = 'shader-bg-canvas light-shader-canvas';
        _canvas.setAttribute('aria-hidden', 'true');

        _renderer = new THREE.WebGLRenderer({
            canvas: _canvas,
            alpha: true,
            antialias: false,
            powerPreference: 'low-power'
        });

        var gl = _renderer.getContext();
        if (!gl) { console.error('[SunsetShader] No WebGL'); return; }

        _renderer.setPixelRatio(Math.min(window.devicePixelRatio, PIXEL_RATIO_CAP));
        var w = window.innerWidth, h = window.innerHeight;
        _renderer.setSize(w, h);
        _renderer.setClearColor(0x000000, 0);

        var pr = Math.min(window.devicePixelRatio, PIXEL_RATIO_CAP);
        var rw = Math.floor(w * pr), rh = Math.floor(h * pr);

        _noiseTex = createNoiseTexture(1024);

        var rtOpts = {
            minFilter: THREE.LinearFilter,
            magFilter: THREE.LinearFilter,
            format: THREE.RGBAFormat,
            type: THREE.UnsignedByteType
        };
        _rtA = new THREE.WebGLRenderTarget(rw, rh, rtOpts);
        _rtB = new THREE.WebGLRenderTarget(rw, rh, rtOpts);

        _bufScene = new THREE.Scene();
        _bufCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

        _bufMat = new THREE.ShaderMaterial({
            vertexShader: LIGHT_BUF_VS,
            fragmentShader: LIGHT_BUF_FS,
            uniforms: {
                iChannel0: { value: _noiseTex },
                iChannel1: { value: _rtA.texture },
                iTime: { value: 0.0 },
                iResolution: { value: new THREE.Vector2(rw, rh) }
            }
        });
        _bufScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), _bufMat));

        _scene = new THREE.Scene();
        _camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

        _finalMat = new THREE.ShaderMaterial({
            vertexShader: LIGHT_FINAL_VS,
            fragmentShader: LIGHT_FINAL_FS,
            uniforms: {
                iChannel0: { value: _rtA.texture },
                uOpacity: { value: 0.0 }
            },
            transparent: true,
            depthTest: false,
            depthWrite: false
        });
        _scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), _finalMat));

        _startTime = performance.now() / 1000.0;
        _pingPong = false;
        _frameCount = 0;
        _inited = true;

        console.log('[SunsetShader] Init OK');
    }

    /* ══════════════════════════════════════════════════════
       RENDER LOOP
       ══════════════════════════════════════════════════════ */
    function renderLoop() {
        if (!_renderer) return;
        if (!_active && _currentOpacity <= 0.01) { pauseLoop(); return; }
        _animId = requestAnimationFrame(renderLoop);

        try {
            var t = performance.now() / 1000.0 - _startTime;
            _frameCount++;

            var readRT, writeRT;
            if (_pingPong) { readRT = _rtB; writeRT = _rtA; }
            else { readRT = _rtA; writeRT = _rtB; }

            _bufMat.uniforms.iTime.value = t;
            _bufMat.uniforms.iChannel1.value = readRT.texture;

            _renderer.setRenderTarget(writeRT);
            _renderer.render(_bufScene, _bufCamera);

            _finalMat.uniforms.iChannel0.value = writeRT.texture;

            var diff = _targetOpacity - _currentOpacity;
            if (Math.abs(diff) > 0.005) _currentOpacity += diff * 0.08;
            else _currentOpacity = _targetOpacity;
            _finalMat.uniforms.uOpacity.value = _currentOpacity;

            _renderer.setRenderTarget(null);
            _renderer.render(_scene, _camera);

            _pingPong = !_pingPong;

            if (_targetOpacity <= 0.01 && _currentOpacity <= 0.01) pauseLoop();
        } catch (e) {
            console.warn('[SunsetShader] Render err:', e);
            pauseLoop();
        }
    }

    function pauseLoop() {
        if (_animId) { cancelAnimationFrame(_animId); _animId = null; }
    }

    function resumeLoop() {
        if (!_animId && _active && _renderer) renderLoop();
    }

    /* ══════════════════════════════════════════════════════
       ACTIVATE / DEACTIVATE
       ══════════════════════════════════════════════════════ */
    function activate() {
        if (!_inited || _active) return;
        _active = true;
        _targetOpacity = 1.0;
        _currentOpacity = 0;

        if (!_canvas.parentNode) {
            document.body.insertBefore(_canvas, document.body.firstChild);
        }
        _canvas.classList.add('shader-bg-visible');
        document.documentElement.classList.add('shader-active');

        resumeLoop();
        console.log('[SunsetShader] Activated');
    }

    function deactivate() {
        if (!_active) return;
        _active = false;

        if (_canvas) _canvas.classList.remove('shader-bg-visible');

        // Set target to 0 - renderLoop will keep running until fade completes
        _targetOpacity = 0;

        console.log('[SunsetShader] Deactivated');
    }

    /* ══════════════════════════════════════════════════════
       RESIZE
       ══════════════════════════════════════════════════════ */
    function resize(w, h) {
        if (!_renderer) return;
        var dpr = Math.min(window.devicePixelRatio, PIXEL_RATIO_CAP);
        _renderer.setSize(w, h);
        var rw = Math.floor(w * dpr), rh = Math.floor(h * dpr);
        if (_rtA) _rtA.setSize(rw, rh);
        if (_rtB) _rtB.setSize(rw, rh);
        if (_bufMat) _bufMat.uniforms.iResolution.value.set(rw, rh);
    }

    /* ══════════════════════════════════════════════════════
       DISPOSE
       ══════════════════════════════════════════════════════ */
    function dispose() {
        pauseLoop();
        if (_rtA) { _rtA.dispose(); _rtA = null; }
        if (_rtB) { _rtB.dispose(); _rtB = null; }
        if (_noiseTex) { _noiseTex.dispose(); _noiseTex = null; }
        if (_bufMat) { _bufMat.dispose(); _bufMat = null; }
        if (_finalMat) { _finalMat.dispose(); _finalMat = null; }
        if (_renderer) { _renderer.dispose(); _renderer = null; }
        if (_canvas && _canvas.parentNode) _canvas.parentNode.removeChild(_canvas);
        _canvas = null; _scene = null; _camera = null;
        _bufScene = null; _bufCamera = null;
        _pingPong = false; _frameCount = 0;
        _currentOpacity = 0; _targetOpacity = 0;
        _inited = false; _active = false;
        console.log('[SunsetShader] Disposed');
    }

    /* ══════════════════════════════════════════════════════
       PUBLIC API
       ══════════════════════════════════════════════════════ */
    window.SunsetShader = {
        init: init,
        activate: activate,
        deactivate: deactivate,
        dispose: dispose,
        resize: resize,
        isActive: function () { return _active; },
        isInitialized: function () { return _inited; },
        setOpacity: function (v) { _targetOpacity = v; },
        getOpacity: function () { return _currentOpacity; },
        getCanvas: function () { return _canvas; }
    };
})();