/**
 * Shader Background System - Light & Dark Theme
 * Light: Sunset cloud scene (Canvas2D fallback)
 * Dark: Volumetric clouds + ray-marched terrain (6-Pass WebGL)
 */
;(function () {
    'use strict';

    var PIXEL_RATIO_CAP = 1.5;
    var TRANSITION_DURATION = 520;

    /* ── Shared State ── */
    var canvas = null;
    var animFrameId = null;
    var isActive = false;
    var currentOpacity = 0;
    var targetOpacity = 0;
    var resizeHandler = null;
    var visibilityHandler = null;
    var clickHandler = null;
    var mouseMoveHandler = null;
    var mouseDownHandler = null;
    var mouseUpHandler = null;
    var activeTheme = null; // 'light' or 'dark'

    /* ── Three.js Loader ── */
    var _threeReady = false;
    var _threeCallbacks = [];

    function hasThreeJS() {
        return typeof THREE !== 'undefined' && THREE.WebGLRenderer;
    }

    function ensureThreeJS(cb) {
        if (hasThreeJS()) { _threeReady = true; cb(); return; }
        if (_threeCallbacks.length > 0) { _threeCallbacks.push(cb); return; }
        _threeCallbacks.push(cb);
        var s = document.createElement('script');
        s.src = '/static/js/three.min.js';
        s.onload = function () {
            _threeReady = true;
            console.log('[ShaderBG] Three.js loaded, v' + THREE.REVISION);
            for (var i = 0; i < _threeCallbacks.length; i++) {
                try { _threeCallbacks[i](); } catch (e) { console.warn(e); }
            }
            _threeCallbacks = [];
        };
        s.onerror = function () { console.error('[ShaderBG] Three.js load failed'); };
        document.head.appendChild(s);
    }

    /* ═══════════════════════════════════════════
       DARK THEME - Volumetric Clouds (6-Pass)
       ═══════════════════════════════════════════ */

    var darkRenderer = null;
    var darkScenes = {};
    var darkCameras = {};
    var darkTargets = {};
    var darkMaterials = {};
    var darkTextures = {};
    var darkMouse = { x: -1000, y: -1000, down: false, dirty: false };
    var darkFrame = 0;
    var darkStartTime = 0;
    var darkAnimId = null;

    /* ── Common GLSL (prepended to all shaders) ── */
    var COMMON_GLSL = [
        '#define PI 3.1415926535',
        '#define SAT(x) clamp(x, 0., 1.)',
        '#define TERRAIN_FREQ .1',
        '#define TERRAIN_HEIGHT 3.',
        '#define HQ_OCTAVES 12',
        '#define MQ_OCTAVES 7',
        '#define CAMERA_NEAR .001',
        '#define CAMERA_FAR 200.',
        '#define CAMERA_FOV 75.',
        '#define CAMERA_HEIGHT 1.6',
        '#define CAMERA_PITCH .15',
        '#define CAMERA_ZOOM -2.',
        '#define CAMERA_DEPTH -1125.',
        '#define FOG_B .3',
        '#define FOG_C .1',
        '#define SUN_INTENSITY 6.66',
        '#define SUN_COLOR vec3(1.2, 1., .6)',
        '#define SKY_COLOR vec3(.25, .5, 1.75)',
        '#define SUN_SPEED .04',
        '#define EARTH_RADIUS 6378100.',
        '#define CLOUD_BOTTOM 3200.',
        '#define CLOUD_TOP 4800.',
        '#define CLOUD_COVERAGE .48',
        '#define CLOUD_BASE_FREQ .00005',
        '#define CLOUD_DETAIL_FREQ .0015',
        '#define CLOUD_STEPS 18',
        '#define CLOUD_LIGHT_STEPS 6',
        '#define CLOUD_TOP_OFFSET 250.',
        '#define CLOUD_ABSORPTION_TOP 1.8',
        '#define CLOUD_ABSORPTION_BOTTOM 3.6',
        '#define WIND_DIR vec3(.4, .1, 1.)',
        '#define WIND_SPEED 75.',
        '#define CLOUDS_AMBIENT_TOP vec3(1., 1.2, 1.6)',
        '#define CLOUDS_AMBIENT_BOTTOM vec3(.6, .4, .8)',
        '#define BAYER_LIMIT 16',
        '#define BAYER_LIMIT_H 4',
        '',
        'const int bayerFilter[16] = int[](',
        '    0,8,2,10, 12,4,14,6, 3,11,1,9, 15,7,13,5',
        ');',
        '',
        'struct Ray { vec3 origin, direction; };',
        '',
        'float remap(float x, float a, float b, float c, float d){return(((x-a)/(b-a))*(d-c))+c;}',
        'float remap01(float x, float a, float b){return((x-a)/(b-a));}',
        '',
        'bool writeToPixel(vec2 fragCoord, int iFrame){',
        '    ivec2 iF=ivec2(fragCoord); int idx=iFrame%16;',
        '    return(((iF.x+4*iF.y)%16)==bayerFilter[idx]);',
        '}',
        '',
        'mat3 getCameraMatrix(vec3 o, vec3 t){',
        '    vec3 l=normalize(t-o); vec3 r=normalize(cross(l,vec3(0,1,0)));',
        '    vec3 u=normalize(cross(r,l)); return mat3(r,u,-l);',
        '}',
        'Ray getCameraRay(vec2 uv, float tm){',
        '    uv*=(CAMERA_FOV/360.)*PI;',
        '    vec3 o=vec3(0.,CAMERA_HEIGHT,CAMERA_DEPTH);',
        '    vec3 t=vec3(0.,o.y+CAMERA_PITCH,CAMERA_DEPTH-1.2);',
        '    return Ray(o,normalize(getCameraMatrix(o,t)*vec3(uv,CAMERA_ZOOM)));',
        '}',
        'vec3 getSun(vec2 mouse, float t){',
        '    vec2 sp=mouse;',
        '    if(mouse.y<-.95){sp=vec2(cos(mod(t*SUN_SPEED,PI))*.7,0.);sp.y=1.-3.05*sp.x*sp.x;}',
        '    return vec3(sp,max(0.,sp.y*.75+.25));',
        '}',
        'vec3 miePhase(float d, vec3 s){return max(exp(-pow(d,.3))*s-.4,0.);}',
        'vec3 atmosphericScattering(vec2 uv, vec2 sp, bool isSun){',
        '    float sd=distance(uv,sp); float sm=SAT(sd); float dist=uv.y;',
        '    dist=(.5*mix(sm,1.,dist))/dist;',
        '    vec3 ms=miePhase(sd,vec3(1.))*SUN_COLOR;',
        '    vec3 c=max(dist*SKY_COLOR,0.);',
        '    vec3 sun=.0002/pow(length(uv-sp),1.7)*SUN_COLOR;',
        '    c=max(mix(pow(c,.8-c),c/(2.*c+.5-c*1.3),SAT(sp.y*2.5)),0.)+(isSun?(sun+ms):vec3(0.));',
        '    c*=(pow(1.-sm,5.)*10.*SAT(.666-sp.y))+1.5;',
        '    return mix(c,vec3(0.),SAT(distance(sp.y,1.)));',
        '}',
        '#define UI0 1597334673U',
        '#define UI1 3812015801U',
        '#define UI2 uvec2(UI0,UI1)',
        '#define UI3 uvec3(UI0,UI1,2798796415U)',
        '#define UIF (1./float(0xffffffffU))',
        'vec3 hash33(vec3 p){uvec3 q=uvec3(ivec3(p))*UI3;q=(q.x^q.y^q.z)*UI3;return-1.+2.*vec3(q)*UIF;}',
        'float hash13(vec3 p){uvec3 q=uvec3(ivec3(p))*UI3;q*=UI3;uint n=(q.x^q.y^q.z)*UI0;return float(n)*UIF;}',
        'float hash12(vec2 p){uvec2 q=uvec2(ivec2(p))*UI2;uint n=(q.x^q.y)*UI0;return float(n)*UIF;}',
        '',
        'vec3 valueNoiseDerivative(vec2 x, sampler2D smp){',
        '    vec2 f=fract(x); vec2 u=f*f*(3.-2.*f);',
        '    ivec2 p=ivec2(floor(x));',
        '    float a=texelFetch(smp,(p+ivec2(0,0))&255,0).x;',
        '    float b=texelFetch(smp,(p+ivec2(1,0))&255,0).x;',
        '    float c=texelFetch(smp,(p+ivec2(0,1))&255,0).x;',
        '    float d=texelFetch(smp,(p+ivec2(1,1))&255,0).x;',
        '    return vec3(a+(b-a)*u.x+(c-a)*u.y+(a-b-c+d)*u.x*u.y,6.*f*(1.-f)*(vec2(b-a,c-a)+(a-b-c+d)*u.yx));',
        '}',
        'float valueNoise(vec3 x, float freq){',
        '    vec3 i=floor(x); vec3 f=fract(x); f=f*f*(3.-2.*f);',
        '    return mix(mix(mix(hash13(mod(i+vec3(0,0,0),freq)),hash13(mod(i+vec3(1,0,0),freq)),f.x),',
        '        mix(hash13(mod(i+vec3(0,1,0),freq)),hash13(mod(i+vec3(1,1,0),freq)),f.x),f.y),',
        '        mix(mix(hash13(mod(i+vec3(0,0,1),freq)),hash13(mod(i+vec3(1,0,1),freq)),f.x),',
        '        mix(hash13(mod(i+vec3(0,1,1),freq)),hash13(mod(i+vec3(1,1,1),freq)),f.x),f.y),f.z);',
        '}',
        'float worleyNoise(vec3 uv, float freq, bool tileable){',
        '    vec3 id=floor(uv); vec3 p=fract(uv); float md=10000.;',
        '    for(float x=-1.;x<=1.;++x)for(float y=-1.;y<=1.;++y)for(float z=-1.;z<=1.;++z){',
        '        vec3 o=vec3(x,y,z); vec3 h=tileable?hash33(mod(id+o,vec3(freq)))*.4+.3:hash33(id+o)*.4+.3;',
        '        h+=o; vec3 d=p-h; md=min(md,dot(d,d));} return 1.-md;',
        '}',
        'float perlinFbm(vec3 p, float freq, int oct){',
        '    float G=exp2(-.85); float a=1.; float n=0.;',
        '    for(int i=0;i<oct;++i){n+=a*valueNoise(p*freq,freq);freq*=2.;a*=G;} return n;',
        '}',
        'float worleyFbm(vec3 p, float freq, bool tileable){',
        '    return max(0.,(worleyNoise(p*freq,freq,tileable)*.625+worleyNoise(p*freq*2.,freq*2.,tileable)*.25+',
        '        worleyNoise(p*freq*4.,freq*4.,tileable)*.125)*1.1-.1);',
        '}',
    ].join('\n');

    /* ── Full-screen quad vertex shader ── */
    var QUAD_VS = 'precision highp float;in vec3 position;in vec2 uv;out vec2 vUv;void main(){vUv=uv;gl_Position=vec4(position,1.);}';

    /* ── Buffer A: Perlin-Worley noise textures ── */
    var BUFFER_A_FS = [
        COMMON_GLSL,
        'uniform sampler2D iChannel0;',
        'uniform sampler2D iChannel1;',
        'uniform vec2 iResolution;',
        'in vec2 vUv; out vec4 fragColor;',
        'bool resolutionChanged(){return int(texelFetch(iChannel1,ivec2(0),0).r)!=int(iResolution.x);}',
        'void main(){',
        '    vec2 fc=vUv*iResolution;',
        '    if(resolutionChanged()){',
        '        vec2 uv=vUv; vec4 col=vec4(0.);',
        '        col.r+=perlinFbm(vec3(uv,.4),4.,15)*.5; col.r=abs(col.r*2.-1.);',
        '        col.r=remap(col.r,worleyFbm(vec3(uv,.2),4.,true)-1.,1.,0.,1.);',
        '        col.g+=worleyFbm(vec3(uv,.5),8.,true)*.625+worleyFbm(vec3(uv,.5),16.,true)*.25+worleyFbm(vec3(uv,.5),32.,true)*.125;',
        '        col.b=1.-col.g; fragColor=col;',
        '    }else{ fragColor=texelFetch(iChannel0,ivec2(fc),0); }',
        '}',
    ].join('\n');

    /* ── Buffer B: Terrain ray march ── */
    var BUFFER_B_FS = [
        COMMON_GLSL,
        'uniform sampler2D iChannel0;',
        'uniform sampler2D iChannel1;',
        'uniform vec3 iResolution;',
        'uniform float iTime;',
        'uniform vec4 iMouse;',
        'uniform int iFrame;',
        'in vec2 vUv; out vec4 fragColor;',
        'const mat2 m2=mat2(.8,-.6,.6,.8);',
        'float terrainFbm(vec2 uv, int oct, sampler2D smp){',
        '    vec2 p=uv*TERRAIN_FREQ; float a=0.; float b=1.; vec2 d=vec2(0.);',
        '    for(int i=0;i<oct;++i){vec3 n=valueNoiseDerivative(p,smp);d+=n.yz;a+=b*n.x/(1.+dot(d,d));b*=.5;p=m2*p*2.;}',
        '    return smoothstep(-.95,.5,abs(a)*2.-1.)*(abs(a)*2.-1.)*TERRAIN_HEIGHT;',
        '}',
        'vec3 calcNormal(vec3 pos, float freq, float t){',
        '    vec2 e=vec2(.002*t,0.); int nl=int(max(5.,float(HQ_OCTAVES)-(float(HQ_OCTAVES)-1.)*t/CAMERA_FAR));',
        '    return normalize(vec3(terrainFbm(pos.xz-e.xy,nl,iChannel0)-terrainFbm(pos.xz+e.xy,nl,iChannel0),2.*e.x,terrainFbm(pos.xz-e.yx,nl,iChannel0)-terrainFbm(pos.xz+e.yx,nl,iChannel0)));',
        '}',
        'float raymarchShadow(Ray ray){',
        '    float sh=1.; float t=CAMERA_NEAR; vec3 p; float h;',
        '    for(int i=0;i<80;++i){p=ray.origin+t*ray.direction;h=p.y-terrainFbm(p.xz,MQ_OCTAVES,iChannel0);sh=min(sh,8.*h/t);t+=h;if(sh<.001||p.z>CAMERA_FAR)break;}',
        '    return SAT(sh);',
        '}',
        'float raymarchTerrain(Ray ray){',
        '    float t=CAMERA_NEAR,h=0.;',
        '    for(int i=0;i<200;++i){vec3 pos=ray.origin+ray.direction*t;h=pos.y-terrainFbm(pos.xz,MQ_OCTAVES,iChannel0);if(abs(h)<(t*.002)||t>CAMERA_FAR)break;t+=h*.5;}',
        '    return t;',
        '}',
        'void main(){',
        '    vec2 fc=vUv*iResolution.xy; vec2 st=vUv; vec2 uv=(2.*fc-iResolution.xy)/iResolution.y;',
        '    vec2 mouse=(2.*iMouse.xy-iResolution.xy)/iResolution.y;',
        '    bool up=writeToPixel(fc,iFrame);',
        '    vec4 col=textureLod(iChannel1,st,0.);',
        '    if(up){',
        '        Ray ray=getCameraRay(uv,iTime);',
        '        float td=raymarchTerrain(ray);',
        '        vec3 sun=getSun(mouse,iTime); vec3 sd=normalize(vec3(sun.x,sun.z,-1.));',
        '        vec3 sh=normalize(sd+ray.direction); float sDot=max(0.,dot(ray.direction,sd));',
        '        col*=0.;',
        '        if(td>CAMERA_FAR){',
        '            col.rgb+=atmosphericScattering(uv*.5+.225,sun.xy*.5+.225,true);',
        '            col.gb+=.006-uv.y*.0048;',
        '            float t2=iTime*.15; float stars=pow(hash12(fc),4.*iResolution.x);',
        '            float tw=sin(t2*3.7+uv.x-sin(uv.y*20.+t2)*10.)*2.;',
        '            tw*=cos(uv.y+t2*4.4-sin(uv.x*15.+t2)*7.)*1.5; tw=tw*.5+.5;',
        '            col+=max(0.,stars*tw*smoothstep(.075,0.,sun.z)*2.);',
        '        }else{',
        '            vec3 mp=ray.origin+ray.direction*td; vec3 tn=calcNormal(mp,TERRAIN_FREQ,td);',
        '            vec3 rock=vec3(.1,.1,.08); vec3 snow=vec3(.9); vec3 grass=vec3(.02,.1,.05);',
        '            vec3 alb=mix(grass,rock,smoothstep(0.,.1*TERRAIN_HEIGHT,mp.y));',
        '            alb=mix(alb,snow,smoothstep(.4*TERRAIN_HEIGHT,1.4*TERRAIN_HEIGHT,mp.y));',
        '            alb=mix(rock,alb,smoothstep(.4,.7,tn.y));',
        '            float ts=clamp(raymarchShadow(Ray(mp-sd*.001,sd)),0.,8.)+.2;',
        '            float diff=max(dot(sd,tn),0.)*ts;',
        '            float spec=SAT(dot(sh,ray.direction));',
        '            float sa=SAT(.5+.5*tn.y);',
        '            col.rgb+=SUN_INTENSITY*SUN_COLOR*diff;',
        '            col.rgb+=vec3(.5,.7,1.2)*sa;',
        '            col.rgb+=SUN_COLOR*(SAT(.5+.5*dot(normalize(vec3(-sd.x,sd.y,sd.z)),tn)));',
        '            col.rgb*=alb; col.rgb+=SUN_INTENSITY*.4*SUN_COLOR*diff*pow(SAT(spec),16.);',
        '            float fm=FOG_C*exp(-ray.origin.y*FOG_B)*(1.-exp(-pow(td*FOG_B,1.5)*ray.direction.y))/ray.direction.y;',
        '            vec3 fc2=mix(atmosphericScattering(uv*.5+.75,sun.xy*.5+.225,false)*.75,vec3(.8,.6,.3),pow(sDot,8.));',
        '            fc2=mix(vec3(.4,.5,.6),fc2,smoothstep(0.,.1,sun.z));',
        '            col.rgb=mix(col.rgb,fc2,SAT(fm));',
        '            col.rgb*=max(.0,sun.z)+mix(vec3(smoothstep(.1,0.,sun.z))*tn.y,fc2,SAT(fm))*(.012,.024,.048);',
        '        }',
        '        col.a=td;',
        '    }',
        '    fragColor=col;',
        '    if(fc.x<1.&&fc.y<1.)fragColor=vec4(iResolution.x,0.,0.,0.);',
        '}',
    ].join('\n');

    /* ── Buffer C: Cloud ray march ── */
    var BUFFER_C_FS = [
        COMMON_GLSL,
        'uniform sampler2D iChannel0;',
        'uniform sampler2D iChannel1;',
        'uniform sampler2D iChannel2;',
        'uniform sampler2D iChannel3;',
        'uniform vec3 iResolution;',
        'uniform float iTime;',
        'uniform vec4 iMouse;',
        'uniform int iFrame;',
        'in vec2 vUv; out vec4 fragColor;',
        'const vec3 noiseKernel[6]=vec3[](vec3(.3805,.9245,-.0211),vec3(-.5063,-.0359,-.8616),vec3(-.3251,-.9456,.0143),vec3(.0903,-.2738,.9576),vec3(.2813,.4244,-.8607),vec3(-.1685,.1475,.9746));',
        'float raySphereIntersect(Ray ray,float r){float b=2.*dot(ray.origin,ray.direction);float c=dot(ray.origin,ray.origin)-r*r;return(-b+sqrt(b*b-4.*c))*.5;}',
        'float cloudGradient(float h){return smoothstep(0.,.05,h)*smoothstep(1.25,.5,h);}',
        'float cloudHeightFract(float p){return(p-EARTH_RADIUS-CLOUD_BOTTOM)/(CLOUD_TOP-CLOUD_BOTTOM);}',
        'float cloudBase(vec3 p,float y){vec3 n=textureLod(iChannel2,(p.xz-WIND_DIR.xz*iTime*WIND_SPEED)*CLOUD_BASE_FREQ,0.).rgb;float v=y*y*n.b+pow(1.-y,12.);return remap01(n.r-v,n.g-1.,1.);}',
        'float cloudDetail(vec3 p,float c,float y){',
        '    p-=WIND_DIR*3.*iTime*WIND_SPEED;',
        '    float hf=worleyFbm(p,CLOUD_DETAIL_FREQ,false)*.625+worleyFbm(p,CLOUD_DETAIL_FREQ*2.,false)*.25+worleyFbm(p,CLOUD_DETAIL_FREQ*4.,false)*.125;',
        '    hf=mix(hf,1.-hf,y*4.); return remap01(c,hf*.5,1.);',
        '}',
        'float getCloudDensity(vec3 p,float y,bool detail){',
        '    p.xz-=WIND_DIR.xz*y*CLOUD_TOP_OFFSET; float d=cloudBase(p,y);',
        '    d=remap01(d,CLOUD_COVERAGE,1.)*CLOUD_COVERAGE; d*=cloudGradient(y);',
        '    bool dt=(d>0.&&d<.3)&&detail; return dt?cloudDetail(p,d,y):d;',
        '}',
        'float henyeyGreenstein(float sd,float g){float g2=g*g;return(.25/PI)*((1.-g2)/pow(1.+g2-2.*g*sd,1.5));}',
        'float marchToLight(vec3 p,vec3 sd,float sdot,float sh){',
        '    float lrs=11.; vec3 lrd=sd*lrs; vec3 lrd2=lrd*.5; float cs=length(lrd); float td2=0.;',
        '    for(int i=0;i<6;++i){vec3 cp=lrd2+cs*noiseKernel[i]*float(i);float y=cloudHeightFract(length(p));if(y>.95||td2>.95)break;td2+=getCloudDensity(cp+p,y,false)*lrs;lrd2+=lrd;}',
        '    return 32.*exp(-td2*mix(CLOUD_ABSORPTION_BOTTOM,CLOUD_ABSORPTION_TOP,sh))*(1.-exp(-td2*2.));',
        '}',
        'void main(){',
        '    vec2 fc=vUv*iResolution.xy; vec2 st=vUv; vec2 uv=(2.*fc-iResolution.xy)/iResolution.y;',
        '    vec2 mouse=(2.*iMouse.xy-iResolution.xy)/iResolution.y;',
        '    float td=texelFetch(iChannel0,ivec2(fc),0).w;',
        '    vec4 prev=textureLod(iChannel1,st,0.); vec4 col=vec4(0.);',
        '    bool up=writeToPixel(fc,iFrame);',
        '    if(up){',
        '        Ray ray=getCameraRay(uv,iTime); vec3 sun=getSun(mouse,iTime); sun.z=clamp(sun.z,0.,.8);',
        '        vec3 sd=normalize(vec3(sun.x,sun.z,-1.)); float sdot=max(0.,dot(ray.direction,sd));',
        '        float sh=smoothstep(.01,.1,sun.z+.025);',
        '        if(td>CAMERA_FAR){',
        '            ray.origin.y=EARTH_RADIUS;',
        '            float start=raySphereIntersect(ray,EARTH_RADIUS+CLOUD_BOTTOM);',
        '            float end=raySphereIntersect(ray,EARTH_RADIUS+CLOUD_TOP);',
        '            float crd=start; float css=(end-start)/float(CLOUD_STEPS);',
        '            crd+=css*texelFetch(iChannel3,(ivec2(fc)+iFrame*ivec2(113,127))&1023,0).r;',
        '            vec3 skc=atmosphericScattering(vec2(.15,.05),vec2(.5,sun.y*.5+.25),false);',
        '            skc.r*=1.1; skc=SAT(pow(skc*2.1,vec3(4.2)));',
        '            float ssh=smoothstep(.15,.4,sun.z);',
        '            float hg=mix(henyeyGreenstein(sdot,.4),henyeyGreenstein(sdot,-.1),.5);',
        '            hg=max(hg,1.6*henyeyGreenstein(sqrt(sdot),SAT(.8-ssh)));',
        '            hg=mix(pow(sdot,.25),hg,sh);',
        '            vec4 ist=vec4(0.,0.,0.,1.); vec3 amb=vec3(0.);',
        '            for(int i=0;i<CLOUD_STEPS;++i){',
        '                vec3 p=ray.origin+crd*ray.direction; float hf=cloudHeightFract(length(p));',
        '                float den=getCloudDensity(p,hf,true);',
        '                if(den>0.){',
        '                    amb=mix(CLOUDS_AMBIENT_BOTTOM,CLOUDS_AMBIENT_TOP,hf);',
        '                    vec3 lum=(amb*SAT(pow(sun.z+.04,1.4))+skc*.125+(sh*skc+vec3(.0075,.015,.03))*SUN_COLOR*hg*marchToLight(p,sd,sdot,ssh))*den;',
        '                    float tr=exp(-den*css); vec3 is2=(lum-lum*tr)*(1./den);',
        '                    ist.rgb+=ist.a*is2; ist.a*=tr;',
        '                }',
        '                if(ist.a<.05)break; crd+=css;',
        '            }',
        '            float fm=1.-exp(-smoothstep(.15,0.,ray.direction.y)*2.);',
        '            vec3 fogC=atmosphericScattering(uv*.5+.2,sun.xy*.5+.2,false);',
        '            ist.rgb=mix(ist.rgb,fogC*sh,fm); ist.a=mix(ist.a,0.,fm);',
        '            col=vec4(max(ist.rgb,0.),ist.a); col=mix(prev,col,.5);',
        '        }',
        '    }else{col=prev;}',
        '    fragColor=col;',
        '}',
    ].join('\n');

    /* ── Buffer D: TXAA ── */
    var BUFFER_D_FS = [
        'uniform sampler2D iChannel0;',
        'uniform sampler2D iChannel1;',
        'in vec2 vUv; out vec4 fragColor;',
        'const ivec2 offsets[8]=ivec2[](ivec2(-1,-1),ivec2(-1,1),ivec2(1,-1),ivec2(1,1),ivec2(1,0),ivec2(0,-1),ivec2(0,1),ivec2(-1,0));',
        'void main(){',
        '    vec2 fc=vUv*vec2(textureSize(iChannel0,0));',
        '    vec4 cur=textureLod(iChannel0,vUv,0.); vec4 hist=textureLod(iChannel1,vUv,0.);',
        '    vec4 cAvg=cur; vec4 cVar=cur*cur;',
        '    for(int i=0;i<8;i++){vec4 nb=texelFetch(iChannel0,ivec2(fc)+offsets[i],0);cAvg+=nb;cVar+=nb*nb;}',
        '    cAvg/=9.; cVar/=9.; float sig=.75;',
        '    vec4 sigma=sqrt(max(vec4(0.),cVar-cAvg*cAvg));',
        '    vec4 cMin=cAvg-sig*sigma; vec4 cMax=cAvg+sig*sigma;',
        '    hist=clamp(hist,cMin,cMax); fragColor=mix(cur,hist,.95);',
        '}',
    ].join('\n');

    /* ── Image: Final post-processing ── */
    var IMAGE_FS = [
        COMMON_GLSL,
        'uniform sampler2D iChannel0;',
        'uniform sampler2D iChannel1;',
        'uniform vec3 iResolution;',
        'uniform float iTime;',
        'uniform vec4 iMouse;',
        'in vec2 vUv; out vec4 fragColor;',
        '#define texOff vec2(1.75/iResolution.xy)',
        'const float kernel[9]=float[](.0625,.125,.0625,.125,.25,.125,.0625,.125,.0625);',
        'vec4 gaussianBlur(sampler2D buf,vec2 uv){',
        '    vec4 c=vec4(0.);',
        '    vec2 off[9]=vec2[](vec2(-texOff.x,texOff.y),vec2(0.,texOff.y),vec2(texOff.x,texOff.y),vec2(-texOff.x,0.),vec2(0.,0.),vec2(texOff.x,0.),vec2(-texOff.x,-texOff.y),vec2(0.,-texOff.y),vec2(texOff.x,-texOff.y));',
        '    for(int i=0;i<9;i++){c+=textureLod(buf,uv+off[i],0.)*kernel[i];} return c;',
        '}',
        '#define ORB_FLARE_COUNT 8',
        '#define DISTORTION_BARREL 1.3',
        'vec2 GetDistOffset(vec2 uv,vec2 px){',
        '    vec2 tc=uv; vec3 prep=normalize(vec3(tc.y,-tc.x,0.));',
        '    float a=length(tc)*2.221*DISTORTION_BARREL; vec3 o=vec3(px,0.);',
        '    vec3 r=o*cos(a)+cross(prep,o)*sin(a)+prep*dot(prep,o)*(1.-cos(a)); return r.xy;',
        '}',
        'vec3 flare(vec2 uv,vec2 pos,float dist,float size){',
        '    pos=GetDistOffset(uv,pos);',
        '    float r=max(.01-pow(length(uv+(dist-.05)*pos),2.4)*(1./(size*2.)),0.)*6.;',
        '    float g=max(.01-pow(length(uv+dist*pos),2.4)*(1./(size*2.)),0.)*6.;',
        '    float b=max(.01-pow(length(uv+(dist+.05)*pos),2.4)*(1./(size*2.)),0.)*6.;',
        '    return vec3(r,g,b);',
        '}',
        'vec3 ring(vec2 uv,vec2 pos,float dist){',
        '    vec2 uvd=uv*(length(uv));',
        '    float r=max(1./(1.+32.*pow(length(uvd+(dist-.05)*pos),2.)),0.)*.25;',
        '    float g=max(1./(1.+32.*pow(length(uvd+dist*pos),2.)),0.)*.23;',
        '    float b=max(1./(1.+32.*pow(length(uvd+(dist+.05)*pos),2.)),0.)*.21;',
        '    return vec3(r,g,b);',
        '}',
        'vec3 lensflare(vec2 uv,vec2 pos,float br,float size){',
        '    vec3 c=flare(uv,pos,-1.,size)*3.; c+=flare(uv,pos,.5,.8*size)*2.; c+=flare(uv,pos,-.4,.8*size);',
        '    c+=ring(uv,pos,-1.)*.5*size; c+=ring(uv,pos,1.)*.5*size; return c*br;',
        '}',
        '#define NUM_SAMPLES 48',
        '#define DENSITY .768',
        '#define WEIGHT .14',
        '#define DECAY .97',
        'vec3 lightScattering(vec2 uv,vec2 lp,vec3 sun){',
        '    vec2 du=vec2(uv-lp); vec2 st=uv; uv=uv*2.-1.; uv.x*=iResolution.x/iResolution.y;',
        '    du*=1./float(NUM_SAMPLES)*DENSITY; float id=1.; vec3 res=vec3(0.);',
        '    for(int i=0;i<NUM_SAMPLES;i++){st-=du;float ls=textureLod(iChannel1,st,0.).a*smoothstep(2.5,-1.,length(uv-sun.xy));ls*=id*WEIGHT;res+=ls;id*=DECAY;}',
        '    return res*SUN_COLOR*.2;',
        '}',
        'vec3 luminanceReinhard(vec3 c){float l=dot(c,vec3(.2126,.7152,.0722));float t=l/(1.+l);return c*t/l;}',
        'void main(){',
        '    vec2 st=vUv; vec2 fc=vUv*iResolution.xy;',
        '    vec2 uv=(2.*fc-iResolution.xy)/iResolution.y;',
        '    vec2 mouse=(2.*iMouse.xy-iResolution.xy)/iResolution.y;',
        '    vec3 sun=getSun(mouse,iTime);',
        '    vec4 terrain=textureLod(iChannel0,vec2(st.x,st.y-1./iResolution.y),0.);',
        '    vec4 clouds=gaussianBlur(iChannel1,st);',
        '    float cam=clouds.a+(terrain.a>CAMERA_FAR?0.:1.);',
        '    vec2 lsp=vec2(sun.x*iResolution.y/iResolution.x,sun.y)*.5+.5;',
        '    float lfm=textureLod(iChannel1,lsp,0.).a;',
        '    vec3 col=clouds.rgb+terrain.rgb*cam;',
        '    col+=lightScattering(st,lsp,sun)*smoothstep(.01,.16,sun.z)*smoothstep(.3,1.5,terrain.a);',
        '    col+=lensflare(uv,sun.xy,.8.,4.)*vec3(1.4,1.2,1.)*lfm;',
        '    col=mix(col,pow(luminanceReinhard(col),vec3(.4545)),.75);',
        '    col+=hash12(fc)*.004;',
        '    fragColor=vec4(col,1.);',
        '    if(fc.y<2.&&fc.x<2.)fragColor=vec4(.6)*sun.z;',
        '}',
    ].join('\n');

    /* ── Noise texture generation ── */
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

    function createBlueNoiseTexture() {
        // Create a simple blue-noise-like texture
        var size = 1024;
        var data = new Uint8Array(size * size * 4);
        for (var i = 0; i < size * size; i++) {
            var v = Math.floor(Math.random() * 256);
            data[i * 4] = v; data[i * 4 + 1] = v; data[i * 4 + 2] = v; data[i * 4 + 3] = 255;
        }
        var tex = new THREE.DataTexture(data, size, size, THREE.RGBAFormat);
        tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
        tex.magFilter = tex.minFilter = THREE.NearestFilter;
        tex.needsUpdate = true;
        return tex;
    }

    /* ── Dark theme init ── */
    function initDark() {
        if (darkRenderer) return;
        if (!hasThreeJS()) { console.error('[Dark] No Three.js'); return; }

        try {
            canvas = document.createElement('canvas');
            canvas.className = 'shader-bg-canvas';
            canvas.setAttribute('aria-hidden', 'true');

            darkRenderer = new THREE.WebGLRenderer({
                canvas: canvas,
                alpha: true,
                antialias: false,
                powerPreference: 'low-power'
            });

            var gl = darkRenderer.getContext();
            if (!gl) { console.error('[Dark] No WebGL'); return; }

            var isWebGL2 = gl instanceof WebGL2RenderingContext;
            console.log('[Dark] WebGL ' + (isWebGL2 ? '2' : '1'));

            if (!isWebGL2) {
                console.error('[Dark] Requires WebGL2 for texelFetch');
                darkRenderer.dispose();
                darkRenderer = null;
                return;
            }

            darkRenderer.setPixelRatio(Math.min(window.devicePixelRatio, PIXEL_RATIO_CAP));
            var w = window.innerWidth, h = window.innerHeight;
            darkRenderer.setSize(w, h);
            darkRenderer.setClearColor(0x000000, 0);

            var pr = Math.min(window.devicePixelRatio, PIXEL_RATIO_CAP);
            var rw = Math.floor(w * pr), rh = Math.floor(h * pr);

            // Noise textures
            darkTextures.noise = createNoiseTexture(256);
            darkTextures.blueNoise = createBlueNoiseTexture();

            // Create render targets for each pass
            var rtOpts = {
                minFilter: THREE.LinearFilter,
                magFilter: THREE.LinearFilter,
                format: THREE.RGBAFormat,
                type: THREE.FloatType
            };
            var rtOptsSmall = Object.assign({}, rtOpts);

            darkTargets.bufA = [new THREE.WebGLRenderTarget(rw, rh, rtOpts), new THREE.WebGLRenderTarget(rw, rh, rtOpts)];
            darkTargets.bufB = [new THREE.WebGLRenderTarget(rw, rh, rtOpts), new THREE.WebGLRenderTarget(rw, rh, rtOpts)];
            darkTargets.bufC = [new THREE.WebGLRenderTarget(rw, rh, rtOpts), new THREE.WebGLRenderTarget(rw, rh, rtOpts)];
            darkTargets.bufD = [new THREE.WebGLRenderTarget(rw, rh, rtOpts), new THREE.WebGLRenderTarget(rw, rh, rtOpts)];

            // Shared camera
            var cam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

            // Buffer A scene
            darkMaterials.bufA = new THREE.ShaderMaterial({
                vertexShader: QUAD_VS,
                fragmentShader: '#version 300 es\nprecision highp float;\n' + BUFFER_A_FS,
                glslVersion: THREE.GLSL3,
                uniforms: {
                    iChannel0: { value: darkTargets.bufA[0].texture },
                    iChannel1: { value: darkTargets.bufA[0].texture },
                    iResolution: { value: new THREE.Vector2(rw, rh) }
                }
            });
            darkScenes.bufA = new THREE.Scene();
            darkScenes.bufA.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), darkMaterials.bufA));

            // Buffer B scene
            darkMaterials.bufB = new THREE.ShaderMaterial({
                vertexShader: QUAD_VS,
                fragmentShader: '#version 300 es\nprecision highp float;\n' + BUFFER_B_FS,
                glslVersion: THREE.GLSL3,
                uniforms: {
                    iChannel0: { value: darkTextures.noise },
                    iChannel1: { value: darkTargets.bufB[0].texture },
                    iResolution: { value: new THREE.Vector3(rw, rh, 0) },
                    iTime: { value: 0 },
                    iMouse: { value: new THREE.Vector4(-1000, -1000, 0, 0) },
                    iFrame: { value: 0 }
                }
            });
            darkScenes.bufB = new THREE.Scene();
            darkScenes.bufB.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), darkMaterials.bufB));

            // Buffer C scene
            darkMaterials.bufC = new THREE.ShaderMaterial({
                vertexShader: QUAD_VS,
                fragmentShader: '#version 300 es\nprecision highp float;\n' + BUFFER_C_FS,
                glslVersion: THREE.GLSL3,
                uniforms: {
                    iChannel0: { value: darkTargets.bufB[0].texture },
                    iChannel1: { value: darkTargets.bufC[0].texture },
                    iChannel2: { value: darkTargets.bufA[0].texture },
                    iChannel3: { value: darkTextures.blueNoise },
                    iResolution: { value: new THREE.Vector3(rw, rh, 0) },
                    iTime: { value: 0 },
                    iMouse: { value: new THREE.Vector4(-1000, -1000, 0, 0) },
                    iFrame: { value: 0 }
                }
            });
            darkScenes.bufC = new THREE.Scene();
            darkScenes.bufC.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), darkMaterials.bufC));

            // Buffer D scene
            darkMaterials.bufD = new THREE.ShaderMaterial({
                vertexShader: QUAD_VS,
                fragmentShader: '#version 300 es\nprecision highp float;\n' + BUFFER_D_FS,
                glslVersion: THREE.GLSL3,
                uniforms: {
                    iChannel0: { value: darkTargets.bufC[0].texture },
                    iChannel1: { value: darkTargets.bufD[0].texture },
                }
            });
            darkScenes.bufD = new THREE.Scene();
            darkScenes.bufD.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), darkMaterials.bufD));

            // Image scene
            darkMaterials.image = new THREE.ShaderMaterial({
                vertexShader: QUAD_VS,
                fragmentShader: '#version 300 es\nprecision highp float;\n' + IMAGE_FS,
                glslVersion: THREE.GLSL3,
                uniforms: {
                    iChannel0: { value: darkTargets.bufB[0].texture },
                    iChannel1: { value: darkTargets.bufD[0].texture },
                    iResolution: { value: new THREE.Vector3(rw, rh, 0) },
                    iTime: { value: 0 },
                    iMouse: { value: new THREE.Vector4(-1000, -1000, 0, 0) },
                },
                transparent: true,
                depthTest: false,
                depthWrite: false
            });
            darkScenes.image = new THREE.Scene();
            darkScenes.image.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), darkMaterials.image));

            darkCameras.cam = cam;
            darkStartTime = performance.now() / 1000;
            darkFrame = 0;

            // Diagnostic
            console.log('[Dark] PASS DIAGNOSTICS:');
            diagnoseShaderGL(gl, darkMaterials.bufA, 'BufA');
            diagnoseShaderGL(gl, darkMaterials.bufB, 'BufB');
            diagnoseShaderGL(gl, darkMaterials.bufC, 'BufC');
            diagnoseShaderGL(gl, darkMaterials.bufD, 'BufD');
            diagnoseShaderGL(gl, darkMaterials.image, 'Image');
            console.log('[Dark] Init OK');

        } catch (e) {
            console.error('[Dark] Init failed:', e);
            cleanupDark();
        }
    }

    function diagnoseShaderGL(gl, mat, name) {
        try {
            var prog = mat.program;
            if (!prog) { console.log('[DBG:' + name + '] no program'); return; }
            var gprog = prog.program;
            if (!gprog) { console.log('[DBG:' + name + '] no GL program'); return; }
            var vs = prog.vertexShader, fs = prog.fragmentShader;
            if (vs) console.log('[DBG:' + name + '] VS=' + gl.getShaderParameter(vs, gl.COMPILE_STATUS));
            if (fs) {
                var ok = gl.getShaderParameter(fs, gl.COMPILE_STATUS);
                var log = gl.getShaderInfoLog(fs);
                console.log('[DBG:' + name + '] FS=' + ok + ' ' + (log || 'OK').substring(0, 200));
            }
            console.log('[DBG:' + name + '] LINK=' + gl.getProgramParameter(gprog, gl.LINK_STATUS));
        } catch (e) { console.warn('[DBG:' + name + '] err', e); }
    }

    var _darkPingPong = [0, 0, 0, 0]; // indices for bufA, bufB, bufC, bufD

    function renderDark() {
        if (!isActive || !darkRenderer || activeTheme !== 'dark') return;
        darkAnimId = requestAnimationFrame(renderDark);

        try {
            var t = performance.now() / 1000 - darkStartTime;
            darkFrame++;

            var gl = darkRenderer.getContext();
            var rw = darkTargets.bufA[0].width, rh = darkTargets.bufA[0].height;
            var cam = darkCameras.cam;

            // Update mouse uniform
            var mx = darkMouse.x, my = darkMouse.y;
            var mvec = new THREE.Vector4(mx, rh - my, darkMouse.down ? 1 : 0, 0);

            // Pass A: Noise textures (only when resolution changes - render once)
            if (darkFrame <= 2) {
                darkMaterials.bufA.uniforms.iChannel1.value = darkTargets.bufA[_darkPingPong[0]].texture;
                darkRenderer.setRenderTarget(darkTargets.bufA[1 - _darkPingPong[0]]);
                darkRenderer.render(darkScenes.bufA, cam);
                _darkPingPong[0] = 1 - _darkPingPong[0];
                // Update references
                darkMaterials.bufA.uniforms.iChannel0.value = darkTargets.bufA[_darkPingPong[0]].texture;
                darkMaterials.bufC.uniforms.iChannel2.value = darkTargets.bufA[_darkPingPong[0]].texture;
            }

            // Pass B: Terrain
            darkMaterials.bufB.uniforms.iTime.value = t;
            darkMaterials.bufB.uniforms.iMouse.value = mvec;
            darkMaterials.bufB.uniforms.iFrame.value = darkFrame;
            darkMaterials.bufB.uniforms.iChannel1.value = darkTargets.bufB[_darkPingPong[1]].texture;
            darkRenderer.setRenderTarget(darkTargets.bufB[1 - _darkPingPong[1]]);
            darkRenderer.render(darkScenes.bufB, cam);
            _darkPingPong[1] = 1 - _darkPingPong[1];

            // Pass C: Clouds
            darkMaterials.bufC.uniforms.iTime.value = t;
            darkMaterials.bufC.uniforms.iMouse.value = mvec;
            darkMaterials.bufC.uniforms.iFrame.value = darkFrame;
            darkMaterials.bufC.uniforms.iChannel0.value = darkTargets.bufB[_darkPingPong[1]].texture;
            darkMaterials.bufC.uniforms.iChannel1.value = darkTargets.bufC[_darkPingPong[2]].texture;
            darkRenderer.setRenderTarget(darkTargets.bufC[1 - _darkPingPong[2]]);
            darkRenderer.render(darkScenes.bufC, cam);
            _darkPingPong[2] = 1 - _darkPingPong[2];

            // Pass D: TXAA
            darkMaterials.bufD.uniforms.iChannel0.value = darkTargets.bufC[_darkPingPong[2]].texture;
            darkMaterials.bufD.uniforms.iChannel1.value = darkTargets.bufD[_darkPingPong[3]].texture;
            darkRenderer.setRenderTarget(darkTargets.bufD[1 - _darkPingPong[3]]);
            darkRenderer.render(darkScenes.bufD, cam);
            _darkPingPong[3] = 1 - _darkPingPong[3];

            // Image pass: Final composite
            darkMaterials.image.uniforms.iTime.value = t;
            darkMaterials.image.uniforms.iMouse.value = mvec;
            darkMaterials.image.uniforms.iChannel0.value = darkTargets.bufB[_darkPingPong[1]].texture;
            darkMaterials.image.uniforms.iChannel1.value = darkTargets.bufD[_darkPingPong[3]].texture;

            // Smooth opacity
            var diff = targetOpacity - currentOpacity;
            if (Math.abs(diff) > 0.005) currentOpacity += diff * 0.08;
            else currentOpacity = targetOpacity;
            darkMaterials.image.uniforms.uOpacity = { value: currentOpacity };
            // Override - use transparent material
            darkMaterials.image.opacity = currentOpacity;

            darkRenderer.setRenderTarget(null);
            darkRenderer.render(darkScenes.image, cam);

            if (darkFrame <= 3) {
                var err = gl.getError();
                console.log('[Dark] F' + darkFrame + ' op=' + currentOpacity.toFixed(2) + ' gl=' + (err === 0 ? 'OK' : '0x' + err.toString(16)));
            }

            if (targetOpacity <= 0.01 && currentOpacity <= 0.01) pauseDark();
        } catch (e) {
            console.error('[Dark] Render err:', e);
            pauseDark();
        }
    }

    function pauseDark() {
        if (darkAnimId) { cancelAnimationFrame(darkAnimId); darkAnimId = null; }
    }

    function resumeDark() {
        if (!darkAnimId && isActive && darkRenderer && activeTheme === 'dark') renderDark();
    }

    function cleanupDark() {
        pauseDark();
        if (darkTargets.bufA) { darkTargets.bufA[0].dispose(); darkTargets.bufA[1].dispose(); }
        if (darkTargets.bufB) { darkTargets.bufB[0].dispose(); darkTargets.bufB[1].dispose(); }
        if (darkTargets.bufC) { darkTargets.bufC[0].dispose(); darkTargets.bufC[1].dispose(); }
        if (darkTargets.bufD) { darkTargets.bufD[0].dispose(); darkTargets.bufD[1].dispose(); }
        if (darkTextures.noise) darkTextures.noise.dispose();
        if (darkTextures.blueNoise) darkTextures.blueNoise.dispose();
        ['bufA','bufB','bufC','bufD','image'].forEach(function(k) {
            if (darkMaterials[k]) darkMaterials[k].dispose();
        });
        if (darkRenderer) { darkRenderer.dispose(); darkRenderer = null; }
        if (canvas && canvas.parentNode) canvas.parentNode.removeChild(canvas);
        canvas = null; darkTargets = {}; darkMaterials = {}; darkTextures = {};
        darkScenes = {}; darkCameras = {};
        _darkPingPong = [0, 0, 0, 0];
        darkFrame = 0;
    }

    /* ── Mouse interaction for dark shader ── */
    function setupDarkMouse() {
        var EXCLUDE = 'button,input,textarea,select,a,.card,.glass-panel,.topbar,.topnav,.switch,.slider';

        if (clickHandler) return;
        clickHandler = function (e) {
            if (activeTheme !== 'dark' || !darkRenderer) return;
            if (e.target.closest(EXCLUDE)) return;
            darkMouse.x = e.clientX * Math.min(window.devicePixelRatio, PIXEL_RATIO_CAP);
            darkMouse.y = e.clientY * Math.min(window.devicePixelRatio, PIXEL_RATIO_CAP);
            darkMouse.down = true;
            darkMouse.dirty = true;
        };
        mouseDownHandler = function (e) {
            if (activeTheme !== 'dark' || !darkRenderer) return;
            if (e.target.closest(EXCLUDE)) return;
            darkMouse.down = true;
        };
        mouseUpHandler = function () { darkMouse.down = false; };
        mouseMoveHandler = function (e) {
            if (activeTheme !== 'dark' || !darkRenderer || !darkMouse.down) return;
            darkMouse.x = e.clientX * Math.min(window.devicePixelRatio, PIXEL_RATIO_CAP);
            darkMouse.y = e.clientY * Math.min(window.devicePixelRatio, PIXEL_RATIO_CAP);
        };

        document.addEventListener('click', clickHandler);
        document.addEventListener('mousedown', mouseDownHandler);
        document.addEventListener('mouseup', mouseUpHandler);
        document.addEventListener('mousemove', mouseMoveHandler);
    }

    function teardownDarkMouse() {
        if (clickHandler) { document.removeEventListener('click', clickHandler); clickHandler = null; }
        if (mouseDownHandler) { document.removeEventListener('mousedown', mouseDownHandler); mouseDownHandler = null; }
        if (mouseUpHandler) { document.removeEventListener('mouseup', mouseUpHandler); mouseUpHandler = null; }
        if (mouseMoveHandler) { document.removeEventListener('mousemove', mouseMoveHandler); mouseMoveHandler = null; }
    }

    /* ═══════════════════════════════════════════
       LIGHT THEME - WebGL Sunset Cloud Scene
       (Ping-Pong RenderTarget + Three.js)
       ═══════════════════════════════════════════ */

    var lightRenderer = null;
    var lightScene = null;
    var lightCamera = null;
    var lightBufScene = null;
    var lightBufCamera = null;
    var lightRtA = null;
    var lightRtB = null;
    var lightPingPong = false;
    var lightFinalMat = null;
    var lightBufMat = null;
    var lightNoiseTex = null;
    var lightCanvas = null;
    var lightAnimId = null;
    var lightStartTime = 0;
    var lightFrameCount = 0;

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

        '    vec4 fg = vec4(0.);',
        '    if (uv.y < 0.5) {',
        '        for (int i = 0; i < 5; i++) {',
        '            fg += foreground(uv, t + 4.0*float(i)/5.0/60.0) / 5.0;',
        '        }',
        '    }',

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

    function createLightNoiseTexture(size) {
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

    function initLight() {
        if (lightRenderer) return;
        if (!hasThreeJS()) { console.error('[Light] No Three.js'); return; }

        try {
            lightCanvas = document.createElement('canvas');
            lightCanvas.className = 'shader-bg-canvas';
            lightCanvas.setAttribute('aria-hidden', 'true');

            lightRenderer = new THREE.WebGLRenderer({
                canvas: lightCanvas,
                alpha: true,
                antialias: false,
                powerPreference: 'low-power'
            });

            var gl = lightRenderer.getContext();
            if (!gl) { console.error('[Light] No WebGL'); return; }

            lightRenderer.setPixelRatio(Math.min(window.devicePixelRatio, PIXEL_RATIO_CAP));
            var w = window.innerWidth, h = window.innerHeight;
            lightRenderer.setSize(w, h);
            lightRenderer.setClearColor(0x000000, 0);

            var pr = Math.min(window.devicePixelRatio, PIXEL_RATIO_CAP);
            var rw = Math.floor(w * pr), rh = Math.floor(h * pr);

            lightNoiseTex = createLightNoiseTexture(1024);

            var rtOpts = {
                minFilter: THREE.LinearFilter,
                magFilter: THREE.LinearFilter,
                format: THREE.RGBAFormat,
                type: THREE.UnsignedByteType
            };
            lightRtA = new THREE.WebGLRenderTarget(rw, rh, rtOpts);
            lightRtB = new THREE.WebGLRenderTarget(rw, rh, rtOpts);

            lightBufScene = new THREE.Scene();
            lightBufCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

            lightBufMat = new THREE.ShaderMaterial({
                vertexShader: LIGHT_BUF_VS,
                fragmentShader: LIGHT_BUF_FS,
                uniforms: {
                    iChannel0: { value: lightNoiseTex },
                    iChannel1: { value: lightRtA.texture },
                    iTime: { value: 0.0 },
                    iResolution: { value: new THREE.Vector2(rw, rh) }
                }
            });
            lightBufScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), lightBufMat));

            lightScene = new THREE.Scene();
            lightCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

            lightFinalMat = new THREE.ShaderMaterial({
                vertexShader: LIGHT_FINAL_VS,
                fragmentShader: LIGHT_FINAL_FS,
                uniforms: {
                    iChannel0: { value: lightRtA.texture },
                    uOpacity: { value: 0.0 }
                },
                transparent: true,
                depthTest: false,
                depthWrite: false
            });
            lightScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), lightFinalMat));

            lightStartTime = performance.now() / 1000.0;
            lightPingPong = false;
            lightFrameCount = 0;

            console.log('[Light] WebGL init OK');
        } catch (e) {
            console.error('[Light] Init failed:', e);
            cleanupLight();
        }
    }

    function renderLight() {
        if (!isActive || !lightRenderer || activeTheme !== 'light') return;
        lightAnimId = requestAnimationFrame(renderLight);

        try {
            var t = performance.now() / 1000.0 - lightStartTime;
            lightFrameCount++;

            var readRT, writeRT;
            if (lightPingPong) { readRT = lightRtB; writeRT = lightRtA; }
            else { readRT = lightRtA; writeRT = lightRtB; }

            lightBufMat.uniforms.iTime.value = t;
            lightBufMat.uniforms.iChannel1.value = readRT.texture;

            lightRenderer.setRenderTarget(writeRT);
            lightRenderer.render(lightBufScene, lightBufCamera);

            lightFinalMat.uniforms.iChannel0.value = writeRT.texture;

            var diff = targetOpacity - currentOpacity;
            if (Math.abs(diff) > 0.005) currentOpacity += diff * 0.08;
            else currentOpacity = targetOpacity;
            lightFinalMat.uniforms.uOpacity.value = currentOpacity;

            lightRenderer.setRenderTarget(null);
            lightRenderer.render(lightScene, lightCamera);

            lightPingPong = !lightPingPong;

            if (targetOpacity <= 0.01 && currentOpacity <= 0.01) pauseLight();
        } catch (e) {
            console.warn('[Light] Render err:', e);
            pauseLight();
        }
    }

    function pauseLight() {
        if (lightAnimId) { cancelAnimationFrame(lightAnimId); lightAnimId = null; }
    }

    function resumeLight() {
        if (!lightAnimId && isActive && lightRenderer && activeTheme === 'light') renderLight();
    }

    function cleanupLight() {
        pauseLight();
        if (lightRtA) { lightRtA.dispose(); lightRtA = null; }
        if (lightRtB) { lightRtB.dispose(); lightRtB = null; }
        if (lightNoiseTex) { lightNoiseTex.dispose(); lightNoiseTex = null; }
        if (lightBufMat) { lightBufMat.dispose(); lightBufMat = null; }
        if (lightFinalMat) { lightFinalMat.dispose(); lightFinalMat = null; }
        if (lightRenderer) { lightRenderer.dispose(); lightRenderer = null; }
        if (lightCanvas && lightCanvas.parentNode) lightCanvas.parentNode.removeChild(lightCanvas);
        lightCanvas = null; lightScene = null; lightCamera = null;
        lightBufScene = null; lightBufCamera = null;
        lightPingPong = false; lightFrameCount = 0;
    }

    /* ═══════════════════════════════════════════
       ACTIVATE / DEACTIVATE
       ═══════════════════════════════════════════ */

    function activate(theme) {
        if (isActive && activeTheme === theme) return;
        if (isActive) deactivate();

        activeTheme = theme;
        isActive = true;
        targetOpacity = 1.0;
        currentOpacity = 0;

        if (theme === 'dark') {
            ensureThreeJS(function () {
                if (!hasThreeJS()) return;
                initDark();
                if (!darkRenderer) return;
                if (!canvas.parentNode) document.body.insertBefore(canvas, document.body.firstChild);
                canvas.classList.add('shader-bg-visible');
                document.documentElement.classList.add('shader-active');
                setupDarkMouse();
                resumeDark();
                console.log('[Dark] Activated');
            });
        } else {
            ensureThreeJS(function () {
                if (!hasThreeJS()) return;
                initLight();
                if (!lightRenderer) return;
                if (!lightCanvas.parentNode) document.body.insertBefore(lightCanvas, document.body.firstChild);
                lightCanvas.classList.add('shader-bg-visible');
                document.documentElement.classList.add('shader-active');
                resumeLight();
                console.log('[Light] Activated');
            });
        }
    }

    function deactivate() {
        isActive = false;
        if (activeTheme === 'dark') {
            targetOpacity = 0;
            teardownDarkMouse();
            if (canvas) canvas.classList.remove('shader-bg-visible');
            setTimeout(function () {
                cleanupDark();
                console.log('[Dark] Cleaned up');
            }, TRANSITION_DURATION + 200);
        } else {
            if (lightCanvas) lightCanvas.classList.remove('shader-bg-visible');
            cleanupLight();
            console.log('[Light] Cleaned up');
        }
        document.documentElement.classList.remove('shader-active');
        activeTheme = null;
        currentOpacity = 0;
        targetOpacity = 0;
    }

    /* ═══════════════════════════════════════════
       THEME INTEGRATION
       ═══════════════════════════════════════════ */

    function isLightTheme() {
        return document.documentElement.classList.contains('light') ||
            document.documentElement.getAttribute('data-theme') === 'light';
    }

    document.addEventListener('theme-changed', function (e) {
        var theme = (e && e.detail && e.detail.theme) || (isLightTheme() ? 'light' : 'dark');
        // Dark = volumetric clouds, Light = sunset Canvas2D
        if (theme === 'light') {
            // Switching to light: stop dark, start light
            if (activeTheme === 'dark') deactivate();
            setTimeout(function () { activate('light'); }, 50);
        } else {
            // Switching to dark: stop light, start dark
            if (activeTheme === 'light') deactivate();
            setTimeout(function () { activate('dark'); }, 50);
        }
    });

    // Resize handler
    window.addEventListener('resize', function () {
        if (!isActive) return;
        var dpr = Math.min(window.devicePixelRatio || 1, PIXEL_RATIO_CAP);
        if (activeTheme === 'light' && lightRenderer) {
            var w = window.innerWidth, h = window.innerHeight;
            lightRenderer.setSize(w, h);
            var rw = Math.floor(w * dpr), rh = Math.floor(h * dpr);
            lightRtA.setSize(rw, rh);
            lightRtB.setSize(rw, rh);
            lightBufMat.uniforms.iResolution.value.set(rw, rh);
        }
        if (activeTheme === 'dark' && darkRenderer) {
            var w = window.innerWidth, h = window.innerHeight;
            darkRenderer.setSize(w, h);
            var rw = Math.floor(w * dpr), rh = Math.floor(h * dpr);
            // Resize all targets
            ['bufA','bufB','bufC','bufD'].forEach(function (k) {
                if (darkTargets[k]) { darkTargets[k][0].setSize(rw, rh); darkTargets[k][1].setSize(rw, rh); }
            });
            // Update iResolution uniforms
            var v2 = new THREE.Vector2(rw, rh);
            var v3 = new THREE.Vector3(rw, rh, 0);
            if (darkMaterials.bufA) darkMaterials.bufA.uniforms.iResolution.value = v2;
            if (darkMaterials.bufB) darkMaterials.bufB.uniforms.iResolution.value = v3;
            if (darkMaterials.bufC) darkMaterials.bufC.uniforms.iResolution.value = v3;
            if (darkMaterials.image) darkMaterials.image.uniforms.iResolution.value = v3;
        }
    });

    // Visibility handler
    document.addEventListener('visibilitychange', function () {
        if (!isActive) return;
        if (document.hidden) {
            if (activeTheme === 'dark') pauseDark();
            if (activeTheme === 'light' && lightAnimId) { cancelAnimationFrame(lightAnimId); lightAnimId = null; }
        } else {
            if (activeTheme === 'dark') resumeDark();
            if (activeTheme === 'light' && !lightAnimId) renderLight();
        }
    });

    // Initial state
    function initIfReady() {
        var theme = isLightTheme() ? 'light' : 'dark';
        // For light theme, activate immediately
        if (theme === 'light') {
            setTimeout(function () { activate('light'); }, 500);
        }
        // For dark theme, pre-load Three.js then activate
        else {
            ensureThreeJS(function () {
                setTimeout(function () { activate('dark'); }, 500);
            });
        }
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initIfReady);
    } else {
        initIfReady();
    }

    // Debug API
    window.__shaderBG = {
        activate: activate,
        deactivate: deactivate,
        getState: function () {
            return {
                isActive: isActive,
                theme: activeTheme,
                opacity: currentOpacity,
                target: targetOpacity,
                hasThreeJS: hasThreeJS(),
                darkFrame: darkFrame
            };
        }
    };
})();