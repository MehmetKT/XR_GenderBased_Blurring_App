import * as THREE from 'three';
import {BlurStyle, DetectedPersonBox, GenderFilter} from './types.ts';

const MAX_BOXES = 12;

const FULLSCREEN_VERT = /* glsl */ `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = vec4(position.xy, 0.0, 1.0);
  }
`;

const PASS_VERT = /* glsl */ `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const GAUSSIAN_FRAG = /* glsl */ `
  precision highp float;
  uniform sampler2D tDiffuse;
  uniform vec2 uTexelSize;
  uniform vec2 uDirection;
  uniform float uRadius;
  varying vec2 vUv;

  void main() {
    vec2 stepUv = uTexelSize * uDirection * uRadius;
    vec4 sum = texture2D(tDiffuse, vUv) * 0.227027;
    sum += texture2D(tDiffuse, vUv + stepUv * 1.384615) * 0.316216;
    sum += texture2D(tDiffuse, vUv - stepUv * 1.384615) * 0.316216;
    sum += texture2D(tDiffuse, vUv + stepUv * 3.230769) * 0.070270;
    sum += texture2D(tDiffuse, vUv - stepUv * 3.230769) * 0.070270;
    gl_FragColor = vec4(sum.rgb, 1.0);
  }
`;

const COMPOSITE_FRAG = /* glsl */ `
  precision highp float;
  uniform sampler2D tSource;
  uniform sampler2D tBlurNear;
  uniform sampler2D tBlurFar;
  uniform sampler2D tMask;
  uniform vec2 uSourceSize;
  uniform vec2 uMaskTexel;
  uniform float uEnabled;
  uniform float uBlurStrength;
  uniform float uMaskPadding;
  uniform int uBlurStyle;
  uniform int uGenderFilter;
  uniform float uBlurUnknown;
  uniform float uShowBoxes;
  uniform int uBoxCount;
  uniform vec4 uBoxes[${MAX_BOXES}];
  uniform float uBoxGenders[${MAX_BOXES}];
  uniform float uTime;
  varying vec2 vUv;

  float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
  }

  // Samples channel-coded RGBA mask (r = female, g = male, b = unknown).
  vec3 sampleFeatheredMaskRgb(vec2 uv) {
    vec3 center = texture2D(tMask, uv).rgb;
    float pad = mix(1.0, 9.0, uMaskPadding);
    vec2 off = uMaskTexel * pad;

    vec3 maxNbr = center;
    vec3 avgNbr = center * 2.0;

    vec2 dirs[8];
    dirs[0] = vec2(1.0, 0.0);
    dirs[1] = vec2(-1.0, 0.0);
    dirs[2] = vec2(0.0, 1.0);
    dirs[3] = vec2(0.0, -1.0);
    dirs[4] = vec2(0.707, 0.707);
    dirs[5] = vec2(-0.707, 0.707);
    dirs[6] = vec2(0.707, -0.707);
    dirs[7] = vec2(-0.707, -0.707);

    for (int i = 0; i < 8; i++) {
      vec3 s = texture2D(tMask, clamp(uv + dirs[i] * off, 0.001, 0.999)).rgb;
      maxNbr = max(maxNbr, s);
      avgNbr += s;
    }
    avgNbr /= 10.0;

    vec3 combined = mix(avgNbr, maxNbr, clamp(uMaskPadding * 1.15, 0.0, 0.92));
    return smoothstep(vec3(0.16), vec3(0.68), combined);
  }

  float selectGenderMask(vec3 rgbMask) {
    float pending = rgbMask.b * uBlurUnknown;
    if (uGenderFilter == 1) {
      return max(rgbMask.r, pending);
    }
    if (uGenderFilter == 2) {
      return max(rgbMask.g, pending);
    }
    return max(rgbMask.r, max(rgbMask.g, rgbMask.b));
  }

  vec3 computePrivacySample(vec2 uv) {
    vec3 nearBlur = texture2D(tBlurNear, uv).rgb;
    vec3 farBlur = texture2D(tBlurFar, uv).rgb;
    vec3 creamyBlur = mix(nearBlur, farBlur, smoothstep(0.2, 0.85, uBlurStrength));

    if (uBlurStyle == 1) {
      vec2 jitter = (vec2(
        hash12(floor(uv * 220.0) + floor(uTime * 12.0) * 0.01),
        hash12(floor(uv * 220.0).yx + 17.3)
      ) - 0.5) * (0.014 * uBlurStrength);
      vec3 frosted = mix(
        texture2D(tBlurNear, clamp(uv + jitter, 0.002, 0.998)).rgb,
        farBlur,
        0.55
      );
      return mix(frosted, vec3(0.86, 0.91, 0.97), 0.14 * uBlurStrength);
    }

    if (uBlurStyle == 2) {
      float cells = mix(96.0, 26.0, uBlurStrength);
      vec2 aspectGrid = vec2(cells * (uSourceSize.x / max(1.0, uSourceSize.y)), cells);
      vec2 quantizedUv = (floor(uv * aspectGrid) + 0.5) / aspectGrid;
      vec3 mosaic = texture2D(tBlurNear, clamp(quantizedUv, 0.002, 0.998)).rgb;
      return mix(creamyBlur, mosaic, 0.85);
    }

    return creamyBlur;
  }

  vec4 drawDetectionBrackets(vec2 uv) {
    if (uShowBoxes < 0.5) return vec4(0.0);
    vec4 hit = vec4(0.0);
    float thickness = 0.0032;

    for (int i = 0; i < ${MAX_BOXES}; i++) {
      if (i >= uBoxCount) break;
      vec4 box = uBoxes[i];
      float genderCode = uBoxGenders[i];

      float x0 = box.x;
      float x1 = box.z;
      float y0 = 1.0 - box.w;
      float y1 = 1.0 - box.y;

      float bw = max(0.01, x1 - x0);
      float bh = max(0.01, y1 - y0);
      float cornerLenX = min(bw * 0.24, 0.032);
      float cornerLenY = min(bh * 0.18, 0.036);

      bool inHorizBand =
        (uv.x >= x0 - thickness && uv.x <= x0 + cornerLenX) ||
        (uv.x >= x1 - cornerLenX && uv.x <= x1 + thickness);
      bool inVertBand =
        (uv.y >= y0 - thickness && uv.y <= y0 + cornerLenY) ||
        (uv.y >= y1 - cornerLenY && uv.y <= y1 + thickness);

      bool onTopBottom =
        inHorizBand &&
        (abs(uv.y - y0) <= thickness || abs(uv.y - y1) <= thickness);
      bool onLeftRight =
        inVertBand &&
        (abs(uv.x - x0) <= thickness * 0.75 || abs(uv.x - x1) <= thickness * 0.75);

      if (onTopBottom || onLeftRight) {
        // Rose = female (1), Sky = male (2), Amber = unknown (0)
        vec3 color = vec3(0.98, 0.75, 0.14);
        if (genderCode > 0.5 && genderCode < 1.5) {
          color = vec3(0.96, 0.45, 0.71);
        } else if (genderCode >= 1.5) {
          color = vec3(0.22, 0.74, 0.97);
        }

        bool isTargeted =
          (uGenderFilter == 0) ||
          (uGenderFilter == 1 && (genderCode > 0.5 && genderCode < 1.5 || (genderCode < 0.5 && uBlurUnknown > 0.5))) ||
          (uGenderFilter == 2 && (genderCode >= 1.5 || (genderCode < 0.5 && uBlurUnknown > 0.5)));

        hit = vec4(color, isTargeted ? 0.95 : 0.42);
      }
    }
    return hit;
  }

  void main() {
    // Background pixels & unselected-gender individuals read untouched tSource directly.
    vec3 sharp = texture2D(tSource, vUv).rgb;
    vec3 rgbMask = sampleFeatheredMaskRgb(vUv);
    float personMask = selectGenderMask(rgbMask) * uEnabled;

    vec3 blurred = computePrivacySample(vUv);
    vec3 finalRgb = mix(sharp, blurred, personMask * smoothstep(0.0, 0.18, uBlurStrength));

    vec4 bracket = drawDetectionBrackets(vUv);
    finalRgb = mix(finalRgb, bracket.rgb, bracket.a);

    gl_FragColor = vec4(finalRgb, 1.0);
  }
`;

const PASSTHROUGH_SHIELD_FRAG = /* glsl */ `
  precision highp float;
  uniform sampler2D tBlurNear;
  uniform sampler2D tBlurFar;
  uniform sampler2D tMask;
  uniform vec2 uMaskTexel;
  uniform float uEnabled;
  uniform float uBlurStrength;
  uniform float uMaskPadding;
  uniform int uGenderFilter;
  uniform float uBlurUnknown;
  varying vec2 vUv;

  vec3 sampleFeatheredMaskRgb(vec2 uv) {
    vec3 center = texture2D(tMask, uv).rgb;
    float pad = mix(1.0, 9.0, uMaskPadding);
    vec2 off = uMaskTexel * pad;
    vec3 maxNbr = center;
    maxNbr = max(maxNbr, texture2D(tMask, clamp(uv + vec2(off.x, 0.0), 0.001, 0.999)).rgb);
    maxNbr = max(maxNbr, texture2D(tMask, clamp(uv - vec2(off.x, 0.0), 0.001, 0.999)).rgb);
    maxNbr = max(maxNbr, texture2D(tMask, clamp(uv + vec2(0.0, off.y), 0.001, 0.999)).rgb);
    maxNbr = max(maxNbr, texture2D(tMask, clamp(uv - vec2(0.0, off.y), 0.001, 0.999)).rgb);
    vec3 combined = mix(center, maxNbr, clamp(uMaskPadding * 1.1, 0.0, 0.9));
    return smoothstep(vec3(0.18), vec3(0.70), combined);
  }

  float selectGenderMask(vec3 rgbMask) {
    float pending = rgbMask.b * uBlurUnknown;
    if (uGenderFilter == 1) {
      return max(rgbMask.r, pending);
    }
    if (uGenderFilter == 2) {
      return max(rgbMask.g, pending);
    }
    return max(rgbMask.r, max(rgbMask.g, rgbMask.b));
  }

  void main() {
    vec3 rgbMask = sampleFeatheredMaskRgb(vUv);
    float mask = selectGenderMask(rgbMask) * uEnabled;
    if (mask < 0.01) discard;
    vec3 nearBlur = texture2D(tBlurNear, vUv).rgb;
    vec3 farBlur = texture2D(tBlurFar, vUv).rgb;
    vec3 veiled = mix(nearBlur, farBlur, smoothstep(0.2, 0.85, uBlurStrength));
    gl_FragColor = vec4(veiled, mask * 0.96);
  }
`;

/**
 * Multi-pass GPU blur and gender-selective mask compositing pipeline.
 * Blurs only pixels matching the selected gender channel while sampling the
 * untouched original camera feed for background pixels and non-targeted people.
 */
export class SelectiveBlurPipeline {
  readonly compositeTarget: THREE.WebGLRenderTarget;
  readonly passthroughMaterial: THREE.ShaderMaterial;

  private blurNearA: THREE.WebGLRenderTarget;
  private blurNearB: THREE.WebGLRenderTarget;
  private blurFarA: THREE.WebGLRenderTarget;
  private blurFarB: THREE.WebGLRenderTarget;

  private quadGeo = new THREE.PlaneGeometry(2, 2);
  private orthoCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  private quadScene = new THREE.Scene();
  private quadMesh: THREE.Mesh;

  private gaussianMaterial: THREE.ShaderMaterial;
  private compositeMaterial: THREE.ShaderMaterial;
  private boxUniformArray: THREE.Vector4[];
  private boxGenderArray: number[];

  constructor(width = 1024, height = 576) {
    this.compositeTarget = new THREE.WebGLRenderTarget(width, height, {
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      format: THREE.RGBAFormat,
      colorSpace: THREE.SRGBColorSpace,
    });

    const halfW = Math.max(128, Math.floor(width / 2));
    const halfH = Math.max(72, Math.floor(height / 2));
    const quarterW = Math.max(64, Math.floor(width / 4));
    const quarterH = Math.max(36, Math.floor(height / 4));

    const rtOpts: THREE.RenderTargetOptions = {
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      format: THREE.RGBAFormat,
      colorSpace: THREE.SRGBColorSpace,
    };

    this.blurNearA = new THREE.WebGLRenderTarget(halfW, halfH, rtOpts);
    this.blurNearB = new THREE.WebGLRenderTarget(halfW, halfH, rtOpts);
    this.blurFarA = new THREE.WebGLRenderTarget(quarterW, quarterH, rtOpts);
    this.blurFarB = new THREE.WebGLRenderTarget(quarterW, quarterH, rtOpts);

    this.gaussianMaterial = new THREE.ShaderMaterial({
      uniforms: {
        tDiffuse: {value: null},
        uTexelSize: {value: new THREE.Vector2(1 / halfW, 1 / halfH)},
        uDirection: {value: new THREE.Vector2(1, 0)},
        uRadius: {value: 1.5},
      },
      vertexShader: FULLSCREEN_VERT,
      fragmentShader: GAUSSIAN_FRAG,
      depthTest: false,
      depthWrite: false,
    });

    this.boxUniformArray = Array.from({length: MAX_BOXES}, () => new THREE.Vector4());
    this.boxGenderArray = new Array<number>(MAX_BOXES).fill(0);

    this.compositeMaterial = new THREE.ShaderMaterial({
      uniforms: {
        tSource: {value: null},
        tBlurNear: {value: this.blurNearB.texture},
        tBlurFar: {value: this.blurFarB.texture},
        tMask: {value: null},
        uSourceSize: {value: new THREE.Vector2(width, height)},
        uMaskTexel: {value: new THREE.Vector2(1 / 256, 1 / 256)},
        uEnabled: {value: 1},
        uBlurStrength: {value: 0.8},
        uMaskPadding: {value: 0.35},
        uBlurStyle: {value: 0},
        uGenderFilter: {value: 0},
        uBlurUnknown: {value: 1},
        uShowBoxes: {value: 0},
        uBoxCount: {value: 0},
        uBoxes: {value: this.boxUniformArray},
        uBoxGenders: {value: this.boxGenderArray},
        uTime: {value: 0},
      },
      vertexShader: FULLSCREEN_VERT,
      fragmentShader: COMPOSITE_FRAG,
      depthTest: false,
      depthWrite: false,
    });

    this.passthroughMaterial = new THREE.ShaderMaterial({
      uniforms: {
        tBlurNear: {value: this.blurNearB.texture},
        tBlurFar: {value: this.blurFarB.texture},
        tMask: {value: null},
        uMaskTexel: {value: new THREE.Vector2(1 / 256, 1 / 256)},
        uEnabled: {value: 1},
        uBlurStrength: {value: 0.8},
        uMaskPadding: {value: 0.35},
        uGenderFilter: {value: 0},
        uBlurUnknown: {value: 1},
      },
      vertexShader: PASS_VERT,
      fragmentShader: PASSTHROUGH_SHIELD_FRAG,
      transparent: true,
      depthTest: false,
      depthWrite: false,
    });

    this.quadMesh = new THREE.Mesh(this.quadGeo, this.gaussianMaterial);
    this.quadMesh.frustumCulled = false;
    this.quadScene.add(this.quadMesh);
  }

  get compositeTexture(): THREE.Texture {
    return this.compositeTarget.texture;
  }

  render(
    renderer: THREE.WebGLRenderer,
    params: {
      sourceTexture: THREE.Texture;
      maskTexture: THREE.Texture;
      enabled: boolean;
      blurStrength: number;
      maskPadding: number;
      blurStyle: BlurStyle;
      genderFilter: GenderFilter;
      blurUnknown: boolean;
      showBoxes: boolean;
      boxes: DetectedPersonBox[];
      elapsedTime: number;
    },
  ): void {
    const prevTarget = renderer.getRenderTarget();
    const prevXrEnabled = renderer.xr.enabled;
    renderer.xr.enabled = false;

    const radiusNear = 0.9 + params.blurStrength * 1.8;
    const radiusFar = 1.4 + params.blurStrength * 2.8;

    // Pass 1 & 2: Half-resolution separable Gaussian blur.
    this.quadMesh.material = this.gaussianMaterial;
    this.gaussianMaterial.uniforms.tDiffuse.value = params.sourceTexture;
    this.gaussianMaterial.uniforms.uTexelSize.value.set(
      1 / this.blurNearA.width,
      1 / this.blurNearA.height,
    );
    this.gaussianMaterial.uniforms.uDirection.value.set(1, 0);
    this.gaussianMaterial.uniforms.uRadius.value = radiusNear;
    renderer.setRenderTarget(this.blurNearA);
    renderer.render(this.quadScene, this.orthoCam);

    this.gaussianMaterial.uniforms.tDiffuse.value = this.blurNearA.texture;
    this.gaussianMaterial.uniforms.uDirection.value.set(0, 1);
    renderer.setRenderTarget(this.blurNearB);
    renderer.render(this.quadScene, this.orthoCam);

    // Pass 3 & 4: Quarter-resolution wider Gaussian blur for deep privacy anonymity.
    this.gaussianMaterial.uniforms.tDiffuse.value = this.blurNearB.texture;
    this.gaussianMaterial.uniforms.uTexelSize.value.set(
      1 / this.blurFarA.width,
      1 / this.blurFarA.height,
    );
    this.gaussianMaterial.uniforms.uDirection.value.set(1, 0);
    this.gaussianMaterial.uniforms.uRadius.value = radiusFar;
    renderer.setRenderTarget(this.blurFarA);
    renderer.render(this.quadScene, this.orthoCam);

    this.gaussianMaterial.uniforms.tDiffuse.value = this.blurFarA.texture;
    this.gaussianMaterial.uniforms.uDirection.value.set(0, 1);
    renderer.setRenderTarget(this.blurFarB);
    renderer.render(this.quadScene, this.orthoCam);

    // Pass 5: Composite gender-selective blur onto untouched sharp background.
    const boxCount = Math.min(params.boxes.length, MAX_BOXES);
    for (let i = 0; i < boxCount; i++) {
      const b = params.boxes[i];
      this.boxUniformArray[i].set(b.xmin, b.ymin, b.xmax, b.ymax);
      this.boxGenderArray[i] = b.gender === 'female' ? 1 : b.gender === 'male' ? 2 : 0;
    }

    const styleIndex =
      params.blurStyle === BlurStyle.Frosted
        ? 1
        : params.blurStyle === BlurStyle.Mosaic
          ? 2
          : 0;

    const genderFilterIndex =
      params.genderFilter === GenderFilter.Female
        ? 1
        : params.genderFilter === GenderFilter.Male
          ? 2
          : 0;

    const u = this.compositeMaterial.uniforms;
    u.tSource.value = params.sourceTexture;
    u.tMask.value = params.maskTexture;
    u.uEnabled.value = params.enabled ? 1 : 0;
    u.uBlurStrength.value = params.blurStrength;
    u.uMaskPadding.value = params.maskPadding;
    u.uBlurStyle.value = styleIndex;
    u.uGenderFilter.value = genderFilterIndex;
    u.uBlurUnknown.value = params.blurUnknown ? 1 : 0;
    u.uShowBoxes.value = params.showBoxes ? 1 : 0;
    u.uBoxCount.value = boxCount;
    u.uTime.value = params.elapsedTime;

    const maskImg = params.maskTexture.image as {width?: number; height?: number} | undefined;
    const maskW = maskImg?.width || 256;
    const maskH = maskImg?.height || 256;
    u.uMaskTexel.value.set(1 / maskW, 1 / maskH);

    // Sync passthrough shield uniforms too.
    const pu = this.passthroughMaterial.uniforms;
    pu.tMask.value = params.maskTexture;
    pu.uMaskTexel.value.set(1 / maskW, 1 / maskH);
    pu.uEnabled.value = params.enabled ? 1 : 0;
    pu.uBlurStrength.value = params.blurStrength;
    pu.uMaskPadding.value = params.maskPadding;
    pu.uGenderFilter.value = genderFilterIndex;
    pu.uBlurUnknown.value = params.blurUnknown ? 1 : 0;

    this.quadMesh.material = this.compositeMaterial;
    renderer.setRenderTarget(this.compositeTarget);
    renderer.render(this.quadScene, this.orthoCam);

    renderer.xr.enabled = prevXrEnabled;
    renderer.setRenderTarget(prevTarget);
  }

  dispose(): void {
    this.compositeTarget.dispose();
    this.blurNearA.dispose();
    this.blurNearB.dispose();
    this.blurFarA.dispose();
    this.blurFarB.dispose();
    this.quadGeo.dispose();
    this.gaussianMaterial.dispose();
    this.compositeMaterial.dispose();
    this.passthroughMaterial.dispose();
  }
}
