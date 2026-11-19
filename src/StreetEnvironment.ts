import * as THREE from 'three';
import * as xb from 'xrblocks';
import {RoomEnvironment} from 'three/addons/environments/RoomEnvironment.js';
import {DetectedPersonBox, PersonGender} from './types.ts';

interface PedestrianActor {
  id: number;
  gender: Exclude<PersonGender, 'unknown'>;
  root: THREE.Group;
  leftLeg: THREE.Group;
  rightLeg: THREE.Group;
  leftArm: THREE.Group;
  rightArm: THREE.Group;
  torso: THREE.Group;
  head: THREE.Group;
  axis: 'z' | 'x';
  lanePos: number;
  minRange: number;
  maxRange: number;
  direction: 1 | -1;
  speed: number;
  phase: number;
  heightScale: number;
  worldMeshes: THREE.Mesh[];
}

const _projVec = new THREE.Vector3();
const _box3 = new THREE.Box3();
const _corners = Array.from({length: 8}, () => new THREE.Vector3());

/**
 * Procedural 3D city street environment with sharp architectural signage and
 * articulated walking women and men. Outputs an RGBA channel-coded pedestrian
 * mask (R = female, G = male) so Desktop Simulator mode demonstrates
 * gender-based selective blurring while keeping street backgrounds sharp.
 */
export class StreetEnvironment extends xb.Script {
  private streetRoot = new THREE.Group();
  private backgroundGroup = new THREE.Group();
  private pedestriansGroup = new THREE.Group();
  private pedestrians: PedestrianActor[] = [];

  private textures: THREE.Texture[] = [];
  private geometries: THREE.BufferGeometry[] = [];
  private materials: THREE.Material[] = [];
  private envRenderTarget: THREE.WebGLRenderTarget | null = null;

  private colorTarget: THREE.WebGLRenderTarget;
  private maskTarget: THREE.WebGLRenderTarget;
  private feedCamera = new THREE.PerspectiveCamera(58, 16 / 9, 0.1, 120);

  private blackOcclusionMaterial = new THREE.MeshBasicMaterial({
    color: 0x000000,
    depthWrite: true,
    depthTest: true,
  });
  private femaleMaskMaterial = new THREE.MeshBasicMaterial({
    color: 0xff0000,
    depthWrite: true,
    depthTest: true,
  });
  private maleMaskMaterial = new THREE.MeshBasicMaterial({
    color: 0x00ff00,
    depthWrite: true,
    depthTest: true,
  });

  private originalMaterials = new Map<THREE.Mesh, THREE.Material | THREE.Material[]>();
  private savedClearColor = new THREE.Color();
  private savedFog: THREE.Fog | THREE.FogExp2 | null = null;
  private savedBackground: THREE.Color | THREE.Texture | null = null;
  private elapsedWalkTime = 0;

  constructor(width = 1024, height = 576) {
    super();
    this.colorTarget = new THREE.WebGLRenderTarget(width, height, {
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      format: THREE.RGBAFormat,
      colorSpace: THREE.SRGBColorSpace,
    });
    this.maskTarget = new THREE.WebGLRenderTarget(
      Math.max(256, Math.floor(width / 2)),
      Math.max(144, Math.floor(height / 2)),
      {
        minFilter: THREE.LinearFilter,
        magFilter: THREE.LinearFilter,
        format: THREE.RGBAFormat,
      },
    );
  }

  get colorTexture(): THREE.Texture {
    return this.colorTarget.texture;
  }

  get maskTexture(): THREE.Texture {
    return this.maskTarget.texture;
  }

  init() {
    this.streetRoot.name = 'Street environment';
    this.streetRoot.xb = {pointerEvents: 'none'};
    this.add(this.streetRoot);
    this.streetRoot.add(this.backgroundGroup, this.pedestriansGroup);

    this.setupLightingAndEnvironment();
    this.buildRoadAndSidewalks();
    this.buildBuildingsAndSignage();
    this.buildStreetProps();
    this.spawnPedestrians();
  }

  setVisible(visible: boolean) {
    this.streetRoot.visible = visible;
    const scene = xb.core.scene;
    if (!visible && scene) {
      if (scene.fog) scene.fog = null;
      if (scene.background instanceof THREE.Color) {
        scene.background = null;
      }
    } else if (visible && scene) {
      scene.background = new THREE.Color(0xd7e6f5);
      scene.fog = new THREE.FogExp2(0xd7e6f5, 0.015);
    }
  }

  update() {
    if (!this.streetRoot.visible) return;
    const dt = Math.min(xb.getDeltaTime(), 0.05);
    this.elapsedWalkTime += dt;

    for (const actor of this.pedestrians) {
      const posProp = actor.axis;
      let coord = actor.root.position[posProp] + actor.direction * actor.speed * dt;
      if (coord > actor.maxRange) {
        coord = actor.maxRange;
        actor.direction = -1;
      } else if (coord < actor.minRange) {
        coord = actor.minRange;
        actor.direction = 1;
      }
      actor.root.position[posProp] = coord;

      if (actor.axis === 'z') {
        actor.root.rotation.y = actor.direction > 0 ? 0 : Math.PI;
      } else {
        actor.root.rotation.y = actor.direction > 0 ? Math.PI / 2 : -Math.PI / 2;
      }

      const stride = this.elapsedWalkTime * actor.speed * 4.2 + actor.phase;
      const swing = Math.sin(stride) * 0.55;
      actor.leftLeg.rotation.x = swing;
      actor.rightLeg.rotation.x = -swing;
      actor.leftArm.rotation.x = -swing * 0.72;
      actor.rightArm.rotation.x = swing * 0.72;
      actor.torso.position.y = Math.abs(Math.cos(stride)) * 0.025;
      actor.head.rotation.y = Math.sin(stride * 0.35) * 0.08;
    }
  }

  /**
   * Renders the street scene cleanly into colorTarget and a channel-coded
   * gender mask into maskTarget (Red = female, Green = male).
   */
  renderCameraFeed(
    renderer: THREE.WebGLRenderer,
    sourceCamera: THREE.Camera,
    hiddenObjects: THREE.Object3D[],
  ): DetectedPersonBox[] {
    const scene = xb.core.scene;
    if (!scene || !this.streetRoot.visible) return [];

    sourceCamera.updateMatrixWorld();
    this.feedCamera.position.setFromMatrixPosition(sourceCamera.matrixWorld);
    this.feedCamera.quaternion.setFromRotationMatrix(sourceCamera.matrixWorld);
    if (sourceCamera instanceof THREE.PerspectiveCamera) {
      this.feedCamera.fov = Math.min(Math.max(sourceCamera.fov, 45), 70);
    }
    this.feedCamera.aspect = this.colorTarget.width / this.colorTarget.height;
    this.feedCamera.updateProjectionMatrix();
    this.feedCamera.updateMatrixWorld(true);

    const previousVisibility = hiddenObjects.map((obj) => obj.visible);
    for (const obj of hiddenObjects) obj.visible = false;

    const prevTarget = renderer.getRenderTarget();
    const prevXrEnabled = renderer.xr.enabled;
    renderer.xr.enabled = false;

    // 1. Render crisp color street feed.
    renderer.setRenderTarget(this.colorTarget);
    renderer.clear();
    renderer.render(scene, this.feedCamera);

    // 2. Render occlusion-aware gender-channel pedestrian mask.
    this.savedFog = scene.fog;
    this.savedBackground = scene.background as THREE.Color | THREE.Texture | null;
    renderer.getClearColor(this.savedClearColor);
    const savedClearAlpha = renderer.getClearAlpha();

    scene.fog = null;
    scene.background = new THREE.Color(0x000000);
    this.swapMaterialsForMaskPass();

    renderer.setRenderTarget(this.maskTarget);
    renderer.setClearColor(0x000000, 1);
    renderer.clear();
    renderer.render(scene, this.feedCamera);

    this.restoreOriginalMaterials();
    scene.fog = this.savedFog;
    scene.background = this.savedBackground;
    renderer.setClearColor(this.savedClearColor, savedClearAlpha);

    renderer.xr.enabled = prevXrEnabled;
    renderer.setRenderTarget(prevTarget);
    hiddenObjects.forEach((obj, idx) => {
      obj.visible = previousVisibility[idx];
    });

    return this.computeVisibleBoxes();
  }

  private computeVisibleBoxes(): DetectedPersonBox[] {
    const boxes: DetectedPersonBox[] = [];
    for (const actor of this.pedestrians) {
      _box3.setFromObject(actor.root);
      if (_box3.isEmpty()) continue;

      const {min, max} = _box3;
      _corners[0].set(min.x, min.y, min.z);
      _corners[1].set(min.x, min.y, max.z);
      _corners[2].set(min.x, max.y, min.z);
      _corners[3].set(min.x, max.y, max.z);
      _corners[4].set(max.x, min.y, min.z);
      _corners[5].set(max.x, min.y, max.z);
      _corners[6].set(max.x, max.y, min.z);
      _corners[7].set(max.x, max.y, max.z);

      let xmin = 1;
      let ymin = 1;
      let xmax = 0;
      let ymax = 0;
      let anyInFront = false;

      for (const corner of _corners) {
        _projVec.copy(corner).project(this.feedCamera);
        if (_projVec.z < 1 && _projVec.z > -1) {
          anyInFront = true;
          const u = (_projVec.x + 1) * 0.5;
          const vTop = 1 - (_projVec.y + 1) * 0.5;
          xmin = Math.min(xmin, u);
          xmax = Math.max(xmax, u);
          ymin = Math.min(ymin, vTop);
          ymax = Math.max(ymax, vTop);
        }
      }

      if (!anyInFront) continue;
      xmin = Math.max(0, xmin);
      ymin = Math.max(0, ymin);
      xmax = Math.min(1, xmax);
      ymax = Math.min(1, ymax);

      const width = xmax - xmin;
      const height = ymax - ymin;
      if (width > 0.015 && height > 0.03) {
        boxes.push({
          trackId: actor.id,
          xmin,
          ymin,
          xmax,
          ymax,
          score: 0.98,
          gender: actor.gender,
          genderConfidence: 1.0,
        });
      }
    }
    return boxes;
  }

  private swapMaterialsForMaskPass() {
    this.originalMaterials.clear();
    this.backgroundGroup.traverse((obj) => {
      if (obj instanceof THREE.Mesh) {
        this.originalMaterials.set(obj, obj.material);
        obj.material = this.blackOcclusionMaterial;
      }
    });
    for (const actor of this.pedestrians) {
      const maskMat =
        actor.gender === 'female' ? this.femaleMaskMaterial : this.maleMaskMaterial;
      for (const mesh of actor.worldMeshes) {
        this.originalMaterials.set(mesh, mesh.material);
        mesh.material = maskMat;
      }
    }
  }

  private restoreOriginalMaterials() {
    for (const [mesh, mat] of this.originalMaterials.entries()) {
      mesh.material = mat;
    }
    this.originalMaterials.clear();
  }

  private setupLightingAndEnvironment() {
    const renderer = xb.core.renderer;
    const scene = xb.core.scene;
    if (renderer && scene && renderer instanceof THREE.WebGLRenderer) {
      const pmrem = new THREE.PMREMGenerator(renderer);
      this.envRenderTarget = pmrem.fromScene(new RoomEnvironment(), 0.04);
      scene.environment = this.envRenderTarget.texture;
      pmrem.dispose();
    }

    if (scene) {
      scene.background = new THREE.Color(0xd7e6f5);
      scene.fog = new THREE.FogExp2(0xd7e6f5, 0.015);
    }

    const hemi = new THREE.HemisphereLight(0xeef6ff, 0x4b5563, 1.35);
    const sun = new THREE.DirectionalLight(0xfff5e4, 2.2);
    sun.position.set(14, 22, 10);
    const fill = new THREE.DirectionalLight(0xbad7ff, 0.65);
    fill.position.set(-12, 10, -14);

    this.backgroundGroup.add(hemi, sun, fill);
  }

  private track<T extends THREE.BufferGeometry>(geom: T): T {
    this.geometries.push(geom);
    return geom;
  }

  private trackMat<T extends THREE.Material>(mat: T): T {
    this.materials.push(mat);
    return mat;
  }

  private trackTex<T extends THREE.Texture>(tex: T): T {
    tex.colorSpace = THREE.SRGBColorSpace;
    const renderer = xb.core.renderer;
    if (renderer && renderer instanceof THREE.WebGLRenderer) {
      tex.anisotropy = Math.min(8, renderer.capabilities.getMaxAnisotropy());
    }
    this.textures.push(tex);
    return tex;
  }

  private buildRoadAndSidewalks() {
    const roadCanvas = document.createElement('canvas');
    roadCanvas.width = 512;
    roadCanvas.height = 1024;
    const rCtx = roadCanvas.getContext('2d')!;
    rCtx.fillStyle = '#2d3138';
    rCtx.fillRect(0, 0, 512, 1024);

    for (let i = 0; i < 6500; i++) {
      const x = Math.random() * 512;
      const y = Math.random() * 1024;
      const v = 35 + Math.floor(Math.random() * 30);
      rCtx.fillStyle = `rgb(${v},${v + 2},${v + 5})`;
      rCtx.fillRect(x, y, 2, 2);
    }

    rCtx.fillStyle = '#eab308';
    rCtx.fillRect(247, 0, 6, 1024);
    rCtx.fillRect(259, 0, 6, 1024);

    rCtx.fillStyle = '#f1f5f9';
    rCtx.fillRect(24, 0, 10, 1024);
    rCtx.fillRect(478, 0, 10, 1024);

    const roadTex = this.trackTex(new THREE.CanvasTexture(roadCanvas));
    roadTex.wrapS = THREE.ClampToEdgeWrapping;
    roadTex.wrapT = THREE.RepeatWrapping;
    roadTex.repeat.set(1, 6);

    const roadMat = this.trackMat(
      new THREE.MeshStandardMaterial({
        map: roadTex,
        roughness: 0.85,
        metalness: 0.05,
      }),
    );
    const road = new THREE.Mesh(this.track(new THREE.PlaneGeometry(7.2, 64)), roadMat);
    road.rotation.x = -Math.PI / 2;
    road.position.set(0, 0, -14);
    this.backgroundGroup.add(road);

    const stripeMat = this.trackMat(
      new THREE.MeshStandardMaterial({color: 0xf8fafc, roughness: 0.7}),
    );
    const stripeGeo = this.track(new THREE.PlaneGeometry(0.55, 2.4));
    for (let i = -5; i <= 5; i++) {
      const stripe = new THREE.Mesh(stripeGeo, stripeMat);
      stripe.rotation.x = -Math.PI / 2;
      stripe.position.set(i * 0.62, 0.01, -7.2);
      this.backgroundGroup.add(stripe);
    }

    const tileCanvas = document.createElement('canvas');
    tileCanvas.width = 512;
    tileCanvas.height = 512;
    const tCtx = tileCanvas.getContext('2d')!;
    tCtx.fillStyle = '#9ca3af';
    tCtx.fillRect(0, 0, 512, 512);
    tCtx.strokeStyle = '#6b7280';
    tCtx.lineWidth = 4;
    for (let i = 0; i <= 512; i += 128) {
      tCtx.beginPath();
      tCtx.moveTo(i, 0);
      tCtx.lineTo(i, 512);
      tCtx.stroke();
      tCtx.beginPath();
      tCtx.moveTo(0, i);
      tCtx.lineTo(512, i);
      tCtx.stroke();
    }
    const tileTex = this.trackTex(new THREE.CanvasTexture(tileCanvas));
    tileTex.wrapS = THREE.RepeatWrapping;
    tileTex.wrapT = THREE.RepeatWrapping;
    tileTex.repeat.set(3, 24);

    const sidewalkMat = this.trackMat(
      new THREE.MeshStandardMaterial({map: tileTex, roughness: 0.88}),
    );
    const sidewalkGeo = this.track(new THREE.BoxGeometry(5.5, 0.14, 64));

    const leftWalk = new THREE.Mesh(sidewalkGeo, sidewalkMat);
    leftWalk.position.set(-6.35, 0.07, -14);
    const rightWalk = new THREE.Mesh(sidewalkGeo, sidewalkMat);
    rightWalk.position.set(6.35, 0.07, -14);
    this.backgroundGroup.add(leftWalk, rightWalk);
  }

  private createSignTexture(
    title: string,
    subtitle: string,
    bg: string,
    fg: string,
    accent: string,
  ): THREE.CanvasTexture {
    const canvas = document.createElement('canvas');
    canvas.width = 1024;
    canvas.height = 256;
    const ctx = canvas.getContext('2d')!;
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, 1024, 256);

    ctx.strokeStyle = accent;
    ctx.lineWidth = 10;
    ctx.strokeRect(14, 14, 996, 228);

    ctx.fillStyle = fg;
    ctx.font = 'bold 84px "Segoe UI", Roboto, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(title, 512, 102);

    ctx.fillStyle = accent;
    ctx.font = '600 40px "Segoe UI", Roboto, sans-serif';
    ctx.fillText(subtitle, 512, 178);

    return this.trackTex(new THREE.CanvasTexture(canvas));
  }

  private createFacadeTexture(baseColor: string, windowGlow: string): THREE.CanvasTexture {
    const canvas = document.createElement('canvas');
    canvas.width = 512;
    canvas.height = 512;
    const ctx = canvas.getContext('2d')!;
    ctx.fillStyle = baseColor;
    ctx.fillRect(0, 0, 512, 512);

    for (let row = 0; row < 4; row++) {
      for (let col = 0; col < 4; col++) {
        const x = 36 + col * 116;
        const y = 32 + row * 118;
        ctx.fillStyle = '#1f2937';
        ctx.fillRect(x - 4, y - 4, 84, 88);
        ctx.fillStyle = windowGlow;
        ctx.fillRect(x, y, 76, 80);
        ctx.fillStyle = '#334155';
        ctx.fillRect(x + 36, y, 4, 80);
        ctx.fillRect(x, y + 38, 76, 4);
      }
    }
    return this.trackTex(new THREE.CanvasTexture(canvas));
  }

  private buildBuildingsAndSignage() {
    const configs = [
      {
        x: -10.8,
        z: -5,
        w: 6,
        h: 11,
        d: 9,
        wall: '#b87654',
        win: '#cfe8ff',
        sign: ['CAFÉ LUMIÈRE', 'ESPRESSO • BAKERY • 07:00–21:00', '#111827', '#ffffff', '#f59e0b'],
        side: 1,
      },
      {
        x: -11.2,
        z: -15.5,
        w: 6.5,
        h: 14,
        d: 10,
        wall: '#64748b',
        win: '#e2e8f0',
        sign: ['CITY BOOKS', 'ARCHITECTURE • ART • DESIGN', '#0f172a', '#f8fafc', '#38bdf8'],
        side: 1,
      },
      {
        x: -10.6,
        z: -27,
        w: 6,
        h: 12,
        d: 10,
        wall: '#94a3b8',
        win: '#dbeafe',
        sign: ['METRO LINE 4', 'CENTRAL STATION →', '#1e3a8a', '#ffffff', '#93c5fd'],
        side: 1,
      },
      {
        x: 10.8,
        z: -5.5,
        w: 6,
        h: 12,
        d: 9,
        wall: '#cbd5e1',
        win: '#bfdbfe',
        sign: ['OPTIC LAB', 'PRECISION LENSES & CAMERAS', '#0f172a', '#ffffff', '#34d399'],
        side: -1,
      },
      {
        x: 11.2,
        z: -16.5,
        w: 6.5,
        h: 15,
        d: 10,
        wall: '#a16247',
        win: '#fde68a',
        sign: ['MARKET HALL', 'FRESH PRODUCE • OPEN DAILY', '#14532d', '#ffffff', '#86efac'],
        side: -1,
      },
      {
        x: 10.8,
        z: -28,
        w: 6,
        h: 11,
        d: 10,
        wall: '#475569',
        win: '#e0f2fe',
        sign: ['GALLERY 24', 'CONTEMPORARY PHOTOGRAPHY', '#18181b', '#fafafa', '#f472b6'],
        side: -1,
      },
    ] as const;

    for (const b of configs) {
      const group = new THREE.Group();
      group.position.set(b.x, 0, b.z);

      const facadeTex = this.createFacadeTexture(b.wall, b.win);
      const wallMat = this.trackMat(
        new THREE.MeshStandardMaterial({map: facadeTex, roughness: 0.78}),
      );
      const body = new THREE.Mesh(this.track(new THREE.BoxGeometry(b.w, b.h, b.d)), wallMat);
      body.position.y = b.h / 2;
      group.add(body);

      const awningMat = this.trackMat(
        new THREE.MeshStandardMaterial({color: 0x1e293b, roughness: 0.5, metalness: 0.3}),
      );
      const awning = new THREE.Mesh(this.track(new THREE.BoxGeometry(1.1, 0.18, b.d - 0.8)), awningMat);
      awning.position.set(b.side * (b.w / 2 + 0.45), 3.1, 0);
      group.add(awning);

      const signTex = this.createSignTexture(
        b.sign[0],
        b.sign[1],
        b.sign[2],
        b.sign[3],
        b.sign[4],
      );
      const signMat = this.trackMat(
        new THREE.MeshStandardMaterial({map: signTex, roughness: 0.35, metalness: 0.1}),
      );
      const signBoard = new THREE.Mesh(this.track(new THREE.PlaneGeometry(b.d * 0.78, 1.15)), signMat);
      signBoard.position.set(b.side * (b.w / 2 + 0.04), 3.85, 0);
      signBoard.rotation.y = b.side > 0 ? Math.PI / 2 : -Math.PI / 2;
      group.add(signBoard);

      this.backgroundGroup.add(group);
    }
  }

  private buildStreetProps() {
    const poleMat = this.trackMat(
      new THREE.MeshStandardMaterial({color: 0x1f2937, roughness: 0.35, metalness: 0.8}),
    );
    const bulbMat = this.trackMat(
      new THREE.MeshStandardMaterial({
        color: 0xfffbeb,
        emissive: 0xfde68a,
        emissiveIntensity: 0.9,
        roughness: 0.2,
      }),
    );
    const trunkMat = this.trackMat(
      new THREE.MeshStandardMaterial({color: 0x5c4033, roughness: 0.9}),
    );
    const leafMat = this.trackMat(
      new THREE.MeshStandardMaterial({color: 0x3b7a47, roughness: 0.78}),
    );

    const lampPositions = [
      [-4.0, -3],
      [-4.0, -11],
      [-4.0, -20],
      [4.0, -3],
      [4.0, -11],
      [4.0, -20],
    ];

    for (const [x, z] of lampPositions) {
      const lamp = new THREE.Group();
      lamp.position.set(x, 0.14, z);
      const base = new THREE.Mesh(this.track(new THREE.CylinderGeometry(0.11, 0.15, 0.4, 12)), poleMat);
      base.position.y = 0.2;
      const shaft = new THREE.Mesh(this.track(new THREE.CylinderGeometry(0.045, 0.06, 3.6, 12)), poleMat);
      shaft.position.y = 2.0;
      const arm = new THREE.Mesh(this.track(new THREE.BoxGeometry(0.65, 0.06, 0.06)), poleMat);
      arm.position.set(x < 0 ? 0.3 : -0.3, 3.75, 0);
      const lantern = new THREE.Mesh(this.track(new THREE.SphereGeometry(0.14, 14, 12)), bulbMat);
      lantern.position.set(x < 0 ? 0.58 : -0.58, 3.62, 0);
      lamp.add(base, shaft, arm, lantern);
      this.backgroundGroup.add(lamp);
    }

    const treePositions = [
      [-5.3, -7],
      [-5.3, -17],
      [5.3, -7.5],
      [5.3, -17.5],
    ];
    for (const [x, z] of treePositions) {
      const tree = new THREE.Group();
      tree.position.set(x, 0.14, z);
      const trunk = new THREE.Mesh(this.track(new THREE.CylinderGeometry(0.14, 0.2, 2.2, 10)), trunkMat);
      trunk.position.y = 1.1;
      const crownA = new THREE.Mesh(this.track(new THREE.DodecahedronGeometry(1.05, 1)), leafMat);
      crownA.position.set(0, 2.75, 0);
      const crownB = new THREE.Mesh(this.track(new THREE.DodecahedronGeometry(0.8, 1)), leafMat);
      crownB.position.set(0.45, 2.45, 0.3);
      const crownC = new THREE.Mesh(this.track(new THREE.DodecahedronGeometry(0.78, 1)), leafMat);
      crownC.position.set(-0.42, 2.5, -0.28);
      tree.add(trunk, crownA, crownB, crownC);
      this.backgroundGroup.add(tree);
    }
  }

  private spawnPedestrians() {
    const configs: Array<{
      gender: Exclude<PersonGender, 'unknown'>;
      axis: 'z' | 'x';
      lanePos: number;
      start: number;
      minRange: number;
      maxRange: number;
      direction: 1 | -1;
      speed: number;
      coat: number;
      pants: number;
      skin: number;
      hair: number;
      scale: number;
    }> = [
      {
        gender: 'female',
        axis: 'z',
        lanePos: -4.8,
        start: -4.8,
        minRange: -18,
        maxRange: -2.4,
        direction: -1,
        speed: 1.22,
        coat: 0xe11d48,
        pants: 0x1e293b,
        skin: 0xf1c27d,
        hair: 0x451a03,
        scale: 0.94,
      },
      {
        gender: 'male',
        axis: 'z',
        lanePos: 4.9,
        start: -11.5,
        minRange: -19,
        maxRange: -2.5,
        direction: 1,
        speed: 1.35,
        coat: 0x2563eb,
        pants: 0x334155,
        skin: 0xe8be9e,
        hair: 0x1c1917,
        scale: 1.03,
      },
      {
        gender: 'female',
        axis: 'x',
        lanePos: -7.1,
        start: -2.6,
        minRange: -5.2,
        maxRange: 5.2,
        direction: 1,
        speed: 1.15,
        coat: 0x9333ea,
        pants: 0x111827,
        skin: 0xc68642,
        hair: 0x09090b,
        scale: 0.95,
      },
      {
        gender: 'male',
        axis: 'x',
        lanePos: -7.7,
        start: 2.2,
        minRange: -5.2,
        maxRange: 5.2,
        direction: -1,
        speed: 1.12,
        coat: 0x059669,
        pants: 0x1f2937,
        skin: 0x8d5524,
        hair: 0x18181b,
        scale: 1.04,
      },
      {
        gender: 'female',
        axis: 'z',
        lanePos: -5.9,
        start: -10.5,
        minRange: -20,
        maxRange: -3,
        direction: 1,
        speed: 1.18,
        coat: 0xd97706,
        pants: 0x0f172a,
        skin: 0xffdbac,
        hair: 0x713f12,
        scale: 0.93,
      },
      {
        gender: 'male',
        axis: 'z',
        lanePos: 5.9,
        start: -6.5,
        minRange: -18,
        maxRange: -2.4,
        direction: -1,
        speed: 1.28,
        coat: 0x0f766e,
        pants: 0x1e293b,
        skin: 0xf1c27d,
        hair: 0x292524,
        scale: 1.01,
      },
    ];

    configs.forEach((cfg, index) => {
      const actor = this.createPedestrian(index + 1, cfg, index * 1.37);
      if (cfg.axis === 'z') {
        actor.root.position.set(cfg.lanePos, 0.14, cfg.start);
      } else {
        actor.root.position.set(cfg.start, 0.01, cfg.lanePos);
      }
      this.pedestrians.push(actor);
      this.pedestriansGroup.add(actor.root);
    });
  }

  private createPedestrian(
    id: number,
    cfg: {
      gender: Exclude<PersonGender, 'unknown'>;
      axis: 'z' | 'x';
      lanePos: number;
      minRange: number;
      maxRange: number;
      direction: 1 | -1;
      speed: number;
      coat: number;
      pants: number;
      skin: number;
      hair: number;
      scale: number;
    },
    phase: number,
  ): PedestrianActor {
    const root = new THREE.Group();
    root.scale.setScalar(cfg.scale);

    const isFemale = cfg.gender === 'female';
    const coatMat = this.trackMat(new THREE.MeshStandardMaterial({color: cfg.coat, roughness: 0.72}));
    const pantsMat = this.trackMat(new THREE.MeshStandardMaterial({color: cfg.pants, roughness: 0.8}));
    const skinMat = this.trackMat(new THREE.MeshStandardMaterial({color: cfg.skin, roughness: 0.6}));
    const hairMat = this.trackMat(new THREE.MeshStandardMaterial({color: cfg.hair, roughness: 0.75}));
    const shoeMat = this.trackMat(new THREE.MeshStandardMaterial({color: 0x111827, roughness: 0.45}));
    const bagMat = this.trackMat(new THREE.MeshStandardMaterial({color: 0x374151, roughness: 0.65}));

    const worldMeshes: THREE.Mesh[] = [];

    const makeLimb = (w: number, h: number, d: number, mat: THREE.Material) => {
      const pivot = new THREE.Group();
      const mesh = new THREE.Mesh(this.track(new THREE.BoxGeometry(w, h, d)), mat);
      mesh.position.y = -h / 2;
      pivot.add(mesh);
      worldMeshes.push(mesh);
      return pivot;
    };

    const legWidth = isFemale ? 0.12 : 0.145;
    const leftLeg = makeLimb(legWidth, 0.78, 0.14, pantsMat);
    leftLeg.position.set(isFemale ? -0.085 : -0.105, 0.82, 0);
    const leftShoe = new THREE.Mesh(this.track(new THREE.BoxGeometry(0.14, 0.08, 0.23)), shoeMat);
    leftShoe.position.set(0, -0.78, 0.04);
    leftLeg.add(leftShoe);
    worldMeshes.push(leftShoe);

    const rightLeg = makeLimb(legWidth, 0.78, 0.14, pantsMat);
    rightLeg.position.set(isFemale ? 0.085 : 0.105, 0.82, 0);
    const rightShoe = new THREE.Mesh(this.track(new THREE.BoxGeometry(0.14, 0.08, 0.23)), shoeMat);
    rightShoe.position.set(0, -0.78, 0.04);
    rightLeg.add(rightShoe);
    worldMeshes.push(rightShoe);

    const torso = new THREE.Group();
    if (isFemale) {
      // Tailored coat / flared trench silhouette for clear visual distinction.
      const upper = new THREE.Mesh(this.track(new THREE.BoxGeometry(0.36, 0.42, 0.21)), coatMat);
      upper.position.y = 1.19;
      const skirt = new THREE.Mesh(
        this.track(new THREE.CylinderGeometry(0.19, 0.27, 0.34, 14)),
        coatMat,
      );
      skirt.position.y = 0.85;
      const shoulderBag = new THREE.Mesh(
        this.track(new THREE.BoxGeometry(0.1, 0.24, 0.22)),
        bagMat,
      );
      shoulderBag.position.set(0.22, 0.96, 0);
      torso.add(upper, skirt, shoulderBag);
      worldMeshes.push(upper, skirt, shoulderBag);
    } else {
      // Broader shoulders & jacket + backpack.
      const jacket = new THREE.Mesh(this.track(new THREE.BoxGeometry(0.45, 0.58, 0.25)), coatMat);
      jacket.position.y = 1.12;
      const backpack = new THREE.Mesh(this.track(new THREE.BoxGeometry(0.31, 0.38, 0.14)), bagMat);
      backpack.position.set(0, 1.14, -0.18);
      torso.add(jacket, backpack);
      worldMeshes.push(jacket, backpack);
    }

    const armX = isFemale ? 0.23 : 0.285;
    const armW = isFemale ? 0.095 : 0.115;
    const leftArm = makeLimb(armW, 0.52, armW, coatMat);
    leftArm.position.set(-armX, 1.36, 0);
    const leftHand = new THREE.Mesh(this.track(new THREE.SphereGeometry(0.05, 10, 8)), skinMat);
    leftHand.position.set(0, -0.54, 0);
    leftArm.add(leftHand);
    worldMeshes.push(leftHand);

    const rightArm = makeLimb(armW, 0.52, armW, coatMat);
    rightArm.position.set(armX, 1.36, 0);
    const rightHand = new THREE.Mesh(this.track(new THREE.SphereGeometry(0.05, 10, 8)), skinMat);
    rightHand.position.set(0, -0.54, 0);
    rightArm.add(rightHand);
    worldMeshes.push(rightHand);

    const head = new THREE.Group();
    head.position.set(0, 1.54, 0);
    const skull = new THREE.Mesh(this.track(new THREE.SphereGeometry(0.12, 16, 14)), skinMat);
    const hairCap = new THREE.Mesh(
      this.track(new THREE.SphereGeometry(0.128, 16, 10, 0, Math.PI * 2, 0, Math.PI * 0.58)),
      hairMat,
    );
    hairCap.position.y = 0.015;
    head.add(skull, hairCap);
    worldMeshes.push(skull, hairCap);

    if (isFemale) {
      // Longer back hair / ponytail silhouette.
      const backHair = new THREE.Mesh(
        this.track(new THREE.CylinderGeometry(0.075, 0.095, 0.26, 12)),
        hairMat,
      );
      backHair.position.set(0, -0.06, -0.08);
      backHair.rotation.x = 0.18;
      head.add(backHair);
      worldMeshes.push(backHair);
    }

    torso.add(leftArm, rightArm, head);
    root.add(leftLeg, rightLeg, torso);

    return {
      id,
      gender: cfg.gender,
      root,
      leftLeg,
      rightLeg,
      leftArm,
      rightArm,
      torso,
      head,
      axis: cfg.axis,
      lanePos: cfg.lanePos,
      minRange: cfg.minRange,
      maxRange: cfg.maxRange,
      direction: cfg.direction,
      speed: cfg.speed,
      phase,
      heightScale: cfg.scale,
      worldMeshes,
    };
  }

  dispose() {
    this.colorTarget.dispose();
    this.maskTarget.dispose();
    this.envRenderTarget?.dispose();
    this.blackOcclusionMaterial.dispose();
    this.femaleMaskMaterial.dispose();
    this.maleMaskMaterial.dispose();
    for (const g of this.geometries) g.dispose();
    for (const m of this.materials) m.dispose();
    for (const t of this.textures) t.dispose();
  }
}
