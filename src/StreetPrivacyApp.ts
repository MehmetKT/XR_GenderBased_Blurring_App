import * as THREE from 'three';
import * as xb from 'xrblocks';
import {PersonDetector} from './PersonDetector.ts';
import {SelectiveBlurPipeline} from './SelectiveBlurPipeline.ts';
import {StreetEnvironment} from './StreetEnvironment.ts';
import {
  BlurStyle,
  DetectedPersonBox,
  FeedSourceMode,
  GenderFilter,
} from './types.ts';

const BLUR_STYLE_ORDER: BlurStyle[] = [
  BlurStyle.Gaussian,
  BlurStyle.Frosted,
  BlurStyle.Mosaic,
];

const BLUR_STYLE_LABELS: Record<BlurStyle, string> = {
  [BlurStyle.Gaussian]: 'Gaussian',
  [BlurStyle.Frosted]: 'Frosted',
  [BlurStyle.Mosaic]: 'Mosaic',
};

/**
 * Main XR Blocks application script orchestrating:
 * - Real-time person detection & gender-selective blurring on the live camera feed
 *   (or the simulated 3D city street with walking women and men)
 * - GPU RGBA mask filtering preserving background & unselected-gender clarity
 * - Interactive spatial UI viewfinder & ergonomic gender-filter control deck
 * - Optional transparent AR Passthrough Veil for head-worn XR sessions
 */
export class StreetPrivacyApp extends xb.Script {
  private streetEnv = new StreetEnvironment(1024, 576);
  private personDetector = new PersonDetector();
  private pipeline = new SelectiveBlurPipeline(1024, 576);

  private blurEnabled = true;
  private genderFilter: GenderFilter = GenderFilter.All;
  private blurUnknown = true;
  private blurStyle = BlurStyle.Gaussian;
  private blurStrength = 0.8;
  private maskPadding = 0.35;
  private showBoxes = false;
  private followUser = false;
  private arVeilEnabled = false;
  private feedSource: FeedSourceMode = FeedSourceMode.StreetSim;

  private fallbackTexture: THREE.DataTexture;
  private placeholderMask: THREE.DataTexture;

  private viewfinderCard!: xb.UICard;
  private controlsCard!: xb.UICard;
  private passthroughShieldMesh!: THREE.Mesh;

  private viewfinderFollow!: xb.FollowHead;
  private viewfinderFace!: xb.FaceCamera;
  private controlsFollow!: xb.FollowHead;
  private controlsFace!: xb.FaceCamera;

  private statusBadgeText!: xb.UIText;
  private sourceBadgeText!: xb.UIText;
  private feedImage!: xb.UIImage;

  private filterAllBtn!: xb.UIButton;
  private filterWomenBtn!: xb.UIButton;
  private filterMenBtn!: xb.UIButton;

  private privacyToggleBtn!: xb.UIButton;
  private styleBtn!: xb.UIButton;
  private boxesBtn!: xb.UIButton;
  private sourceBtn!: xb.UIButton;
  private pendingBtn!: xb.UIButton;
  private followBtn!: xb.UIButton;
  private veilBtn!: xb.UIButton;
  private blurValueText!: xb.UIText;
  private paddingValueText!: xb.UIText;

  private uiValidated = false;
  private frameCounter = 0;
  private elapsedSeconds = 0;
  private realCameraAttempted = false;

  constructor() {
    super();

    const emptyPixel = new Uint8Array([18, 22, 30, 255]);
    this.fallbackTexture = new THREE.DataTexture(emptyPixel, 1, 1, THREE.RGBAFormat);
    this.fallbackTexture.colorSpace = THREE.SRGBColorSpace;
    this.fallbackTexture.needsUpdate = true;

    const zeroMask = new Uint8Array([0, 0, 0, 255]);
    this.placeholderMask = new THREE.DataTexture(zeroMask, 1, 1, THREE.RGBAFormat);
    this.placeholderMask.needsUpdate = true;
  }

  init() {
    xb.ui.theme = 'grayGlass';

    if (!xb.core.simulatorRunning && xb.core.deviceCamera) {
      this.feedSource = FeedSourceMode.LiveCamera;
    }

    this.add(this.streetEnv);
    this.streetEnv.setVisible(this.feedSource === FeedSourceMode.StreetSim);

    this.buildSpatialUI();
    this.buildPassthroughShield();
    this.syncUIState();
  }

  private buildSpatialUI() {
    // 1. Primary Privacy Viewfinder Card (0.94m wide, 16:9 live processed lens)
    this.statusBadgeText = new xb.UIText({
      text: 'Scanning Street...',
      style: {
        fontSize: 19,
        fontWeight: 600,
        color: '#5eead4',
      },
    });

    this.sourceBadgeText = new xb.UIText({
      text: 'Street Sim',
      style: {
        fontSize: 17,
        color: '#cbd5e1',
      },
    });

    const headerRow = new xb.UIPanel({
      style: {
        width: '100%',
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        paddingLeft: 6,
        paddingRight: 6,
      },
      children: [
        new xb.UIPanel({
          style: {
            flexDirection: 'row',
            alignItems: 'center',
            gap: 10,
          },
          children: [
            new xb.UIIcon({
              icon: 'shield_person',
              style: {fontSize: 24, color: '#5eead4'},
            }),
            new xb.UIText({
              text: 'STREET PRIVACY LENS',
              style: {fontSize: 20, fontWeight: 700, color: '#f8fafc'},
            }),
          ],
        }),
        new xb.UIPanel({
          style: {
            flexDirection: 'row',
            alignItems: 'center',
            gap: 14,
          },
          children: [this.statusBadgeText, this.sourceBadgeText],
        }),
      ],
    });

    this.feedImage = new xb.UIImage({
      src: this.pipeline.compositeTexture,
      ariaLabel: 'Processed gender-selective street privacy camera feed',
      style: {
        width: '100%',
        height: 504,
        borderRadius: 14,
        objectFit: 'cover',
      },
    });
    this.feedImage.xb = {pointerEvents: 'none'};

    this.viewfinderCard = new xb.UICard({
      size: {width: 0.94, height: 'auto'},
      manipulation: true,
      edge: true,
      style: {
        flexDirection: 'column',
        gap: 12,
        padding: 16,
        borderRadius: 22,
        backgroundColor: 'rgba(15, 23, 42, 0.86)',
      },
      children: [headerRow, this.feedImage],
    });
    this.viewfinderCard.position.set(0, 1.6, -1.28);

    this.viewfinderFollow = new xb.FollowHead({
      offset: new THREE.Vector3(0, 0.1, -1.28),
      smoothing: 0.09,
    });
    this.viewfinderFace = new xb.FaceCamera({
      mode: 'capsule',
      capsuleHalfHeight: 0.3,
      smoothing: 0.12,
    });
    this.viewfinderCard.add(this.viewfinderFollow, this.viewfinderFace);
    this.viewfinderFollow.suspend();
    this.viewfinderFace.suspend();

    // 2. Gender Filter Row (top of ergonomic control deck)
    this.filterAllBtn = new xb.UIButton({
      label: 'Blur: All',
      icon: 'groups',
      style: {flexGrow: 1, height: 46, borderRadius: 12},
      onClick: () => {
        this.genderFilter = GenderFilter.All;
        this.syncUIState();
      },
    });

    this.filterWomenBtn = new xb.UIButton({
      label: 'Blur: Women',
      icon: 'woman',
      style: {flexGrow: 1, height: 46, borderRadius: 12},
      onClick: () => {
        this.genderFilter = GenderFilter.Female;
        this.syncUIState();
      },
    });

    this.filterMenBtn = new xb.UIButton({
      label: 'Blur: Men',
      icon: 'man',
      style: {flexGrow: 1, height: 46, borderRadius: 12},
      onClick: () => {
        this.genderFilter = GenderFilter.Male;
        this.syncUIState();
      },
    });

    const genderRow = new xb.UIPanel({
      style: {
        width: '100%',
        flexDirection: 'row',
        gap: 10,
      },
      children: [this.filterAllBtn, this.filterWomenBtn, this.filterMenBtn],
    });

    // 3. Primary Processing Row
    this.privacyToggleBtn = new xb.UIButton({
      label: 'Privacy: ON',
      icon: 'blur_on',
      style: {flexGrow: 1, height: 46, borderRadius: 12},
      onClick: () => {
        this.blurEnabled = !this.blurEnabled;
        this.syncUIState();
      },
    });

    this.styleBtn = new xb.UIButton({
      label: 'Style: Gaussian',
      icon: 'auto_fix_high',
      style: {flexGrow: 1, height: 46, borderRadius: 12},
      onClick: () => {
        const idx = BLUR_STYLE_ORDER.indexOf(this.blurStyle);
        this.blurStyle = BLUR_STYLE_ORDER[(idx + 1) % BLUR_STYLE_ORDER.length];
        this.syncUIState();
      },
    });

    this.boxesBtn = new xb.UIButton({
      label: 'Boxes: Off',
      icon: 'crop_free',
      style: {flexGrow: 1, height: 46, borderRadius: 12},
      onClick: () => {
        this.showBoxes = !this.showBoxes;
        this.syncUIState();
      },
    });

    const primaryRow = new xb.UIPanel({
      style: {
        width: '100%',
        flexDirection: 'row',
        gap: 10,
      },
      children: [this.privacyToggleBtn, this.styleBtn, this.boxesBtn],
    });

    this.blurValueText = new xb.UIText({
      text: '80%',
      style: {fontSize: 18, fontWeight: 600, color: '#93c5fd'},
    });

    const blurSliderPanel = new xb.UIPanel({
      style: {
        width: '100%',
        flexDirection: 'column',
        gap: 6,
      },
      children: [
        new xb.UIPanel({
          style: {
            width: '100%',
            flexDirection: 'row',
            justifyContent: 'space-between',
          },
          children: [
            new xb.UIText({
              text: 'Blur Intensity (Target Gender)',
              style: {fontSize: 18, color: '#e2e8f0'},
            }),
            this.blurValueText,
          ],
        }),
        new xb.UISlider({
          ariaLabel: 'Blur Intensity',
          min: 15,
          max: 100,
          step: 5,
          value: 80,
          style: {width: '100%', height: 34},
          onInput: (val) => {
            this.blurStrength = val / 100;
            this.blurValueText.text = `${Math.round(val)}%`;
          },
          onChange: (val) => {
            this.blurStrength = val / 100;
            this.blurValueText.text = `${Math.round(val)}%`;
          },
        }),
      ],
    });

    this.paddingValueText = new xb.UIText({
      text: '35%',
      style: {fontSize: 18, fontWeight: 600, color: '#93c5fd'},
    });

    const paddingSliderPanel = new xb.UIPanel({
      style: {
        width: '100%',
        flexDirection: 'column',
        gap: 6,
      },
      children: [
        new xb.UIPanel({
          style: {
            width: '100%',
            flexDirection: 'row',
            justifyContent: 'space-between',
          },
          children: [
            new xb.UIText({
              text: 'Silhouette Edge Padding',
              style: {fontSize: 18, color: '#e2e8f0'},
            }),
            this.paddingValueText,
          ],
        }),
        new xb.UISlider({
          ariaLabel: 'Silhouette Edge Padding',
          min: 0,
          max: 100,
          step: 5,
          value: 35,
          style: {width: '100%', height: 34},
          onInput: (val) => {
            this.maskPadding = val / 100;
            this.paddingValueText.text = `${Math.round(val)}%`;
          },
          onChange: (val) => {
            this.maskPadding = val / 100;
            this.paddingValueText.text = `${Math.round(val)}%`;
          },
        }),
      ],
    });

    this.sourceBtn = new xb.UIButton({
      label: 'Feed: Sim',
      icon: 'videocam',
      style: {flexGrow: 1, height: 44, borderRadius: 12},
      onClick: () => {
        void this.toggleFeedSource();
      },
    });

    this.pendingBtn = new xb.UIButton({
      label: 'Pending: Blur',
      icon: 'help_outline',
      style: {flexGrow: 1, height: 44, borderRadius: 12},
      onClick: () => {
        this.blurUnknown = !this.blurUnknown;
        this.syncUIState();
      },
    });

    this.followBtn = new xb.UIButton({
      label: 'Follow: Off',
      icon: 'center_focus_strong',
      style: {flexGrow: 1, height: 44, borderRadius: 12},
      onClick: () => {
        this.toggleFollowMode();
      },
    });

    this.veilBtn = new xb.UIButton({
      label: 'AR Veil: Off',
      icon: 'visibility_off',
      style: {flexGrow: 1, height: 44, borderRadius: 12},
      onClick: () => {
        this.arVeilEnabled = !this.arVeilEnabled;
        this.syncUIState();
      },
    });

    const secondaryRow = new xb.UIPanel({
      style: {
        width: '100%',
        flexDirection: 'row',
        gap: 8,
      },
      children: [this.sourceBtn, this.pendingBtn, this.followBtn, this.veilBtn],
    });

    this.controlsCard = new xb.UICard({
      size: {width: 0.94, height: 'auto'},
      manipulation: true,
      edge: true,
      style: {
        flexDirection: 'column',
        gap: 11,
        padding: 16,
        borderRadius: 22,
        backgroundColor: 'rgba(15, 23, 42, 0.88)',
      },
      children: [
        genderRow,
        primaryRow,
        blurSliderPanel,
        paddingSliderPanel,
        secondaryRow,
      ],
    });
    this.controlsCard.position.set(0, 0.96, -1.18);
    this.controlsCard.rotation.x = -0.22;

    this.controlsFollow = new xb.FollowHead({
      offset: new THREE.Vector3(0, -0.45, -1.18),
      smoothing: 0.09,
    });
    this.controlsFace = new xb.FaceCamera({
      mode: 'capsule',
      capsuleHalfHeight: 0.3,
      smoothing: 0.12,
    });
    this.controlsCard.add(this.controlsFollow, this.controlsFace);
    this.controlsFollow.suspend();
    this.controlsFace.suspend();

    this.add(this.viewfinderCard, this.controlsCard);
  }

  private buildPassthroughShield() {
    const shieldGeo = new THREE.PlaneGeometry(1.6, 0.9);
    this.passthroughShieldMesh = new THREE.Mesh(
      shieldGeo,
      this.pipeline.passthroughMaterial,
    );
    this.passthroughShieldMesh.name = 'AR Passthrough Privacy Veil';
    this.passthroughShieldMesh.visible = false;
    this.passthroughShieldMesh.renderOrder = 2;
    this.passthroughShieldMesh.xb = {pointerEvents: 'none'};

    this.passthroughShieldMesh.add(
      new xb.FollowHead({
        offset: new THREE.Vector3(0, 0, -1.1),
        smoothing: 0.35,
      }),
      new xb.FaceCamera({
        mode: 'spherical',
        smoothing: 0.35,
      }),
    );
    this.add(this.passthroughShieldMesh);
  }

  private async toggleFeedSource() {
    if (this.feedSource === FeedSourceMode.StreetSim) {
      this.feedSource = FeedSourceMode.LiveCamera;
      this.streetEnv.setVisible(false);
      await this.switchToRealWebcamIfNeeded();
      this.personDetector.ensureInitialized();
    } else {
      this.feedSource = FeedSourceMode.StreetSim;
      this.streetEnv.setVisible(true);
      this.personDetector.clear();
    }
    this.syncUIState();
  }

  private async switchToRealWebcamIfNeeded() {
    if (this.realCameraAttempted) return;
    this.realCameraAttempted = true;
    const cam = xb.core.deviceCamera;
    if (!cam) return;

    try {
      const devices = await cam.getAvailableVideoDevices();
      const physicalWebcam = devices.find(
        (d) => d.groupId !== 'simulator' && d.deviceId !== '',
      );
      if (physicalWebcam) {
        await cam.setDeviceId(physicalWebcam.deviceId);
      } else if (xb.core.simulatorRunning) {
        await cam.setFacingMode('user');
      }
    } catch (err) {
      console.warn('Could not switch to physical camera, continuing with active stream:', err);
    }
  }

  private toggleFollowMode() {
    this.followUser = !this.followUser;
    if (this.followUser) {
      this.viewfinderFollow.resume();
      this.viewfinderFace.resume();
      this.controlsFollow.resume();
      this.controlsFace.resume();
    } else {
      this.viewfinderFollow.suspend();
      this.viewfinderFace.suspend();
      this.controlsFollow.suspend();
      this.controlsFace.suspend();
    }
    this.syncUIState();
  }

  private syncUIState() {
    const inactiveBg = 'rgba(30, 41, 59, 0.72)';

    this.filterAllBtn.style.backgroundColor =
      this.genderFilter === GenderFilter.All ? 'rgba(13, 148, 136, 0.88)' : inactiveBg;
    this.filterWomenBtn.style.backgroundColor =
      this.genderFilter === GenderFilter.Female ? 'rgba(219, 39, 119, 0.88)' : inactiveBg;
    this.filterMenBtn.style.backgroundColor =
      this.genderFilter === GenderFilter.Male ? 'rgba(37, 99, 235, 0.88)' : inactiveBg;

    this.privacyToggleBtn.label = this.blurEnabled ? 'Privacy: ON' : 'Privacy: OFF';
    this.privacyToggleBtn.icon = this.blurEnabled ? 'blur_on' : 'blur_off';
    this.privacyToggleBtn.style.backgroundColor = this.blurEnabled
      ? 'rgba(13, 148, 136, 0.78)'
      : 'rgba(180, 83, 9, 0.78)';

    this.styleBtn.label = `Style: ${BLUR_STYLE_LABELS[this.blurStyle]}`;
    this.boxesBtn.label = this.showBoxes ? 'Boxes: On' : 'Boxes: Off';
    this.sourceBtn.label =
      this.feedSource === FeedSourceMode.StreetSim ? 'Feed: Sim' : 'Feed: Camera';
    this.pendingBtn.label = this.blurUnknown ? 'Pending: Blur' : 'Pending: Clear';
    this.pendingBtn.style.backgroundColor = this.blurUnknown
      ? 'rgba(217, 119, 6, 0.78)'
      : inactiveBg;
    this.followBtn.label = this.followUser ? 'Follow: On' : 'Follow: Off';
    this.veilBtn.label = this.arVeilEnabled ? 'AR Veil: On' : 'AR Veil: Off';
    this.passthroughShieldMesh.visible = this.arVeilEnabled;

    this.sourceBadgeText.text =
      this.feedSource === FeedSourceMode.StreetSim ? 'Street Sim' : 'Live Camera';
  }

  update() {
    const renderer = xb.core.renderer;
    const camera = xb.core.camera;
    if (!(renderer instanceof THREE.WebGLRenderer) || !camera) return;

    const dt = xb.getDeltaTime();
    this.elapsedSeconds += dt;
    const nowMs = performance.now();

    const inRealXRSession =
      renderer.xr.isPresenting && !xb.core.simulatorRunning;
    if (inRealXRSession && this.feedSource === FeedSourceMode.StreetSim) {
      this.feedSource = FeedSourceMode.LiveCamera;
      this.streetEnv.setVisible(false);
      this.syncUIState();
    }

    let sourceTexture: THREE.Texture = this.fallbackTexture;
    let maskTexture: THREE.Texture = this.placeholderMask;
    let boxes: DetectedPersonBox[] = [];
    let femaleCount = 0;
    let maleCount = 0;
    let unknownCount = 0;

    if (this.feedSource === FeedSourceMode.StreetSim) {
      boxes = this.streetEnv.renderCameraFeed(renderer, camera, [
        this.viewfinderCard,
        this.controlsCard,
        this.passthroughShieldMesh,
        xb.core.input.reticles,
      ]);
      sourceTexture = this.streetEnv.colorTexture;
      maskTexture = this.streetEnv.maskTexture;
      for (const b of boxes) {
        if (b.gender === 'female') femaleCount++;
        else if (b.gender === 'male') maleCount++;
        else unknownCount++;
      }
    } else {
      const deviceCam = xb.core.deviceCamera;
      this.personDetector.update(deviceCam, nowMs);

      if (deviceCam && deviceCam.state === 'streaming') {
        sourceTexture = deviceCam.texture;
      }
      maskTexture = this.personDetector.maskTexture;
      boxes = this.personDetector.boxes;
      femaleCount = this.personDetector.femaleCount;
      maleCount = this.personDetector.maleCount;
      unknownCount = this.personDetector.unknownCount;
    }

    this.pipeline.render(renderer, {
      sourceTexture,
      maskTexture,
      enabled: this.blurEnabled,
      blurStrength: this.blurStrength,
      maskPadding: this.maskPadding,
      blurStyle: this.blurStyle,
      genderFilter: this.genderFilter,
      blurUnknown: this.blurUnknown,
      showBoxes: this.showBoxes,
      boxes,
      elapsedTime: this.elapsedSeconds,
    });

    this.updateStatusBadge(femaleCount, maleCount, unknownCount);

    this.frameCounter++;
    if (!this.uiValidated && this.frameCounter === 10) {
      this.uiValidated = true;
      const report = xb.ui.validate();
      if (!report.ok) {
        console.warn('Spatial UI validation issues:', report.issues);
      }
    }
  }

  private updateStatusBadge(
    femaleCount: number,
    maleCount: number,
    unknownCount: number,
  ) {
    if (!this.blurEnabled) {
      this.statusBadgeText.text = 'Blur Paused • Raw Feed';
      this.statusBadgeText.style.color = '#fbbf24';
      return;
    }

    if (this.feedSource === FeedSourceMode.LiveCamera) {
      const camState = xb.core.deviceCamera?.state;
      if (camState && camState !== 'streaming') {
        this.statusBadgeText.text = 'Camera Initializing...';
        this.statusBadgeText.style.color = '#93c5fd';
        return;
      }
      if (!this.personDetector.isReady && !this.personDetector.hasError) {
        this.statusBadgeText.text = 'Loading AI Detector...';
        this.statusBadgeText.style.color = '#93c5fd';
        return;
      }
      if (this.personDetector.aiError && this.genderFilter !== GenderFilter.All) {
        const shortErr = this.personDetector.aiError.includes('GEMINI_API_KEY')
          ? 'Set GEMINI_API_KEY for Gender AI'
          : 'Gender AI Unavailable';
        this.statusBadgeText.text = shortErr;
        this.statusBadgeText.style.color = '#f87171';
        return;
      }
    }

    const total = femaleCount + maleCount + unknownCount;
    if (total === 0) {
      this.statusBadgeText.text = 'Clear • 0 People';
      this.statusBadgeText.style.color = '#94a3b8';
      return;
    }

    if (this.genderFilter === GenderFilter.Female) {
      const suffix = unknownCount > 0 ? ` • ${unknownCount}?` : '';
      this.statusBadgeText.text = `Blurring ${femaleCount} ${femaleCount === 1 ? 'Woman' : 'Women'} • ${maleCount} Men Clear${suffix}`;
      this.statusBadgeText.style.color = '#f472b6';
      return;
    }

    if (this.genderFilter === GenderFilter.Male) {
      const suffix = unknownCount > 0 ? ` • ${unknownCount}?` : '';
      this.statusBadgeText.text = `Blurring ${maleCount} ${maleCount === 1 ? 'Man' : 'Men'} • ${femaleCount} Women Clear${suffix}`;
      this.statusBadgeText.style.color = '#38bdf8';
      return;
    }

    const breakdown = unknownCount > 0 ? `${femaleCount}W / ${maleCount}M / ${unknownCount}?` : `${femaleCount}W / ${maleCount}M`;
    this.statusBadgeText.text = `Blurring ${total} ${total === 1 ? 'Person' : 'People'} (${breakdown})`;
    this.statusBadgeText.style.color = '#5eead4';
  }

  dispose() {
    this.personDetector.dispose();
    this.pipeline.dispose();
    this.fallbackTexture.dispose();
    this.placeholderMask.dispose();
    this.passthroughShieldMesh?.geometry.dispose();
  }
}
