import {FilesetResolver, ObjectDetector} from '@mediapipe/tasks-vision';
import * as THREE from 'three';
import * as xb from 'xrblocks';
import {DetectedPersonBox, PersonGender} from './types.ts';

const MEDIAPIPE_WASM_URL =
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.34/wasm';
const PERSON_DETECTOR_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite';

const MASK_WIDTH = 256;
const MASK_HEIGHT = 256;
const DETECTOR_INTERVAL_MS = 90;
const CLASSIFY_INTERVAL_MS = 1400;
const CLASSIFY_STALE_MS = 5000;
const CLASSIFY_ERROR_BACKOFF_MS = 6000;
const TRACK_TTL_MS = 650;

export type GenderClassifierStatus = 'idle' | 'classifying' | 'ready' | 'error';

interface TrackedPerson {
  trackId: number;
  box: {xmin: number; ymin: number; xmax: number; ymax: number; score: number};
  gender: PersonGender;
  genderConfidence: number;
  lastSeenMs: number;
  lastClassifiedMs: number;
}

function computeIoU(
  a: {xmin: number; ymin: number; xmax: number; ymax: number},
  b: {xmin: number; ymin: number; xmax: number; ymax: number},
): number {
  const ix0 = Math.max(a.xmin, b.xmin);
  const iy0 = Math.max(a.ymin, b.ymin);
  const ix1 = Math.min(a.xmax, b.xmax);
  const iy1 = Math.min(a.ymax, b.ymax);
  const iw = Math.max(0, ix1 - ix0);
  const ih = Math.max(0, iy1 - iy0);
  const inter = iw * ih;
  if (inter <= 0) return 0;
  const areaA = Math.max(1e-6, (a.xmax - a.xmin) * (a.ymax - a.ymin));
  const areaB = Math.max(1e-6, (b.xmax - b.xmin) * (b.ymax - b.ymin));
  return inter / (areaA + areaB - inter);
}

/**
 * Real-time person detector & gender tracker for the live camera feed:
 * - Combines XR Blocks' MediaPipe pixel Segmenter (`xb.world.segmentation`)
 *   with MediaPipe ObjectDetector (`person` class).
 * - Maintains stable per-person `trackId`s across frames using IoU + centroid
 *   matching.
 * - Periodically calls `/api/classify-gender` (server-side Gemini Vision proxy)
 *   to classify each tracked person as `'female'`, `'male'`, or `'unknown'`.
 * - Encodes each person into a channel-coded RGBA GPU mask texture:
 *     R = female, G = male, B = unknown/pending.
 */
export class PersonDetector {
  readonly maskTexture: THREE.DataTexture;

  private maskBuffer = new Uint8Array(MASK_WIDTH * MASK_HEIGHT * 4);
  // 3 channels per pixel: [female, male, unknown]
  private smoothBuffer = new Float32Array(MASK_WIDTH * MASK_HEIGHT * 3);

  private objectDetector: ObjectDetector | null = null;
  private initPromise: Promise<void> | null = null;
  private initFailed = false;
  private inFlight = false;
  private lastRunMs = 0;

  private tracks: TrackedPerson[] = [];
  private nextTrackId = 1;
  private latestBoxes: DetectedPersonBox[] = [];
  private latestInferenceMs = 0;
  private segmenterPersonPresent = false;

  private classifyInFlight = false;
  private lastClassifyMs = 0;
  private lastClassifyErrorMs = 0;
  private _aiStatus: GenderClassifierStatus = 'idle';
  private _aiError: string | null = null;

  private disposed = false;

  constructor() {
    this.maskTexture = new THREE.DataTexture(
      this.maskBuffer,
      MASK_WIDTH,
      MASK_HEIGHT,
      THREE.RGBAFormat,
      THREE.UnsignedByteType,
    );
    this.maskTexture.flipY = true;
    this.maskTexture.minFilter = THREE.LinearFilter;
    this.maskTexture.magFilter = THREE.LinearFilter;
    this.maskTexture.wrapS = THREE.ClampToEdgeWrapping;
    this.maskTexture.wrapT = THREE.ClampToEdgeWrapping;
    this.maskTexture.needsUpdate = true;
  }

  get isReady(): boolean {
    return this.objectDetector !== null || Boolean(xb.world.segmentation?.latestMask);
  }

  get hasError(): boolean {
    return this.initFailed;
  }

  get boxes(): DetectedPersonBox[] {
    return this.latestBoxes;
  }

  get personCount(): number {
    if (this.latestBoxes.length > 0) return this.latestBoxes.length;
    return this.segmenterPersonPresent ? 1 : 0;
  }

  get femaleCount(): number {
    let count = 0;
    for (const b of this.latestBoxes) {
      if (b.gender === 'female') count++;
    }
    return count;
  }

  get maleCount(): number {
    let count = 0;
    for (const b of this.latestBoxes) {
      if (b.gender === 'male') count++;
    }
    return count;
  }

  get unknownCount(): number {
    if (this.latestBoxes.length === 0 && this.segmenterPersonPresent) return 1;
    let count = 0;
    for (const b of this.latestBoxes) {
      if (b.gender === 'unknown') count++;
    }
    return count;
  }

  get aiStatus(): GenderClassifierStatus {
    return this._aiStatus;
  }

  get aiError(): string | null {
    return this._aiError;
  }

  get inferenceMs(): number {
    return this.latestInferenceMs;
  }

  ensureInitialized(): void {
    if (this.initPromise || this.initFailed || this.disposed) return;
    this.initPromise = this.initDetector();
  }

  private async initDetector(): Promise<void> {
    try {
      const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_URL);
      if (this.disposed) return;
      this.objectDetector = await ObjectDetector.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: PERSON_DETECTOR_MODEL_URL,
          delegate: 'GPU',
        },
        runningMode: 'IMAGE',
        scoreThreshold: 0.34,
        maxResults: 12,
        categoryAllowlist: ['person'],
      });
    } catch (err) {
      try {
        const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_URL);
        if (this.disposed) return;
        this.objectDetector = await ObjectDetector.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: PERSON_DETECTOR_MODEL_URL,
            delegate: 'CPU',
          },
          runningMode: 'IMAGE',
          scoreThreshold: 0.34,
          maxResults: 12,
          categoryAllowlist: ['person'],
        });
      } catch (fallbackErr) {
        console.warn(
          'MediaPipe ObjectDetector unavailable, using segmentation only:',
          err,
          fallbackErr,
        );
        this.initFailed = true;
      }
    }
  }

  /**
   * Updates detection & gender classification against the live device camera
   * and refreshes the channel-coded RGBA person mask texture.
   */
  update(deviceCamera: xb.XRDeviceCamera | undefined, nowMs: number): void {
    if (this.disposed) return;
    this.ensureInitialized();

    const streaming = Boolean(deviceCamera && deviceCamera.state === 'streaming');

    if (
      streaming &&
      deviceCamera &&
      this.objectDetector &&
      !this.inFlight &&
      nowMs - this.lastRunMs >= DETECTOR_INTERVAL_MS
    ) {
      this.lastRunMs = nowMs;
      void this.runObjectDetection(deviceCamera, nowMs);
    }

    this.rebuildFusedMask();

    if (streaming && deviceCamera && this.shouldRunGenderClassification(nowMs)) {
      this.lastClassifyMs = nowMs;
      void this.classifyTrackedPeople(deviceCamera);
    }
  }

  private shouldRunGenderClassification(nowMs: number): boolean {
    if (this.classifyInFlight || this.tracks.length === 0) return false;
    if (
      this._aiStatus === 'error' &&
      nowMs - this.lastClassifyErrorMs < CLASSIFY_ERROR_BACKOFF_MS
    ) {
      return false;
    }

    const hasUnclassified = this.tracks.some((t) => t.lastClassifiedMs === 0);
    if (hasUnclassified && nowMs - this.lastClassifyMs >= 450) {
      return true;
    }

    const hasStaleTrack = this.tracks.some(
      (t) => nowMs - t.lastClassifiedMs >= CLASSIFY_STALE_MS,
    );
    return hasStaleTrack && nowMs - this.lastClassifyMs >= CLASSIFY_INTERVAL_MS;
  }

  private async runObjectDetection(
    deviceCamera: xb.XRDeviceCamera,
    nowMs: number,
  ): Promise<void> {
    if (!this.objectDetector || this.inFlight || this.disposed) return;
    this.inFlight = true;
    const t0 = performance.now();
    try {
      const imageData = await deviceCamera.captureSnapshot({
        outputFormat: 'imageData',
        width: 512,
        height: 384,
      });
      if (!imageData || this.disposed || !this.objectDetector) return;

      const result = this.objectDetector.detect(imageData);
      const imgW = imageData.width || 512;
      const imgH = imageData.height || 384;
      const rawBoxes: Array<{
        xmin: number;
        ymin: number;
        xmax: number;
        ymax: number;
        score: number;
      }> = [];

      for (const det of result.detections) {
        const bb = det.boundingBox;
        const cat = det.categories?.[0];
        if (!bb || !cat) continue;
        const xmin = Math.max(0, Math.min(1, bb.originX / imgW));
        const ymin = Math.max(0, Math.min(1, bb.originY / imgH));
        const xmax = Math.max(0, Math.min(1, (bb.originX + bb.width) / imgW));
        const ymax = Math.max(0, Math.min(1, (bb.originY + bb.height) / imgH));
        if (xmax - xmin > 0.015 && ymax - ymin > 0.03) {
          rawBoxes.push({xmin, ymin, xmax, ymax, score: cat.score});
        }
      }

      // Close-up fallback: if ObjectDetector sees 0 boxes (e.g. tight headshot)
      // but pixel segmentation shows a person, derive a box from segmentation.
      if (rawBoxes.length === 0) {
        const segBox = this.computeSegmentationFallbackBox();
        if (segBox) rawBoxes.push(segBox);
      }

      this.updateTracks(rawBoxes, nowMs);
      this.latestInferenceMs = performance.now() - t0;
    } catch {
      // Transient frame errors are ignored while stream settles.
    } finally {
      this.inFlight = false;
    }
  }

  private computeSegmentationFallbackBox(): {
    xmin: number;
    ymin: number;
    xmax: number;
    ymax: number;
    score: number;
  } | null {
    const segMask = xb.world.segmentation?.latestMask;
    if (!segMask || segMask.width <= 0 || segMask.height <= 0) return null;

    let minX = segMask.width;
    let minY = segMask.height;
    let maxX = -1;
    let maxY = -1;
    let count = 0;

    for (let y = 0; y < segMask.height; y++) {
      for (let x = 0; x < segMask.width; x++) {
        if (segMask.data[y * segMask.width + x] >= 1) {
          count++;
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
        }
      }
    }

    const total = segMask.width * segMask.height;
    if (count < total * 0.015 || maxX <= minX || maxY <= minY) return null;

    return {
      xmin: Math.max(0, minX / segMask.width - 0.02),
      ymin: Math.max(0, minY / segMask.height - 0.02),
      xmax: Math.min(1, (maxX + 1) / segMask.width + 0.02),
      ymax: Math.min(1, (maxY + 1) / segMask.height + 0.02),
      score: 0.72,
    };
  }

  private updateTracks(
    detections: Array<{
      xmin: number;
      ymin: number;
      xmax: number;
      ymax: number;
      score: number;
    }>,
    nowMs: number,
  ): void {
    const matchedTrackIds = new Set<number>();
    const updatedTracks: TrackedPerson[] = [];

    for (const det of detections) {
      let bestTrack: TrackedPerson | null = null;
      let bestScore = 0;

      const detCx = (det.xmin + det.xmax) * 0.5;
      const detCy = (det.ymin + det.ymax) * 0.5;

      for (const track of this.tracks) {
        if (matchedTrackIds.has(track.trackId)) continue;
        const iou = computeIoU(det, track.box);
        const tCx = (track.box.xmin + track.box.xmax) * 0.5;
        const tCy = (track.box.ymin + track.box.ymax) * 0.5;
        const dist = Math.hypot(detCx - tCx, detCy - tCy);
        const score = iou > 0.15 ? iou + 0.25 : dist < 0.16 ? 0.2 - dist : 0;

        if (score > bestScore) {
          bestScore = score;
          bestTrack = track;
        }
      }

      if (bestTrack) {
        matchedTrackIds.add(bestTrack.trackId);
        // Smooth box coordinates slightly to reduce jitter.
        bestTrack.box = {
          xmin: bestTrack.box.xmin * 0.25 + det.xmin * 0.75,
          ymin: bestTrack.box.ymin * 0.25 + det.ymin * 0.75,
          xmax: bestTrack.box.xmax * 0.25 + det.xmax * 0.75,
          ymax: bestTrack.box.ymax * 0.25 + det.ymax * 0.75,
          score: det.score,
        };
        bestTrack.lastSeenMs = nowMs;
        updatedTracks.push(bestTrack);
      } else {
        updatedTracks.push({
          trackId: this.nextTrackId++,
          box: det,
          gender: 'unknown',
          genderConfidence: 0,
          lastSeenMs: nowMs,
          lastClassifiedMs: 0,
        });
      }
    }

    // Keep recently lost tracks briefly to survive single-frame detector drops.
    for (const track of this.tracks) {
      if (!matchedTrackIds.has(track.trackId) && nowMs - track.lastSeenMs <= TRACK_TTL_MS) {
        updatedTracks.push(track);
      }
    }

    this.tracks = updatedTracks.slice(0, 12);
    this.syncLatestBoxes();
  }

  private syncLatestBoxes(): void {
    this.latestBoxes = this.tracks.map((t) => ({
      trackId: t.trackId,
      xmin: t.box.xmin,
      ymin: t.box.ymin,
      xmax: t.box.xmax,
      ymax: t.box.ymax,
      score: t.box.score,
      gender: t.gender,
      genderConfidence: t.genderConfidence,
    }));
  }

  private async classifyTrackedPeople(deviceCamera: xb.XRDeviceCamera): Promise<void> {
    if (this.classifyInFlight || this.disposed) return;
    const tracksSnapshot = this.tracks.slice(0, 12);
    if (tracksSnapshot.length === 0) return;

    this.classifyInFlight = true;
    this._aiStatus = 'classifying';

    try {
      const snapshot = await deviceCamera.captureSnapshot({
        outputFormat: 'base64',
        mimeType: 'image/jpeg',
        quality: 0.78,
        width: 640,
        height: 360,
      });
      if (!snapshot || this.disposed) return;

      let mimeType = 'image/jpeg';
      let imageBase64 = snapshot;
      const dataUrlMatch = snapshot.match(/^data:([^;,]+)[^,]*,(.*)$/s);
      if (dataUrlMatch) {
        mimeType = dataUrlMatch[1];
        imageBase64 = dataUrlMatch[2];
      }

      const payload = {
        imageBase64,
        mimeType,
        boxes: tracksSnapshot.map((t) => ({
          id: t.trackId,
          xmin: t.box.xmin,
          ymin: t.box.ymin,
          xmax: t.box.xmax,
          ymax: t.box.ymax,
        })),
      };

      const res = await fetch('/api/classify-gender', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload),
      });

      const body = (await res.json().catch(() => ({}))) as {
        classifications?: Array<{id: number; gender: PersonGender; confidence?: number}>;
        error?: string;
      };

      if (!res.ok) {
        throw new Error(body.error || `Classifier failed (${res.status})`);
      }

      const finishedAt = performance.now();
      const map = new Map<number, {gender: PersonGender; confidence: number}>();
      for (const item of body.classifications ?? []) {
        if (item && typeof item.id === 'number') {
          const g: PersonGender =
            item.gender === 'female' || item.gender === 'male' ? item.gender : 'unknown';
          map.set(item.id, {
            gender: g,
            confidence: typeof item.confidence === 'number' ? item.confidence : 0.75,
          });
        }
      }

      for (const track of this.tracks) {
        const resItem = map.get(track.trackId);
        if (resItem) {
          track.gender = resItem.gender;
          track.genderConfidence = resItem.confidence;
          track.lastClassifiedMs = finishedAt;
        }
      }

      this.syncLatestBoxes();
      this._aiStatus = 'ready';
      this._aiError = null;
    } catch (err) {
      this._aiStatus = 'error';
      this._aiError = err instanceof Error ? err.message : String(err);
      this.lastClassifyErrorMs = performance.now();
    } finally {
      this.classifyInFlight = false;
    }
  }

  private findChannelForPixel(nx: number, ny: number): 0 | 1 | 2 {
    let bestChannel: 0 | 1 | 2 = 2;
    let bestDistSq = Infinity;

    for (const box of this.latestBoxes) {
      const inside =
        nx >= box.xmin - 0.03 &&
        nx <= box.xmax + 0.03 &&
        ny >= box.ymin - 0.03 &&
        ny <= box.ymax + 0.03;
      if (!inside) continue;

      const cx = (box.xmin + box.xmax) * 0.5;
      const cy = (box.ymin + box.ymax) * 0.5;
      const dSq = (nx - cx) * (nx - cx) + (ny - cy) * (ny - cy);
      if (dSq < bestDistSq) {
        bestDistSq = dSq;
        bestChannel = box.gender === 'female' ? 0 : box.gender === 'male' ? 1 : 2;
      }
    }

    return bestChannel;
  }

  private rebuildFusedMask(): void {
    const segMask = xb.world.segmentation?.latestMask ?? null;
    const totalPixels = MASK_WIDTH * MASK_HEIGHT;

    // 1. Sample pixel segmentation mask & assign each foreground pixel to its
    //    matched tracked person's gender channel (R=female, G=male, B=unknown).
    let segForegroundCount = 0;
    for (let y = 0; y < MASK_HEIGHT; y++) {
      const srcY = segMask
        ? Math.min(segMask.height - 1, Math.floor((y / MASK_HEIGHT) * segMask.height))
        : 0;
      const ny = (y + 0.5) / MASK_HEIGHT;
      for (let x = 0; x < MASK_WIDTH; x++) {
        const dstBase = (y * MASK_WIDTH + x) * 3;
        let t0 = 0;
        let t1 = 0;
        let t2 = 0;

        if (segMask) {
          const srcX = Math.min(
            segMask.width - 1,
            Math.floor((x / MASK_WIDTH) * segMask.width),
          );
          const category = segMask.data[srcY * segMask.width + srcX];
          if (category >= 1) {
            segForegroundCount++;
            const nx = (x + 0.5) / MASK_WIDTH;
            const ch = this.findChannelForPixel(nx, ny);
            if (ch === 0) t0 = 1;
            else if (ch === 1) t1 = 1;
            else t2 = 1;
          }
        }

        this.smoothBuffer[dstBase] = this.smoothBuffer[dstBase] * 0.42 + t0 * 0.58;
        this.smoothBuffer[dstBase + 1] = this.smoothBuffer[dstBase + 1] * 0.42 + t1 * 0.58;
        this.smoothBuffer[dstBase + 2] = this.smoothBuffer[dstBase + 2] * 0.42 + t2 * 0.58;
      }
    }

    this.segmenterPersonPresent = segForegroundCount > totalPixels * 0.008;

    // 2. Reinforce each detected person bounding box with a soft capsule in its
    //    gender channel while damping opposing channels for clean switching.
    for (const box of this.latestBoxes) {
      const ch: 0 | 1 | 2 = box.gender === 'female' ? 0 : box.gender === 'male' ? 1 : 2;
      const bx0 = Math.max(0, Math.floor(box.xmin * MASK_WIDTH));
      const bx1 = Math.min(MASK_WIDTH - 1, Math.ceil(box.xmax * MASK_WIDTH));
      const by0 = Math.max(0, Math.floor(box.ymin * MASK_HEIGHT));
      const by1 = Math.min(MASK_HEIGHT - 1, Math.ceil(box.ymax * MASK_HEIGHT));

      const cx = (box.xmin + box.xmax) * 0.5;
      const cy = (box.ymin + box.ymax) * 0.5;
      const rx = Math.max(0.01, (box.xmax - box.xmin) * 0.48);
      const ry = Math.max(0.01, (box.ymax - box.ymin) * 0.5);

      for (let y = by0; y <= by1; y++) {
        const ny = (y + 0.5) / MASK_HEIGHT;
        const dy = (ny - cy) / ry;
        for (let x = bx0; x <= bx1; x++) {
          const nx = (x + 0.5) / MASK_WIDTH;
          const dx = (nx - cx) / rx;
          const distSq = dx * dx + dy * dy * 0.82;
          if (distSq < 1.15) {
            const capsuleWeight = Math.max(0, Math.min(1, (1.15 - distSq) * 2.2));
            const base = (y * MASK_WIDTH + x) * 3;
            for (let c = 0; c < 3; c++) {
              if (c === ch) {
                if (capsuleWeight > this.smoothBuffer[base + c]) {
                  this.smoothBuffer[base + c] =
                    this.smoothBuffer[base + c] * 0.32 + capsuleWeight * 0.68;
                }
              } else {
                this.smoothBuffer[base + c] *= 0.45;
              }
            }
          }
        }
      }
    }

    // 3. Write quantized RGBA channels to DataTexture.
    for (let i = 0; i < totalPixels; i++) {
      const s = i * 3;
      const d = i * 4;
      this.maskBuffer[d] = Math.round(Math.max(0, Math.min(1, this.smoothBuffer[s])) * 255);
      this.maskBuffer[d + 1] = Math.round(
        Math.max(0, Math.min(1, this.smoothBuffer[s + 1])) * 255,
      );
      this.maskBuffer[d + 2] = Math.round(
        Math.max(0, Math.min(1, this.smoothBuffer[s + 2])) * 255,
      );
      this.maskBuffer[d + 3] = 255;
    }
    this.maskTexture.needsUpdate = true;
  }

  clear(): void {
    this.tracks = [];
    this.latestBoxes = [];
    this.segmenterPersonPresent = false;
    this._aiStatus = 'idle';
    this._aiError = null;
    this.smoothBuffer.fill(0);
    for (let i = 0; i < MASK_WIDTH * MASK_HEIGHT; i++) {
      const d = i * 4;
      this.maskBuffer[d] = 0;
      this.maskBuffer[d + 1] = 0;
      this.maskBuffer[d + 2] = 0;
      this.maskBuffer[d + 3] = 255;
    }
    this.maskTexture.needsUpdate = true;
  }

  dispose(): void {
    this.disposed = true;
    this.objectDetector?.close();
    this.objectDetector = null;
    this.maskTexture.dispose();
  }
}
