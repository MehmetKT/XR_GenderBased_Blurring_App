export enum BlurStyle {
  Gaussian = 'gaussian',
  Frosted = 'frosted',
  Mosaic = 'mosaic',
}

export enum FeedSourceMode {
  StreetSim = 'streetSim',
  LiveCamera = 'liveCamera',
}

export enum GenderFilter {
  All = 'all',
  Female = 'female',
  Male = 'male',
}

export type PersonGender = 'female' | 'male' | 'unknown';

export interface DetectedPersonBox {
  trackId?: number;
  /** Normalized [0..1] coordinates in image space (origin top-left). */
  xmin: number;
  ymin: number;
  xmax: number;
  ymax: number;
  score: number;
  gender: PersonGender;
  genderConfidence?: number;
}
