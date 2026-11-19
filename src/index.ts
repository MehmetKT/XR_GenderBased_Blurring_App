import * as xb from 'xrblocks';
import {StreetPrivacyApp} from './StreetPrivacyApp.ts';

document.addEventListener('DOMContentLoaded', () => {
  const options = new xb.Options();
  options.enableCamera('environment');
  options.enableSegmentation();
  options.world.segmentation.backendConfig.mediapipe.wasmFilesUrl =
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.34/wasm';
  options.world.segmentation.pollingIntervalMs = 50;
  options.enableReticles();

  // Start the Desktop Simulator in a clean empty backdrop so our 3D City Street
  // renders without an overlapping indoor loft room.
  const streetManifestUrl = `data:application/json,${encodeURIComponent(
    JSON.stringify({name: 'City Street', objects: []}),
  )}`;
  options.simulator.environments = [
    {name: 'City Street', manifestPath: streetManifestUrl},
    ...options.simulator.environments,
  ];
  options.simulator.activeEnvironmentIndex = 0;

  options.setAppTitle('Street Privacy Lens');
  options.setAppDescription(
    'Detects and blurs people in real time while preserving crisp street and background clarity.',
  );

  xb.add(new StreetPrivacyApp());
  void xb.init(options);
});
