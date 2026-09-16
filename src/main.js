const MAX_LIGHTS = 8;
const IMAGE_SIZE = { width: 1448, height: 1086 };
const CAMERA = {
  focalLength: { x: 1135, y: 1135 },
  principalPoint: { x: 724, y: 543 },
  nearDepth: 0.7,
  farDepth: 6.0,
};

const shader = /* wgsl */ `
struct SceneUniforms {
  ambientIntensity: f32,
  roughness: f32,
  useDepthNormals: f32,
  shadowSampleCount: f32,
  focalLength: vec2f,
  principalPoint: vec2f,
  nearDepth: f32,
  farDepth: f32,
  lightCount: f32,
  outputMode: f32,
};

struct Light {
  position: vec3f,
  enabled: f32,
  color: vec3f,
  intensity: f32,
  radius: f32,
  sourceRadius: f32,
};

struct LightBlock {
  values: array<Light, ${MAX_LIGHTS}>,
}

struct LightingContribution {
  diffuse: vec3f,
  specular: vec3f,
}

@group(0) @binding(0) var<uniform> scene: SceneUniforms;
@group(0) @binding(1) var<uniform> lights: LightBlock;
@group(0) @binding(2) var radianceTexture: texture_2d<f32>;
@group(0) @binding(3) var depthTexture: texture_2d<f32>;
@group(0) @binding(4) var normalTexture: texture_2d<f32>;
@group(0) @binding(5) var linearSampler: sampler;

const MINIMUM_SHADOW_VISIBILITY: f32 = 0.16;
const SPECULAR_STRENGTH: f32 = 0.28;
const SHARPEST_SPECULAR_EXPONENT: f32 = 96.0;
const SOFTEST_SPECULAR_EXPONENT: f32 = 4.0;
const SOFT_SHADOW_RAY_COUNT: i32 = 5;

struct VertexOut {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};

@vertex
fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOut {
  var positions = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  var uvs = array<vec2f, 3>(vec2f(0.0, 1.0), vec2f(2.0, 1.0), vec2f(0.0, -1.0));
  var output: VertexOut;
  output.position = vec4f(positions[vertexIndex], 0.0, 1.0);
  output.uv = uvs[vertexIndex];
  return output;
}

fn srgbToLinear(c: vec3f) -> vec3f {
  let low = c / 12.92;
  let high = pow((c + vec3f(0.055)) / 1.055, vec3f(2.4));
  return select(high, low, c <= vec3f(0.04045));
}

fn linearToSrgb(c: vec3f) -> vec3f {
  let v = max(c, vec3f(0.0));
  let low = v * 12.92;
  let high = 1.055 * pow(v, vec3f(1.0 / 2.4)) - vec3f(0.055);
  return select(high, low, v <= vec3f(0.0031308));
}

fn readDepth(uv: vec2f) -> f32 {
  let sample = textureSampleLevel(depthTexture, linearSampler, clamp(uv, vec2f(0.001), vec2f(0.999)), 0.0).rgb;
  return dot(sample, vec3f(0.299, 0.587, 0.114));
}

fn depthToZ(depth: f32) -> f32 {
  return mix(scene.farDepth, scene.nearDepth, clamp(depth, 0.0, 1.0));
}

fn unproject(uv: vec2f, z: f32) -> vec3f {
  let imageSize = vec2f(textureDimensions(depthTexture));
  let pixel = uv * imageSize;
  return vec3f(
    (pixel.x - scene.principalPoint.x) * z / scene.focalLength.x,
    -(pixel.y - scene.principalPoint.y) * z / scene.focalLength.y,
    z
  );
}

fn projectToUv(position: vec3f) -> vec2f {
  let imageSize = vec2f(textureDimensions(depthTexture));
  let pixel = vec2f(
    scene.focalLength.x * position.x / position.z + scene.principalPoint.x,
    scene.principalPoint.y - scene.focalLength.y * position.y / position.z
  );
  return pixel / imageSize;
}

fn depthNormal(uv: vec2f) -> vec3f {
  let dims = vec2f(textureDimensions(depthTexture));
  let texel = 1.35 / dims;
  let leftUv = uv - vec2f(texel.x, 0.0);
  let rightUv = uv + vec2f(texel.x, 0.0);
  let upUv = uv - vec2f(0.0, texel.y);
  let downUv = uv + vec2f(0.0, texel.y);
  let left = unproject(leftUv, depthToZ(readDepth(leftUv)));
  let right = unproject(rightUv, depthToZ(readDepth(rightUv)));
  let up = unproject(upUv, depthToZ(readDepth(upUv)));
  let down = unproject(downUv, depthToZ(readDepth(downUv)));
  return normalize(cross(right - left, down - up));
}

fn providedNormal(uv: vec2f) -> vec3f {
  let encoded = textureSampleLevel(normalTexture, linearSampler, uv, 0.0).rgb;
  let decoded = encoded * 2.0 - vec3f(1.0);
  // Standard tangent-map convention -> camera space, where -Z faces the camera.
  return normalize(vec3f(decoded.x, -decoded.y, -abs(decoded.z) - 0.08));
}

fn shadowVisibility(surfaceUv: vec2f, surface: vec3f, lightPosition: vec3f) -> f32 {
  let maximumSampleCount = clamp(i32(scene.shadowSampleCount), 0, 64);
  if (maximumSampleCount == 0) { return 1.0; }

  let imageSize = vec2f(textureDimensions(depthTexture));
  let lightUv = projectToUv(lightPosition);
  let rayLengthPixels = length((lightUv - surfaceUv) * imageSize);
  let pixelStride = 1.5;
  let requiredSampleCount = max(4, i32(ceil(rayLengthPixels / pixelStride)));
  let sampleCount = min(requiredSampleCount, maximumSampleCount);
  var strongestOcclusion = 0.0;
  for (var step = 0; step < sampleCount; step = step + 1) {
    let t = (f32(step) + 0.5) / f32(sampleCount);
    let sampleUv = mix(surfaceUv, lightUv, t);
    let inverseRayDepth = mix(1.0 / surface.z, 1.0 / lightPosition.z, t);
    let rayDepth = 1.0 / inverseRayDepth;
    if (all(sampleUv > vec2f(0.002)) && all(sampleUv < vec2f(0.998)) && rayDepth > scene.nearDepth) {
      let sceneZ = depthToZ(readDepth(sampleUv));
      let bias = 0.035 + rayDepth * 0.012;
      let depthSeparation = rayDepth - sceneZ;
      let transitionWidth = 0.06 + rayDepth * 0.01;
      let sampleOcclusion = smoothstep(bias, bias + transitionWidth, depthSeparation);
      strongestOcclusion = max(strongestOcclusion, sampleOcclusion);
      if (strongestOcclusion > 0.999) { break; }
    }
  }
  return mix(1.0, MINIMUM_SHADOW_VISIBILITY, strongestOcclusion);
}

fn softShadowVisibility(
  surfaceUv: vec2f,
  surfacePosition: vec3f,
  lightPosition: vec3f,
  lightDirection: vec3f,
  sourceRadius: f32
) -> f32 {
  if (sourceRadius <= 0.001) {
    return shadowVisibility(surfaceUv, surfacePosition, lightPosition);
  }

  let referenceAxis = select(
    vec3f(0.0, 1.0, 0.0),
    vec3f(1.0, 0.0, 0.0),
    abs(lightDirection.y) > 0.95
  );
  let diskTangent = normalize(cross(lightDirection, referenceAxis));
  let diskBitangent = normalize(cross(lightDirection, diskTangent));
  let diskSamples = array<vec2f, SOFT_SHADOW_RAY_COUNT>(
    vec2f(0.0, 0.0),
    vec2f(1.0, 0.0),
    vec2f(0.0, 1.0),
    vec2f(-1.0, 0.0),
    vec2f(0.0, -1.0)
  );

  var visibilitySum = 0.0;
  for (var sampleIndex = 0; sampleIndex < SOFT_SHADOW_RAY_COUNT; sampleIndex = sampleIndex + 1) {
    let diskOffset = diskSamples[sampleIndex] * sourceRadius;
    let samplePosition = lightPosition
      + diskTangent * diskOffset.x
      + diskBitangent * diskOffset.y;
    visibilitySum += shadowVisibility(surfaceUv, surfacePosition, samplePosition);
  }
  return visibilitySum / f32(SOFT_SHADOW_RAY_COUNT);
}

fn pointLightPosition(light: Light) -> vec3f {
  let lightUv = vec2f(
    0.5 + 0.5 * light.position.x,
    0.5 - 0.5 * light.position.y
  );
  return unproject(lightUv, light.position.z);
}

fn distanceAttenuation(distanceToLight: f32, lightRadius: f32) -> f32 {
  let radiusFade = clamp(1.0 - distanceToLight / lightRadius, 0.0, 1.0);
  let inverseSquareFalloff = 1.0 / (0.32 + 0.22 * distanceToLight * distanceToLight);
  return radiusFade * radiusFade * inverseSquareFalloff;
}

fn diffuseFactor(normal: vec3f, lightDirection: vec3f) -> f32 {
  return max(dot(normal, lightDirection), 0.0);
}

fn specularFactor(normal: vec3f, lightDirection: vec3f, viewDirection: vec3f) -> f32 {
  let halfwayDirection = normalize(lightDirection + viewDirection);
  let exponent = mix(SHARPEST_SPECULAR_EXPONENT, SOFTEST_SPECULAR_EXPONENT, scene.roughness);
  let highlight = pow(max(dot(normal, halfwayDirection), 0.0), exponent);
  let roughnessDimming = 1.0 - scene.roughness * 0.72;
  return highlight * roughnessDimming;
}

fn evaluatePointLight(
  surfaceUv: vec2f,
  surfacePosition: vec3f,
  normal: vec3f,
  viewDirection: vec3f,
  light: Light
) -> LightingContribution {
  let lightPosition = pointLightPosition(light);
  let surfaceToLight = lightPosition - surfacePosition;
  let distanceToLight = max(length(surfaceToLight), 0.001);
  let lightDirection = surfaceToLight / distanceToLight;

  let attenuation = distanceAttenuation(distanceToLight, light.radius);
  let visibility = softShadowVisibility(
    surfaceUv,
    surfacePosition,
    lightPosition,
    lightDirection,
    light.sourceRadius
  );
  let incidentLight = light.color * light.intensity * attenuation * visibility;

  let diffuse = incidentLight * diffuseFactor(normal, lightDirection);
  let specular = incidentLight * specularFactor(normal, lightDirection, viewDirection) * SPECULAR_STRENGTH;
  return LightingContribution(diffuse, specular);
}

fn reinhardToneMap(color: vec3f) -> vec3f {
  return color / (vec3f(1.0) + color);
}

@fragment
fn fragmentMain(input: VertexOut) -> @location(0) vec4f {
  let uv = input.uv;
  let radiance = textureSampleLevel(radianceTexture, linearSampler, uv, 0.0).rgb;
  let depth = readDepth(uv);
  let suppliedNormal = providedNormal(uv);
  let reconstructedNormal = depthNormal(uv);
  let normal = select(suppliedNormal, reconstructedNormal, scene.useDepthNormals > 0.5);

  if (scene.outputMode > 0.5 && scene.outputMode < 1.5) { return vec4f(radiance, 1.0); }
  if (scene.outputMode > 1.5 && scene.outputMode < 2.5) { return vec4f(vec3f(depth), 1.0); }
  if (scene.outputMode > 2.5) { return vec4f(normal * 0.5 + vec3f(0.5), 1.0); }

  let baseColor = srgbToLinear(radiance);
  let surfacePosition = unproject(uv, depthToZ(depth));
  let viewDirection = normalize(-surfacePosition);
  var diffuseEnergy = vec3f(scene.ambientIntensity);
  var specularEnergy = vec3f(0.0);

  for (var index = 0; index < ${MAX_LIGHTS}; index = index + 1) {
    let light = lights.values[index];
    if (f32(index) < scene.lightCount && light.enabled > 0.5) {
      let contribution = evaluatePointLight(uv, surfacePosition, normal, viewDirection, light);
      diffuseEnergy += contribution.diffuse;
      specularEnergy += contribution.specular;
    }
  }

  let linearColor = baseColor * diffuseEnergy + specularEnergy;
  let displayColor = linearToSrgb(reinhardToneMap(linearColor));
  return vec4f(displayColor, 1.0);
}
`;

const $ = (selector) => document.querySelector(selector);
const canvas = $("#gpu-canvas");
const handles = $("#light-handles");
const status = $("#gpu-status");
const loadingCover = $("#loading-cover");

const presets = {
  studio: [
    { name: "Key light", color: "#fff1d6", intensity: 4.5, x: -0.38, y: 0.32, z: 1.3, radius: 4.2, sourceRadius: 0.16, enabled: true },
    { name: "Fill light", color: "#9fc8ff", intensity: 2.6, x: 0.56, y: 0.06, z: 1.72, radius: 4.6, sourceRadius: 0.28, enabled: true },
    { name: "Rim light", color: "#ffd28f", intensity: 3.2, x: 0.2, y: 0.65, z: 3.05, radius: 3.8, sourceRadius: 0.12, enabled: true },
  ],
  sunset: [
    { name: "Window sun", color: "#ff9a48", intensity: 7.2, x: -0.68, y: 0.46, z: 1.1, radius: 5.7, sourceRadius: 0.22, enabled: true },
    { name: "Sky fill", color: "#7b98e8", intensity: 2.0, x: 0.54, y: 0.68, z: 2.7, radius: 5.8, sourceRadius: 0.36, enabled: true },
  ],
  neon: [
    { name: "Magenta", color: "#ff3fa4", intensity: 6.5, x: -0.53, y: 0.02, z: 1.25, radius: 4.2, sourceRadius: 0.14, enabled: true },
    { name: "Cyan", color: "#38dcff", intensity: 6.1, x: 0.58, y: 0.1, z: 1.35, radius: 4.4, sourceRadius: 0.14, enabled: true },
    { name: "Violet rim", color: "#815dff", intensity: 4.0, x: 0.05, y: 0.73, z: 3.1, radius: 4.8, sourceRadius: 0.2, enabled: true },
  ],
};

let lights = structuredClone(presets.studio);
let selectedIndex = 0;
let outputMode = 0;
let gpu = null;
let dirty = true;
let lastFrame = performance.now();
let frameSamples = [];

function hexToLinear(hex) {
  const value = hex.replace("#", "");
  const srgb = [0, 2, 4].map((offset) => parseInt(value.slice(offset, offset + 2), 16) / 255);
  return srgb.map((channel) => channel <= 0.04045 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4);
}

function clamp(value, min, max) { return Math.min(max, Math.max(min, value)); }

function updateRangeFill(input) {
  const percentage = ((Number(input.value) - Number(input.min)) / (Number(input.max) - Number(input.min))) * 100;
  input.style.setProperty("--fill", `${percentage}%`);
}

function markDirty() { dirty = true; }

function renderLightList() {
  const bulbIcon = `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 18h6M10 21h4M8.5 14.5A6 6 0 1 1 15.5 14.5C14.6 15.3 14 16.1 14 18h-4c0-1.9-.6-2.7-1.5-3.5Z"/></svg>`;
  $("#light-list").innerHTML = lights.map((light, index) => `
    <button class="light-item ${index === selectedIndex ? "selected" : ""}" data-light-index="${index}" type="button">
      <span class="light-bulb" style="--light:${light.color}">${bulbIcon}</span>
      <span class="light-copy"><b>${light.name}</b><small>${light.intensity.toFixed(1)} EV · ${light.z.toFixed(2)} m</small></span>
      <label class="mini-switch" aria-label="Enable ${light.name}"><input type="checkbox" ${light.enabled ? "checked" : ""} data-enable="${index}"/><i></i></label>
    </button>
  `).join("");

  document.querySelectorAll("[data-light-index]").forEach((element) => {
    element.addEventListener("click", () => selectLight(Number(element.dataset.lightIndex)));
  });
  document.querySelectorAll("[data-enable]").forEach((input) => {
    input.addEventListener("click", (event) => event.stopPropagation());
    input.addEventListener("change", () => {
      lights[Number(input.dataset.enable)].enabled = input.checked;
      renderHandles();
      markDirty();
    });
  });
}

function renderHandles() {
  handles.innerHTML = lights.map((light, index) => `
    <button class="light-handle ${index === selectedIndex ? "selected" : ""} ${light.enabled ? "" : "disabled"}"
      type="button" data-handle="${index}" data-index="L${index + 1}" style="--light:${light.color};left:${(light.x + 1) * 50}%;top:${(1 - light.y) * 50}%" aria-label="Move ${light.name}"></button>
  `).join("");
  document.querySelectorAll("[data-handle]").forEach((handle) => {
    handle.addEventListener("pointerdown", beginDrag);
    handle.addEventListener("keydown", (event) => {
      const index = Number(handle.dataset.handle);
      const delta = event.shiftKey ? 0.08 : 0.025;
      if (event.key === "ArrowLeft") lights[index].x -= delta;
      else if (event.key === "ArrowRight") lights[index].x += delta;
      else if (event.key === "ArrowUp") lights[index].y += delta;
      else if (event.key === "ArrowDown") lights[index].y -= delta;
      else return;
      event.preventDefault();
      lights[index].x = clamp(lights[index].x, -0.95, 0.95);
      lights[index].y = clamp(lights[index].y, -0.95, 0.95);
      selectLight(index);
      markDirty();
    });
  });
}

function beginDrag(event) {
  const dragHandle = event.currentTarget;
  const index = Number(dragHandle.dataset.handle);
  selectLight(index, false);
  dragHandle.setPointerCapture(event.pointerId);
  const move = (moveEvent) => {
    const bounds = canvas.getBoundingClientRect();
    lights[index].x = clamp(((moveEvent.clientX - bounds.left) / bounds.width) * 2 - 1, -0.95, 0.95);
    lights[index].y = clamp(1 - ((moveEvent.clientY - bounds.top) / bounds.height) * 2, -0.95, 0.95);
    dragHandle.style.left = `${(lights[index].x + 1) * 50}%`;
    dragHandle.style.top = `${(1 - lights[index].y) * 50}%`;
    markDirty();
  };
  const end = () => {
    dragHandle.removeEventListener("pointermove", move);
    dragHandle.removeEventListener("pointerup", end);
    dragHandle.removeEventListener("pointercancel", end);
  };
  dragHandle.addEventListener("pointermove", move);
  dragHandle.addEventListener("pointerup", end);
  dragHandle.addEventListener("pointercancel", end);
}

function selectLight(index, refreshHandles = true) {
  selectedIndex = clamp(index, 0, lights.length - 1);
  const light = lights[selectedIndex];
  $("#selected-name").textContent = light.name;
  $("#selected-swatch").style.background = light.color;
  $("#selected-swatch").style.color = light.color;
  $("#light-color").value = light.color;
  $("#color-value").textContent = light.color.toUpperCase();
  $("#intensity").value = light.intensity;
  $("#depth-position").value = light.z;
  $("#radius").value = light.radius;
  $("#source-radius").value = light.sourceRadius;
  updateEditorOutputs();
  renderLightList();
  if (refreshHandles) {
    renderHandles();
  } else {
    handles.querySelectorAll("[data-handle]").forEach((handle) => {
      handle.classList.toggle("selected", Number(handle.dataset.handle) === selectedIndex);
    });
  }
}

function updateEditorOutputs() {
  $("#intensity-value").textContent = Number($("#intensity").value).toFixed(1);
  $("#depth-value").textContent = `${Number($("#depth-position").value).toFixed(2)} m`;
  $("#radius-value").textContent = `${Number($("#radius").value).toFixed(1)} m`;
  $("#source-radius-value").textContent = `${Number($("#source-radius").value).toFixed(2)} m`;
  $("#shadow-samples-value").textContent = $("#shadow-samples").value;
  $("#shadow-samples").disabled = !$("#shadows").checked;
  document.querySelectorAll('input[type="range"]').forEach(updateRangeFill);
}

function applyPreset(name) {
  lights = structuredClone(presets[name]);
  selectedIndex = 0;
  document.querySelectorAll("[data-preset]").forEach((button) => button.classList.toggle("active", button.dataset.preset === name));
  selectLight(0);
  markDirty();
}

function wireInterface() {
  renderLightList();
  renderHandles();
  selectLight(0);

  document.querySelectorAll(".view-tab").forEach((tab) => tab.addEventListener("click", () => {
    outputMode = Number(tab.dataset.view);
    document.querySelectorAll(".view-tab").forEach((item) => {
      const active = item === tab;
      item.classList.toggle("active", active);
      item.setAttribute("aria-selected", String(active));
    });
    handles.hidden = outputMode !== 0;
    $("#viewport-hint").hidden = outputMode !== 0;
    markDirty();
  }));

  document.querySelectorAll("[data-preset]").forEach((button) => button.addEventListener("click", () => applyPreset(button.dataset.preset)));

  $("#add-light").addEventListener("click", () => {
    if (lights.length >= MAX_LIGHTS) return;
    const palette = ["#b4ffea", "#ffb0d8", "#c7b5ff", "#ffe098"];
    lights.push({ name: `Point light ${lights.length + 1}`, color: palette[lights.length % palette.length], intensity: 3.0, x: 0, y: 0.4, z: 1.6, radius: 4, sourceRadius: 0.15, enabled: true });
    selectLight(lights.length - 1);
    markDirty();
  });

  $("#remove-light").addEventListener("click", () => {
    if (lights.length <= 1) return;
    lights.splice(selectedIndex, 1);
    selectLight(Math.min(selectedIndex, lights.length - 1));
    markDirty();
  });

  $("#light-color").addEventListener("input", (event) => {
    lights[selectedIndex].color = event.target.value;
    $("#color-value").textContent = event.target.value.toUpperCase();
    $("#selected-swatch").style.background = event.target.value;
    renderLightList();
    renderHandles();
    markDirty();
  });

  const lightRanges = [
    ["#intensity", "intensity"],
    ["#depth-position", "z"],
    ["#radius", "radius"],
    ["#source-radius", "sourceRadius"],
  ];
  lightRanges.forEach(([selector, property]) => $(selector).addEventListener("input", (event) => {
    lights[selectedIndex][property] = Number(event.target.value);
    updateEditorOutputs();
    renderLightList();
    markDirty();
  }));

  ["#normal-source", "#ambient", "#roughness", "#shadows", "#shadow-samples"].forEach((selector) => $(selector).addEventListener("input", () => {
    $("#ambient-value").textContent = Number($("#ambient").value).toFixed(2);
    $("#roughness-value").textContent = Number($("#roughness").value).toFixed(2);
    $("#shadow-samples-value").textContent = $("#shadow-samples").value;
    $("#shadow-samples").disabled = !$("#shadows").checked;
    updateRangeFill($("#ambient"));
    updateRangeFill($("#roughness"));
    updateRangeFill($("#shadow-samples"));
    markDirty();
  }));

  $("#reset-camera").addEventListener("click", () => applyPreset("studio"));
  window.addEventListener("resize", resizeCanvas);
  updateEditorOutputs();
}

function resizeCanvas() {
  const bounds = canvas.getBoundingClientRect();
  if (!bounds.width) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.floor(bounds.width * dpr));
  const height = Math.max(1, Math.floor(bounds.height * dpr));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
    markDirty();
  }
}

async function loadTexture(device, url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not load ${url}`);
  const bitmap = await createImageBitmap(await response.blob(), { colorSpaceConversion: "none" });
  const texture = device.createTexture({
    size: [bitmap.width, bitmap.height, 1],
    format: "rgba8unorm",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
  });
  device.queue.copyExternalImageToTexture({ source: bitmap }, { texture }, [bitmap.width, bitmap.height]);
  bitmap.close();
  return texture;
}

async function initializeWebGPU() {
  if (!navigator.gpu) throw new Error("WebGPU is not available in this browser. Try current Chrome, Edge, or another WebGPU-enabled browser.");
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("No compatible WebGPU adapter was found.");
  const device = await adapter.requestDevice();
  const context = canvas.getContext("webgpu");
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: "opaque" });

  const [radiance, depth, normal] = await Promise.all([
    loadTexture(device, "/public/radiance.png"),
    loadTexture(device, "/public/depth.png"),
    loadTexture(device, "/public/normal.png"),
  ]);

  const module = device.createShaderModule({ label: "Relighting shader", code: shader });
  const compilation = await module.getCompilationInfo();
  const errors = compilation.messages.filter((message) => message.type === "error");
  if (errors.length) throw new Error(errors.map((message) => `WGSL ${message.lineNum}:${message.linePos} ${message.message}`).join("\n"));

  const pipeline = device.createRenderPipeline({
    label: "Full-screen relighting pipeline",
    layout: "auto",
    vertex: { module, entryPoint: "vertexMain" },
    fragment: { module, entryPoint: "fragmentMain", targets: [{ format }] },
    primitive: { topology: "triangle-list" },
  });

  const sceneBuffer = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const lightBuffer = device.createBuffer({ size: MAX_LIGHTS * 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const sampler = device.createSampler({ magFilter: "linear", minFilter: "linear", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge" });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: sceneBuffer } },
      { binding: 1, resource: { buffer: lightBuffer } },
      { binding: 2, resource: radiance.createView() },
      { binding: 3, resource: depth.createView() },
      { binding: 4, resource: normal.createView() },
      { binding: 5, resource: sampler },
    ],
  });

  device.lost.then((info) => showError(`WebGPU device lost: ${info.message}`));
  return { device, context, pipeline, sceneBuffer, lightBuffer, bindGroup };
}

function uploadUniforms() {
  const shadowSampleCount = $("#shadows").checked ? Number($("#shadow-samples").value) : 0;
  const sceneData = new Float32Array([
    Number($("#ambient").value),
    Number($("#roughness").value),
    Number($("#normal-source").value),
    shadowSampleCount,
    CAMERA.focalLength.x,
    CAMERA.focalLength.y,
    CAMERA.principalPoint.x,
    CAMERA.principalPoint.y,
    CAMERA.nearDepth,
    CAMERA.farDepth,
    lights.length,
    outputMode,
  ]);
  gpu.device.queue.writeBuffer(gpu.sceneBuffer, 0, sceneData);

  const lightData = new Float32Array(MAX_LIGHTS * 12);
  lights.forEach((light, index) => {
    const offset = index * 12;
    const color = hexToLinear(light.color);
    lightData.set([light.x, light.y, light.z, light.enabled ? 1 : 0], offset);
    lightData.set([color[0], color[1], color[2], light.intensity], offset + 4);
    lightData.set([light.radius, light.sourceRadius, 0, 0], offset + 8);
  });
  gpu.device.queue.writeBuffer(gpu.lightBuffer, 0, lightData);
}

function renderFrame(now) {
  if (!gpu) return;
  const delta = now - lastFrame;
  lastFrame = now;
  if (delta < 100) {
    frameSamples.push(delta);
    if (frameSamples.length > 45) frameSamples.shift();
    if (frameSamples.length === 45) {
      const average = frameSamples.reduce((sum, value) => sum + value, 0) / frameSamples.length;
      $("#frame-ms").textContent = `${average.toFixed(1)} ms`;
    }
  }

  if (dirty) {
    uploadUniforms();
    const encoder = gpu.device.createCommandEncoder({ label: "Relight frame" });
    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: gpu.context.getCurrentTexture().createView(), clearValue: { r: 0.02, g: 0.02, b: 0.025, a: 1 }, loadOp: "clear", storeOp: "store" }],
    });
    pass.setPipeline(gpu.pipeline);
    pass.setBindGroup(0, gpu.bindGroup);
    pass.draw(3);
    pass.end();
    gpu.device.queue.submit([encoder.finish()]);
    dirty = false;
  }
  requestAnimationFrame(renderFrame);
}

function showError(message) {
  status.className = "gpu-status error";
  status.querySelector("b").textContent = "WEBGPU ERROR";
  loadingCover.innerHTML = `<b style="max-width:420px;padding:24px;text-align:center;line-height:1.6;color:#f0a0a0">${message}</b>`;
  loadingCover.classList.remove("hidden");
}

async function main() {
  wireInterface();
  resizeCanvas();
  try {
    gpu = await initializeWebGPU();
    status.className = "gpu-status ready";
    status.querySelector("b").textContent = "WEBGPU READY";
    loadingCover.classList.add("hidden");
    dirty = true;
    requestAnimationFrame(renderFrame);
  } catch (error) {
    console.error(error);
    showError(error.message);
  }
}

main();
