// script.js (P3 Integrated & Corrected)

// --- Three.js & Core P2 Rendering ---
let scene, camera, renderer, plane, material, clock;
let boat;
let isSailing = false;

// --- Journey, Cycle, and P2 UI Management ---
const CYCLE_DURATION = 10; // seconds
let cycleTimer = 0;
let journeyLog = [];
let journeyLogUI, logEntriesList;

// --- P3 Console & Analysis UI ---
let shipwrightsConsole, consoleTab, phrase1Input, phrase2Input, launchButtonQuick, launchButtonFull, methodCardsContainer, aggregationViewRadios;
let infoPanel, infoPanelTab; // ADDED: Variable for the new info panel
let currentLaunchData = null; // To store the full backend response
let currentSelectedMethodCard = null;

// --- Color constants ---
const TURBULENT_COLOR = new THREE.Color(0xff8844);
const SMOOTH_COLOR = new THREE.Color(0x66ddff);

// --- Particle/FX Variables (from P2) ---
const FORGE_PARTICLE_COUNT = 50;
let forgeParticles = [];
const MAX_TRAIL_SEGMENTS = 300;
let trailSegments = [];
let trailPoolIndex = 0;
let timeSinceLastSpawn = 0;
const WIND_STREAK_COUNT = 200;
let windStreaks = [];
const MAX_JOURNEY_NODES = 100;
let journeyNodes = [];
let journeyNodePoolIndex = 0;
let constellationLines = [];

// --- Game State & Other Globals (from P2) ---
const gameState = { r: 0, theta: 0, polarity: 0.5, worldOffset: new THREE.Vector2(0, 0) };
const MIN_ZOOM = 0.6;
const MAX_ZOOM = 2.5;
let targetZoom = 1.0;

// --- Shader Code (Unchanged) ---
const vertexShader = `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;
const fragmentShader = `
  varying vec2 vUv;
  uniform float u_time;
  uniform vec2 u_resolution;
  uniform float u_scale;
  uniform vec2 u_scroll;
  uniform vec2 u_offset;
  uniform float u_shockwave_time;
  uniform float u_shockwave_strength;
  vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
  vec2 mod289(vec2 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
  vec3 permute(vec3 x) { return mod289(((x*34.0)+1.0)*x); }
  float snoise(vec2 v) { const vec4 C = vec4(0.211324865405187, 0.366025403784439, -0.577350269189626, 0.024390243902439); vec2 i  = floor(v + dot(v, C.yy) ); vec2 x0 = v -   i + dot(i, C.xx); vec2 i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0); vec4 x12 = x0.xyxy + C.xxzz; x12.xy -= i1; i = mod289(i); vec3 p = permute( permute( i.y + vec3(0.0, i1.y, 1.0 )) + i.x + vec3(0.0, i1.x, 1.0 )); vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy), dot(x12.zw,x12.zw)), 0.0); m = m*m; m = m*m; vec3 x = 2.0 * fract(p * C.www) - 1.0; vec3 h = abs(x) - 0.5; vec3 ox = floor(x + 0.5); vec3 a0 = x - ox; m *= 1.79284291400159 - 0.85373472095314 * ( a0*a0 + h*h ); vec3 g; g.x  = a0.x  * x0.x  + h.x  * x0.y; g.yz = a0.yz * x12.wz + h.yz * x12.zx; return 130.0 * dot(m, g); }
  float fbm(vec2 st) { float value = 0.0; float amplitude = 0.5; for (int i = 0; i < 6; i++) { value += amplitude * snoise(st); st *= 2.0; amplitude *= 0.5; } return value; }
  float hash(vec2 p) { p = fract(p * vec2(443.897, 441.423)); p += dot(p, p.yx + 19.19); return fract((p.x + p.y) * p.x); }
  float triangle_noise(vec2 st) { vec2 i = floor(st); vec2 f = fract(st); float h = hash(i); float d = step(f.x, f.y + h*0.2); return mix(f.x, 1.0 - f.y, d) + h * 0.5; }
  void main() {
    vec2 st = vUv;
    if (u_shockwave_time >= 0.0) { vec2 center = vec2(0.5); float dist = distance(st, center); float ripple = sin(dist * 15.0 - u_shockwave_time * 20.0); float shockwave_effect = ripple * (1.0 - u_shockwave_time) * u_shockwave_strength; st += normalize(st - center) * shockwave_effect * 0.1; }
    vec2 st_nebula = st * 3.0 + u_offset * 0.2; st_nebula.x += u_time * 0.01; float nebula_pattern = fbm(st_nebula); vec3 nebula_color1 = vec3(0.0, 0.01, 0.05); vec3 nebula_color2 = vec3(0.05, 0.0, 0.2); vec3 nebula_color3 = vec3(0.2, 0.1, 0.3); vec3 nebula_color = mix(mix(nebula_color1, nebula_color2, smoothstep(-0.2, 0.2, nebula_pattern)), nebula_color3, smoothstep(0.1, 0.5, nebula_pattern)); vec2 st_crystal = st * u_resolution / u_resolution.y; st_crystal += u_offset; st_crystal *= u_scale; st_crystal += u_time * u_scroll; float n_crystal = 0.0; n_crystal += triangle_noise(st_crystal * 1.0) * 0.5; n_crystal += triangle_noise(st_crystal * 2.0) * 0.25; n_crystal += triangle_noise(st_crystal * 4.0) * 0.125; n_crystal += triangle_noise(st_crystal * 8.0) * 0.06; float crystal_mask = smoothstep(0.2, 0.8, n_crystal); vec3 crystal_color1 = vec3(0.01, 0.0, 0.05); vec3 crystal_color2 = vec3(0.4, 0.15, 0.5); vec3 crystal_color = mix(crystal_color1, crystal_color2, crystal_mask); vec3 final_color = nebula_color + crystal_color; gl_FragColor = vec4(final_color, 1.0);
  }
`;

function init() {
    clock = new THREE.Clock(); scene = new THREE.Scene(); camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000); camera.position.z = targetZoom; renderer = new THREE.WebGLRenderer({ antialias: true }); renderer.setSize(window.innerWidth, window.innerHeight); 
    
    document.getElementById('gamespace-wrapper').appendChild(renderer.domElement);

    window.addEventListener('resize', onWindowResize, false); window.addEventListener('wheel', onMouseWheel, { passive: false });
    const uniforms = { u_time: { value: 0.0 }, u_resolution: { value: new THREE.Vector2(window.innerWidth, window.innerHeight) }, u_scale: { value: 1.0 }, u_scroll: { value: new THREE.Vector2(0.05, 0.02) }, u_offset: { value: gameState.worldOffset }, u_shockwave_time: { value: -1.0 }, u_shockwave_strength: { value: 0.15 } };
    material = new THREE.ShaderMaterial({ vertexShader, fragmentShader, uniforms });
    const geometry = new THREE.PlaneGeometry(15, 15); plane = new THREE.Mesh(geometry, material); plane.position.z = -0.1; scene.add(plane);
    createBoat(); createTrail(); createForgeParticles(); createWindStreaks(); createJourneyNodes();
    journeyLogUI = document.getElementById('journey-log-ui'); logEntriesList = document.getElementById('log-entries');
    setupConsoleUI();
    animate();
}

function setupConsoleUI() {
    shipwrightsConsole = document.getElementById('shipwrights-console');
    consoleTab = document.getElementById('console-tab');
    phrase1Input = document.getElementById('phrase1');
    phrase2Input = document.getElementById('phrase2');
    launchButtonQuick = document.getElementById('launchButtonQuick');
    launchButtonFull = document.getElementById('launchButtonFull');
    methodCardsContainer = document.getElementById('methodCardsContainer');
    aggregationViewRadios = document.querySelectorAll('input[name="aggregationView"]');

    // ADDED: Get info panel elements
    infoPanel = document.getElementById('info-panel');
    infoPanelTab = document.getElementById('info-panel-tab');

    // ADDED: Add event listener for the info panel tab
    infoPanelTab.addEventListener('click', () => infoPanel.classList.toggle('panel-collapsed'));

    consoleTab.addEventListener('click', () => { shipwrightsConsole.classList.toggle('console-collapsed'); });
    launchButtonQuick.addEventListener('click', () => launchAnalysis('quick'));
    launchButtonFull.addEventListener('click', () => launchAnalysis('full'));
    aggregationViewRadios.forEach(radio => { radio.addEventListener('change', updateAllCardsView); });
}

async function launchAnalysis(analysis_depth) {
    const p1Val = phrase1Input.value.trim(); const p2Val = phrase2Input.value.trim();
    if (!p1Val || !p2Val) { alert("Please provide both Concept 1 and Concept 2."); return; }
    launchButtonQuick.disabled = true; launchButtonFull.disabled = true;
    const originalText = analysis_depth === 'quick' ? launchButtonQuick.textContent : launchButtonFull.textContent;
    const buttonToAnimate = analysis_depth === 'quick' ? launchButtonQuick : launchButtonFull;
    buttonToAnimate.textContent = 'Analyzing...';
    methodCardsContainer.innerHTML = '<p class="placeholder-text">Analyzing winds... This may take a moment.</p>';
    currentLaunchData = null; currentSelectedMethodCard = null;
    await getAndDisplayAnalysis(p1Val, p2Val, analysis_depth);
    buttonToAnimate.textContent = originalText;
    launchButtonQuick.disabled = false; launchButtonFull.disabled = false;
}

async function getAndDisplayAnalysis(phrase1, phrase2, analysis_depth) {
    try {
        const API_URL = "https://apocryphaleditor-synestheticvoyage.hf.space";
        const response = await fetch(`${API_URL}/get_hull_analysis_by_methods`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ phrase1, phrase2, analysis_depth }), });
        if (!response.ok) { const errorText = await response.text(); throw new Error(`Backend Error: ${response.status} - ${errorText}`); }
        const data = await response.json();
        if (data.error || !data.results || data.results.length === 0) { throw new Error(`Backend response error: ${data.error || 'No results found.'}`); }
        currentLaunchData = data.results; methodCardsContainer.innerHTML = '';
        currentLaunchData.forEach(methodData => { const cardElement = createMethodCard(methodData); methodCardsContainer.appendChild(cardElement); });
        updateAllCardsView();
    } catch (error) { console.error("Analysis failed:", error); methodCardsContainer.innerHTML = `<p class="placeholder-text" style="color: #ff8844;">Analysis failed. Please check the backend connection and console for details.<br><small>${error.message}</small></p>`; currentLaunchData = null; }
}

function createMethodCard(methodData) {
    const card = document.createElement('div'); card.className = 'method-card'; card.dataset.methodId = methodData.method_id;
    card.innerHTML = `
        <h3 class="method-name">${methodData.method_name}</h3>
        <div class="polarity-meter"><div class="pole-label">Safe Pole (Ideal)</div><div class="meter-track"><div class="separation-bar north-bar"></div><div class="separation-bar south-bar"></div><div class="center-line"></div></div><div class="pole-label">Unsafe Pole (Ideal)</div><div class="polarity-score-text">Polarity: N/A</div></div>
        <div class="boat-visuals-container"><div class="magnitude-container"><div class="magnitude-label-value-line"><span class="magnitude-label">Magnitude r=</span><span class="magnitude-value">N/A</span></div><div class="magnitude-bar-track"><div class="magnitude-bar-fill"></div></div></div><div class="orientation-container"><svg class="mini-compass-svg" width="60" height="60"></svg><div class="theta-value-text">θ=N/A</div></div></div>
        <button class="set-sail-button">Set Course & Sail</button>`;
    
    card.querySelector('.set-sail-button').addEventListener('click', () => {
        if (isSailing) return;
        const selectedAggView = document.querySelector('input[name="aggregationView"]:checked').value;
        const variantData = methodData.aggregation_variants.find(v => v.aggregation_applied === selectedAggView);
        if (!variantData || variantData.error) { alert('Cannot sail with this method; data is invalid or contains an error.'); return; }
        
        const backend_theta_deg = variantData.boat_theta || 0;
        gameState.theta = backend_theta_deg * (Math.PI / 180);

        gameState.r = variantData.boat_r;
        gameState.polarity = variantData.polarity_score;
        
        const logEntry = { promptA: phrase1Input.value, promptB: phrase2Input.value, method: methodData.method_name, r: gameState.r, theta: gameState.theta, polarity: gameState.polarity };
        journeyLog.push(logEntry);
        updateJourneyLogUI();

        triggerWordForgeAnimation(logEntry.promptA, logEntry.promptB, () => {
            triggerShockwave(); cycleTimer = 0; isSailing = true; shipwrightsConsole.classList.add('console-collapsed');
        });
        
        if (currentSelectedMethodCard) currentSelectedMethodCard.classList.remove('selected-method-card');
        card.classList.add('selected-method-card'); currentSelectedMethodCard = card;
    });
    return card;
}

function updateMethodCardDisplay(cardElement, fullMethodData, selectedAggView) {
    const variantData = fullMethodData.aggregation_variants.find(v => v.aggregation_applied === selectedAggView);
    const scoreTextEl = cardElement.querySelector('.polarity-score-text'); const northBar = cardElement.querySelector('.north-bar'); const southBar = cardElement.querySelector('.south-bar'); const magnitudeValueText = cardElement.querySelector('.magnitude-value'); const magnitudeBarFill = cardElement.querySelector('.magnitude-bar-fill'); const miniCompassSvgEl = cardElement.querySelector('.mini-compass-svg'); const thetaValueText = cardElement.querySelector('.theta-value-text'); const sailButton = cardElement.querySelector('.set-sail-button');

    if (!variantData || variantData.error) { scoreTextEl.textContent = `Polarity: ${variantData ? 'Error' : 'N/A'}`; northBar.style.width = '0%'; southBar.style.width = '0%'; magnitudeValueText.textContent = 'N/A'; magnitudeBarFill.style.width = '0%'; thetaValueText.textContent = 'θ=N/A'; sailButton.disabled = true; miniCompassSvgEl.innerHTML = `<circle cx="30" cy="30" r="25" fill="rgba(255,255,255,0.1)" stroke="rgba(255,255,255,0.3)"></circle><text x="30" y="12" text-anchor="middle" fill="#ccc" font-size="8px">N</text>`; return; }
    sailButton.disabled = false;
    const polarity = variantData.polarity_score || 0; scoreTextEl.textContent = `Polarity: ${polarity.toFixed(3)}`; const barWidth = polarity * 50; northBar.style.width = `${barWidth}%`; southBar.style.width = `${barWidth}%`;
    const boatR = variantData.boat_r || 0; magnitudeValueText.textContent = boatR.toFixed(2); magnitudeBarFill.style.width = `${Math.min(boatR * 100, 100)}%`;
    
    const backend_theta_deg = variantData.boat_theta || 0;
    const display_theta_deg = (90 - backend_theta_deg + 360) % 360;
    thetaValueText.textContent = `θ=${display_theta_deg.toFixed(1)}°`;
    
    const r = 25, c = 30;
    const display_theta_rad = display_theta_deg * (Math.PI / 180);
    const arrowLength = r * 0.7;
    const x2 = c + arrowLength * Math.sin(display_theta_rad);
    const y2 = c - arrowLength * Math.cos(display_theta_rad);
    miniCompassSvgEl.innerHTML = `<circle cx="${c}" cy="${c}" r="${r}" fill="rgba(255,255,255,0.1)" stroke="rgba(255,255,255,0.3)"/><text x="${c}" y="${c - r + 10}" text-anchor="middle" fill="#ccc" font-size="8px">N</text><line x1="${c}" y1="${c}" x2="${x2}" y2="${y2}" stroke="#ffdda3" stroke-width="2"/>`;
}

function updateAllCardsView() {
    if (!currentLaunchData) return;
    const selectedAggView = document.querySelector('input[name="aggregationView"]:checked').value;
    const cards = methodCardsContainer.querySelectorAll('.method-card');
    cards.forEach(cardElement => {
        const methodId = cardElement.dataset.methodId; const fullMethodData = currentLaunchData.find(m => m.method_id === methodId);
        if (fullMethodData) { updateMethodCardDisplay(cardElement, fullMethodData, selectedAggView); }
    });
}

function animate() { requestAnimationFrame(animate); const deltaTime = clock.getDelta(); camera.position.z += (targetZoom - camera.position.z) * 0.1; material.uniforms.u_time.value = clock.getElapsedTime(); if (material.uniforms.u_shockwave_time.value >= 0) { material.uniforms.u_shockwave_time.value += deltaTime * 0.8; if (material.uniforms.u_shockwave_time.value > 1) { material.uniforms.u_shockwave_time.value = -1; } } updateBoat(deltaTime); updateTrail(deltaTime); updateForgeParticles(deltaTime); updateWindStreaks(deltaTime); updateJourneyNodes(deltaTime); updateConstellationLines(deltaTime); if (isSailing) { cycleTimer += deltaTime; if (cycleTimer >= CYCLE_DURATION) endCycleAndPrepareForNext(); } renderer.render(scene, camera); }

function updateBoat(deltaTime) {
    if (!boat) return;
    boat.position.set(0, 0, 0.1);
    if (isSailing) {
        const targetRotation = gameState.theta + Math.PI;
        boat.rotation.z += (targetRotation - boat.rotation.z) * 0.1;

        const moveSpeed = gameState.r * 0.2;
        const moveX = Math.cos(gameState.theta) * moveSpeed * deltaTime;
        const moveY = Math.sin(gameState.theta) * moveSpeed * deltaTime;
        gameState.worldOffset.x -= moveX;
        gameState.worldOffset.y -= moveY;

        timeSinceLastSpawn += deltaTime;
        const spawnInterval = (1.1 - gameState.r) * 0.05 + 0.001;
        if (gameState.r > 0.05 && timeSinceLastSpawn > spawnInterval) { spawnTrailSegment(); timeSinceLastSpawn = 0; }
    }
    updateBoatVisuals();
}

function spawnTrailSegment() {
    const segment = trailSegments[trailPoolIndex];
    segment.is_active = true; segment.visible = true;
    segment.position.set(0, 0, 0.05);
    segment.maxLife = segment.life = Math.max(0.5, gameState.r * 2.0);
    const trailSpeed = gameState.r * 0.15;

    const trailAngle = gameState.theta + Math.PI;
    segment.velocity.set(Math.cos(trailAngle) * trailSpeed, Math.sin(trailAngle) * trailSpeed);

    if (gameState.polarity < 0.3) {
        const spread = (1 - gameState.polarity) * 0.3;
        segment.velocity.x += (Math.random() - 0.5) * spread;
        segment.velocity.y += (Math.random() - 0.5) * spread;
    }
    trailPoolIndex = (trailPoolIndex + 1) % MAX_TRAIL_SEGMENTS;
}

function updateWindStreaks(deltaTime) {
    const baseSpeed = 4.0;
    const speed = baseSpeed * (0.5 + gameState.r);
    
    const mathAngle = gameState.theta;

    for (const streak of windStreaks) {
        streak.position.x += Math.cos(mathAngle) * speed * deltaTime;
        streak.position.y += Math.sin(mathAngle) * speed * deltaTime;
        streak.material.opacity = gameState.r * 0.5;
        streak.scale.x = 0.5 + gameState.r * 1.5;
        streak.rotation.z = mathAngle;
        const bounds = {x: 3.5, y: 2.5};
        if (streak.position.x > bounds.x) streak.position.x = -bounds.x;
        if (streak.position.x < -bounds.x) streak.position.x = bounds.x;
        if (streak.position.y > bounds.y) streak.position.y = -bounds.y;
        if (streak.position.y < -bounds.y) streak.position.y = -bounds.y;
    }
}

function endCycleAndPrepareForNext() { spawnJourneyNode(); isSailing = false; gameState.r = 0; shipwrightsConsole.classList.remove('console-collapsed'); phrase1Input.focus(); }

function updateJourneyLogUI() {
    if (journeyLog.length > 0) { journeyLogUI.classList.remove('hidden'); }
    logEntriesList.innerHTML = '';
    for (const entry of journeyLog) {
        const listItem = document.createElement('li');
        
        const math_theta_rad = entry.theta;
        const display_theta_deg = (90 - (math_theta_rad * 180 / Math.PI) + 360) % 360;
        
        const r_val = entry.r.toFixed(2); const polarity_val = entry.polarity.toFixed(2);
        listItem.innerHTML = `<div class="log-prompts">"${entry.promptA}" / "${entry.promptB}"</div><div class="log-method" style="font-size:0.8em; color: #ccc;">Method: <em>${entry.method}</em></div><div class="log-metrics">r: <span class="metric-r">${r_val}</span> | θ: <span class="metric-theta">${display_theta_deg.toFixed(1)}°</span> | pol: <span class="metric-polarity">${polarity_val}</span></div>`;
        logEntriesList.appendChild(listItem);
    }
    journeyLogUI.scrollTop = journeyLogUI.scrollHeight;
}

function triggerWordForgeAnimation(textA, textB, onComplete) {
    const container = document.getElementById('forge-animation-container'); const promptA_el = phrase1Input; const promptB_el = phrase2Input; container.innerHTML = ''; const animationDuration = 1200;
    const createLetters = (text, rect) => {
        Array.from(text).forEach(char => {
            if (char === ' ') return;
            const letter = document.createElement('span'); letter.className = 'flying-letter'; letter.textContent = char; const consoleRect = shipwrightsConsole.getBoundingClientRect();
            letter.style.left = `${consoleRect.left + rect.left + (Math.random() * rect.width)}px`; letter.style.top = `${rect.top + (Math.random() * rect.height)}px`;
            letter.style.animation = `fly-to-center ${animationDuration}ms forwards ease-in`; letter.style.animationDelay = `${Math.random() * 300}ms`;
            container.appendChild(letter);
        });
    };
    createLetters(textA, promptA_el.getBoundingClientRect()); createLetters(textB, promptB_el.getBoundingClientRect());
    setTimeout(() => { container.innerHTML = ''; triggerForgeParticleBurst(); if (onComplete) onComplete(); }, animationDuration + 200);
}

function createBoat() { const shape = new THREE.Shape(); shape.moveTo(0.1, 0); shape.lineTo(-0.05, 0.05); shape.lineTo(-0.05, -0.05); shape.closePath(); const geometry = new THREE.ShapeGeometry(shape); const material = new THREE.MeshBasicMaterial({ color: 0xffeeaa, blending: THREE.AdditiveBlending, transparent: true, opacity: 0.9 }); boat = new THREE.Mesh(geometry, material); boat.position.z = 0.1; scene.add(boat); }
function updateBoatVisuals() { const baseOpacity = 0.5 + gameState.r * 0.5; const boatColor = new THREE.Color(); boatColor.lerpColors(TURBULENT_COLOR, SMOOTH_COLOR, gameState.polarity); boat.material.color.set(boatColor); if (isSailing && gameState.polarity < 0.3) { const jitterAmount = (1 - gameState.polarity) * 0.005; boat.position.x += (Math.random() - 0.5) * jitterAmount; boat.position.y += (Math.random() - 0.5) * jitterAmount; boat.material.opacity = baseOpacity - (Math.random() * 0.4); } else { boat.material.opacity = baseOpacity; } }
function spawnJourneyNode() { const node = journeyNodes[journeyNodePoolIndex]; node.is_active = true; node.visible = true; node.worldPos.copy(gameState.worldOffset); const nodeColor = new THREE.Color(); nodeColor.lerpColors(TURBULENT_COLOR, SMOOTH_COLOR, gameState.polarity); node.material.color.set(nodeColor); node.scale.set(0,0,0); if (journeyLog.length > 0) { const prevNodeIndex = (journeyNodePoolIndex - 1 + MAX_JOURNEY_NODES) % MAX_JOURNEY_NODES; const prevNode = journeyNodes[prevNodeIndex]; if(prevNode.is_active) { const lineMaterial = new THREE.LineBasicMaterial({ color: nodeColor, blending: THREE.AdditiveBlending, transparent: true, opacity: 0.7 }); const points = [new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 0, 0)]; const geometry = new THREE.BufferGeometry().setFromPoints(points); const line = new THREE.Line(geometry, lineMaterial); line.worldStart = new THREE.Vector2().copy(prevNode.worldPos); line.worldEnd = new THREE.Vector2().copy(node.worldPos); constellationLines.push(line); scene.add(line); } } journeyNodePoolIndex = (journeyNodePoolIndex + 1) % MAX_JOURNEY_NODES; }
function updateConstellationLines(deltaTime) { const deepParallaxFactor = 0.2; for (const line of constellationLines) { const startX = (line.worldStart.x - gameState.worldOffset.x) * deepParallaxFactor; const startY = (line.worldStart.y - gameState.worldOffset.y) * deepParallaxFactor; const endX = (line.worldEnd.x - gameState.worldOffset.x) * deepParallaxFactor; const endY = (line.worldEnd.y - gameState.worldOffset.y) * deepParallaxFactor; const positions = line.geometry.attributes.position.array; positions[0] = startX; positions[1] = startY; positions[3] = endX; positions[4] = endY; line.geometry.attributes.position.needsUpdate = true; } }
function onWindowResize() { camera.aspect = window.innerWidth / window.innerHeight; camera.updateProjectionMatrix(); renderer.setSize(window.innerWidth, window.innerHeight); material.uniforms.u_resolution.value.set(window.innerWidth, window.innerHeight); }
function onMouseWheel(event) { event.preventDefault(); const zoomSpeed = 0.005; targetZoom += event.deltaY * zoomSpeed; targetZoom = THREE.MathUtils.clamp(targetZoom, MIN_ZOOM, MAX_ZOOM); }
function createTrail() { const segmentGeometry = new THREE.PlaneGeometry(0.04, 0.04); const segmentMaterial = new THREE.MeshBasicMaterial({ color: 0xffaacc, blending: THREE.AdditiveBlending, transparent: true }); for (let i = 0; i < MAX_TRAIL_SEGMENTS; i++) { const segment = new THREE.Mesh(segmentGeometry, segmentMaterial.clone()); segment.visible = false; segment.is_active = false; segment.life = 0; segment.maxLife = 1; segment.velocity = new THREE.Vector2(0, 0); trailSegments.push(segment); scene.add(segment); } }
function updateTrail(deltaTime) { for (const segment of trailSegments) { if (segment.is_active) { segment.life -= deltaTime; segment.position.x += segment.velocity.x * deltaTime; segment.position.y += segment.velocity.y * deltaTime; if (segment.life <= 0) { segment.is_active = false; segment.visible = false; } else { segment.material.opacity = (segment.life / segment.maxLife) * 0.7; } } } }
function createForgeParticles() { const particleGeometry = new THREE.PlaneGeometry(0.08, 0.08); const particleMaterial = new THREE.MeshBasicMaterial({ color: 0xffffff, blending: THREE.AdditiveBlending, transparent: true }); for (let i = 0; i < FORGE_PARTICLE_COUNT; i++) { const particle = new THREE.Mesh(particleGeometry, particleMaterial.clone()); particle.visible = false; particle.is_active = false; particle.life = 0; particle.velocity = new THREE.Vector3(0, 0, 0); forgeParticles.push(particle); scene.add(particle); } }
function triggerForgeParticleBurst() { for (const particle of forgeParticles) { particle.is_active = true; particle.visible = true; particle.life = Math.random() * 0.5 + 0.3; particle.position.set(0, 0, 0.2); const angle = Math.random() * Math.PI * 2; const speed = Math.random() * 1.5 + 0.5; particle.velocity.set( Math.cos(angle) * speed, Math.sin(angle) * speed, 0 ); particle.material.opacity = 1.0; } }
function updateForgeParticles(deltaTime) { for (const particle of forgeParticles) { if (particle.is_active) { particle.life -= deltaTime; if (particle.life <= 0) { particle.is_active = false; particle.visible = false; } else { particle.position.x += particle.velocity.x * deltaTime; particle.position.y += particle.velocity.y * deltaTime; particle.material.opacity = particle.life / 0.8; } } } }
function createWindStreaks() { const streakMaterial = new THREE.MeshBasicMaterial({ color: 0xffffff, blending: THREE.AdditiveBlending, transparent: true, opacity: 0.5 }); for (let i = 0; i < WIND_STREAK_COUNT; i++) { const streakGeometry = new THREE.PlaneGeometry(0.2, 0.002); const streak = new THREE.Mesh(streakGeometry, streakMaterial.clone()); streak.position.set((Math.random() - 0.5) * 4, (Math.random() - 0.5) * 4, 0); windStreaks.push(streak); scene.add(streak); } }
function createJourneyNodes() { const nodeGeometry = new THREE.CircleGeometry(0.02, 16); const nodeMaterial = new THREE.MeshBasicMaterial({ blending: THREE.AdditiveBlending, transparent: true }); for (let i = 0; i < MAX_JOURNEY_NODES; i++) { const node = new THREE.Mesh(nodeGeometry, nodeMaterial.clone()); node.visible = false; node.is_active = false; node.worldPos = new THREE.Vector2(0,0); journeyNodes.push(node); scene.add(node); } }
function updateJourneyNodes(deltaTime) { for (const node of journeyNodes) { if (node.is_active) { const deepParallaxFactor = 0.2; node.position.x = (node.worldPos.x - gameState.worldOffset.x) * deepParallaxFactor; node.position.y = (node.worldPos.y - gameState.worldOffset.y) * deepParallaxFactor; if(node.scale.x < 1.0) { node.scale.x += deltaTime * 0.5; node.scale.y += deltaTime * 0.5; node.scale.z += deltaTime * 0.5; } } } }
function triggerShockwave() { material.uniforms.u_shockwave_time.value = 0; }

// --- Run Application ---
init();