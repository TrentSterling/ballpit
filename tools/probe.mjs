// Physics test rig. Boots the pit in a real browser over CDP, runs each mode for
// a fixed time, then reads the particle buffer back and JUDGES it.
//
// The judgement is the point. A screenshot cannot tell you whether a dam break
// spread or collapsed; a height profile can, and it can do it while you are
// asleep. Each mode declares what it should look like when the physics is right,
// and the run fails when it does not.
//
//   node tools/probe.mjs [url] [secondsPerMode]

import { spawn } from 'node:child_process';
import { writeFileSync, existsSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';

const URL_ = process.argv[2] ?? 'http://localhost:8100/';
const SECONDS = Number(process.argv[3]) || 5;
const OUT = join(import.meta.dirname, '..', 'shots');
const ARCHIVE = join(OUT, 'archive');
const STAMP = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 16);
if (!existsSync(OUT)) mkdirSync(OUT);
if (!existsSync(ARCHIVE)) mkdirSync(ARCHIVE);

const CHROME = [
  'C:/Program Files/Google/Chrome/Application/chrome.exe',
  'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe',
  `C:/Users/${process.env.USERNAME ?? ''}/AppData/Local/Google/Chrome/Application/chrome.exe`,
].find(existsSync);
if (!CHROME) { console.error('chrome not found'); process.exit(2); }

const PORT = 9334;
const profile = join(process.env.TEMP ?? '.', 'ballpit-cdp-profile');
const child = spawn(CHROME, [
  `--remote-debugging-port=${PORT}`,
  `--user-data-dir=${profile}`,
  '--no-first-run', '--no-default-browser-check',
  '--enable-unsafe-webgpu',
  '--window-size=1600,900',
  // Without these, Chrome treats the window as occluded the moment anything
  // covers it and stops requestAnimationFrame entirely: zero frames, no error.
  '--disable-features=CalculateNativeWinOcclusion',
  '--disable-backgrounding-occluded-windows',
  '--disable-renderer-backgrounding',
  '--disable-background-timer-throttling',
  URL_,
], { stdio: 'ignore', detached: false });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const avg = (a) => a.reduce((s, v) => s + v, 0) / a.length;

// Always open our OWN tab. Never adopt one that was already there.
async function ownTab() {
  for (let i = 0; i < 40; i++) {
    try {
      const r = await fetch(`http://127.0.0.1:${PORT}/json/new?${encodeURIComponent(URL_)}`, { method: 'PUT' });
      if (r.ok) return await r.json();
    } catch {}
    await sleep(250);
  }
  throw new Error('could not open a CDP tab');
}

const page = await ownTab();
const ws = new WebSocket(page.webSocketDebuggerUrl);
const logs = [];
let id = 0;
await new Promise((r) => ws.addEventListener('open', r));

const pending = new Map();
ws.addEventListener('message', (ev) => {
  const msg = JSON.parse(ev.data);
  if (msg.method === 'Runtime.consoleAPICalled') {
    logs.push(`[${msg.params.type}] ${msg.params.args.map((a) => a.value ?? a.description ?? a.type).join(' ')}`);
  } else if (msg.method === 'Runtime.exceptionThrown') {
    const d = msg.params.exceptionDetails;
    logs.push(`[EXCEPTION] ${d.exception?.description ?? d.text}`);
  } else if (msg.method === 'Log.entryAdded') {
    logs.push(`[${msg.params.entry.level}] ${msg.params.entry.text}`);
  } else if (msg.id && pending.has(msg.id)) {
    pending.get(msg.id)(msg.result);
    pending.delete(msg.id);
  }
});

const call = (method, params = {}) => new Promise((res) => {
  const myId = ++id;
  pending.set(myId, res);
  ws.send(JSON.stringify({ id: myId, method, params }));
});

const evalJS = async (expr) => {
  const r = await call('Runtime.evaluate', { expression: expr, returnByValue: true, awaitPromise: true });
  if (r?.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description ?? 'eval failed');
  return r?.result?.value;
};

await call('Runtime.enable');
await call('Log.enable');
await call('Page.enable');
await call('Page.reload', { ignoreCache: true });
await sleep(3500);

const alive = await evalJS('typeof globalThis.__ballpit === "function"');
if (!alive) {
  console.error('no heartbeat: the sim never booted');
  for (const l of logs.slice(0, 30)) console.error(l);
  ws.close();
  try { child.kill(); } catch {}
  process.exit(1);
}

// What each mode has to look like once it has run. These are the assertions that
// separate "it renders" from "it is a fluid".
const CHECKS = {
  0: {
    name: 'DAM BREAK',
    // The column must collapse and run: a fluid crosses most of the floor and
    // its surface slopes down away from the start. A jammed solver leaves a
    // block standing at the left wall.
    test: (p) => {
      const spread = p.maxX / p.worldW;
      const left = avg(p.profile.slice(0, 6));
      const right = avg(p.profile.slice(-6));
      return [
        [spread > 0.85, `front reached ${(spread * 100).toFixed(0)}% of the floor (want >85%)`],
        [left > right, `surface slopes down to the right (${left.toFixed(1)} -> ${right.toFixed(1)})`],
        [p.maxY < p.worldH * 0.8, `settled below the ceiling (top ${p.maxY.toFixed(1)}/${p.worldH})`],
      ];
    },
  },
  1: {
    name: 'FALL',
    // A dropped slab must come to rest as a heap, and it must actually rest.
    //
    // "At rest" is measured as POSITION drift between two probes a second apart,
    // not as velocity. Velocity here is (p - prev) / h with h around a
    // thousandth of a second, so a jiggle of one hundredth of a radius reads as
    // half a unit per second. Judging rest by that number condemns a pile that
    // is visually stone still.
    settle: true,
    test: (p) => [
      [p.drift < p.radius * 0.5, `settled: mean drift ${p.drift.toFixed(4)} over 1.2s (want < ${(p.radius * 0.5).toFixed(4)}, half a radius)`],
      [p.maxY > p.radius * 10, `pile has height ${p.maxY.toFixed(1)} (want > ${(p.radius * 10).toFixed(1)}: not a pancake)`],
    ],
  },
  2: {
    name: 'RIVER',
    // Pushed sideways, the body has to bank up against the far wall rather than
    // grind to a halt in the middle.
    test: (p) => {
      const left = avg(p.profile.slice(0, 6));
      const right = avg(p.profile.slice(-6));
      return [
        [right > left, `banked up downstream (${left.toFixed(1)} -> ${right.toFixed(1)})`],
        [p.maxX > p.worldW * 0.9, `reached the far wall (${p.maxX.toFixed(1)}/${p.worldW})`],
      ];
    },
  },
  3: {
    name: 'SWIRL',
    // Under a rotating body force the whole mass must stay in motion. If it
    // stalls, contacts are eating the drive.
    test: (p) => [
      [p.movingPct > 70, `${p.movingPct}% in motion (want >70%)`],
      [p.meanSpeed > 2, `mean speed ${p.meanSpeed} (want >2)`],
    ],
  },
  4: {
    name: 'PEGS',
    // A timed pour through static obstacles. Two things can go wrong and both
    // are silent: the spawn schedule releases everyone at once, or particles
    // pass straight through the pegs.
    seconds: 12,
    test: (p) => [
      [p.awakePct > 20 && p.awakePct < 100, `pour is staged: ${p.awakePct}% released so far (want a partial release)`],
      [p.pegOverlaps === 0, `${p.pegOverlaps} particles inside an obstacle (want 0)`],
      [p.maxX > p.worldW * 0.5, `spread past the peg field (${p.maxX.toFixed(1)}/${p.worldW})`],
    ],
  },
  5: {
    name: 'SHOVE',
    // A mixed crowd driven sideways: one particle in four is five times as
    // heavy. If the inverse-mass split in the contact solve works, the heavy
    // minority ploughs to the front and holds it. This is the crowd-game
    // property — brutes reaching the objective ahead of runners — and it is the
    // only thing mass actually buys in a non-penetration solver.
    // NOTE the direction. The heavy phase TRAILS and sits LOWER, which is the
    // opposite of the obvious guess. At a free surface the phase that takes the
    // larger share of every correction is the one that gets squirted up and
    // over the top, so the LIGHT particles surf forward while the heavy ones
    // stay in the packed body. Mass buys you resistance to displacement, not a
    // battering ram.
    seconds: 12,
    test: (p) => [
      [p.densePct > 15 && p.densePct < 40, `mixed crowd: ${p.densePct}% heavy`],
      [p.denseMeanY < p.lightMeanY, `heavy phase rides lower (y ${p.denseMeanY} vs ${p.lightMeanY})`],
      [p.lightMeanX - p.denseMeanX > p.worldW * 0.01, `light phase surfs ahead by ${(p.lightMeanX - p.denseMeanX).toFixed(2)} (want > ${(p.worldW * 0.01).toFixed(2)}: mass is affecting the solve)`],
    ],
  },
};

function sparkline(profile, worldH) {
  const ramp = ' .:-=+*#%@';
  return profile
    .map((h) => ramp[Math.min(ramp.length - 1, Math.floor((h / worldH) * ramp.length))])
    .join('');
}

let failed = 0;
for (const key of Object.keys(CHECKS)) {
  const mode = Number(key);
  await evalJS(`__ballpitSet('mode', ${mode})`);
  await sleep((CHECKS[mode].seconds ?? SECONDS) * 1000);

  const h = JSON.parse(await evalJS('JSON.stringify(__ballpit())'));
  const p = JSON.parse(await evalJS('__ballpitProbe().then(JSON.stringify)'));
  if (CHECKS[mode].settle) p.drift = await evalJS('__ballpitDrift(1200)');

  console.log(`\n=== ${CHECKS[mode].name} ===`);
  console.log(`  ${h.count.toLocaleString()} particles  r=${h.radius}  fill=${h.fillPct}%  ${h.substeps}x${h.iterations} steps  compute=${h.computeMs}ms`);
  console.log(`  overlap=${h.compressionPct}%  speed-capped=${h.cappedPct}%  hash-dropped=${h.droppedPct}%`);
  console.log(`  surface |${sparkline(p.profile, p.worldH)}|  maxY=${p.maxY} maxX=${p.maxX} moving=${p.movingPct}%`);

  for (const [ok, msg] of CHECKS[mode].test(p)) {
    console.log(`  ${ok ? 'PASS' : 'FAIL'}  ${msg}`);
    if (!ok) failed++;
  }
  if (h.compressionPct > 12) {
    console.log(`  FAIL  overlap ${h.compressionPct}% of a diameter: the solver is not converging`);
    failed++;
  }

  const shot = await call('Page.captureScreenshot', { format: 'png' });
  const slug = CHECKS[mode].name.toLowerCase().replace(/ /g, '-');
  const png = Buffer.from(shot.data, 'base64');
  // Stable name for the README and the landing page, plus a dated copy that is
  // never overwritten. The archive is the visual history of the solver: it is
  // how you can see a regression as a picture instead of a number.
  writeFileSync(join(OUT, `${slug}.png`), png);
  writeFileSync(join(ARCHIVE, `${STAMP}-${slug}.png`), png);
  console.log(`  shot: ${join(OUT, `${slug}.png`)}`);
}

const errs = logs.filter((l) => /EXCEPTION|\[error\]|\[SEVERE\]/i.test(l));
console.log(`\n--- ${failed} failed checks, ${errs.length} console errors ---`);
for (const l of errs.slice(0, 20)) console.log(l);

ws.close();
// Only ever kill the throwaway profile's own process. Never a broad taskkill:
// the user's normal Chrome windows are not ours to close.
try { child.kill(); } catch {}
process.exit(failed || errs.length ? 1 : 0);
