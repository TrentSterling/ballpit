// Renders og-image.png (1200x630) by photographing the LIVE sim: boots the
// pit with a healthy giant count, lets a dam break curl for a moment, pauses,
// hides the panel, injects the title card, and screenshots the result. Real
// physics as cover art; no compositing pipeline to drift out of date.
//
//   node serve.mjs 8100     (in another terminal)
//   node tools/og.mjs

import { spawn } from 'node:child_process';
import { writeFileSync, existsSync } from 'node:fs';
import { join } from 'node:path';

const URL_ = 'http://localhost:8100/?giants=400';
const OUT = join(import.meta.dirname, '..', 'og-image.png');
const WAVE_MS = 1300;          // how far into the collapse the shutter fires

const CHROME = [
  'C:/Program Files/Google/Chrome/Application/chrome.exe',
  'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe',
  `C:/Users/${process.env.USERNAME ?? ''}/AppData/Local/Google/Chrome/Application/chrome.exe`,
].find(existsSync);
if (!CHROME) { console.error('chrome not found'); process.exit(2); }

const PORT = 9336;
const child = spawn(CHROME, [
  `--remote-debugging-port=${PORT}`,
  `--user-data-dir=${join(process.env.TEMP ?? '.', 'ballpit-og-profile')}`,
  '--no-first-run', '--no-default-browser-check', '--enable-unsafe-webgpu',
  '--window-size=1280,760',
  '--window-position=-32000,-32000',
  '--disable-features=CalculateNativeWinOcclusion',
  '--disable-backgrounding-occluded-windows',
  '--disable-renderer-backgrounding',
  '--disable-background-timer-throttling',
  URL_,
], { stdio: 'ignore' });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

let page;
for (let i = 0; i < 40 && !page; i++) {
  try {
    const r = await fetch(`http://127.0.0.1:${PORT}/json/new?${encodeURIComponent(URL_)}`, { method: 'PUT' });
    if (r.ok) page = await r.json();
  } catch {}
  if (!page) await sleep(250);
}

const ws = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((r) => ws.addEventListener('open', r));
let id = 0;
const pending = new Map();
ws.addEventListener('message', (ev) => {
  const m = JSON.parse(ev.data);
  if (m.id && pending.has(m.id)) { pending.get(m.id)(m.result); pending.delete(m.id); }
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
await call('Page.enable');
await call('Emulation.setDeviceMetricsOverride', {
  width: 1200, height: 630, deviceScaleFactor: 1, mobile: false,
});
await call('Page.reload', { ignoreCache: true });

for (let i = 0; i < 40; i++) {
  if (await evalJS('typeof globalThis.__ballpit === "function"').catch(() => false)) break;
  await sleep(250);
}
// Warm up past the pipeline-compile hitches, or they eat the wave timer and
// the shutter fires on a column that has barely started to fall.
for (let i = 0; i < 40; i++) {
  if ((await evalJS('__ballpit().frames').catch(() => 0)) > 40) break;
  await sleep(250);
}

// Fresh dam break, shutter mid-curl, then dress the set.
await evalJS(`(async () => {
  __ballpitSet('mode', 0);
  await new Promise((r) => setTimeout(r, ${WAVE_MS}));
  __ballpitSet('paused', true);
  document.getElementById('panel').hidden = true;
  document.getElementById('help').hidden = true;
  const o = document.createElement('div');
  o.innerHTML = \`
    <div style="position:fixed;inset:0;z-index:50;background:linear-gradient(to top,
      rgba(8,12,17,0.94) 0%, rgba(8,12,17,0.5) 40%, rgba(8,12,17,0) 72%)"></div>
    <div style="position:fixed;left:60px;bottom:52px;z-index:51;font-family:Consolas,'DejaVu Sans Mono',monospace">
      <div style="color:#7ee0ff;font-size:92px;letter-spacing:20px;line-height:1">BALLPIT</div>
      <div style="color:#dce6f2;font-size:31px;margin-top:14px;letter-spacing:1px">GPU contact physics testbed</div>
      <div style="color:#7d90a6;font-size:24px;margin-top:14px">20,000 balls
        <span style="color:#7ee0ff;font-size:20px;vertical-align:2px">&#9679;</span> giants
        <span style="color:#f05ad2;font-size:20px;vertical-align:2px">&#9679;</span> WebGPU compute, right in the browser</div>
    </div>\`;
  document.body.appendChild(o);
  return true;
})()`);
await sleep(400);      // one settled render of the paused frame + overlay

const shot = await call('Page.captureScreenshot', {
  format: 'png',
  clip: { x: 0, y: 0, width: 1200, height: 630, scale: 1 },
});
writeFileSync(OUT, Buffer.from(shot.data, 'base64'));
console.log(`wrote ${OUT}`);

ws.close();
try { child.kill(); } catch {}
process.exit(0);
