// Ad-hoc probe: boot the pit, wait, then evaluate an arbitrary expression.
//
//   node tools/eval.mjs "<js>" [seconds] [url]

import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { join } from 'node:path';

const EXPR = process.argv[2] ?? '__ballpit()';
const SECONDS = Number(process.argv[3]) || 4;
const URL_ = process.argv[4] ?? 'http://localhost:8100/';

const CHROME = [
  'C:/Program Files/Google/Chrome/Application/chrome.exe',
  'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe',
  `C:/Users/${process.env.USERNAME ?? ''}/AppData/Local/Google/Chrome/Application/chrome.exe`,
].find(existsSync);

const PORT = 9335;
const child = spawn(CHROME, [
  `--remote-debugging-port=${PORT}`,
  `--user-data-dir=${join(process.env.TEMP ?? '.', 'ballpit-eval-profile')}`,
  '--no-first-run', '--no-default-browser-check', '--enable-unsafe-webgpu',
  '--window-size=1600,900',
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
  if (m.method === 'Runtime.exceptionThrown') console.log(`[EXCEPTION] ${m.params.exceptionDetails.exception?.description}`);
  if (m.method === 'Runtime.consoleAPICalled') console.log(`[${m.params.type}] ${m.params.args.map((a) => a.value ?? a.description).join(' ')}`);
  if (m.id && pending.has(m.id)) { pending.get(m.id)(m.result); pending.delete(m.id); }
});
const call = (method, params = {}) => new Promise((res) => {
  const myId = ++id;
  pending.set(myId, res);
  ws.send(JSON.stringify({ id: myId, method, params }));
});

await call('Runtime.enable');
await call('Page.enable');
await call('Page.reload', { ignoreCache: true });
await sleep(SECONDS * 1000);

const r = await call('Runtime.evaluate', {
  expression: `(async () => JSON.stringify(await (${EXPR}), null, 1))()`,
  returnByValue: true,
  awaitPromise: true,
});
console.log(r?.exceptionDetails ? `ERROR ${r.exceptionDetails.exception?.description}` : r?.result?.value);

ws.close();
try { child.kill(); } catch {}
process.exit(0);
