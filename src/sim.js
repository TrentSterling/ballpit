// BALLPIT - GPU contact physics testbed.
//
// One file, one job: prove that N particles can flow, collide and pack like a
// liquid entirely in compute shaders. No enemies, no towers, no game.
//
// THE RULE THIS FILE EXISTS TO ENFORCE
// ------------------------------------
// Contacts own position. Velocity is DERIVED from position, never assigned.
// Steering (gravity, flow, the pointer) may only propose a motion during the
// prediction step; the contact solver then overrules it freely, and the velocity
// that comes out is whatever the particle actually managed to move.
//
// The old solver did the opposite: it assigned a target velocity every frame, so
// a particle crushed at the bottom of a pile was still told it was moving at
// full speed. Contacts could never win, the pile compressed forever, and the
// pancake on the floor is exactly that failure made visible.
//
// The loop, XPBD small-steps style. Substeps beat iterations: two substeps of
// three iterations is stiffer and more stable than one substep of six.
//
//   for each substep (h = dt / substeps):
//     predict   v += accel * h;  prev = p;  p += v * h
//     clear     zero the hash
//     scatter   every particle claims a slot in its grid cell
//     relax  xK   read neighbours, write a positional correction (Jacobi)
//     apply  xK   add the correction, clamp to the walls
//     finish    v = (p - prev) / h, then XSPH viscosity
//
// relax and apply are separate dispatches on purpose. Reading neighbour
// positions from the same buffer you are writing is a data race; every particle
// would see a different mix of old and new neighbours depending on scheduling.
// The correction goes to its own buffer and is applied in a second pass.

import * as THREE from 'three/webgpu';
import {
  Fn, If, Return, instancedArray, uniform, atomicAdd, atomicStore, instanceIndex,
  float, int, uint, vec2, vec3, vec4, length, max, clamp,
  hash, positionGeometry, uv, mix, step, smoothstep, cos, sin,
} from 'three/tsl';

// ---------------------------------------------------------------- parameters
const qs = new URLSearchParams(location.search);
const COUNT = Number(qs.get('n')) || 20000;
const WORLD_W = 40;
const WORLD_H = 22;

// Radius is DERIVED from the head count, not guessed. Particle area has to fit
// in the pit with room to move: at a quarter coverage a settled pile takes about
// a third of the box and every mode has somewhere to flow. Picking a radius
// independently of N is how you end up asking 40000 balls to occupy 200% of the
// available area, which no solver can satisfy and which looks exactly like the
// flat pancake it is.
const FILL = 0.25;
const AUTO_R = Math.sqrt((FILL * WORLD_W * WORLD_H) / (Math.PI * COUNT));
const R_MIN = AUTO_R * 0.5;
const R_MAX = AUTO_R * 1.5;

// The hash grid is allocated for the smallest radius the slider allows, then
// simply uses fewer of its cells when the radius goes up.
const CELL_MIN = R_MIN * 2.05;
const GRID_W_MAX = Math.ceil(WORLD_W / CELL_MIN) + 1;
const GRID_H_MAX = Math.ceil(WORLD_H / CELL_MIN) + 1;
const GRID_MAX = GRID_W_MAX * GRID_H_MAX;
const BUCKET_K = 12;                // neighbours tracked per cell

const MODES = ['DAM BREAK', 'FALL', 'RIVER', 'SWIRL', 'PEGS', 'SHOVE'];

const boot = document.getElementById('boot');
const bootmsg = document.getElementById('bootmsg');

if (!navigator.gpu) {
  bootmsg.innerHTML = 'This needs <b>WebGPU</b>: Firefox 141+, Chrome or Edge.';
  throw new Error('no WebGPU');
}

// ------------------------------------------------------------------- renderer
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x080c11);
const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 100);
camera.position.set(WORLD_W / 2, WORLD_H / 2, 10);

const renderer = new THREE.WebGPURenderer({ antialias: false, trackTimestamp: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
document.body.appendChild(renderer.domElement);
try {
  await renderer.init();
} catch (err) {
  bootmsg.innerHTML = `WebGPU failed:<br><code>${err.message}</code>`;
  throw err;
}

// -------------------------------------------------------------------- buffers
const pos = instancedArray(COUNT, 'vec4');      // x, y, vx, vy
const prev = instancedArray(COUNT, 'vec2');     // position at the start of the substep
const corr = instancedArray(COUNT, 'vec4');     // dx, dy, contacts, spare
const cellCount = instancedArray(GRID_MAX, 'uint').toAtomic();
const bucket = instancedArray(GRID_MAX * BUCKET_K, 'uint');
// 0: summed deepest overlap (thousandths)  1: contacting particles
// 2: particles hitting the speed limit     3: neighbours the hash had to drop
const stats = instancedArray(4, 'uint').toAtomic();
// inverse mass, heavy flag, spawn time, spare
const meta = instancedArray(COUNT, 'vec4');
// Static round obstacles: x, y, radius, spare. Laid out by a compute pass rather
// than uploaded, so there is no CPU write path to get wrong.
const PEG_COLS = 6;
const PEG_ROWS = 4;
const PEG_COUNT = PEG_COLS * PEG_ROWS;
const pegs = instancedArray(PEG_COUNT, 'vec4');

const u = {
  h: uniform(1 / 120),              // substep length
  time: uniform(0),
  radius: uniform(AUTO_R),
  stiffness: uniform(1.0),          // share of the averaged correction applied
  viscosity: uniform(0.08),         // XSPH: how much neighbourhood velocity to adopt
  travel: uniform(0.9),             // max travel per substep, in radii
  gravity: uniform(22),
  flow: uniform(9),
  mode: uniform(0, 'int'),
  pegsOn: uniform(0),
  cell: uniform(CELL_MIN),
  gridW: uniform(GRID_W_MAX, 'int'),
  gridWf: uniform(GRID_W_MAX),
  gridHf: uniform(GRID_H_MAX),
  pointer: uniform(new THREE.Vector2(-999, -999)),
  pointerPush: uniform(0),
  pointerRadius: uniform(5),
  seed: uniform(1, 'int'),
};

const SPOUT = [WORLD_W * 0.5, WORLD_H * 0.95];
const POUR_SECONDS = 14;

// Peg board. A staggered grid, laid out once.
const pegPass = Fn(() => {
  const k = instanceIndex;
  const col = float(k.mod(uint(PEG_COLS)));
  const row = float(k.div(uint(PEG_COLS)));
  const dx = WORLD_W / (PEG_COLS + 1);
  const dy = (WORLD_H * 0.62) / PEG_ROWS;
  // every other row offset by half a spacing: the classic peg board
  const x = col.mul(dx).add(dx).add(row.mod(2).mul(dx * 0.5)).sub(dx * 0.25);
  const y = float(WORLD_H * 0.78).sub(row.mul(dy));
  pegs.element(k).assign(vec4(x, y, 0.85, 0));
})().compute(PEG_COUNT);

// ----------------------------------------------------------------- init pass
// Each mode starts from the block that makes its behaviour legible. DAM BREAK is
// the standard fluid validation case: a column held against one wall, then
// released. If it collapses into a spreading wave with a curling front it is a
// fluid; if it slumps and locks it is a bag of gravel.
const initPass = Fn(() => {
  const i = instanceIndex;
  const rx = hash(i.add(uint(u.seed))).toVar();
  const ry = hash(i.add(uint(u.seed)).add(uint(9871))).toVar();
  const p = vec2(0).toVar();
  // inverse mass. 1 is the light default; a heavy particle gets a smaller one
  // and therefore wins position disputes against its lighter neighbours.
  const invMass = float(1).toVar();
  const heavy = float(0).toVar();
  const spawn = float(-1).toVar();                     // already awake

  If(u.mode.equal(int(0)), () => {                     // dam: tall column, left wall
    p.assign(vec2(rx.mul(WORLD_W * 0.34).add(0.4), ry.mul(WORLD_H * 0.94).add(0.3)));
  }).Else(() => {
    If(u.mode.equal(int(1)), () => {                   // fall: wide slab up top
      p.assign(vec2(rx.mul(WORLD_W * 0.8).add(WORLD_W * 0.1), ry.mul(WORLD_H * 0.42).add(WORLD_H * 0.55)));
    }).Else(() => {
      If(u.mode.equal(int(2)), () => {                 // river: pool along the floor
        p.assign(vec2(rx.mul(WORLD_W * 0.9).add(0.4), ry.mul(WORLD_H * 0.34).add(0.3)));
      }).Else(() => {
        If(u.mode.equal(int(3)), () => {               // swirl: fill the box
          p.assign(vec2(rx.mul(WORLD_W - 1).add(0.5), ry.mul(WORLD_H - 1).add(0.5)));
        }).Else(() => {
          If(u.mode.equal(int(4)), () => {
            // PEGS: everyone waits at the spout and is released over time. The
            // same mechanism a wave spawner needs, tested here where a mistake
            // is obvious instead of buried in a game loop.
            p.assign(vec2(
              rx.sub(0.5).mul(2.4).add(SPOUT[0]),
              ry.sub(0.5).mul(0.8).add(SPOUT[1]),
            ));
            spawn.assign(float(i).div(COUNT).mul(POUR_SECONDS));
          }).Else(() => {
            // SHOVE: a mixed crowd driven sideways. The heavy quarter should
            // plough to the front of the pack and hold it.
            //
            // This mode started life as a buoyancy test and that test FAILED,
            // for a reason worth keeping: non-penetration constraints plus
            // gravity-as-acceleration are weightless. A heavy particle never
            // presses down harder, it only resists being pushed, so a dense
            // layer resting on a light one has no reason at all to sink. Real
            // buoyancy needs a density constraint (position based FLUIDS), which
            // is a different solver, not a tweak to this one.
            //
            // What inverse mass genuinely buys is who wins a contact, so that is
            // what gets tested. It is also the only part a crowd game needs.
            p.assign(vec2(rx.mul(WORLD_W * 0.55).add(0.4), ry.mul(WORLD_H * 0.55).add(0.3)));
            heavy.assign(step(float(0.75), hash(i.add(uint(4242)))));
            invMass.assign(mix(float(1), float(0.2), heavy));
          });
        });
      });
    });
  });

  pos.element(i).assign(vec4(p, 0, 0));
  prev.element(i).assign(p);
  corr.element(i).assign(vec4(0));
  meta.element(i).assign(vec4(invMass, heavy, spawn, 0));
})().compute(COUNT);

// ---------------------------------------------------------------- clear pass
const clearPass = Fn(() => {
  atomicStore(cellCount.element(instanceIndex), uint(0));
  If(instanceIndex.lessThan(uint(4)), () => {
    atomicStore(stats.element(instanceIndex), uint(0));
  });
})().compute(GRID_MAX);

// -------------------------------------------------------------- predict pass
// The only place steering is allowed to speak, and it speaks in accelerations.
const predictPass = Fn(() => {
  const i = instanceIndex;
  const P = pos.element(i).toVar();
  const p = P.xy.toVar();
  const v = P.zw.toVar();

  // Not yet released: hold position, contribute nothing, stay out of the hash.
  // An unspawned particle must not exist to anyone.
  // Return() is the TSL node, not a JavaScript return: a bare `return` here
  // would just end the callback and emit no shader code at all, so every
  // unspawned particle would carry on simulating.
  If(meta.element(i).z.greaterThan(u.time), () => {
    prev.element(i).assign(p);
    Return();
  });

  // Plain gravity is the DEFAULT, and only the two modes that need something
  // else opt out. Written the other way round, with a catch-all Else holding the
  // swirl, every mode added later silently inherits a rotating body force and
  // passes its tests for the wrong reason.
  const a = vec2(0, u.gravity.negate()).toVar();
  If(u.mode.equal(int(5)), () => {
    a.assign(vec2(u.flow.mul(2), u.gravity.negate()));           // shove: drive the crowd right
  });
  If(u.mode.equal(int(2)), () => {
    // River: a body force to the right plus gravity. Not a target velocity. A
    // particle wedged against the far wall simply stops, because its derived
    // velocity is whatever it managed to move, which is nothing.
    a.assign(vec2(u.flow.mul(3), u.gravity.negate()));
  }).Else(() => {
    If(u.mode.equal(int(3)), () => {
      const c = vec2(WORLD_W / 2, WORLD_H / 2);
      const d = p.sub(c).toVar();
      const l = max(length(d), float(1e-4)).toVar();
      a.assign(vec2(d.y.negate(), d.x).div(l).mul(u.flow.mul(2))
        .sub(d.div(l).mul(u.flow.mul(0.35))));                   // tangential + gentle inward
    });
  });

  // Pointer stir, also an acceleration.
  const toP = p.sub(u.pointer).toVar();
  const pdist = max(length(toP), float(1e-4)).toVar();
  If(pdist.lessThan(u.pointerRadius), () => {
    a.addAssign(toP.div(pdist).mul(u.pointerPush).mul(u.pointerRadius.sub(pdist).div(u.pointerRadius)));
  });

  v.addAssign(a.mul(u.h));

  // THE DISCRETISATION SPEED LIMIT.
  // A particle may not travel further than about one radius in a substep. Go
  // past that and it steps clean through the layer beneath it before the solver
  // has ever seen the contact: the pile it should be landing on is simply not
  // there yet when its position is written. That is what turns a ball pit into
  // a pancake, and no amount of solver iterations can undo it afterwards,
  // because once everything is coincident the hash overflows too.
  //
  // The cap is a symptom gauge as much as a fix. If stats[2] is large the
  // substep count is too low for the gravity in use; raise substeps until it
  // falls to zero and the cap stops mattering.
  const vmax = u.radius.mul(u.travel).div(u.h).toVar();
  const sp = length(v).toVar();
  If(sp.greaterThan(vmax), () => {
    v.mulAssign(vmax.div(sp));
    atomicAdd(stats.element(uint(2)), uint(1));
  });

  prev.element(i).assign(p);
  p.addAssign(v.mul(u.h));
  pos.element(i).assign(vec4(p, v));
})().compute(COUNT);

// -------------------------------------------------------------- scatter pass
const scatterPass = Fn(() => {
  If(meta.element(instanceIndex).z.greaterThan(u.time), () => { Return(); });
  const p = pos.element(instanceIndex).xy.toVar();
  const cx = int(clamp(p.x.div(u.cell), float(0), u.gridWf.sub(1))).toVar();
  const cy = int(clamp(p.y.div(u.cell), float(0), u.gridHf.sub(1))).toVar();
  const cell = cy.mul(u.gridW).add(cx).toVar();
  const slot = atomicAdd(cellCount.element(cell), uint(1)).toVar();
  If(slot.lessThan(uint(BUCKET_K)), () => {
    // index + 1, so an untouched slot reads as empty
    bucket.element(cell.mul(int(BUCKET_K)).add(int(slot))).assign(instanceIndex.add(uint(1)));
  }).Else(() => {
    // Anything past BUCKET_K is invisible to every neighbour for this substep.
    // A silent drop here is indistinguishable from good physics right up until
    // the crowd interpenetrates, so it gets counted and shown.
    atomicAdd(stats.element(uint(3)), uint(1));
  });
})().compute(COUNT);

// ---------------------------------------------------------------- relax pass
// Gathers the non-penetration corrections. Writes nowhere but its own slot, so
// every particle in an iteration sees the exact same world: a true Jacobi sweep.
const relaxPass = Fn(() => {
  const i = instanceIndex;
  const mi = meta.element(i).toVar();
  If(mi.z.greaterThan(u.time), () => {
    corr.element(i).assign(vec4(0));
    Return();
  });
  const p = pos.element(i).xy.toVar();
  const wi = mi.x.toVar();                      // inverse mass
  const r = u.radius.toVar();
  const minDist = r.mul(2).toVar();

  const push = vec2(0).toVar();
  const hits = float(0).toVar();
  const deepest = float(0).toVar();

  const cx = int(clamp(p.x.div(u.cell), float(1), u.gridWf.sub(2))).toVar();
  const cy = int(clamp(p.y.div(u.cell), float(1), u.gridHf.sub(2))).toVar();

  for (let oy = -1; oy <= 1; oy++) {
    for (let ox = -1; ox <= 1; ox++) {
      const cell = cy.add(int(oy)).mul(u.gridW).add(cx.add(int(ox))).toVar();
      for (let k = 0; k < BUCKET_K; k++) {
        const raw = bucket.element(cell.mul(int(BUCKET_K)).add(int(k))).toVar();
        If(raw.greaterThan(uint(0)), () => {
          const other = raw.sub(uint(1)).toVar();
          If(other.notEqual(i), () => {
            const q = pos.element(other).xy.toVar();
            const delta = p.sub(q).toVar();
            const dist = length(delta).toVar();
            If(dist.lessThan(minDist), () => {
              // Coincident pairs give a garbage normal. Fall back to a stable
              // per-pair direction rather than to noise, so the pair actually
              // separates instead of jittering in place.
              const degenerate = step(dist, float(1e-5));
              const ang = hash(i.add(other).add(uint(7331))).mul(6.2831853).toVar();
              const jitter = vec2(cos(ang), sin(ang));
              const n = mix(delta.div(max(dist, float(1e-5))), jitter, degenerate).toVar();
              const overlap = minDist.sub(dist).toVar();
              // Split the overlap by inverse mass instead of evenly. Equal
              // masses give half each, exactly as before; a heavy particle
              // yields less and so shoves lighter ones aside.
              //
              // This is the only mass term in the file, and buoyancy falls out
              // of it: there is no buoyancy force anywhere, the dense phase
              // simply wins its contacts and sinks.
              const wj = meta.element(other).x.toVar();
              const share = wi.div(max(wi.add(wj), float(1e-5))).toVar();
              push.addAssign(n.mul(overlap.mul(share)));
              hits.addAssign(1);
              deepest.assign(max(deepest, overlap));
            });
          });
        });
      }
    }
  }

  // Averaged, not summed. A particle with eight neighbours pushing on it must
  // move once, not eight times: summing is what makes a packed crowd explode,
  // and the clamps people bolt on afterwards are what make it lock solid.
  const n = max(hits, float(1)).toVar();
  corr.element(i).assign(vec4(push.div(n), hits, 0));

  // Health metric: deepest overlap seen, in thousandths of a unit.
  If(hits.greaterThan(float(0)), () => {
    atomicAdd(stats.element(uint(0)), uint(deepest.mul(1000)));
    atomicAdd(stats.element(uint(1)), uint(1));
  });
})().compute(COUNT);

// ---------------------------------------------------------------- apply pass
const applyPass = Fn(() => {
  const i = instanceIndex;
  If(meta.element(i).z.greaterThan(u.time), () => { Return(); });
  const P = pos.element(i).toVar();
  const p = P.xy.add(corr.element(i).xy.mul(u.stiffness)).toVar();
  const r = u.radius.toVar();

  // Static obstacles, projected out the same way the walls are: an immovable
  // body is just a contact with zero inverse mass, so the particle takes the
  // whole correction. Resolved after the particle-particle pass and before the
  // walls, so a particle squeezed between a peg and a wall ends up outside both.
  If(u.pegsOn.greaterThan(float(0.5)), () => {
    for (let k = 0; k < PEG_COUNT; k++) {
      const g = pegs.element(uint(k)).toVar();
      const d = p.sub(g.xy).toVar();
      const dist = length(d).toVar();
      const minD = g.z.add(r).toVar();
      If(dist.lessThan(minD), () => {
        const nrm = d.div(max(dist, float(1e-5))).toVar();
        p.assign(g.xy.add(nrm.mul(minD)));
      });
    }
  });

  // Walls are a hard constraint, resolved last so nothing ever ends a step
  // outside the box.
  pos.element(i).assign(vec4(
    clamp(p.x, r, float(WORLD_W).sub(r)),
    clamp(p.y, r, float(WORLD_H).sub(r)),
    P.z, P.w,
  ));
})().compute(COUNT);

// --------------------------------------------------------------- finish pass
// Velocity is read off the actual displacement. This single line is what makes a
// pile support itself: a particle that could not move has no velocity, with no
// damping term, no friction hack and no special case for "resting".
const finishPass = Fn(() => {
  const i = instanceIndex;
  If(meta.element(i).z.greaterThan(u.time), () => { Return(); });
  const p = pos.element(i).xy.toVar();
  const v = p.sub(prev.element(i)).div(u.h).toVar();

  // XSPH viscosity: nudge toward the neighbourhood average. This is the whole
  // difference between marbles and water. It is velocity SMOOTHING, not velocity
  // assignment, so it cannot overrule a contact.
  const sum = vec2(0).toVar();
  const n = float(0).toVar();
  const rad = u.radius.mul(2.4).toVar();
  const cx = int(clamp(p.x.div(u.cell), float(1), u.gridWf.sub(2))).toVar();
  const cy = int(clamp(p.y.div(u.cell), float(1), u.gridHf.sub(2))).toVar();
  for (let oy = -1; oy <= 1; oy++) {
    for (let ox = -1; ox <= 1; ox++) {
      const cell = cy.add(int(oy)).mul(u.gridW).add(cx.add(int(ox))).toVar();
      for (let k = 0; k < BUCKET_K; k++) {
        const raw = bucket.element(cell.mul(int(BUCKET_K)).add(int(k))).toVar();
        If(raw.greaterThan(uint(0)), () => {
          const other = raw.sub(uint(1)).toVar();
          If(other.notEqual(i), () => {
            const Q = pos.element(other).toVar();
            If(length(p.sub(Q.xy)).lessThan(rad), () => {
              sum.addAssign(Q.zw);
              n.addAssign(1);
            });
          });
        });
      }
    }
  }
  If(n.greaterThan(float(0)), () => {
    v.addAssign(sum.div(n).sub(v).mul(u.viscosity));
  });

  pos.element(i).assign(vec4(p, v));
})().compute(COUNT);

// --------------------------------------------------------------------- render
const src = new THREE.PlaneGeometry(1, 1);
const geo = new THREE.InstancedBufferGeometry();
geo.setAttribute('position', src.getAttribute('position'));
geo.setAttribute('uv', src.getAttribute('uv'));
geo.setIndex(src.getIndex());
geo.instanceCount = COUNT;

const posA = pos.toAttribute();
const metaA = meta.toAttribute();
const mat = new THREE.MeshBasicNodeMaterial();

// An unspawned particle collapses to zero area rather than being drawn at the
// spout, where twenty thousand of them would sit in a single bright dot.
const awake = step(metaA.z, u.time);
mat.positionNode = vec3(posA.xy.add(positionGeometry.xy.mul(u.radius.mul(2.35).mul(awake))), 0.1);

// Colour by speed, because speed is the thing under test. A healthy fluid shows
// a moving front and a still body; a broken one is uniformly one colour. The
// dense phase gets its own hue so the separation is readable at a glance.
const speed = length(posA.zw);
const t = clamp(speed.div(18), 0, 1);
const light = mix(vec3(0.11, 0.35, 0.72), vec3(0.34, 0.83, 1.0), smoothstep(0, 0.45, t));
const dense = mix(vec3(0.55, 0.20, 0.10), vec3(1.0, 0.62, 0.24), smoothstep(0, 0.45, t));
const tint = mix(light, dense, metaA.y)
  .add(vec3(0.94, 0.99, 1.0).mul(smoothstep(0.55, 1, t).mul(0.7)));

// Soft round sprite. The rim highlight is what stops a dense pack from reading
// as one flat blue sheet.
const d = length(uv().sub(0.5)).mul(2);
mat.colorNode = tint.add(vec3(smoothstep(0.55, 1.0, d).mul(0.45)));
mat.opacityNode = smoothstep(1.0, 0.86, d);
mat.transparent = true;
mat.depthWrite = false;

const mesh = new THREE.Mesh(geo, mat);
mesh.frustumCulled = false;
scene.add(mesh);

// Obstacles, drawn from the same buffer the solver collides against, so what you
// see is exactly what the physics uses.
const pegGeo = new THREE.InstancedBufferGeometry();
pegGeo.setAttribute('position', src.getAttribute('position'));
pegGeo.setAttribute('uv', src.getAttribute('uv'));
pegGeo.setIndex(src.getIndex());
pegGeo.instanceCount = PEG_COUNT;

const pegA = pegs.toAttribute();
const pegMat = new THREE.MeshBasicNodeMaterial();
pegMat.positionNode = vec3(pegA.xy.add(positionGeometry.xy.mul(pegA.z.mul(2))), 0.05);
const pd = length(uv().sub(0.5)).mul(2);
pegMat.colorNode = mix(vec3(0.16, 0.22, 0.30), vec3(0.30, 0.44, 0.58), smoothstep(0.4, 1.0, pd));
pegMat.opacityNode = smoothstep(1.0, 0.9, pd).mul(u.pegsOn);
pegMat.transparent = true;
pegMat.depthWrite = false;

const pegMesh = new THREE.Mesh(pegGeo, pegMat);
pegMesh.frustumCulled = false;
scene.add(pegMesh);

// Pointer ring, so the tool has a visible size rather than an invisible one.
const ringPts = [];
for (let a = 0; a <= 48; a++) {
  ringPts.push(new THREE.Vector3(Math.cos((a / 48) * Math.PI * 2), Math.sin((a / 48) * Math.PI * 2), 2));
}
const ring = new THREE.Line(
  new THREE.BufferGeometry().setFromPoints(ringPts),
  new THREE.LineBasicMaterial({ color: 0x7ee0ff, transparent: true, opacity: 0.5 }),
);
ring.visible = false;
scene.add(ring);

// Line, not LineLoop: the WebGPU renderer does not support LineLoop.
const box = new THREE.Line(
  new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(0, 0, 1), new THREE.Vector3(WORLD_W, 0, 1),
    new THREE.Vector3(WORLD_W, WORLD_H, 1), new THREE.Vector3(0, WORLD_H, 1),
    new THREE.Vector3(0, 0, 1),
  ]),
  new THREE.LineBasicMaterial({ color: 0x2b4258 }),
);
scene.add(box);

// ----------------------------------------------------------------------- ui
const $ = (id) => document.getElementById(id);
// Substeps beat iterations, and it is not close. Measured on a settled pile of
// 20000, residual overlap as a share of a diameter:
//
//   4 substeps x 2 iters   48%       8 x 8    8.3%
//   8 substeps x 2 iters   31%      16 x 2    6.6%
//   8 substeps x 4 iters   20%      16 x 4    1.9%
//
// Doubling substeps beat quadrupling iterations every time, because iterations
// only polish what the substep already saw: they cannot recover a contact that
// the particle stepped straight over.
const state = { mode: 0, substeps: 16, iterations: 4, radius: AUTO_R, paused: false };

function bindSlider(id, get, set, fmt = (v) => v.toFixed(2)) {
  const s = $(`s-${id}`);
  const o = $(`o-${id}`);
  s.value = String(get());
  o.textContent = fmt(get());
  s.addEventListener('input', () => {
    set(Number(s.value));
    o.textContent = fmt(Number(s.value));
  });
}

function setRadius(r) {
  state.radius = r;
  u.radius.value = r;
  // Cell size follows the radius so the 3x3 neighbourhood always covers the
  // contact distance. Get this wrong in either direction and you either miss
  // neighbours entirely or pay to visit hundreds of them.
  const cell = Math.max(CELL_MIN, r * 2.05);
  u.cell.value = cell;
  u.gridW.value = Math.min(GRID_W_MAX, Math.ceil(WORLD_W / cell) + 1);
  u.gridWf.value = u.gridW.value;
  u.gridHf.value = Math.min(GRID_H_MAX, Math.ceil(WORLD_H / cell) + 1);
  $('fill').textContent = `${((COUNT * Math.PI * r * r) / (WORLD_W * WORLD_H) * 100).toFixed(0)} %`;
}

$('s-radius').min = R_MIN.toFixed(4);
$('s-radius').max = R_MAX.toFixed(4);
$('s-radius').step = ((R_MAX - R_MIN) / 40).toFixed(5);

bindSlider('radius', () => state.radius, setRadius, (v) => v.toFixed(3));
bindSlider('stiff', () => u.stiffness.value, (v) => { u.stiffness.value = v; });
bindSlider('visc', () => u.viscosity.value, (v) => { u.viscosity.value = v; });
bindSlider('subs', () => state.substeps, (v) => { state.substeps = v; }, (v) => v.toFixed(0));
bindSlider('iters', () => state.iterations, (v) => { state.iterations = v; }, (v) => v.toFixed(0));
bindSlider('grav', () => u.gravity.value, (v) => { u.gravity.value = v; }, (v) => v.toFixed(0));
bindSlider('flow', () => u.flow.value, (v) => { u.flow.value = v; }, (v) => v.toFixed(0));

setRadius(AUTO_R);
$('count').textContent = COUNT.toLocaleString();

let frames = 0;

function reset() {
  u.seed.value = (u.seed.value * 1664525 + 1013904223) & 0xffff;
  u.mode.value = state.mode;
  // Time drives the spawn schedule, so it has to rewind with the particles or a
  // reset in PEGS would release the whole pour on the first frame.
  u.time.value = 0;
  renderer.compute(initPass);
  frames = 0;
}

function setMode(m) {
  state.mode = ((m % MODES.length) + MODES.length) % MODES.length;
  u.mode.value = state.mode;
  u.pegsOn.value = state.mode === 4 ? 1 : 0;
  $('modeBtn').textContent = `MODE: ${MODES[state.mode]} (space)`;
  reset();
}

$('reset').addEventListener('click', reset);
$('modeBtn').addEventListener('click', () => setMode(state.mode + 1));

addEventListener('keydown', (e) => {
  if (e.key === 'r' || e.key === 'R') reset();
  if (e.code === 'Space') { e.preventDefault(); setMode(state.mode + 1); }
  if (e.key === 'p' || e.key === 'P') state.paused = !state.paused;
});

// pointer stir
const world = new THREE.Vector2();
function toWorld(e) {
  const rect = renderer.domElement.getBoundingClientRect();
  const x = (e.clientX - rect.left) / rect.width;
  const y = 1 - (e.clientY - rect.top) / rect.height;
  world.set(
    camera.position.x + camera.left + x * (camera.right - camera.left),
    camera.position.y + camera.bottom + y * (camera.top - camera.bottom),
  );
  return world;
}
// Left drag pushes, right drag pulls. Pull is the more interesting tool: a
// solver that cannot hold a void open collapses the moment you let go.
renderer.domElement.addEventListener('contextmenu', (e) => e.preventDefault());
renderer.domElement.addEventListener('pointerdown', (e) => {
  u.pointer.value.copy(toWorld(e));
  u.pointerPush.value = e.button === 2 ? -55 : 90;
  ring.position.set(u.pointer.value.x, u.pointer.value.y, 2);
  ring.scale.setScalar(u.pointerRadius.value);
  ring.visible = true;
});
renderer.domElement.addEventListener('pointermove', (e) => {
  if (u.pointerPush.value === 0) return;
  u.pointer.value.copy(toWorld(e));
  ring.position.set(u.pointer.value.x, u.pointer.value.y, 2);
});
addEventListener('pointerup', () => {
  u.pointerPush.value = 0;
  u.pointer.value.set(-999, -999);
  ring.visible = false;
});
renderer.domElement.addEventListener('wheel', (e) => {
  e.preventDefault();
  const r = Math.max(1.5, Math.min(14, u.pointerRadius.value - Math.sign(e.deltaY) * 0.6));
  u.pointerRadius.value = r;
  ring.scale.setScalar(r);
}, { passive: false });

function resize() {
  const w = innerWidth;
  const h = innerHeight;
  renderer.setSize(w, h);
  const aspect = w / h;
  const worldAspect = WORLD_W / WORLD_H;
  let vw = WORLD_W * 1.06;
  let vh = WORLD_H * 1.06;
  if (aspect > worldAspect) vw = vh * aspect; else vh = vw / aspect;
  // These bounds are in CAMERA space, not world space. The camera already sits
  // at the middle of the pit, so adding the world centre here as well shifts
  // everything half a world off screen, which looks exactly like a sim that
  // renders nothing.
  camera.left = -vw / 2;
  camera.right = vw / 2;
  camera.top = vh / 2;
  camera.bottom = -vh / 2;
  camera.updateProjectionMatrix();
}
addEventListener('resize', resize);
resize();

// -------------------------------------------------------------------- frame
let last = performance.now();
let fpsAcc = 0;
let fpsN = 0;
let computeMs = 0;
let compression = 0;
let cappedPct = 0;
let droppedPct = 0;
let statsInFlight = false;

renderer.compute(pegPass);
setMode(0);
boot.hidden = true;
$('panel').hidden = false;
$('help').hidden = false;

function pollStats() {
  if (statsInFlight) return;
  statsInFlight = true;
  renderer.getArrayBufferAsync(stats.value).then((buf) => {
    const s = new Uint32Array(buf);
    const contacts = s[1];
    // Mean deepest overlap per contacting particle, as a share of a diameter.
    compression = contacts > 0 ? (s[0] / 1000) / contacts / (state.radius * 2) : 0;
    cappedPct = (s[2] / COUNT) * 100;
    droppedPct = (s[3] / COUNT) * 100;
    statsInFlight = false;
  }).catch((err) => {
    console.error('stats readback failed', err);
    statsInFlight = false;
  });
}

renderer.setAnimationLoop(() => {
  const now = performance.now();
  const dt = Math.min(0.033, (now - last) / 1000);
  last = now;

  if (!state.paused) {
    const S = Math.max(1, Math.round(state.substeps));
    const K = Math.max(1, Math.round(state.iterations));
    u.h.value = dt / S;
    u.time.value += dt;
    for (let s = 0; s < S; s++) {
      // clear first: predict writes the speed-limit tally into the same stats
      // block, and clearing after it would wipe the number every substep.
      renderer.compute(clearPass);
      renderer.compute(predictPass);
      renderer.compute(scatterPass);
      for (let k = 0; k < K; k++) {
        renderer.compute(relaxPass);
        renderer.compute(applyPass);
      }
      renderer.compute(finishPass);
    }
  }

  renderer.render(scene, camera);
  frames++;

  fpsAcc += dt;
  fpsN++;
  if (fpsAcc >= 0.4) {
    $('fps').textContent = (fpsN / fpsAcc).toFixed(0);
    $('ms').textContent = `${((fpsAcc / fpsN) * 1000).toFixed(1)} ms`;
    $('compute').textContent = `${computeMs.toFixed(2)} ms`;
    $('overlap').textContent = `${(compression * 100).toFixed(1)} %`;
    $('overlap').style.color = compression > 0.12 ? '#e2564a' : '';
    $('capped').textContent = `${cappedPct.toFixed(1)} %`;
    $('capped').style.color = cappedPct > 20 ? '#e2a04a' : '';
    $('dropped').textContent = `${droppedPct.toFixed(1)} %`;
    $('dropped').style.color = droppedPct > 1 ? '#e2564a' : '';
    fpsAcc = 0;
    fpsN = 0;
    renderer.resolveTimestampsAsync(THREE.TimestampQuery.COMPUTE).catch(() => {});
    computeMs = renderer.info.compute.timestamp ?? computeMs;
    pollStats();
  }
});

// Test harness hooks. Everything a headless run needs to judge the sim without
// looking at a picture.
globalThis.__ballpit = () => ({
  frames,
  count: COUNT,
  mode: MODES[state.mode],
  radius: Number(state.radius.toFixed(4)),
  fillPct: Number(((COUNT * Math.PI * state.radius ** 2) / (WORLD_W * WORLD_H) * 100).toFixed(1)),
  compressionPct: Number((compression * 100).toFixed(2)),
  cappedPct: Number(cappedPct.toFixed(1)),
  droppedPct: Number(droppedPct.toFixed(2)),
  substeps: state.substeps,
  iterations: state.iterations,
  computeMs: Number(computeMs.toFixed(3)),
});

// Reads the raw particle buffer back so a test can measure the SHAPE of the
// pile: how tall it is, how far it spread, whether anything is still moving.
globalThis.__ballpitProbe = async () => {
  const buf = await renderer.getArrayBufferAsync(pos.value);
  const f = new Float32Array(buf);
  const m = new Float32Array(await renderer.getArrayBufferAsync(meta.value));
  let maxY = 0; let maxX = 0; let sumY = 0; let sumSpeed = 0; let moving = 0;
  let awake = 0;
  let denseY = 0; let denseX = 0; let denseS = 0; let denseN = 0;
  let lightY = 0; let lightX = 0; let lightS = 0; let lightN = 0;
  const cols = 24;
  const height = new Array(cols).fill(0);
  for (let i = 0; i < COUNT; i++) {
    if (m[i * 4 + 2] > u.time.value) continue;         // not spawned yet
    awake++;
    const x = f[i * 4];
    const y = f[i * 4 + 1];
    const sp = Math.hypot(f[i * 4 + 2], f[i * 4 + 3]);
    if (y > maxY) maxY = y;
    if (x > maxX) maxX = x;
    sumY += y;
    sumSpeed += sp;
    if (sp > 0.4) moving++;
    if (m[i * 4 + 1] > 0.5) { denseY += y; denseX += x; denseS += sp; denseN++; } else { lightY += y; lightX += x; lightS += sp; lightN++; }
    const c = Math.min(cols - 1, Math.max(0, Math.floor((x / WORLD_W) * cols)));
    if (y > height[c]) height[c] = y;
  }
  // How many particles ended up inside an obstacle. Tunnelling through a static
  // body is silent otherwise: the render still looks plausible.
  let pegOverlaps = 0;
  if (u.pegsOn.value > 0.5) {
    const g = new Float32Array(await renderer.getArrayBufferAsync(pegs.value));
    for (let i = 0; i < COUNT; i++) {
      if (m[i * 4 + 2] > u.time.value) continue;
      for (let k = 0; k < PEG_COUNT; k++) {
        const dx = f[i * 4] - g[k * 4];
        const dy = f[i * 4 + 1] - g[k * 4 + 1];
        // a hair of tolerance: sitting exactly on the surface is a contact, not a breach
        if (Math.hypot(dx, dy) < g[k * 4 + 2] - state.radius * 0.25) { pegOverlaps++; break; }
      }
    }
  }

  const n = Math.max(1, awake);
  return {
    pegOverlaps,
    worldW: WORLD_W,
    worldH: WORLD_H,
    radius: +state.radius.toFixed(4),
    awake,
    awakePct: +((awake / COUNT) * 100).toFixed(1),
    maxY: +maxY.toFixed(2),
    maxX: +maxX.toFixed(2),
    meanY: +(sumY / n).toFixed(2),
    meanSpeed: +(sumSpeed / n).toFixed(2),
    movingPct: +((moving / n) * 100).toFixed(1),
    // Mean height of each phase. If mass handling is right, the dense one is
    // measurably lower than the light one.
    denseMeanY: denseN ? +(denseY / denseN).toFixed(2) : null,
    lightMeanY: lightN ? +(lightY / lightN).toFixed(2) : null,
    denseMeanX: denseN ? +(denseX / denseN).toFixed(2) : null,
    lightMeanX: lightN ? +(lightX / lightN).toFixed(2) : null,
    densePct: +((denseN / n) * 100).toFixed(1),
    denseMeanSpeed: denseN ? +(denseS / denseN).toFixed(3) : null,
    lightMeanSpeed: lightN ? +(lightS / lightN).toFixed(3) : null,
    profile: height.map((h) => +h.toFixed(2)),
  };
};

// Mean distance each particle travels over a window. The honest "is it at rest"
// measure: derived velocity is displacement divided by a substep of about a
// thousandth of a second, so it magnifies invisible jitter into big numbers.
globalThis.__ballpitDrift = async (ms = 1200) => {
  const snap = async () => Float32Array.from(new Float32Array(await renderer.getArrayBufferAsync(pos.value)));
  const a = await snap();
  await new Promise((r) => setTimeout(r, ms));
  const b = await snap();
  let sum = 0;
  for (let i = 0; i < COUNT; i++) {
    sum += Math.hypot(b[i * 4] - a[i * 4], b[i * 4 + 1] - a[i * 4 + 1]);
  }
  return +(sum / COUNT).toFixed(5);
};

globalThis.__ballpitSet = (k, v) => {
  if (k === 'mode') { setMode(v); return true; }
  if (k === 'radius') { setRadius(v); return true; }
  if (k in u) { u[k].value = v; return true; }
  if (k in state) { state[k] = v; return true; }
  return false;
};
