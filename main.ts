// roblox-up-and-coming-forecast-01: a drift-diffusion forecaster for Roblox's
// Up-and-Coming top 10 (Node 22, standard library only).
//
// Method (see README.md for the full write-up):
//  Each game on the chart is a random walk in rank space. From the history we
//  estimate, per game, a drift (ranks per hour) and a diffusion scale:
//    - drift comes from two signals, both measured over the last 24 hours and
//      centred on the chart median so the model is zero-sum across games:
//        * rank velocity (least-squares slope of rank vs time). Velocity
//          persists only weakly (practice OLS: ~0.10 of it survives), so its
//          weight is small.
//        * log-playing growth (least-squares slope of log(playing) vs time).
//          Playing growth is the leading signal for breakouts (practice OLS:
//          a game growing 0.05 log-points/hour improves ~3.6 ranks/hour), so
//          it carries the larger weight. Sweep The Moon's 8x playing surge in
//          the last 18h of the practice history is the textbook case: rank 29
//          at the close, #1 within a day.
//    - diffusion is Laplace (fat tails: the chart mostly creeps, sometimes a
//      game jumps 20+ ranks in hours), with scale growing as sqrt(hours)
//      and BANDS by rank at the close: top games are sticky (mean |rank
//      change| 0.20/hour), tail games jump (0.93/hour), so the scale is
//      much tighter for the games that matter most. Practice residuals put
//      a top-10 game's rank uncertainty at sd 7.6 after 24h, 12.5 after 48h.
//    - a baseline slide: top games drift down as new games enter above them
//      (+3.3 ranks in 24h on average for a top-10 game), fading by rank 40.
//    - games age out of the chart 28 days after they were created: when the
//      history carries `details.created`, a game's mass is zeroed after its
//      28th day. (The practice archive carries no details, so this binds only
//      on the final history's house-recorder reads.)
//  For each of the window's 72 hourly slots we compute every game's Laplace
//  probability of each place 1..10, then average over the slots: one honest
//  forecast, since every slot's read counts equally. Each place's named
//  probabilities are capped so ~12% stays unnamed, the share the games no
//  forecaster can name (brand-new entrants: 8 of the 100 practice answer
//  slots) split at each read.
//  `top10` orders games by expected place (mass outside the top 10 counts
//  as place 11).
//
//  The program is deterministic: no randomness, no clock, no network; the
//  same history always yields the same forecast byte for byte.
//
//  Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
//  stand in for /input and /output:
//    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// --- Tunable constants (calibrated on the practice window; see README) ---
/** Hours of recent history used for velocity / playing-growth fits. */
const VEL_WINDOW_H = 24;
/** Weight on centred rank velocity (ranks/hour per rank/hour). */
const DRIFT_V = 0.1;
/** Weight on centred log-playing growth (ranks/hour per log-point/hour). */
const DRIFT_G = 6;
/** Largest |drift| allowed, ranks per hour. */
const DRIFT_CLAMP = 1.0;
/**
 * Laplace scale b(t) = B0 + B1 * sqrt(t), by the game's rank at the close.
 * Top games are sticky (mean |rank change| 0.20/hour); tail games jump
 * (0.93/hour). Practice: a top-10 game's rank moves sd 7.6 in 24h, 12.5
 * in 48h; the bands below scale up from there.
 */
const BANDS = [
  { maxRank: 10, b0: 1.5, b1: 0.5 },
  { maxRank: 30, b0: 3.0, b1: 1.3 },
  { maxRank: Infinity, b0: 5.0, b1: 2.5 },
];
/**
 * Baseline slide: a top game drifts down as new games enter above it
 * (practice: a top-10 game slides +3.3 ranks in 24h on average). Fades
 * to zero by rank 40; the centred velocity/growth drift sits on top.
 */
const SLIDE0 = 0.14;
const SLIDE_RANK = 40;
/** Largest total named probability at one place; the rest stays unnamed. */
const NAMED_CAP = 0.92;
/** Smallest p that earns a row (rows are floored to 6 decimals). */
const P_MIN = 0.001;
/** Rank penalty for a game that dropped off the last chart. */
const DROPOUT_PENALTY = 5;
/** Games seen in the top this many ranks within the last 72h stay candidates. */
const DROPOUT_LOOKBACK_H = 72;
const DROPOUT_TOP = 40;

const MICRO = 1_000_000;

interface Row {
  rank: number;
  universe_id: number;
  name?: string;
  playing?: number | null;
}

interface Snapshot {
  at: string;
  ok?: boolean;
  up_and_coming: Row[] | null;
  details?: Record<string, { created?: string } | null> | null;
}

const task = JSON.parse(readFileSync(join(INPUT_DIR, 'task.json'), 'utf8')) as {
  places: number;
  output: string;
  window: { from: string; until: string; slot_minutes: number; slots: number };
};
const lines = readFileSync(join(INPUT_DIR, 'history.ndjson'), 'utf8')
  .split('\n')
  .filter((l) => l !== '');
const snapshots = lines
  .map((l) => JSON.parse(l) as Snapshot)
  .filter((s) => s.ok !== false && Array.isArray(s.up_and_coming))
  .sort((a, b) => Date.parse(a.at) - Date.parse(b.at));
if (snapshots.length === 0) throw new Error('the history holds no usable snapshot');
const last = snapshots[snapshots.length - 1]!;
const tClose = Date.parse(last.at);
const windowFrom = Date.parse(task.window.from);
const slotHours = (task.window.slot_minutes ?? 60) / 60;
const slots = task.window.slots ?? 72;
const hours: number[] = [];
for (let s = 1; s <= slots; s++) hours.push(s * slotHours);

// --- Per-game series ---
interface Obs {
  tH: number; // hours relative to the close (<= 0)
  rank: number;
  playing: number | null;
}
const series = new Map<number, Obs[]>();
const created = new Map<number, number>(); // universe_id -> created ms
for (const s of snapshots) {
  const tH = (Date.parse(s.at) - tClose) / 3_600_000;
  for (const r of s.up_and_coming!) {
    const list = series.get(r.universe_id) ?? [];
    list.push({ tH, rank: r.rank, playing: typeof r.playing === 'number' ? r.playing : null });
    series.set(r.universe_id, list);
  }
  if (s.details) {
    for (const [uid, d] of Object.entries(s.details)) {
      if (d && d.created && !created.has(Number(uid))) created.set(Number(uid), Date.parse(d.created));
    }
  }
}
const lastChart = new Map<number, number>(); // universe_id -> rank at close
for (const r of last.up_and_coming!) lastChart.set(r.universe_id, r.rank);

// --- Least-squares slope of y vs t over observations, with exponential
// recency weights (half-life SLOPE_HALF_LIFE_H): in a chart this fast, what
// a game did in the last few hours predicts the next hours far better than
// what it did yesterday. ---
const SLOPE_HALF_LIFE_H = 8;

function slope(obs: Array<{ t: number; y: number }>): number {
  const n = obs.length;
  if (n < 2) return 0;
  const tMax = Math.max(...obs.map((o) => o.t));
  let sw = 0;
  let mt = 0;
  let my = 0;
  for (const o of obs) {
    const w = Math.pow(0.5, (tMax - o.t) / SLOPE_HALF_LIFE_H);
    sw += w;
    mt += w * o.t;
    my += w * o.y;
  }
  mt /= sw;
  my /= sw;
  let num = 0;
  let den = 0;
  for (const o of obs) {
    const w = Math.pow(0.5, (tMax - o.t) / SLOPE_HALF_LIFE_H);
    num += w * (o.t - mt) * (o.y - my);
    den += w * (o.t - mt) * (o.t - mt);
  }
  return den === 0 ? 0 : num / den;
}

function median(xs: number[]): number {
  if (xs.length === 0) return 0;
  const s = [...xs].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 === 1 ? s[m]! : (s[m - 1]! + s[m]!) / 2;
}

// --- Candidate games: last chart + recent top-40 dropouts ---
interface Game {
  uid: number;
  r0: number;
  drift: number;
  exitHour: number; // hours after window.from when the game ages out (Infinity = never)
}
const candidates = new Map<number, Game>();
const velList: number[] = [];
const growList: number[] = [];
const feats = new Map<number, { v: number; g: number; n: number }>();
for (const [uid, obs] of series) {
  const inLast = lastChart.has(uid);
  const lastObs = obs[obs.length - 1]!;
  const lastSeenH = -lastObs.tH; // hours before the close
  if (!inLast) {
    if (lastSeenH > DROPOUT_LOOKBACK_H || lastObs.rank > DROPOUT_TOP) continue;
  }
  const recent = obs.filter((o) => -o.tH <= VEL_WINDOW_H);
  const v = slope(recent.map((o) => ({ t: o.tH, y: o.rank })));
  const playObs = recent.filter((o) => o.playing !== null && (o.playing as number) > 0);
  const g = playObs.length >= 3 ? slope(playObs.map((o) => ({ t: o.tH, y: Math.log(o.playing as number) }))) : 0;
  feats.set(uid, { v, g, n: recent.length });
  velList.push(v);
  growList.push(g);
}
const medV = median(velList);
const medG = median(growList);
for (const [uid, f] of feats) {
  const inLast = lastChart.has(uid);
  const lastObs = series.get(uid)![series.get(uid)!.length - 1]!;
  const r0 = inLast ? lastChart.get(uid)! : lastObs.rank + DROPOUT_PENALTY;
  // Shrink the drift when the game has few recent observations.
  const shrink = Math.min(1, f.n / 4);
  const slide = SLIDE0 * Math.max(0, 1 - r0 / SLIDE_RANK);
  const drift = Math.max(
    -DRIFT_CLAMP,
    Math.min(DRIFT_CLAMP, slide + (DRIFT_V * (f.v - medV) - DRIFT_G * (f.g - medG)) * shrink),
  );
  let exitHour = Infinity;
  const c = created.get(uid);
  if (c !== undefined) exitHour = (c + 28 * 24 * 3_600_000 - windowFrom) / 3_600_000;
  candidates.set(uid, { uid, r0, drift, exitHour });
}

// --- Laplace CDF ---
function laplaceCdf(x: number, mu: number, b: number): number {
  if (x < mu) return 0.5 * Math.exp((x - mu) / b);
  return 1 - 0.5 * Math.exp(-(x - mu) / b);
}

// --- Time-averaged place probabilities ---
// placeProbs[place] = Map(uid -> p)
const placeProbs: Array<Map<number, number>> = [];
for (let k = 0; k < task.places; k++) placeProbs.push(new Map());
for (const game of candidates.values()) {
  const acc = new Array<number>(task.places).fill(0);
  let liveHours = 0;
  const band = BANDS.find((b) => game.r0 <= b.maxRank) ?? BANDS[BANDS.length - 1]!;
  for (const t of hours) {
    if (t >= game.exitHour) continue; // aged out of the chart
    liveHours++;
    const mu = game.r0 + game.drift * t;
    const b = band.b0 + band.b1 * Math.sqrt(t);
    for (let k = 1; k <= task.places; k++) {
      acc[k - 1]! += laplaceCdf(k + 0.5, mu, b) - laplaceCdf(k - 0.5, mu, b);
    }
  }
  if (liveHours === 0) continue;
  for (let k = 1; k <= task.places; k++) {
    const p = acc[k - 1]! / hours.length;
    if (p >= P_MIN) placeProbs[k - 1]!.set(game.uid, p);
  }
}

// --- Cap each place's named sum at NAMED_CAP (the rest stays unnamed) ---
const probabilities: Array<{ universe_id: number; place: number; p: number }> = [];
for (let k = 1; k <= task.places; k++) {
  const m = placeProbs[k - 1]!;
  let sum = 0;
  for (const p of m.values()) sum += p;
  const scale = sum > NAMED_CAP ? NAMED_CAP / sum : 1;
  const rows: Array<{ universe_id: number; place: number; p: number }> = [];
  for (const [uid, p] of [...m.entries()].sort((a, b) => a[0] - b[0])) {
    const q = Math.floor(p * scale * MICRO) / MICRO;
    if (q > 0) rows.push({ universe_id: uid, place: k, p: q });
  }
  probabilities.push(...rows);
}

// --- top10 by expected place (mass outside the top 10 counts as place 11) ---
const expected = new Map<number, number>();
for (const game of candidates.values()) {
  let exp = 0;
  let mass = 0;
  for (let k = 1; k <= task.places; k++) {
    const p = placeProbs[k - 1]!.get(game.uid) ?? 0;
    exp += k * p;
    mass += p;
  }
  expected.set(game.uid, exp + 11 * (1 - mass));
}
const top10 = [...expected.entries()]
  .sort((a, b) => (a[1] - b[1]) || (a[0] - b[0]))
  .slice(0, task.places)
  .map(([uid]) => uid);

writeFileSync(join(OUTPUT_DIR, task.output), `${JSON.stringify({ probabilities, top10 })}\n`);
