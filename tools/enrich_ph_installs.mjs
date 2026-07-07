#!/usr/bin/env node
/* Enrich the Product Hunt cohort (which INCLUDES flops) with real Google Play
 * install counts, by matching each PH app to its Play listing by name.
 *
 * Runs on a GH Actions runner. For each PH app: search Google Play by name,
 * accept only a confident title match, then fetch its install count. This gives
 * an unbiased launch list (PH) + a real outcome (installs) — far less
 * survivor-biased than search-sourcing the list itself.
 *
 * Output: data/ph_play_enriched.csv (only PH apps found on Google Play).
 * Caveats: (1) name matching is fuzzy -> we keep only high/med confidence and
 * record it; (2) only PH apps that shipped to Play are covered; (3) installs are
 * cumulative (PH window 2022-24 is tight, so age spread is small).
 */
import fs from "node:fs";

const IN = "data/producthunt_cohort.csv";
const OUT = "data/ph_play_enriched.csv";
const COUNTRY = "us";
const MAX = parseInt(process.argv[2] || "99999", 10);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// --- csv helpers ---
function lineFields(line) {
  const out = []; let f = "", q = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (q) { if (c === '"' && line[i + 1] === '"') { f += '"'; i++; } else if (c === '"') q = false; else f += c; }
    else { if (c === '"') q = true; else if (c === ",") { out.push(f); f = ""; } else f += c; }
  }
  out.push(f); return out;
}
function readPH() {
  const lines = fs.readFileSync(IN, "utf8").split(/\r?\n/).filter((l) => l && !l.startsWith("#"));
  const header = lineFields(lines[0]);
  return lines.slice(1).map((ln) => {
    const f = lineFields(ln); const o = {};
    header.forEach((h, i) => (o[h] = f[i] ?? "")); return o;
  });
}
const csvCell = (v) => `"${String(v ?? "").replace(/"/g, '""').replace(/\r?\n/g, " ").trim()}"`;

// --- name matching ---
const STOP = new Set(["the", "a", "an", "app", "ai", "for", "and", "of", "to", "your", "my"]);
const norm = (s) => String(s).toLowerCase().replace(/[^a-z0-9 ]/g, " ").replace(/\s+/g, " ").trim();
const toks = (s) => norm(s).split(" ").filter((t) => t.length >= 3 && !STOP.has(t));
function matchConf(phName, playTitle) {
  const a = norm(phName), b = norm(playTitle);
  if (!a || !b) return null;
  if (a === b) return "high";
  const at = toks(phName), bt = new Set(toks(playTitle));
  if (at.length >= 1 && at.every((t) => bt.has(t))) return "med";   // all PH tokens present in title
  return null;
}

const outCols = ["phId", "name", "description", "topics", "createdAt", "votesCount",
  "matchedAppId", "matchedTitle", "matchConf", "minInstalls", "genre", "released", "free", "offersIAP"];

async function main() {
  const gplay = (await import("google-play-scraper")).default;
  const rows = readPH();
  console.log(`PH apps: ${rows.length}`);
  const out = [];
  let processed = 0, matched = 0;

  for (const r of rows) {
    if (processed >= MAX) break;
    processed++;
    const name = r.name;
    if (!name || norm(name).length < 4) continue;
    try {
      const res = await gplay.search({ term: name, num: 4, country: COUNTRY, throttle: 5 });
      let hit = null, conf = null;
      for (const a of res) { const c = matchConf(name, a.title); if (c) { hit = a; conf = c; break; } }
      if (hit) {
        const d = await gplay.app({ appId: hit.appId, country: COUNTRY });
        out.push({
          phId: r.id, name, description: r.description, topics: r.topics,
          createdAt: r.createdAt, votesCount: r.votesCount,
          matchedAppId: d.appId, matchedTitle: d.title, matchConf: conf,
          minInstalls: d.minInstalls ?? 0, genre: d.genre, released: d.released,
          free: d.free ? 1 : 0, offersIAP: d.offersIAP ? 1 : 0,
        });
        matched++;
        await sleep(150);
      }
    } catch (e) { /* skip */ }
    if (processed % 100 === 0) console.log(`processed ${processed}/${rows.length} matched ${matched}`);
    await sleep(250);
  }

  if (out.length === 0) { console.error("no matches"); process.exit(1); }
  fs.mkdirSync("data", { recursive: true });
  const meta = `# ph->play enriched | matched=${out.length}/${processed} | label=minInstalls | list=PH(includes flops)\n`;
  const body = out.map((x) => outCols.map((c) => csvCell(x[c])).join(",")).join("\n");
  fs.writeFileSync(OUT, meta + outCols.join(",") + "\n" + body + "\n");
  console.log(`wrote ${OUT}: ${out.length} matched apps (of ${processed} PH apps)`);
}
main().catch((e) => { console.error(e.message); process.exit(1); });
