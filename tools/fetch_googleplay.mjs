#!/usr/bin/env node
/* Build a Google Play cohort with a REAL outcome label (install counts).
 *
 * Runs on a GitHub Actions runner (open internet). Uses google-play-scraper to
 * (1) gather a varied set of app IDs via many search terms (surfaces the long
 * tail, not just top-chart survivors), then (2) fetch each app's details:
 * install count, release date, description, genre, monetization.
 *
 *   node tools/fetch_googleplay.mjs <maxApps> <country>
 *
 * Label = minInstalls (cumulative). Caveats (documented in results): cumulative
 * vs launch-date, and survivorship (delisted apps are absent from Play).
 */
import fs from "node:fs";

const MAX = parseInt(process.argv[2] || "1200", 10);
const COUNTRY = process.argv[3] || "us";
const FILE = "data/googleplay_cohort.csv";

// diverse search terms across categories + our taxonomy, to get variety incl. long tail
const TERMS = [
  "habit tracker", "budget app", "meditation", "workout planner", "language learning",
  "photo editor", "recipe app", "podcast player", "note taking", "to do list",
  "ai chat assistant", "ai image generator", "invoice maker", "expense tracker", "sleep tracker",
  "journal diary", "flashcards study", "meal planner", "run tracker", "focus timer",
  "resume builder", "video editor", "crypto wallet", "stock portfolio", "dating chat",
  "social network", "habit streak", "mental health", "calorie counter", "period tracker",
  "kids learning", "coding practice", "music maker", "pdf scanner", "password manager",
  "vpn privacy", "weather widget", "plant identifier", "car maintenance", "grocery list",
  "freelance invoice", "team chat", "crm sales", "email marketing", "ai writing assistant",
  "screen recorder", "wallpaper hd", "step counter", "water reminder", "flight tracker",
];

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const csvCell = (v) => `"${String(v ?? "").replace(/"/g, '""').replace(/\r?\n/g, " ").trim()}"`;
const cols = ["appId", "title", "description", "genre", "released", "releasedYear",
  "minInstalls", "score", "ratings", "free", "price", "offersIAP"];

async function main() {
  const gplay = (await import("google-play-scraper")).default;

  // 1) collect app IDs from searches
  const ids = new Set();
  for (const term of TERMS) {
    try {
      const res = await gplay.search({ term, num: 40, country: COUNTRY, throttle: 5 });
      for (const a of res) ids.add(a.appId);
      console.log(`search "${term}": ${res.length} (total ids ${ids.size})`);
    } catch (e) {
      console.log(`search "${term}" failed: ${e.message}`);
    }
    await sleep(400);
    if (ids.size >= MAX * 1.6) break; // gather a bit more than needed (some get filtered)
  }

  // 2) fetch details
  const rows = [];
  let done = 0;
  for (const appId of ids) {
    if (rows.length >= MAX) break;
    try {
      const d = await gplay.app({ appId, country: COUNTRY });
      const yr = (String(d.released || "").match(/(\d{4})/) || [])[1] || "";
      if (!yr) { done++; continue; }                 // need a date for the time-split
      rows.push({
        appId: d.appId, title: d.title, description: d.description, genre: d.genre,
        released: d.released, releasedYear: yr, minInstalls: d.minInstalls ?? 0,
        score: d.score ?? "", ratings: d.ratings ?? 0, free: d.free ? 1 : 0,
        price: d.price ?? 0, offersIAP: d.offersIAP ? 1 : 0,
      });
    } catch (e) {
      // 404 / removed apps -> skip
    }
    done++;
    if (done % 50 === 0) console.log(`details ${done}/${ids.size} (kept ${rows.length})`);
    await sleep(150);
  }

  if (rows.length === 0) { console.error("no apps fetched"); process.exit(1); }
  fs.mkdirSync("data", { recursive: true });
  const meta = `# googleplay cohort | ${COUNTRY} | n=${rows.length} | label=minInstalls(cumulative) | note=survivorship: delisted apps absent\n`;
  const body = rows.map((r) => cols.map((c) => csvCell(r[c])).join(",")).join("\n");
  fs.writeFileSync(FILE, meta + cols.join(",") + "\n" + body + "\n");
  console.log(`wrote ${FILE}: ${rows.length} apps`);
}
main().catch((e) => { console.error(e.message); process.exit(1); });
