#!/usr/bin/env node
/* Extract each app's ICON color for the existing Google Play cohort.
 *
 * Runs on a GH Actions runner. Re-runs the same searches to recover each
 * cohort app's icon URL (fast — search results include the icon), downloads the
 * icon, computes its average colour (sharp resize-to-1x1 trick, flattened on
 * white to handle transparency), and buckets the hue. Writes data/icon_colors.csv
 * to be joined to data/googleplay_cohort.csv by appId.
 */
import fs from "node:fs";

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
const COUNTRY = "us";
const FILE = "data/icon_colors.csv";
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function cohortIds() {
  const ids = new Set();
  if (!fs.existsSync("data/googleplay_cohort.csv")) return ids;
  for (const ln of fs.readFileSync("data/googleplay_cohort.csv", "utf8").split(/\r?\n/)) {
    if (!ln || ln.startsWith("#") || ln.startsWith("appId,")) continue;
    const m = ln.match(/^"((?:[^"]|"")*)"/);
    if (m) ids.add(m[1]);
  }
  return ids;
}

function rgb2hsv(r, g, b) {
  r /= 255; g /= 255; b /= 255;
  const mx = Math.max(r, g, b), mn = Math.min(r, g, b), d = mx - mn;
  let h = 0;
  if (d !== 0) {
    if (mx === r) h = ((g - b) / d) % 6;
    else if (mx === g) h = (b - r) / d + 2;
    else h = (r - g) / d + 4;
    h *= 60; if (h < 0) h += 360;
  }
  return { h, s: mx === 0 ? 0 : d / mx, v: mx };
}
function bucket(h, s, v) {
  if (v < 0.22) return "dark";
  if (s < 0.18) return v > 0.8 ? "white" : "gray";
  if (h < 15 || h >= 345) return "red";
  if (h < 45) return "orange";
  if (h < 70) return "yellow";
  if (h < 165) return "green";
  if (h < 200) return "teal";
  if (h < 260) return "blue";
  if (h < 290) return "purple";
  return "pink";
}

async function main() {
  const gplay = (await import("google-play-scraper")).default;
  const sharp = (await import("sharp")).default;
  const ids = cohortIds();
  console.log(`cohort ids: ${ids.size}`);

  // recover icon URLs via search
  const iconMap = new Map();
  for (const term of TERMS) {
    try {
      const res = await gplay.search({ term, num: 40, country: COUNTRY, throttle: 5 });
      for (const a of res) if (ids.has(a.appId) && a.icon && !iconMap.has(a.appId)) iconMap.set(a.appId, a.icon);
    } catch (e) { /* skip */ }
    await sleep(300);
  }
  console.log(`icon URLs recovered: ${iconMap.size}/${ids.size}`);

  const rows = [];
  let done = 0;
  for (const [appId, url] of iconMap) {
    try {
      const res = await fetch(url);
      const buf = Buffer.from(await res.arrayBuffer());
      const { data } = await sharp(buf).flatten({ background: "#ffffff" }).resize(1, 1).raw().toBuffer({ resolveWithObject: true });
      const [r, g, b] = [data[0], data[1], data[2]];
      const { h, s, v } = rgb2hsv(r, g, b);
      rows.push({ appId, r, g, b, hue: Math.round(h), sat: s.toFixed(3), val: v.toFixed(3), color: bucket(h, s, v) });
    } catch (e) { /* skip bad image */ }
    done++;
    if (done % 100 === 0) console.log(`icons ${done}/${iconMap.size} (kept ${rows.length})`);
    await sleep(60);
  }

  if (rows.length === 0) { console.error("no icons processed"); process.exit(1); }
  const cols = ["appId", "r", "g", "b", "hue", "sat", "val", "color"];
  fs.mkdirSync("data", { recursive: true });
  const body = rows.map((x) => cols.map((c) => `"${x[c]}"`).join(",")).join("\n");
  fs.writeFileSync(FILE, `# icon colors | n=${rows.length}\n` + cols.join(",") + "\n" + body + "\n");
  console.log(`wrote ${FILE}: ${rows.length} icons`);
}
main().catch((e) => { console.error(e.message); process.exit(1); });
