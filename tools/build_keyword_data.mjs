#!/usr/bin/env node
/* Build docs/keyword_data.js from a CSV export of any keyword tool.
 *
 * Usage:
 *   node tools/build_keyword_data.mjs path/to/keywords.csv
 *
 * The CSV must have a header row containing (case-insensitive) a keyword
 * column and at least one of volume / difficulty. Recognised column names:
 *   keyword:    keyword, term, query
 *   volume:     volume, search volume, search_volume, avg. monthly searches
 *   difficulty: difficulty, kd, keyword difficulty, competition
 *
 * Only keywords listed in IdeaRL.KW are kept (others are ignored). This lets
 * you paste a broad export and it self-filters to the ones the model uses.
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const csvPath = process.argv[2];
if (!csvPath) { console.error("usage: node tools/build_keyword_data.mjs <csv>"); process.exit(1); }

// the exact keyword set the model grounds on (mirrors IdeaRL.KW values)
const WANTED = new Set([
  "ai agent", "longevity", "adhd", "creator economy", "financial independence", "dating app",
  "ai companion", "ai image generator", "habit tracker", "gamification app", "sleep tracker", "marketplace app",
  "ai chatbot", "short video app", "augmented reality app", "widget app", "smartwatch app", "utility app",
]);

function splitCSVLine(line) {
  const out = []; let cur = "", q = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (q) { if (ch === '"' && line[i + 1] === '"') { cur += '"'; i++; } else if (ch === '"') q = false; else cur += ch; }
    else { if (ch === '"') q = true; else if (ch === ",") { out.push(cur); cur = ""; } else cur += ch; }
  }
  out.push(cur); return out;
}
const norm = (s) => String(s).trim().toLowerCase();
const num = (s) => { const n = parseFloat(String(s).replace(/[, %]/g, "")); return Number.isFinite(n) ? n : null; };

const rows = fs.readFileSync(csvPath, "utf8").split(/\r?\n/).filter(Boolean).map(splitCSVLine);
const header = rows.shift().map(norm);
const findCol = (names) => header.findIndex((h) => names.includes(h));
const kwCol = findCol(["keyword", "term", "query"]);
const volCol = findCol(["volume", "search volume", "search_volume", "avg. monthly searches", "avg monthly searches"]);
const kdCol = findCol(["difficulty", "kd", "keyword difficulty", "competition"]);
if (kwCol < 0) { console.error("no keyword column found; header was:", header); process.exit(1); }

const data = {};
for (const r of rows) {
  const kw = norm(r[kwCol]);
  if (!WANTED.has(kw)) continue;
  const rec = {};
  if (volCol >= 0 && num(r[volCol]) != null) rec.volume = Math.round(num(r[volCol]));
  if (kdCol >= 0 && num(r[kdCol]) != null) rec.difficulty = Math.round(num(r[kdCol]));
  if (Object.keys(rec).length) data[kw] = rec;
}

const src = `Keyword-tool export: ${path.basename(csvPath)}`;
const out = `/* Auto-generated from ${path.basename(csvPath)} by tools/build_keyword_data.mjs. */\nglobalThis.KEYWORD_DATA_SOURCE = ${JSON.stringify(src)};\nglobalThis.KEYWORD_DATA = ${JSON.stringify(data, null, 2)};\n`;
const outPath = path.join(__dirname, "..", "docs", "keyword_data.js");
fs.writeFileSync(outPath, out);
console.log(`wrote ${Object.keys(data).length}/${WANTED.size} grounded keywords -> ${outPath}`);
