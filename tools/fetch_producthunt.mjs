#!/usr/bin/env node
/* Fetch/append a multi-category Product Hunt launch cohort into
 * data/producthunt_cohort.csv (merged, deduped by id).
 *
 * Runs on a GitHub Actions runner (open internet); PH_TOKEN from a repo secret.
 * NEWEST order within a date window (cohort includes flops). Appends to any
 * existing CSV so several runs (spaced past the 15-min rate limit) accumulate
 * across categories. Saves partial + exits cleanly on rate limit.
 *
 *   PH_TOKEN=xxx node tools/fetch_producthunt.mjs <topicsCsv|all> <after> <before> <perTopic>
 *   e.g. node tools/fetch_producthunt.mjs "productivity,fintech,health-fitness" 2022-01-01 2024-01-01 300
 *
 * Label captured = votesCount (buzz proxy), for the pilot.
 */
import fs from "node:fs";

const API = "https://api.producthunt.com/v2/api/graphql";
const TOKEN = process.env.PH_TOKEN;
if (!TOKEN) { console.error("ERROR: PH_TOKEN env var not set (add it as a repo secret)."); process.exit(1); }

const topics = (process.argv[2] || "artificial-intelligence").split(",").map((s) => s.trim()).filter(Boolean);
const after = (process.argv[3] || "2022-01-01") + "T00:00:00Z";
const before = (process.argv[4] || "2024-01-01") + "T00:00:00Z";
const PER_TOPIC = parseInt(process.argv[5] || "300", 10);
const FILE = "data/producthunt_cohort.csv";
const cols = ["id", "name", "tagline", "description", "topics", "createdAt", "votesCount", "commentsCount", "website", "url"];

const buildQuery = (useTopic) => `
query Cohort($after: DateTime!, $before: DateTime!, $cursor: String${useTopic ? ", $topic: String!" : ""}) {
  posts(order: NEWEST, postedAfter: $after, postedBefore: $before, first: 20, after: $cursor${useTopic ? ", topic: $topic" : ""}) {
    edges { node {
      id name tagline description votesCount commentsCount createdAt website url
      topics(first: 5) { edges { node { name } } }
    } }
    pageInfo { endCursor hasNextPage }
  }
}`;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function gql(query, variables) {
  for (let attempt = 0; attempt < 4; attempt++) {
    const res = await fetch(API, {
      method: "POST",
      headers: { "Content-Type": "application/json", "Authorization": `Bearer ${TOKEN}`, "Accept": "application/json" },
      body: JSON.stringify({ query, variables }),
    });
    if (res.status === 429 || res.status === 503) { await sleep(5000 * (attempt + 1)); continue; }
    const json = await res.json();
    if (json.errors) {
      const msg = JSON.stringify(json.errors);
      if (/rate|complexity|limit/i.test(msg) && attempt < 3) { await sleep(5000 * (attempt + 1)); continue; }
      if (/rate|complexity|limit/i.test(msg)) return null;
      throw new Error("GraphQL error: " + msg);
    }
    return json.data.posts;
  }
  return null;
}

const csvCell = (v) => `"${String(v ?? "").replace(/"/g, '""').replace(/\r?\n/g, " ").trim()}"`;
const csvLine = (n) => cols.map((c) => {
  if (c === "topics") return csvCell((n.topics?.edges || []).map((t) => t.node.name).join("|"));
  return csvCell(n[c]);
}).join(",");

function loadExisting() {
  if (!fs.existsSync(FILE)) return { ids: new Set(), lines: [] };
  const raw = fs.readFileSync(FILE, "utf8").split(/\r?\n/);
  const ids = new Set(); const lines = [];
  for (const ln of raw) {
    if (!ln || ln.startsWith("#") || ln.startsWith("id,")) continue;
    const m = ln.match(/^"((?:[^"]|"")*)"/); // first field = id
    if (m) { ids.add(m[1]); lines.push(ln); }
  }
  return { ids, lines };
}

async function main() {
  const { ids, lines } = loadExisting();
  console.log(`existing rows: ${lines.length}`);
  const newLines = [];
  let stopped = false;

  for (const t of topics) {
    if (stopped) break;
    const useTopic = t.toLowerCase() !== "all";
    const query = buildQuery(useTopic);
    let cursor = null, got = 0, page = 0;
    while (got < PER_TOPIC) {
      const variables = { after, before, cursor };
      if (useTopic) variables.topic = t;
      let posts;
      try { posts = await gql(query, variables); }
      catch (e) { console.log(`[${t}] error ${e.message} — saving partial`); stopped = true; break; }
      if (!posts) { console.log(`[${t}] rate limit — saving partial`); stopped = true; break; }
      for (const e of posts.edges) {
        const n = e.node;
        if (ids.has(n.id)) continue;
        ids.add(n.id); newLines.push(csvLine(n)); got++;
      }
      page++;
      console.log(`[${t}] page ${page}: +${posts.edges.length} (topic new ${got}, total new ${newLines.length})`);
      if (!posts.pageInfo.hasNextPage) break;
      cursor = posts.pageInfo.endCursor;
      await sleep(2000);
    }
  }

  const all = [...lines, ...newLines];
  if (all.length === 0) { console.error("no rows"); process.exit(1); }
  fs.mkdirSync("data", { recursive: true });
  const meta = `# producthunt cohort | topics=${topics.join("+")}+prev | ${after}..${before} | n=${all.length} (+${newLines.length} new) | label=votesCount(buzz proxy)\n`;
  fs.writeFileSync(FILE, meta + cols.join(",") + "\n" + all.join("\n") + "\n");
  console.log(`wrote ${FILE}: ${all.length} rows (+${newLines.length} new)${stopped ? " [stopped early on rate limit]" : ""}`);
}
main().catch((e) => { console.error(e.message); process.exit(1); });
