#!/usr/bin/env node
/* Fetch a Product Hunt launch cohort into data/producthunt_cohort.csv.
 *
 * Runs on a GitHub Actions runner (open internet); reads the token from the
 * PH_TOKEN env var (a repo secret). Pulls launches in NEWEST order within a
 * date window so the cohort includes flops (no survivorship/vote bias), then
 * writes a CSV for the outcome-learning pipeline.
 *
 *   PH_TOKEN=xxx node tools/fetch_producthunt.mjs <topicSlug|all> <after> <before> <max>
 *   e.g. node tools/fetch_producthunt.mjs artificial-intelligence 2022-01-01 2024-01-01 1500
 *
 * Outcome label captured = votesCount (a BUZZ proxy, not revenue). Honest for a
 * pilot that tests the pipeline + whether any signal exists.
 */
import fs from "node:fs";

const API = "https://api.producthunt.com/v2/api/graphql";
const TOKEN = process.env.PH_TOKEN;
if (!TOKEN) { console.error("ERROR: PH_TOKEN env var not set (add it as a repo secret)."); process.exit(1); }

const topic = (process.argv[2] || "artificial-intelligence").trim();
const after = (process.argv[3] || "2022-01-01") + "T00:00:00Z";
const before = (process.argv[4] || "2024-01-01") + "T00:00:00Z";
const MAX = parseInt(process.argv[5] || "1500", 10);

const useTopic = topic && topic.toLowerCase() !== "all";

const QUERY = `
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

async function gql(cursor) {
  const variables = { after, before, cursor };
  if (useTopic) variables.topic = topic;
  for (let attempt = 0; attempt < 5; attempt++) {
    const res = await fetch(API, {
      method: "POST",
      headers: { "Content-Type": "application/json", "Authorization": `Bearer ${TOKEN}`, "Accept": "application/json" },
      body: JSON.stringify({ query: QUERY, variables }),
    });
    if (res.status === 429 || res.status === 503) {
      const wait = 5000 * (attempt + 1);
      console.log(`rate-limited (${res.status}); waiting ${wait}ms…`);
      await sleep(wait); continue;
    }
    const json = await res.json();
    if (json.errors) {
      // complexity/rate errors -> back off; otherwise fail loudly
      const msg = JSON.stringify(json.errors);
      if (/rate|complexity|limit/i.test(msg) && attempt < 3) { await sleep(5000 * (attempt + 1)); continue; }
      if (/rate|complexity|limit/i.test(msg)) return null; // exhausted -> stop, keep partial
      throw new Error("GraphQL error: " + msg);
    }
    return json.data.posts;
  }
  return null; // rate-limited past retries -> signal caller to stop & save
}

const csvCell = (v) => `"${String(v ?? "").replace(/"/g, '""').replace(/\r?\n/g, " ").trim()}"`;
const cols = ["id", "name", "tagline", "description", "topics", "createdAt", "votesCount", "commentsCount", "website", "url"];

async function main() {
  const rows = [];
  let cursor = null, page = 0, stoppedEarly = false;
  while (rows.length < MAX) {
    let posts;
    try { posts = await gql(cursor); }
    catch (e) { console.log("fetch error: " + e.message + " — saving partial"); stoppedEarly = true; break; }
    if (!posts) { console.log("rate limit reached — saving partial cohort"); stoppedEarly = true; break; }
    for (const e of posts.edges) {
      const n = e.node;
      rows.push({
        id: n.id, name: n.name, tagline: n.tagline, description: n.description,
        topics: (n.topics?.edges || []).map((t) => t.node.name).join("|"),
        createdAt: n.createdAt, votesCount: n.votesCount, commentsCount: n.commentsCount,
        website: n.website, url: n.url,
      });
    }
    page++;
    console.log(`page ${page}: +${posts.edges.length} (total ${rows.length})`);
    if (!posts.pageInfo.hasNextPage) break;
    cursor = posts.pageInfo.endCursor;
    await sleep(2000); // stay under the complexity budget
  }
  if (rows.length === 0) { console.error("no rows fetched"); process.exit(1); }
  console.log(`collected ${rows.length} rows${stoppedEarly ? " (stopped early on rate limit)" : ""}`);
  fs.mkdirSync("data", { recursive: true });
  const header = cols.join(",");
  const body = rows.slice(0, MAX).map((r) => cols.map((c) => csvCell(r[c])).join(",")).join("\n");
  const meta = `# producthunt cohort | topic=${topic} | ${after}..${before} | n=${Math.min(rows.length, MAX)} | label=votesCount(buzz proxy)\n`;
  fs.writeFileSync("data/producthunt_cohort.csv", meta + header + "\n" + body + "\n");
  console.log(`wrote data/producthunt_cohort.csv (${Math.min(rows.length, MAX)} rows)`);
}
main().catch((e) => { console.error(e.message); process.exit(1); });
