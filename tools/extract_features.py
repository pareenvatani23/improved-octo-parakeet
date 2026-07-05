#!/usr/bin/env python3
"""Extract model features from a Product Hunt cohort CSV.

Reads data/producthunt_cohort.csv and writes data/features.csv:
  - taxonomy features (our 6 dimensions) inferred from name+tagline+description+topics
    by transparent keyword rules (reproducible; upgradeable to LLM labels later)
  - derived features (word_count, has_ai, launch year/month)
  - label: `votes` (buzz) and a binary `hit` = top-decile votes within the cohort

Deliberately keyword-based so it is deterministic and auditable for the pilot.
"""
import csv, re, sys, math
from pathlib import Path

IN = Path("data/producthunt_cohort.csv")
OUT = Path("data/features.csv")

# taxonomy value -> keyword cues (lowercased, substring match on the app text)
TAXO = {
    "trend": {
        "ai_agents": ["agent", "autonomous", "auto-gpt", "does it for you", "on your behalf"],
        "creator": ["creator", "influencer", "content creator", "youtuber", "newsletter", "audience"],
        "mental_health": ["mental health", "adhd", "anxiety", "therapy", "mindful", "wellbeing", "focus", "mood"],
        "productivity": ["productivity", "workflow", "getting things done", "task", "notes", "docs", "automation"],
        "dating_social": ["dating", "match", "relationship", "friends", "social network", "community"],
        "finance": ["finance", "money", "invest", "budget", "crypto", "trading", "payments", "fintech"],
        "other": [],
    },
    "audience": {
        "developers": ["developer", "engineer", "api", "sdk", "code", "devtool", "github"],
        "creators_solo": ["creator", "solopreneur", "freelancer", "indie", "maker"],
        "businesses": ["team", "business", "b2b", "enterprise", "sales", "crm", "startup", "saas"],
        "consumers": ["personal", "everyday", "your life", "for you", "consumer"],
        "students": ["student", "study", "learn", "course", "education", "exam"],
        "other": [],
    },
    "mechanic": {
        "ai_generator": ["generate", "generator", "create images", "text to", "ai-powered", "produce"],
        "ai_assistant": ["assistant", "copilot", "chatbot", "chat with", "ai companion", "coach"],
        "tracker": ["track", "tracker", "analytics", "dashboard", "insight", "monitor", "metrics"],
        "marketplace": ["marketplace", "directory", "browse", "discover", "curated list", "hire"],
        "social_feed": ["feed", "share", "community", "social", "post", "collaborate"],
        "utility": ["tool", "utility", "convert", "extension", "plugin", "widget"],
    },
    "loop": {
        "shareable_output": ["share", "shareable", "export", "publish", "showcase", "portfolio"],
        "invite_collab": ["invite", "collaborate", "team", "workspace", "together"],
        "community": ["community", "forum", "leaderboard", "challenge", "compete"],
        "none": [],
    },
    "money": {
        "subscription": ["subscription", "/mo", "per month", "monthly", "plan", "pro plan"],
        "freemium": ["free", "freemium", "credits", "free tier", "upgrade"],
        "b2b": ["enterprise", "contact sales", "seats", "per seat", "b2b"],
        "one_time": ["one-time", "lifetime", "buy once", "pay once"],
        "unknown": [],
    },
    "format": {
        "chat_voice": ["chat", "voice", "conversation", "talk to", "assistant"],
        "browser_ext": ["extension", "chrome", "browser", "plugin"],
        "mobile": ["ios", "android", "mobile", "app store", "iphone"],
        "web": ["web", "webapp", "web app", "platform", "dashboard", "saas"],
        "api_dev": ["api", "sdk", "developer", "integration"],
        "other": [],
    },
}

AI_CUES = ["ai", "gpt", "llm", "machine learning", "neural", "genai", "artificial intelligence"]


def pick(text, options):
    """Return the taxonomy value whose cues have the most hits; last (catch-all) if none."""
    best, best_n = None, 0
    for val, cues in options.items():
        n = sum(1 for c in cues if c and c in text)
        if n > best_n:
            best, best_n = val, n
    if best is None:
        best = list(options.keys())[-1]  # catch-all bucket
    return best


def read_cohort(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    reader = csv.DictReader(lines)
    for r in reader:
        rows.append(r)
    return rows


def main():
    if not IN.exists():
        print(f"missing {IN}", file=sys.stderr); sys.exit(1)
    rows = read_cohort(IN)
    if not rows:
        print("empty cohort", file=sys.stderr); sys.exit(1)

    # binary hit = top-decile votes within the cohort
    votes = sorted(int(r.get("votesCount") or 0) for r in rows)
    cutoff = votes[int(len(votes) * 0.9)] if len(votes) > 10 else max(votes)

    out_rows = []
    for r in rows:
        text = " ".join([r.get("name", ""), r.get("tagline", ""), r.get("description", ""), r.get("topics", "")]).lower()
        v = int(r.get("votesCount") or 0)
        created = r.get("createdAt", "")[:10]
        year = created[:4]
        feat = {dim: pick(text, opts) for dim, opts in TAXO.items()}
        feat.update({
            "has_ai": int(any(c in text for c in AI_CUES)),
            "word_count": len(text.split()),
            "year": year,
            "votes": v,
            "hit": int(v >= cutoff and cutoff > 0),
            "created": created,
            "name": r.get("name", ""),
        })
        out_rows.append(feat)

    cols = list(TAXO.keys()) + ["has_ai", "word_count", "year", "created", "votes", "hit", "name"]
    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(out_rows)
    n_hit = sum(r["hit"] for r in out_rows)
    print(f"wrote {OUT}: {len(out_rows)} rows, {n_hit} hits (top-decile, cutoff={cutoff} votes)")


if __name__ == "__main__":
    main()
