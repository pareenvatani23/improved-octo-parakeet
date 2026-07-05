#!/usr/bin/env python3
"""Extract model features from a cohort CSV (Google Play or Product Hunt).

Auto-detects the source:
  - data/googleplay_cohort.csv  -> label = minInstalls (REAL adoption), date = released
  - data/producthunt_cohort.csv -> label = votes (buzz proxy),         date = createdAt

Writes data/features.csv: our 6-dim taxonomy (keyword-extracted from the app's
text) + derived features, plus `label`, binary `hit` (top-decile label) and a
sortable `created` date for the time-split.

IMPORTANT: we deliberately EXCLUDE rating-count / score / installs from the
features — those are the label in disguise (popularity) and would leak.
"""
import csv, sys, math
from datetime import datetime
from pathlib import Path

# taxonomy value -> keyword cues (substring match on lowercased app text)
TAXO = {
    "trend": {
        "ai": ["ai", "gpt", "chatbot", "artificial intelligence", "machine learning"],
        "creator": ["creator", "influencer", "content", "youtuber", "streamer", "portfolio"],
        "mental_health": ["mental health", "adhd", "anxiety", "therapy", "mindful", "meditat", "mood", "calm"],
        "productivity": ["productivity", "workflow", "task", "notes", "to-do", "to do", "planner", "organize"],
        "fitness_health": ["fitness", "workout", "calorie", "diet", "run", "steps", "sleep", "health", "period"],
        "finance": ["finance", "money", "budget", "invest", "crypto", "trading", "expense", "invoice", "wallet"],
        "dating_social": ["dating", "match", "friends", "social", "chat", "community"],
        "other": [],
    },
    "audience": {
        "developers": ["developer", "engineer", "api", "sdk", "code", "coding"],
        "creators_solo": ["creator", "freelancer", "indie", "solopreneur", "small business"],
        "businesses": ["team", "business", "b2b", "enterprise", "sales", "crm", "invoice"],
        "students": ["student", "study", "learn", "course", "exam", "flashcard", "kids"],
        "consumers": ["personal", "everyday", "your", "daily", "life"],
        "other": [],
    },
    "mechanic": {
        "ai_generator": ["generate", "generator", "create", "ai-powered", "text to"],
        "ai_assistant": ["assistant", "copilot", "chatbot", "chat with", "coach"],
        "tracker": ["track", "tracker", "counter", "log", "monitor", "analytics", "dashboard"],
        "marketplace": ["marketplace", "directory", "browse", "discover", "hire", "shop"],
        "social_feed": ["feed", "share", "community", "social", "post"],
        "utility": ["tool", "utility", "convert", "scanner", "editor", "maker", "widget", "manager"],
    },
    "loop": {
        "shareable_output": ["share", "shareable", "export", "publish", "showcase"],
        "invite_collab": ["invite", "collaborate", "team", "together", "sync"],
        "community": ["community", "leaderboard", "challenge", "compete", "streak"],
        "none": [],
    },
    "money": {
        "subscription": ["subscription", "per month", "monthly", "premium", "pro"],
        "freemium": ["free", "credits", "unlock", "upgrade"],
        "one_time": ["one-time", "lifetime", "buy once", "purchase"],
        "unknown": [],
    },
    "format": {
        "chat_voice": ["chat", "voice", "conversation", "assistant"],
        "camera_ar": ["camera", "scan", "ar", "photo", "identify"],
        "widget": ["widget", "lockscreen", "home screen"],
        "wearable": ["watch", "wear os", "wearable", "band"],
        "tracker_ui": ["tracker", "dashboard", "chart"],
        "other": [],
    },
}
AI_CUES = ["ai", "gpt", "llm", "machine learning", "neural", "genai", "artificial intelligence"]

GP = Path("data/googleplay_cohort.csv")
PH = Path("data/producthunt_cohort.csv")
OUT = Path("data/features.csv")


def pick(text, options):
    best, best_n = None, 0
    for val, cues in options.items():
        n = sum(1 for c in cues if c and c in text)
        if n > best_n:
            best, best_n = val, n
    return best if best is not None else list(options.keys())[-1]


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    return list(csv.DictReader(lines))


def parse_date_gp(s):
    s = (s or "").strip()
    for fmt in ("%b %d, %Y", "%d %b %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    # fallback: any 4-digit year
    import re
    m = re.search(r"(\d{4})", s)
    return f"{m.group(1)}-01-01" if m else ""


def main():
    if GP.exists():
        src, rows = "googleplay", read_rows(GP)
        text_of = lambda r: " ".join([r.get("title", ""), r.get("description", ""), r.get("genre", "")]).lower()
        label_of = lambda r: int(float(r.get("minInstalls") or 0))
        date_of = lambda r: parse_date_gp(r.get("released", ""))
        extra = lambda r: {"free": int(r.get("free") or 0), "offersIAP": int(r.get("offersIAP") or 0)}
        name_of = lambda r: r.get("title", "")
        lbl_name = "installs"
    elif PH.exists():
        src, rows = "producthunt", read_rows(PH)
        text_of = lambda r: " ".join([r.get("name", ""), r.get("tagline", ""), r.get("description", ""), r.get("topics", "")]).lower()
        label_of = lambda r: int(r.get("votesCount") or 0)
        date_of = lambda r: (r.get("createdAt", "") or "")[:10]
        extra = lambda r: {}
        name_of = lambda r: r.get("name", "")
        lbl_name = "votes"
    else:
        print("no cohort CSV found in data/", file=sys.stderr); sys.exit(1)

    rows = [r for r in rows if date_of(r)]
    labels = sorted(label_of(r) for r in rows)
    cutoff = labels[int(len(labels) * 0.9)] if len(labels) > 10 else max(labels)

    out_rows = []
    extra_keys = set()
    for r in rows:
        t = text_of(r)
        lab = label_of(r)
        feat = {dim: pick(t, opts) for dim, opts in TAXO.items()}
        ex = extra(r); extra_keys |= set(ex.keys())
        feat.update(ex)
        feat.update({
            "has_ai": int(any(c in t for c in AI_CUES)),
            "word_count": len(t.split()),
            "created": date_of(r),
            "label": lab,
            "hit": int(lab >= cutoff and cutoff > 0),
            "name": name_of(r),
        })
        out_rows.append(feat)

    cols = list(TAXO.keys()) + sorted(extra_keys) + ["has_ai", "word_count", "created", "label", "hit", "name"]
    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(out_rows)
    n_hit = sum(r["hit"] for r in out_rows)
    print(f"source={src} label={lbl_name}: wrote {OUT}: {len(out_rows)} rows, {n_hit} hits "
          f"(top-decile, cutoff={cutoff})")


if __name__ == "__main__":
    main()
