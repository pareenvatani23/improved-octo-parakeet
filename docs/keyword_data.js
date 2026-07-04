/* Keyword metrics for the App-Idea RL model.
 *
 * SOURCE: approximate, search-informed estimates compiled July 2026. These are
 * order-of-magnitude US monthly search volumes and 0-100 difficulty scores,
 * reconciled from public web signals — NOT tool-grade figures from a single
 * keyword provider (Ahrefs/Semrush access was unavailable). Treat as a grounded
 * hypothesis layer, not exact market data. Replace this file with a keyword-tool
 * export via tools/build_keyword_data.mjs for precise numbers.
 *
 * volume = approx. US monthly searches; difficulty = keyword difficulty 0..100.
 */
globalThis.KEYWORD_DATA_SOURCE = "Approximate, search-informed estimates (Jul 2026) — not tool-grade";
globalThis.KEYWORD_DATA = {
  // trend hooks
  "ai agent":               { volume: 110000, difficulty: 66 },
  "longevity":              { volume: 60000,  difficulty: 42 },
  "adhd":                   { volume: 500000, difficulty: 58 },
  "creator economy":        { volume: 12000,  difficulty: 44 },
  "financial independence": { volume: 33000,  difficulty: 40 },
  "dating app":             { volume: 135000, difficulty: 82 },
  // core mechanics
  "ai companion":           { volume: 45000,  difficulty: 56 },
  "ai image generator":     { volume: 350000, difficulty: 82 },
  "habit tracker":          { volume: 70000,  difficulty: 56 },
  "gamification app":       { volume: 2400,   difficulty: 34 },
  "sleep tracker":          { volume: 45000,  difficulty: 52 },
  "marketplace app":        { volume: 6600,   difficulty: 46 },
  // formats
  "ai chatbot":             { volume: 250000, difficulty: 78 },
  "short video app":        { volume: 8100,   difficulty: 46 },
  "augmented reality app":  { volume: 5400,   difficulty: 42 },
  "widget app":             { volume: 27000,  difficulty: 40 },
  "smartwatch app":         { volume: 18000,  difficulty: 42 },
  "utility app":            { volume: 3600,   difficulty: 34 },
};
