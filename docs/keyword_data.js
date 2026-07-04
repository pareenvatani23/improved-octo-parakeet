/* Keyword metrics for the App-Idea RL model.
 *
 * SOURCE: approximate, search-informed estimates compiled July 2026. These are
 * order-of-magnitude US monthly search volumes and 0-100 difficulty scores,
 * reconciled from public web signals — NOT tool-grade figures from a single
 * keyword provider (Ahrefs/Semrush access was unavailable). Treat as a grounded
 * hypothesis layer, not exact market data. Replace this file with a keyword-tool
 * export via tools/build_keyword_data.mjs for precise numbers.
 *
 * volume = approx. US monthly searches; difficulty = keyword difficulty 0..100;
 * growth = directional momentum -1..+1 (declining .. surging), a real early-
 * demand signal — here estimated from public trend direction, not a Trends API.
 */
globalThis.KEYWORD_DATA_SOURCE = "Approximate, search-informed estimates (Jul 2026) — not tool-grade";
globalThis.KEYWORD_DATA = {
  // trend hooks
  "ai agent":               { volume: 110000, difficulty: 66, growth: 0.9 },
  "longevity":              { volume: 60000,  difficulty: 42, growth: 0.5 },
  "adhd":                   { volume: 500000, difficulty: 58, growth: 0.4 },
  "creator economy":        { volume: 12000,  difficulty: 44, growth: 0.1 },
  "financial independence": { volume: 33000,  difficulty: 40, growth: 0.1 },
  "dating app":             { volume: 135000, difficulty: 82, growth: -0.3 },
  // core mechanics
  "ai companion":           { volume: 45000,  difficulty: 56, growth: 0.7 },
  "ai image generator":     { volume: 350000, difficulty: 82, growth: 0.3 },
  "habit tracker":          { volume: 70000,  difficulty: 56, growth: 0.1 },
  "gamification app":       { volume: 2400,   difficulty: 34, growth: 0.0 },
  "sleep tracker":          { volume: 45000,  difficulty: 52, growth: 0.2 },
  "marketplace app":        { volume: 6600,   difficulty: 46, growth: 0.0 },
  // formats
  "ai chatbot":             { volume: 250000, difficulty: 78, growth: 0.4 },
  "short video app":        { volume: 8100,   difficulty: 46, growth: 0.2 },
  "augmented reality app":  { volume: 5400,   difficulty: 42, growth: 0.1 },
  "widget app":             { volume: 27000,  difficulty: 40, growth: 0.0 },
  "smartwatch app":         { volume: 18000,  difficulty: 42, growth: 0.1 },
  "utility app":            { volume: 3600,   difficulty: 34, growth: -0.1 },
};
