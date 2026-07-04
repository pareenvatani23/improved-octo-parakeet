/* Live keyword metrics for the App-Idea RL model.
 *
 * Populate this map to ground the model in REAL data. Keys are the target
 * keywords from IdeaRL.KW; each value carries:
 *   volume     — average monthly search volume (integer)
 *   difficulty — keyword difficulty / competition, 0..100
 *
 * When present, `volume` overrides a feature's demand and `difficulty`
 * overrides its competition. Anything missing falls back to the modeled
 * estimate. Generate this file from a keyword-tool export with
 * tools/build_keyword_data.mjs, or edit by hand, e.g.:
 *
 *   globalThis.KEYWORD_DATA = {
 *     "ai agent":  { volume: 74000, difficulty: 61 },
 *     "adhd":      { volume: 301000, difficulty: 43 },
 *   };
 *
 * It is intentionally empty right now — no live keyword-data provider was
 * available in the build environment (Ahrefs/Semrush MCP required a paid plan).
 */
globalThis.KEYWORD_DATA = {};
