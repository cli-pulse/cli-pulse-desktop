//! Incremental scan cache — per-provider on-disk state that lets the
//! scanner skip files whose (mtime, size) haven't changed since last run
//! and resume parsing at a saved byte offset for files that have grown.
//!
//! Schema is a straight port of the Swift `CostUsageCache` in the macOS
//! app (`CLIPulseCore/CostUsageCache.swift`), kept deliberately simple so
//! the two implementations stay interchangeable at the concept level
//! (not binary — this is a Rust-only v1 file living at its own path to
//! avoid cross-app interference).
//!
//! Cache location:
//!   macOS:   ~/Library/Caches/dev.clipulse.desktop/cost-usage/{provider}-v1.json
//!   Linux:   ~/.cache/dev.clipulse.desktop/cost-usage/{provider}-v1.json
//!   Windows: %LOCALAPPDATA%\dev.clipulse.desktop\cost-usage\{provider}-v1.json
//!
//! Packed slot layout:
//!   Codex:  [input, cached, output, cost_nanos]
//!   Claude: [input, cache_read, cache_create, output, cost_nanos, msgs]
//!
//! `cost_nanos` is per-request cost accumulated during parse, stored at
//! a billion-scale (i.e. `$0.01 → 10_000_000`). Pre-aggregation of cost
//! is essential because tiered pricing (Claude's 200K tier, Codex's 272K
//! long-context tier) must be evaluated per request, not on the day's
//! aggregated tokens, and because Codex rates depend on the request's date
//! (see scanner.rs and pricing.rs).
//!
//! Each provider's cache also records the version of the rules that produced
//! it (`rules_version`). A cache written under other rules is discarded on
//! load, so a rules change reaches every file, including files that have not
//! changed on disk.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

const CACHE_SCHEMA_VERSION: u32 = 1;

/// Version of the rules that turn one provider's log lines into cached
/// numbers. Bump it when a change alters what an unchanged file contributes;
/// every cache written under an older version is then rebuilt from the logs.
///
/// Per provider, so a Codex change does not force a rescan of Claude's logs,
/// which are usually far larger.
///
/// Codex history:
/// - 0: day-level Codex cost, the cumulative baseline followed every drop.
/// - 1: per-request cost with dated and long-context rates (packed slot 3);
///   the cumulative counter can no longer re-count a gap; state advances on
///   out-of-range events too; the rollout id and event span are recorded so
///   a second copy of a rollout is counted once.
pub fn rules_version(provider: &str) -> u32 {
    match provider {
        "codex" => 1,
        _ => 0,
    }
}

pub type Packed = Vec<i64>;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CodexTotals {
    pub input: i64,
    pub cached: i64,
    pub output: i64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FileEntry {
    pub mtime_unix_ms: i64,
    pub size: i64,
    /// This file's own contribution: `day -> model -> packed`. Kept per
    /// file (not only at cache-top) so we can subtract it when the file
    /// rotates / shrinks / disappears.
    pub days: HashMap<String, HashMap<String, Packed>>,
    #[serde(default)]
    pub parsed_bytes: Option<i64>,
    #[serde(default)]
    pub last_model: Option<String>,
    /// Codex: the previous cumulative `total_token_usage` snapshot.
    #[serde(default)]
    pub last_totals: Option<CodexTotals>,
    #[serde(default)]
    pub session_id: Option<String>,
    /// Codex: the highest cumulative snapshot seen in this file, per
    /// component. See `scanner::CodexCounter`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peak_totals: Option<CodexTotals>,
    /// Codex: `session_meta.payload.id`, the id of this rollout. Unlike
    /// `session_id`, which a sub-agent's rollout shares with its parent, this
    /// is the rollout's own. Used only to recognise a second copy of the same
    /// rollout (see `scanner::codex_duplicate_copies`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rollout_id: Option<String>,
    /// Codex: time of the first and last token event in the whole file (not
    /// just the scan window), Unix ms.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_event_ms: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_event_ms: Option<i64>,
    /// Codex: number of token events in the whole file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub event_count: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostUsageCache {
    #[serde(default = "default_version")]
    pub version: u32,
    /// `rules_version(provider)` at the time this cache was built. Absent in
    /// caches written before it existed, which read as 0.
    #[serde(default)]
    pub rules_version: u32,
    #[serde(default)]
    pub last_scan_unix_ms: i64,
    #[serde(default)]
    pub files: HashMap<String, FileEntry>,
    /// Aggregate across all tracked files: `day -> model -> packed`.
    /// Scanner emits DailyEntries straight from this without walking
    /// `files` — much cheaper on warm scans.
    #[serde(default)]
    pub days: HashMap<String, HashMap<String, Packed>>,
}

fn default_version() -> u32 {
    CACHE_SCHEMA_VERSION
}

impl Default for CostUsageCache {
    fn default() -> Self {
        Self {
            version: CACHE_SCHEMA_VERSION,
            rules_version: 0,
            last_scan_unix_ms: 0,
            files: HashMap::new(),
            days: HashMap::new(),
        }
    }
}

impl CostUsageCache {
    /// An empty cache stamped with `provider`'s current rules. Every fresh
    /// cache the scanner builds must come from here: one built from
    /// `Default` would be discarded by the next `load` and re-parsed from
    /// scratch on every scan.
    pub fn for_provider(provider: &str) -> Self {
        Self {
            rules_version: rules_version(provider),
            ..Self::default()
        }
    }
}

// ========================================================================
// IO
// ========================================================================

fn cache_root(override_dir: Option<&Path>) -> Option<PathBuf> {
    if let Some(o) = override_dir {
        return Some(o.to_path_buf());
    }
    dirs::cache_dir().map(|d| d.join("dev.clipulse.desktop").join("cost-usage"))
}

pub fn cache_path(provider: &str, override_dir: Option<&Path>) -> Option<PathBuf> {
    cache_root(override_dir).map(|d| d.join(format!("{}-v{}.json", provider, CACHE_SCHEMA_VERSION)))
}

/// Cache directory for the compare-mode **baseline** scan (v0.10.2). Compare
/// mode needs a wide, fixed window (up to 180 days) so the frontend can slice
/// "current N days" vs "previous N days". But the main scan prunes its cache to
/// the currently-selected window on every run (`prune_days`), and unchanged
/// files are skipped without re-parsing — so a baseline scan *sharing* the main
/// cache would find the older days already pruned away and silently under-report
/// the previous period. Giving the baseline its own namespace keeps it a stable,
/// always-wide, incrementally-warmed cache that never fights the main scan's
/// pruning. Returns `None` only when no OS cache dir is available (the scan then
/// runs uncached but still correct).
pub fn compare_cache_dir() -> Option<PathBuf> {
    cache_root(None).map(|d| d.join("compare"))
}

pub fn load(provider: &str, override_dir: Option<&Path>) -> CostUsageCache {
    let fresh = || CostUsageCache::for_provider(provider);
    let path = match cache_path(provider, override_dir) {
        Some(p) => p,
        None => return fresh(),
    };
    if !path.exists() {
        return fresh();
    }
    let text = match fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) => {
            log::warn!("cache::load read failed ({}): {e}", path.display());
            return fresh();
        }
    };
    let rules = rules_version(provider);
    match serde_json::from_str::<CostUsageCache>(&text) {
        Ok(cache) if cache.version == CACHE_SCHEMA_VERSION && cache.rules_version == rules => cache,
        Ok(cache) => {
            log::info!(
                "cache::load({provider}) written under schema {} / rules {}, current is {CACHE_SCHEMA_VERSION} / {rules}; rebuilding",
                cache.version,
                cache.rules_version,
            );
            fresh()
        }
        Err(e) => {
            log::warn!(
                "cache::load parse failed ({}): {e} — starting fresh",
                path.display()
            );
            fresh()
        }
    }
}

pub fn save(
    provider: &str,
    cache: &CostUsageCache,
    override_dir: Option<&Path>,
) -> anyhow::Result<()> {
    let path = cache_path(provider, override_dir)
        .ok_or_else(|| anyhow::anyhow!("no cache dir available"))?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    // Atomic replace via tmp file — same pattern as Swift's NSFileManager.replaceItemAt.
    let tmp = path.with_extension("json.tmp");
    let text = serde_json::to_string(cache)?;
    fs::write(&tmp, text)?;
    fs::rename(&tmp, &path)?;
    Ok(())
}

#[allow(dead_code)]
pub fn wipe_all(override_dir: Option<&Path>) -> anyhow::Result<()> {
    if let Some(root) = cache_root(override_dir) {
        if root.exists() {
            fs::remove_dir_all(root)?;
        }
    }
    Ok(())
}

// ========================================================================
// Packed-slot arithmetic — mirrors Swift addPacked/applyFileDays/mergeFileDays.
// ========================================================================

/// Elementwise `a + sign*b`, clamped at 0. Handles mismatched slot counts
/// by treating missing slots as 0 in the shorter operand.
pub fn add_packed(a: &Packed, b: &Packed, sign: i64) -> Packed {
    let len = a.len().max(b.len());
    let mut out = vec![0i64; len];
    for (i, slot) in out.iter_mut().enumerate() {
        let av = a.get(i).copied().unwrap_or(0);
        let bv = b.get(i).copied().unwrap_or(0);
        *slot = (av + sign * bv).max(0);
    }
    out
}

/// Apply a file's per-day packed contribution to the aggregate `cache.days`,
/// either adding (`sign=+1`) when registering a file's parse output or
/// subtracting (`sign=-1`) when evicting a stale entry.
pub fn apply_file_days(
    cache: &mut CostUsageCache,
    file_days: &HashMap<String, HashMap<String, Packed>>,
    sign: i64,
) {
    for (day, models) in file_days {
        let day_models = cache.days.entry(day.clone()).or_default();
        for (model, packed) in models {
            let existing = day_models.get(model).cloned().unwrap_or_default();
            let merged = add_packed(&existing, packed, sign);
            if merged.iter().all(|v| *v == 0) {
                day_models.remove(model);
            } else {
                day_models.insert(model.clone(), merged);
            }
        }
        if day_models.is_empty() {
            cache.days.remove(day);
        }
    }
}

/// Add a parse-delta's per-day contribution onto an existing file entry's
/// `days` map (for the incremental-parse path — merge new tail's output
/// into what was already cached for the same file).
pub fn merge_file_days(
    existing: &mut HashMap<String, HashMap<String, Packed>>,
    delta: &HashMap<String, HashMap<String, Packed>>,
) {
    for (day, models) in delta {
        let day_models = existing.entry(day.clone()).or_default();
        for (model, packed) in models {
            let merged = add_packed(day_models.get(model).unwrap_or(&Vec::new()), packed, 1);
            if merged.iter().all(|v| *v == 0) {
                day_models.remove(model);
            } else {
                day_models.insert(model.clone(), merged);
            }
        }
        if day_models.is_empty() {
            existing.remove(day);
        }
    }
}

/// Drop days outside the current scan range from the aggregate — prevents
/// the cache from growing without bound. File-level `days` are also pruned.
pub fn prune_days(cache: &mut CostUsageCache, since_key: &str, until_key: &str) {
    cache
        .days
        .retain(|k, _| k.as_str() >= since_key && k.as_str() <= until_key);
    for entry in cache.files.values_mut() {
        entry
            .days
            .retain(|k, _| k.as_str() >= since_key && k.as_str() <= until_key);
    }
}

// ========================================================================
// Decision: what to do with a JSONL file on a scan
// ========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileAction {
    /// File's (mtime, size) match the cache — skip parsing, reuse days.
    Unchanged,
    /// File has grown (or was never cached) — parse incrementally from
    /// the last saved offset. If no prior offset, starts at 0.
    Incremental { start_offset: i64 },
    /// File shrank, mtime older than cached, or any other oddity —
    /// subtract the cached contribution and do a full re-parse.
    FullReparse,
}

pub fn decide_action(entry: Option<&FileEntry>, mtime_ms: i64, size: i64) -> FileAction {
    let entry = match entry {
        Some(e) => e,
        None => return FileAction::Incremental { start_offset: 0 },
    };
    if entry.mtime_unix_ms == mtime_ms && entry.size == size {
        return FileAction::Unchanged;
    }
    // File grew and mtime advanced (or same mtime but bigger, e.g. append without
    // stat update): try incremental from the last parsed offset.
    if size > entry.size {
        let off = entry.parsed_bytes.unwrap_or(entry.size);
        if off >= 0 && off <= size {
            return FileAction::Incremental { start_offset: off };
        }
    }
    FileAction::FullReparse
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_packed_basic() {
        let a = vec![10, 20, 30];
        let b = vec![1, 2, 3];
        assert_eq!(add_packed(&a, &b, 1), vec![11, 22, 33]);
        assert_eq!(add_packed(&a, &b, -1), vec![9, 18, 27]);
    }

    #[test]
    fn add_packed_clamps_at_zero() {
        let a = vec![5, 0];
        let b = vec![10, 0];
        assert_eq!(add_packed(&a, &b, -1), vec![0, 0]);
    }

    #[test]
    fn add_packed_handles_mismatched_lengths() {
        let a = vec![10, 20];
        let b = vec![1, 2, 3, 4];
        assert_eq!(add_packed(&a, &b, 1), vec![11, 22, 3, 4]);
    }

    #[test]
    fn decide_action_unchanged_when_mtime_and_size_match() {
        let entry = FileEntry {
            mtime_unix_ms: 1000,
            size: 500,
            days: HashMap::new(),
            parsed_bytes: Some(500),
            ..Default::default()
        };
        assert_eq!(
            decide_action(Some(&entry), 1000, 500),
            FileAction::Unchanged
        );
    }

    #[test]
    fn decide_action_incremental_when_file_grew() {
        let entry = FileEntry {
            mtime_unix_ms: 1000,
            size: 500,
            days: HashMap::new(),
            parsed_bytes: Some(500),
            ..Default::default()
        };
        let action = decide_action(Some(&entry), 2000, 800);
        assert_eq!(action, FileAction::Incremental { start_offset: 500 });
    }

    #[test]
    fn decide_action_full_reparse_when_file_shrank() {
        let entry = FileEntry {
            mtime_unix_ms: 1000,
            size: 500,
            days: HashMap::new(),
            parsed_bytes: Some(500),
            ..Default::default()
        };
        assert_eq!(
            decide_action(Some(&entry), 2000, 400),
            FileAction::FullReparse
        );
    }

    #[test]
    fn decide_action_new_file_starts_at_zero() {
        let action = decide_action(None, 1000, 500);
        assert_eq!(action, FileAction::Incremental { start_offset: 0 });
    }

    #[test]
    fn apply_file_days_plus_then_minus_is_zero() {
        let mut cache = CostUsageCache::default();
        let mut file_days: HashMap<String, HashMap<String, Packed>> = HashMap::new();
        let mut models = HashMap::new();
        models.insert("gpt-5".to_string(), vec![100, 0, 50]);
        file_days.insert("2026-04-24".to_string(), models);

        apply_file_days(&mut cache, &file_days, 1);
        assert_eq!(cache.days["2026-04-24"]["gpt-5"], vec![100, 0, 50]);

        apply_file_days(&mut cache, &file_days, -1);
        assert!(cache.days.is_empty()); // evicted when all slots hit 0
    }

    #[test]
    fn save_load_roundtrip() {
        let tmp = std::env::temp_dir().join(format!("cli-pulse-cache-test-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();

        let mut cache = CostUsageCache {
            last_scan_unix_ms: 1234567890,
            ..CostUsageCache::for_provider("codex")
        };
        let mut models = HashMap::new();
        models.insert("gpt-5".to_string(), vec![1000, 0, 200]);
        cache.days.insert("2026-04-24".to_string(), models);

        save("codex", &cache, Some(&tmp)).unwrap();
        let loaded = load("codex", Some(&tmp));
        assert_eq!(loaded.version, CACHE_SCHEMA_VERSION);
        assert_eq!(loaded.last_scan_unix_ms, 1234567890);
        assert_eq!(loaded.days["2026-04-24"]["gpt-5"], vec![1000, 0, 200]);

        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn codex_cache_from_older_rules_is_rebuilt_but_claude_is_kept() {
        let tmp = std::env::temp_dir().join(format!(
            "cli-pulse-cache-rules-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&tmp).unwrap();
        let seeded = || {
            let mut cache = CostUsageCache::default();
            let mut models = HashMap::new();
            models.insert("m".to_string(), vec![1000, 0, 200]);
            cache.days.insert("2026-04-24".to_string(), models);
            cache
        };

        // A Codex cache written before the rules version existed (reads as
        // 0) holds day-priced numbers: it must not be reused.
        save("codex", &seeded(), Some(&tmp)).unwrap();
        let codex = load("codex", Some(&tmp));
        assert!(codex.days.is_empty(), "stale Codex cache was reused");
        assert_eq!(codex.rules_version, rules_version("codex"));
        assert_eq!(rules_version("codex"), 1);

        // The same file on the Claude side is still current: no rescan.
        save("claude", &seeded(), Some(&tmp)).unwrap();
        let claude = load("claude", Some(&tmp));
        assert_eq!(claude.days["2026-04-24"]["m"], vec![1000, 0, 200]);

        // And a Codex cache written under the current rules is kept.
        let mut current = seeded();
        current.rules_version = rules_version("codex");
        save("codex", &current, Some(&tmp)).unwrap();
        assert_eq!(
            load("codex", Some(&tmp)).days["2026-04-24"]["m"],
            vec![1000, 0, 200]
        );

        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn older_cache_json_without_the_new_fields_still_parses() {
        // A file entry written before this version has none of the Codex
        // identity fields; they read as absent instead of failing the load.
        let json = r#"{"version":1,"last_scan_unix_ms":5,"files":{"/a.jsonl":{"mtime_unix_ms":1,"size":2,"days":{},"parsed_bytes":2,"last_model":null,"last_totals":null,"session_id":null}},"days":{}}"#;
        let cache: CostUsageCache = serde_json::from_str(json).unwrap();
        assert_eq!(cache.rules_version, 0);
        let entry = &cache.files["/a.jsonl"];
        assert!(entry.rollout_id.is_none() && entry.peak_totals.is_none());
        assert!(entry.first_event_ms.is_none() && entry.event_count.is_none());
    }

    #[test]
    fn load_handles_missing_file_gracefully() {
        let tmp =
            std::env::temp_dir().join(format!("cli-pulse-cache-missing-{}", std::process::id()));
        let cache = load("codex", Some(&tmp));
        assert_eq!(cache.version, CACHE_SCHEMA_VERSION);
        assert!(cache.files.is_empty());
        assert!(cache.days.is_empty());
    }

    /// v0.5.4 — `wipe_all` clears the cost-usage directory so the
    /// "Clear local caches" Danger Zone action is a true reset, not
    /// just a no-op (the layer above invalidates the in-memory
    /// `DASHBOARD_CACHE` separately). Pin the round-trip:
    /// save → wipe_all → load returns default (cache miss).
    #[test]
    fn wipe_all_removes_persisted_cache() {
        let tmp = std::env::temp_dir().join(format!("cli-pulse-wipe-test-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();

        // Seed a non-empty cache so wipe has something to actually clear.
        let mut cache = CostUsageCache::default();
        let mut models = HashMap::new();
        models.insert("gpt-5".to_string(), vec![1000, 0, 200]);
        cache.days.insert("2026-04-24".to_string(), models);
        save("codex", &cache, Some(&tmp)).unwrap();

        let path = cache_path("codex", Some(&tmp)).unwrap();
        assert!(path.exists(), "cache file should exist after save");

        wipe_all(Some(&tmp)).unwrap();

        // After wipe, load returns a fresh default — no v0.5.4 surprise
        // where users hit "Clear caches" but old days re-render.
        let loaded = load("codex", Some(&tmp));
        assert!(loaded.days.is_empty(), "load post-wipe must be empty");
        assert!(!path.exists(), "cache file must be removed by wipe_all");

        std::fs::remove_dir_all(&tmp).ok();
    }

    /// v0.5.4 — `wipe_all` is idempotent: clearing an already-empty
    /// directory must not error. Important because the Danger Zone
    /// action's caller treats wipe_all errors as a hard fail; an
    /// idempotent guarantee here means re-clicking "Clear caches" on
    /// a fresh install doesn't surface a confusing error.
    #[test]
    fn wipe_all_is_idempotent_on_missing_dir() {
        let tmp = std::env::temp_dir().join(format!("cli-pulse-wipe-empty-{}", std::process::id()));
        // Note: tmp NEVER created — wipe_all called against a non-
        // existent path. Must succeed, not error out.
        let result = wipe_all(Some(&tmp));
        assert!(
            result.is_ok(),
            "wipe_all on missing dir should be Ok, got {result:?}"
        );
    }

    #[test]
    fn prune_days_drops_out_of_range() {
        let mut cache = CostUsageCache::default();
        let make = || {
            let mut m = HashMap::new();
            m.insert("gpt-5".to_string(), vec![100, 0, 50]);
            m
        };
        cache.days.insert("2026-01-01".into(), make());
        cache.days.insert("2026-04-24".into(), make());
        cache.days.insert("2026-12-31".into(), make());

        prune_days(&mut cache, "2026-04-01", "2026-04-30");
        assert_eq!(cache.days.len(), 1);
        assert!(cache.days.contains_key("2026-04-24"));
    }

    #[test]
    fn compare_cache_dir_is_namespaced_beside_main_cache() {
        // Skip when the OS provides no cache dir (compare_cache_dir → None, and
        // the scan just runs uncached). Always Some on CI runners.
        if let (Some(dir), Some(main)) = (compare_cache_dir(), cache_path("codex", None)) {
            // The baseline lives in its own "compare" subfolder so it never
            // shares — and never gets pruned by — the main scan's cache.
            assert!(
                dir.ends_with("compare"),
                "expected .../compare, got {dir:?}"
            );
            // ...but under the SAME cost-usage root as the main provider caches
            // (dir is `<root>/compare`, main is `<root>/codex-vN.json`).
            assert_eq!(
                dir.parent(),
                main.parent(),
                "compare dir must sit beside the main provider caches"
            );
            // And it must NOT collide with a real provider cache file.
            assert_ne!(dir, main);
        }
    }
}
