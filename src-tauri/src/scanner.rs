//! JSONL log scanner for Codex (OpenAI) and Claude (Anthropic) CLI tools.
//!
//! Ported from Swift `CostUsageScanner` in the macOS app. Warm scans are
//! incremental: files whose (mtime, size) are unchanged since the last
//! run are skipped entirely; files that have grown are parsed only from
//! their previous size forward (via `cache::FileAction::Incremental`).
//! State lives in per-provider JSON caches at `cache::cache_path(...)`.
//!
//! Output schema matches what the Swift scanner produces; the macOS app
//! still uploads daily usage via `upsert_daily_usage`. Tauri-side daily
//! upload is paused in v0.2.14 and returns in v0.3.1 via a multi-device
//! aware path.
//!
//! Cost invariant (both providers):
//! Cost is the sum of per-request costs. NEVER compute cost from
//! day-aggregated tokens: Claude sonnet-4-5 / sonnet-4-6 have a 200K-token
//! tier and several Codex models a 272K long-context tier, and a day's total
//! crosses them when no single request does. Codex rates also depend on the
//! request's date (a repriced model keeps its old rate for older requests).
//! Per-request cost is accumulated during parse into `cost_nanos` (scaled by
//! 1e9): Claude packed slot [4], Codex packed slot [3].

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use chrono::{DateTime, Datelike, NaiveDate};
use serde::{Deserialize, Serialize};
use walkdir::WalkDir;

use crate::cache::{self, CodexTotals, CostUsageCache, FileAction, FileEntry, Packed};
use crate::paths;
use crate::pricing;
use crate::wsl::{classify_origin, Origin};

pub const CLAUDE_MSG_BUCKET_MODEL: &str = "__claude_msg__";
/// `cost_nanos` units per USD.
const COST_SCALE: f64 = 1_000_000_000.0;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DailyEntry {
    pub date: String,     // "YYYY-MM-DD" in local TZ
    pub provider: String, // "Codex" or "Claude"
    pub model: String,    // normalized model name
    pub input_tokens: i64,
    pub cached_tokens: i64,
    pub output_tokens: i64,
    pub cost_usd: Option<f64>,
    pub message_count: i64,
}

/// Per-origin usage totals (native Windows/macOS/Linux vs. a WSL distro). Lets
/// the UI surface the otherwise-silent WSL merge — Windows users running the
/// CLIs inside WSL can see how much of their usage comes from there. `tokens` is
/// I/O tokens (input + output, EXCLUDING cached) over the scan window, matching
/// `ScanResult.total_tokens` and the Overview so the figures reconcile.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OriginUsage {
    /// "native" or "wsl".
    pub kind: String,
    /// The distro name for `kind == "wsl"`; `None` for native.
    pub distro: Option<String>,
    pub tokens: i64,
    /// Number of tracked files contributing (with in-range tokens).
    pub files: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanResult {
    pub entries: Vec<DailyEntry>,
    pub total_cost_usd: f64,
    pub total_tokens: i64,
    pub today_key: String,
    pub days_scanned: u32,
    pub files_scanned: u32,
    /// Files present in the cache whose (mtime, size) matched and were
    /// therefore skipped entirely. Useful for benchmarking the incremental
    /// path and reported in the scan report.
    pub files_cached: u32,
    /// Usage split by physical origin (native vs. each WSL distro), derived
    /// from the cached file paths. Empty when nothing has usage; the UI only
    /// surfaces the split when a WSL entry is present. See `origin_usage`.
    #[serde(default)]
    pub origin_usage: Vec<OriginUsage>,
}

#[derive(Debug, Clone)]
pub struct ScanOptions {
    pub days: u32,
    pub force_rescan: bool,
    pub cache_dir: Option<PathBuf>,
    /// Test-only: override the Codex sessions roots scanned. Production
    /// callers leave this `None` so the platform-default paths are used.
    pub codex_roots_override: Option<Vec<PathBuf>>,
    /// Test-only: override the Claude projects roots scanned.
    pub claude_roots_override: Option<Vec<PathBuf>>,
    /// Test-only: pin "today" so the date-range derivation is deterministic
    /// regardless of the CI runner's clock or timezone. Production leaves
    /// this `None` and the scanner reads `chrono::Local::now()`.
    pub today_override: Option<NaiveDate>,
}

impl Default for ScanOptions {
    fn default() -> Self {
        Self {
            days: 30,
            force_rescan: false,
            cache_dir: None,
            codex_roots_override: None,
            claude_roots_override: None,
            today_override: None,
        }
    }
}

pub fn scan(days: u32) -> anyhow::Result<ScanResult> {
    scan_with_options(ScanOptions {
        days,
        ..Default::default()
    })
}

pub fn scan_with_options(opts: ScanOptions) -> anyhow::Result<ScanResult> {
    // IMPORTANT: per-event day classification (`parse_day_key_local`) converts
    // each JSONL timestamp into the *local* timezone before keying it. The
    // range filter MUST use the same local-clock anchor or events get
    // wrongly excluded near midnight in non-UTC timezones (e.g. JST users
    // between 00:00 and 09:00 local: UTC date trails local date by one,
    // so "today's" events were tagged 2026-04-25 by parse_day_key_local
    // but the filter range only ran through 2026-04-24 — entire morning of
    // usage went missing). Caught by Codex review post v0.2.1.
    let today = opts
        .today_override
        .unwrap_or_else(|| chrono::Local::now().date_naive());
    let since = today
        .checked_sub_signed(chrono::Duration::days(opts.days as i64))
        .unwrap_or(today);
    let range = DateRange {
        since_key: fmt_date(since),
        until_key: fmt_date(today),
    };
    let today_key = fmt_date(today);

    let (codex_cache, codex_files_scanned, codex_files_cached) =
        scan_codex_provider(&opts, &range)?;
    let (claude_cache, claude_files_scanned, claude_files_cached) =
        scan_claude_provider(&opts, &range)?;

    let entries = emit_entries(&codex_cache, &claude_cache, &range);
    let origin_usage = origin_usage(&codex_cache, &claude_cache, &range);

    let total_cost_usd: f64 = entries.iter().filter_map(|e| e.cost_usd).sum();
    let total_tokens: i64 = entries
        .iter()
        .map(|e| e.input_tokens + e.output_tokens)
        .sum();

    Ok(ScanResult {
        entries,
        total_cost_usd,
        total_tokens,
        today_key,
        days_scanned: opts.days,
        files_scanned: codex_files_scanned + claude_files_scanned,
        files_cached: codex_files_cached + claude_files_cached,
        origin_usage,
    })
}

/// Sum a single cached file's in-range **I/O tokens** (input + output). `io_slots`
/// are the packed indices holding input and output for that provider — Codex
/// `[input, cached, output]` → `[0, 2]`; Claude `[input, cache_read, cache_create,
/// output, cost_nanos, msgs]` → `[0, 3]`. Cached tokens are deliberately EXCLUDED
/// to match the app-wide "tokens" convention (`ScanResult.total_tokens` and the
/// Overview both count input+output only; cached is a separate metric), so the
/// origin split reconciles with every other token figure in the UI.
fn sum_file_tokens(entry: &FileEntry, io_slots: &[usize], range: &DateRange) -> i64 {
    let mut total = 0i64;
    for (day, models) in &entry.days {
        if !in_range(day, range) {
            continue;
        }
        for packed in models.values() {
            for &i in io_slots {
                total += packed.get(i).copied().unwrap_or(0);
            }
        }
    }
    total
}

/// Split I/O-token usage by physical origin (native vs. each WSL distro) by
/// walking the per-file cache entries and classifying each file's absolute path.
/// Because the cache retains every tracked file (including ones skipped as
/// unchanged on a warm scan), this reflects the full window without a re-parse
/// and needs no cache-schema change. Files with no in-range tokens (e.g. Claude's
/// synthetic message bucket) contribute nothing. Sorted: native first, then
/// distros by descending tokens then name, for a stable UI order.
fn origin_usage(
    codex_cache: &CostUsageCache,
    claude_cache: &CostUsageCache,
    range: &DateRange,
) -> Vec<OriginUsage> {
    // key: (kind, distro) -> (tokens, files). I/O slots per provider: Codex
    // input=0/output=2; Claude input=0/output=3 (cached + cost + msgs excluded).
    // A second copy of a Codex rollout is left out here exactly as it is left
    // out of the totals, so the split still adds up to them.
    let codex_copies = codex_duplicate_copies(codex_cache);
    let no_copies = HashSet::new();
    let mut acc: HashMap<(String, Option<String>), (i64, u32)> = HashMap::new();
    for (cache, io_slots, skip) in [
        (codex_cache, [0usize, 2usize].as_slice(), &codex_copies),
        (claude_cache, [0usize, 3usize].as_slice(), &no_copies),
    ] {
        for (path, entry) in &cache.files {
            if skip.contains(path) {
                continue;
            }
            let tokens = sum_file_tokens(entry, io_slots, range);
            if tokens == 0 {
                continue;
            }
            let key = match classify_origin(path) {
                Origin::Native => ("native".to_string(), None),
                Origin::Wsl(distro) => ("wsl".to_string(), Some(distro)),
            };
            let slot = acc.entry(key).or_insert((0, 0));
            slot.0 += tokens;
            slot.1 += 1;
        }
    }
    let mut out: Vec<OriginUsage> = acc
        .into_iter()
        .map(|((kind, distro), (tokens, files))| OriginUsage {
            kind,
            distro,
            tokens,
            files,
        })
        .collect();
    out.sort_by(|a, b| {
        // native first, then WSL distros by tokens desc, then distro name.
        let a_native = a.kind == "native";
        let b_native = b.kind == "native";
        b_native
            .cmp(&a_native)
            .then(b.tokens.cmp(&a.tokens))
            .then(a.distro.cmp(&b.distro))
    });
    out
}

#[derive(Debug, Clone)]
struct DateRange {
    since_key: String,
    until_key: String,
}

fn fmt_date(d: NaiveDate) -> String {
    format!("{:04}-{:02}-{:02}", d.year(), d.month(), d.day())
}

fn parse_day_key_local(ts: &str) -> Option<String> {
    if let Ok(dt) = DateTime::parse_from_rfc3339(ts) {
        let local = dt.with_timezone(&chrono::Local);
        return Some(fmt_date(local.date_naive()));
    }
    if ts.len() >= 10 {
        if let Ok(d) = NaiveDate::parse_from_str(&ts[..10], "%Y-%m-%d") {
            return Some(fmt_date(d));
        }
    }
    None
}

fn in_range(day: &str, r: &DateRange) -> bool {
    day >= r.since_key.as_str() && day <= r.until_key.as_str()
}

fn now_unix_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

fn file_stat(path: &Path) -> Option<(i64, i64)> {
    let meta = path.metadata().ok()?;
    let size = meta.len() as i64;
    let mtime = meta
        .modified()
        .ok()?
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .ok()?;
    Some((mtime, size))
}

fn path_key(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

// ========================================================================
// Codex scanning
// ========================================================================

fn scan_codex_provider(
    opts: &ScanOptions,
    range: &DateRange,
) -> anyhow::Result<(CostUsageCache, u32, u32)> {
    let mut cache = if opts.force_rescan {
        CostUsageCache::for_provider("codex")
    } else {
        cache::load("codex", opts.cache_dir.as_deref())
    };

    let mut files_scanned = 0u32;
    let mut files_cached = 0u32;
    let mut seen_paths: std::collections::HashSet<String> = std::collections::HashSet::new();

    let roots: Vec<PathBuf> = opts
        .codex_roots_override
        .clone()
        .unwrap_or_else(paths::codex_sessions_roots);
    for root in roots {
        if !root.exists() {
            continue;
        }
        for walk_entry in WalkDir::new(&root).into_iter().filter_map(Result::ok) {
            if !walk_entry.file_type().is_file() {
                continue;
            }
            let p = walk_entry.path();
            if p.extension().and_then(|s| s.to_str()) != Some("jsonl") {
                continue;
            }
            if let Some(file_date) = codex_date_from_path(p, &root) {
                if !in_range(&file_date, range) {
                    continue;
                }
            }
            let key = path_key(p);
            seen_paths.insert(key.clone());

            let (mtime, size) = match file_stat(p) {
                Some(s) => s,
                None => continue,
            };

            let action = cache::decide_action(cache.files.get(&key), mtime, size);
            match action {
                FileAction::Unchanged => {
                    files_cached += 1;
                    continue;
                }
                FileAction::Incremental { start_offset } => {
                    let previous = cache.files.get(&key);
                    let resume = previous.map(CodexResume::from_entry).unwrap_or_default();
                    let mut merged_days = previous.map(|e| e.days.clone()).unwrap_or_default();
                    let parsed = parse_codex_file(p, range, start_offset, resume);
                    if !parsed.file_days.is_empty() {
                        cache::apply_file_days(&mut cache, &parsed.file_days, 1);
                    }
                    cache::merge_file_days(&mut merged_days, &parsed.file_days);
                    cache
                        .files
                        .insert(key.clone(), parsed.into_entry(mtime, size, merged_days));
                    files_scanned += 1;
                }
                FileAction::FullReparse => {
                    if let Some(old) = cache.files.get(&key).cloned() {
                        cache::apply_file_days(&mut cache, &old.days, -1);
                    }
                    let mut parsed = parse_codex_file(p, range, 0, CodexResume::default());
                    cache::apply_file_days(&mut cache, &parsed.file_days, 1);
                    let days = std::mem::take(&mut parsed.file_days);
                    cache
                        .files
                        .insert(key.clone(), parsed.into_entry(mtime, size, days));
                    files_scanned += 1;
                }
            }
        }
    }

    // Evict files that vanished from disk since last scan.
    let stale: Vec<String> = cache
        .files
        .keys()
        .filter(|k| !seen_paths.contains(*k))
        .cloned()
        .collect();
    for key in stale {
        if let Some(old) = cache.files.remove(&key) {
            cache::apply_file_days(&mut cache, &old.days, -1);
        }
    }

    cache::prune_days(&mut cache, &range.since_key, &range.until_key);

    // The aggregate is rebuilt from the files that count, so a second copy of
    // a rollout (see `codex_duplicate_copies`) never reaches it. The running
    // additions above keep the per-file bookkeeping uniform with Claude's; this
    // makes the aggregate exact whatever they did.
    let copies = codex_duplicate_copies(&cache);
    if !copies.is_empty() {
        log::info!(
            "codex: {} file(s) are copies of another tracked rollout and are counted once",
            copies.len()
        );
    }
    rebuild_days_from_files(&mut cache, &copies);

    cache.last_scan_unix_ms = now_unix_ms();
    if let Err(e) = cache::save("codex", &cache, opts.cache_dir.as_deref()) {
        log::warn!("cache::save(codex) failed: {e}");
    }
    Ok((cache, files_scanned, files_cached))
}

/// Replace the aggregate `cache.days` with the sum of every tracked file's own
/// days, leaving out the paths in `skip`.
fn rebuild_days_from_files(cache: &mut CostUsageCache, skip: &HashSet<String>) {
    let mut days: HashMap<String, HashMap<String, Packed>> = HashMap::new();
    for (path, entry) in &cache.files {
        if skip.contains(path) {
            continue;
        }
        cache::merge_file_days(&mut days, &entry.days);
    }
    cache.days = days;
}

/// Paths of Codex rollout files that are a second copy of another tracked
/// file, and so must not be counted.
///
/// Codex moves a finished rollout from `sessions/` to `archived_sessions/`, and
/// the scan walks both (and each WSL distro's `~/.codex`). A move is harmless,
/// but a rollout that exists in two places at once (copied instead of moved, a
/// restored or synced folder, a WSL home linked to the Windows one) would be
/// counted twice, because files are otherwise told apart only by path.
///
/// Two files are copies of one rollout when they carry the same rollout id
/// (`session_meta.payload.id`) AND their token events overlap in time. The id
/// alone is not enough: an editor can start a rollout in one file and continue
/// it in another under the same id, with no event in common, and both halves
/// are real usage. Files with no token events are never copies of anything.
///
/// Among overlapping copies the most complete one is kept: most token events,
/// then the larger final totals, then the first path in sort order.
///
/// Sub-agent rollouts are not affected: each has its own rollout id (their
/// `session_id` is the parent's, which is why this does not use it).
pub(crate) fn codex_duplicate_copies(cache: &CostUsageCache) -> HashSet<String> {
    let mut by_rollout: HashMap<&str, Vec<(&String, &FileEntry)>> = HashMap::new();
    for (path, entry) in &cache.files {
        if let (Some(id), Some(_), Some(_)) = (
            entry.rollout_id.as_deref(),
            entry.first_event_ms,
            entry.last_event_ms,
        ) {
            by_rollout.entry(id).or_default().push((path, entry));
        }
    }

    fn completeness(e: &FileEntry) -> (i64, i64, i64) {
        let totals = e.peak_totals.or(e.last_totals).unwrap_or_default();
        (e.event_count.unwrap_or(0), totals.input, totals.output)
    }

    let mut copies = HashSet::new();
    for (_, mut files) in by_rollout {
        if files.len() < 2 {
            continue;
        }
        files.sort_by(|(a_path, a), (b_path, b)| {
            completeness(b)
                .cmp(&completeness(a))
                .then_with(|| a_path.cmp(b_path))
        });
        let mut kept: Vec<(i64, i64)> = Vec::new();
        for (path, entry) in files {
            let (Some(start), Some(end)) = (entry.first_event_ms, entry.last_event_ms) else {
                continue;
            };
            if kept.iter().any(|&(s, e)| start <= e && s <= end) {
                copies.insert(path.clone());
            } else {
                kept.push((start, end));
            }
        }
    }
    copies
}

fn codex_date_from_path(path: &Path, root: &Path) -> Option<String> {
    let rel = path.strip_prefix(root).ok()?;
    let parts: Vec<&str> = rel.iter().filter_map(|s| s.to_str()).collect();
    if parts.len() < 3 {
        return None;
    }
    let (y, m, d) = (parts[0], parts[1], parts[2]);
    if y.len() == 4 && m.len() == 2 && d.len() == 2 {
        Some(format!("{y}-{m}-{d}"))
    } else {
        None
    }
}

/// Turns Codex's cumulative `total_token_usage` snapshots into per-request
/// deltas without ever counting the same tokens twice.
///
/// A rollout's counter normally only grows, and each snapshot minus the
/// previous one is the request's usage. Two things break that:
///
/// - **The counter restarts.** It drops to a small value and climbs again from
///   there; the requests after the drop are real usage.
/// - **Two counters interleave in one file** (for example several agents
///   writing to one rollout): the snapshots jump between a high and a low
///   series.
///
/// The old rule, `current - previous` with the baseline following every drop,
/// handles the restart but not the interleaving: each jump back up re-counts
/// the whole gap between the two series, so the usage is counted again on
/// every flip. Counting only growth above the highest snapshot seen (a pure
/// high-water mark) never re-counts, but it drops everything a restarted
/// counter does until it passes the old peak.
///
/// So each component is counted as:
/// - at or above the peak: growth above the peak;
/// - below the peak: growth since the previous snapshot, never negative.
///
/// A restarted counter keeps being counted, a jump back up to a higher series
/// cannot re-count the gap, a repeated snapshot counts zero, and no delta is
/// ever negative. It undercounts in two shapes: a lower series is not counted
/// on an event that directly follows a higher one, and a restarted counter
/// that climbs past the old peak loses the part of that one request that lies
/// below the peak.
///
/// What it does not try to recognise is a lower series that replays snapshots
/// already counted (copied history). A pure high-water mark would drop those,
/// but only by also dropping every restarted counter, and a restart is the
/// shape real logs show: after each drop we found, every later snapshot grew
/// by exactly that request's own `last_token_usage`.
///
/// The high-water mark follows CodexBar's `CodexTotalsTracker` (MIT; see the
/// notice in pricing.rs).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct CodexCounter {
    /// The previous snapshot.
    pub prev: Option<CodexTotals>,
    /// The highest snapshot seen so far, per component.
    pub peak: Option<CodexTotals>,
}

impl CodexCounter {
    /// A `total_token_usage` snapshot. Returns the tokens it adds.
    pub fn observe_total(&mut self, current: CodexTotals) -> CodexTotals {
        let delta = match (self.prev, self.peak.or(self.prev)) {
            (Some(prev), Some(peak)) => CodexTotals {
                input: counter_delta(current.input, prev.input, peak.input),
                cached: counter_delta(current.cached, prev.cached, peak.cached),
                output: counter_delta(current.output, prev.output, peak.output),
            },
            _ => clamp_totals(current),
        };
        self.advance_to(current);
        delta
    }

    /// An event with only `last_token_usage` (the request's own usage).
    /// Counted as is; the snapshot is advanced by it, so a later
    /// `total_token_usage` that already includes it does not count it again.
    pub fn observe_last(&mut self, last: CodexTotals) -> CodexTotals {
        let delta = clamp_totals(last);
        let advanced = match self.prev {
            Some(p) => CodexTotals {
                input: p.input + delta.input,
                cached: p.cached + delta.cached,
                output: p.output + delta.output,
            },
            None => delta,
        };
        self.advance_to(advanced);
        delta
    }

    fn advance_to(&mut self, snapshot: CodexTotals) {
        self.peak = Some(match self.peak.or(self.prev) {
            Some(peak) => CodexTotals {
                input: peak.input.max(snapshot.input),
                cached: peak.cached.max(snapshot.cached),
                output: peak.output.max(snapshot.output),
            },
            None => snapshot,
        });
        self.prev = Some(snapshot);
    }
}

fn counter_delta(current: i64, prev: i64, peak: i64) -> i64 {
    if current >= peak {
        current - peak
    } else {
        (current - prev).max(0)
    }
}

fn clamp_totals(t: CodexTotals) -> CodexTotals {
    CodexTotals {
        input: t.input.max(0),
        cached: t.cached.max(0),
        output: t.output.max(0),
    }
}

fn is_zero(t: &CodexTotals) -> bool {
    t.input == 0 && t.cached == 0 && t.output == 0
}

/// First and last token-event time of a whole file, and how many there were.
#[derive(Debug, Clone, Copy, Default)]
struct EventSpan {
    first_ms: Option<i64>,
    last_ms: Option<i64>,
    count: i64,
}

impl EventSpan {
    fn observe(&mut self, at_ms: Option<i64>) {
        self.count += 1;
        if let Some(t) = at_ms {
            self.first_ms = Some(self.first_ms.map_or(t, |f| f.min(t)));
            self.last_ms = Some(self.last_ms.map_or(t, |l| l.max(t)));
        }
    }
}

/// What a parse carries over from the part of a file already parsed.
#[derive(Debug, Clone, Default)]
struct CodexResume {
    model: Option<String>,
    counter: CodexCounter,
    span: EventSpan,
    session_id: Option<String>,
    rollout_id: Option<String>,
}

impl CodexResume {
    fn from_entry(e: &FileEntry) -> Self {
        Self {
            model: e.last_model.clone(),
            counter: CodexCounter {
                prev: e.last_totals,
                peak: e.peak_totals,
            },
            span: EventSpan {
                first_ms: e.first_event_ms,
                last_ms: e.last_event_ms,
                count: e.event_count.unwrap_or(0),
            },
            session_id: e.session_id.clone(),
            rollout_id: e.rollout_id.clone(),
        }
    }
}

struct CodexParseResult {
    parsed_bytes: i64,
    file_days: HashMap<String, HashMap<String, Packed>>,
    /// State at the end of the parse, including what was resumed from.
    state: CodexResume,
}

impl CodexParseResult {
    fn into_entry(
        self,
        mtime_unix_ms: i64,
        size: i64,
        days: HashMap<String, HashMap<String, Packed>>,
    ) -> FileEntry {
        let s = self.state;
        FileEntry {
            mtime_unix_ms,
            size,
            days,
            parsed_bytes: Some(self.parsed_bytes),
            last_model: s.model,
            last_totals: s.counter.prev,
            session_id: s.session_id,
            peak_totals: s.counter.peak,
            rollout_id: s.rollout_id,
            first_event_ms: s.span.first_ms,
            last_event_ms: s.span.last_ms,
            event_count: Some(s.span.count),
        }
    }
}

fn parse_unix_ms(ts: &str) -> Option<i64> {
    DateTime::parse_from_rfc3339(ts)
        .ok()
        .map(|d| d.timestamp_millis())
}

fn codex_totals(v: &serde_json::Value) -> CodexTotals {
    CodexTotals {
        input: json_i64(v, "input_tokens"),
        cached: json_i64_or(v, &["cached_input_tokens", "cache_read_input_tokens"]),
        output: json_i64(v, "output_tokens"),
    }
}

fn parse_codex_file(
    path: &Path,
    range: &DateRange,
    start_offset: i64,
    resume: CodexResume,
) -> CodexParseResult {
    let mut out = CodexParseResult {
        parsed_bytes: start_offset,
        file_days: HashMap::new(),
        state: resume,
    };

    let mut file = match File::open(path) {
        Ok(f) => f,
        Err(_) => return out,
    };
    if start_offset > 0 && file.seek(SeekFrom::Start(start_offset as u64)).is_err() {
        return out;
    }
    let mut reader = BufReader::with_capacity(256 * 1024, file);

    let state = &mut out.state;
    let mut bytes_seen: i64 = 0;
    let mut buf: Vec<u8> = Vec::with_capacity(4096);

    // IMPORTANT: don't use `reader.lines()` here — it strips `\r\n` AND `\n`
    // but doesn't tell us how many bytes were actually consumed. On Windows
    // CRLF JSONLs that under-counted by 1 byte per line, so the cached
    // `parsed_bytes` drifted and the next incremental scan would seek into
    // the middle of a line. read_until returns the exact byte count
    // including the terminator, which we strip ourselves.
    loop {
        buf.clear();
        let n = match reader.read_until(b'\n', &mut buf) {
            Ok(0) => break,
            Ok(n) => n,
            Err(_) => continue,
        };
        bytes_seen += n as i64;
        while matches!(buf.last(), Some(&b'\n') | Some(&b'\r')) {
            buf.pop();
        }
        if buf.is_empty() {
            continue;
        }
        let line: &str = match std::str::from_utf8(&buf) {
            Ok(s) => s,
            Err(_) => continue,
        };
        if !line.contains("\"type\":\"event_msg\"")
            && !line.contains("\"type\":\"turn_context\"")
            && !line.contains("\"type\":\"session_meta\"")
        {
            continue;
        }
        let obj: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let ty = obj.get("type").and_then(|v| v.as_str()).unwrap_or("");

        if ty == "session_meta" {
            if let Some(payload) = obj.get("payload") {
                if state.session_id.is_none() {
                    state.session_id = payload
                        .get("session_id")
                        .and_then(|v| v.as_str())
                        .or_else(|| payload.get("sessionId").and_then(|v| v.as_str()))
                        .or_else(|| payload.get("id").and_then(|v| v.as_str()))
                        .map(String::from);
                }
                if state.rollout_id.is_none() {
                    state.rollout_id = payload.get("id").and_then(|v| v.as_str()).map(String::from);
                }
            }
            continue;
        }

        if ty == "turn_context" {
            if let Some(payload) = obj.get("payload") {
                if let Some(m) = payload.get("model").and_then(|v| v.as_str()) {
                    state.model = Some(m.to_string());
                } else if let Some(info) = payload.get("info") {
                    if let Some(m) = info.get("model").and_then(|v| v.as_str()) {
                        state.model = Some(m.to_string());
                    }
                }
            }
            continue;
        }

        if ty != "event_msg" {
            continue;
        }
        let payload = match obj.get("payload") {
            Some(p) => p,
            None => continue,
        };
        if payload.get("type").and_then(|v| v.as_str()) != Some("token_count") {
            continue;
        }
        let ts = match obj.get("timestamp").and_then(|v| v.as_str()) {
            Some(t) => t,
            None => continue,
        };
        let day = match parse_day_key_local(ts) {
            Some(d) => d,
            None => continue,
        };

        let info = payload.get("info");
        let total = info.and_then(|i| i.get("total_token_usage"));
        let last = info.and_then(|i| i.get("last_token_usage"));
        // The counter advances on EVERY event, in the scan window or not. If
        // it only advanced on in-window events, the first in-window event of a
        // rollout that began before the window would be measured from zero,
        // and everything the rollout used before the window would land on the
        // window's first day.
        let delta = if let Some(total) = total {
            state.counter.observe_total(codex_totals(total))
        } else if let Some(last) = last {
            state.counter.observe_last(codex_totals(last))
        } else {
            continue;
        };
        let at_ms = parse_unix_ms(ts);
        state.span.observe(at_ms);

        if !in_range(&day, range) || is_zero(&delta) {
            continue;
        }

        let model = info
            .and_then(|i| i.get("model").and_then(|v| v.as_str()))
            .or_else(|| info.and_then(|i| i.get("model_name").and_then(|v| v.as_str())))
            .or_else(|| payload.get("model").and_then(|v| v.as_str()))
            .or_else(|| obj.get("model").and_then(|v| v.as_str()))
            .map(String::from)
            .or_else(|| state.model.clone())
            .unwrap_or_else(|| "gpt-5".to_string());

        let cached = delta.cached.min(delta.input);
        // Priced per request, at the rates of the request's own time: the
        // long-context tier is a property of one request, and a repriced
        // model's rate depends on when the request was made.
        let cost_nanos = pricing::codex_cost_usd(&model, delta.input, cached, delta.output, at_ms)
            .map(|c| (c * COST_SCALE).round() as i64)
            .unwrap_or(0);

        let norm_model = pricing::normalize_codex_model(&model);
        let day_models = out.file_days.entry(day).or_default();
        let packed = day_models
            .entry(norm_model)
            .or_insert_with(|| vec![0, 0, 0, 0]);
        while packed.len() < 4 {
            packed.push(0);
        }
        packed[0] += delta.input;
        packed[1] += cached;
        packed[2] += delta.output;
        packed[3] += cost_nanos;
    }

    out.parsed_bytes = start_offset + bytes_seen;
    out
}

// ========================================================================
// Claude scanning
// ========================================================================

fn scan_claude_provider(
    opts: &ScanOptions,
    range: &DateRange,
) -> anyhow::Result<(CostUsageCache, u32, u32)> {
    let mut cache = if opts.force_rescan {
        CostUsageCache::for_provider("claude")
    } else {
        cache::load("claude", opts.cache_dir.as_deref())
    };

    let mut files_scanned = 0u32;
    let mut files_cached = 0u32;
    let mut seen_paths: std::collections::HashSet<String> = std::collections::HashSet::new();

    let roots: Vec<PathBuf> = opts
        .claude_roots_override
        .clone()
        .unwrap_or_else(paths::claude_projects_roots);
    for root in roots {
        if !root.exists() {
            continue;
        }
        for walk_entry in WalkDir::new(&root).into_iter().filter_map(Result::ok) {
            if !walk_entry.file_type().is_file() {
                continue;
            }
            let p = walk_entry.path();
            if p.extension().and_then(|s| s.to_str()) != Some("jsonl") {
                continue;
            }
            let key = path_key(p);
            seen_paths.insert(key.clone());

            let (mtime, size) = match file_stat(p) {
                Some(s) => s,
                None => continue,
            };

            let action = cache::decide_action(cache.files.get(&key), mtime, size);
            match action {
                FileAction::Unchanged => {
                    files_cached += 1;
                    continue;
                }
                FileAction::Incremental { start_offset } => {
                    let parsed = parse_claude_file(p, range, start_offset);
                    if !parsed.file_days.is_empty() {
                        cache::apply_file_days(&mut cache, &parsed.file_days, 1);
                    }
                    let mut merged_days = cache
                        .files
                        .get(&key)
                        .map(|e| e.days.clone())
                        .unwrap_or_default();
                    cache::merge_file_days(&mut merged_days, &parsed.file_days);
                    cache.files.insert(
                        key.clone(),
                        FileEntry {
                            mtime_unix_ms: mtime,
                            size,
                            days: merged_days,
                            parsed_bytes: Some(parsed.parsed_bytes),
                            ..Default::default()
                        },
                    );
                    files_scanned += 1;
                }
                FileAction::FullReparse => {
                    if let Some(old) = cache.files.get(&key).cloned() {
                        cache::apply_file_days(&mut cache, &old.days, -1);
                    }
                    let parsed = parse_claude_file(p, range, 0);
                    cache::apply_file_days(&mut cache, &parsed.file_days, 1);
                    cache.files.insert(
                        key.clone(),
                        FileEntry {
                            mtime_unix_ms: mtime,
                            size,
                            days: parsed.file_days,
                            parsed_bytes: Some(parsed.parsed_bytes),
                            ..Default::default()
                        },
                    );
                    files_scanned += 1;
                }
            }
        }
    }

    let stale: Vec<String> = cache
        .files
        .keys()
        .filter(|k| !seen_paths.contains(*k))
        .cloned()
        .collect();
    for key in stale {
        if let Some(old) = cache.files.remove(&key) {
            cache::apply_file_days(&mut cache, &old.days, -1);
        }
    }

    cache::prune_days(&mut cache, &range.since_key, &range.until_key);
    cache.last_scan_unix_ms = now_unix_ms();
    if let Err(e) = cache::save("claude", &cache, opts.cache_dir.as_deref()) {
        log::warn!("cache::save(claude) failed: {e}");
    }
    Ok((cache, files_scanned, files_cached))
}

struct ClaudeParseResult {
    parsed_bytes: i64,
    file_days: HashMap<String, HashMap<String, Packed>>,
}

fn parse_claude_file(path: &Path, range: &DateRange, start_offset: i64) -> ClaudeParseResult {
    let mut out = ClaudeParseResult {
        parsed_bytes: start_offset,
        file_days: HashMap::new(),
    };

    let mut file = match File::open(path) {
        Ok(f) => f,
        Err(_) => return out,
    };
    if start_offset > 0 && file.seek(SeekFrom::Start(start_offset as u64)).is_err() {
        return out;
    }
    let mut reader = BufReader::with_capacity(256 * 1024, file);

    let mut seen_keys: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut bytes_seen: i64 = 0;
    let mut buf: Vec<u8> = Vec::with_capacity(4096);

    // CRLF-safe line iteration. See parse_codex_file for why we don't use
    // `reader.lines()`.
    loop {
        buf.clear();
        let n = match reader.read_until(b'\n', &mut buf) {
            Ok(0) => break,
            Ok(n) => n,
            Err(_) => continue,
        };
        bytes_seen += n as i64;
        while matches!(buf.last(), Some(&b'\n') | Some(&b'\r')) {
            buf.pop();
        }
        if buf.is_empty() {
            continue;
        }
        let line: &str = match std::str::from_utf8(&buf) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let is_assistant = line.contains("\"type\":\"assistant\"");
        let is_user = line.contains("\"type\":\"user\"");
        if !is_assistant && !is_user {
            continue;
        }

        let obj: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let ty = obj.get("type").and_then(|v| v.as_str()).unwrap_or("");
        let ts = match obj.get("timestamp").and_then(|v| v.as_str()) {
            Some(t) => t,
            None => continue,
        };
        let day = match parse_day_key_local(ts) {
            Some(d) => d,
            None => continue,
        };
        if !in_range(&day, range) {
            continue;
        }

        if ty == "user" {
            bump_msg(&mut out.file_days, &day);
            continue;
        }

        // type == "assistant" — always count against msg bucket even
        // for streaming chunks (matches Claude Code UI semantics).
        bump_msg(&mut out.file_days, &day);

        let message = match obj.get("message") {
            Some(m) => m,
            None => continue,
        };
        let model = match message.get("model").and_then(|v| v.as_str()) {
            Some(m) => m.to_string(),
            None => continue,
        };
        let usage = match message.get("usage") {
            Some(u) => u,
            None => continue,
        };

        // Token dedup: streaming chunks re-report cumulative usage —
        // count each (message.id, requestId) only once for tokens.
        let message_id = message.get("id").and_then(|v| v.as_str());
        let request_id = obj.get("requestId").and_then(|v| v.as_str());
        if let (Some(mid), Some(rid)) = (message_id, request_id) {
            let key = format!("{mid}:{rid}");
            if !seen_keys.insert(key) {
                continue;
            }
        }

        let input = json_i64(usage, "input_tokens").max(0);
        let cache_create = json_i64(usage, "cache_creation_input_tokens").max(0);
        let cache_read = json_i64(usage, "cache_read_input_tokens").max(0);
        let output = json_i64(usage, "output_tokens").max(0);
        if input == 0 && cache_create == 0 && cache_read == 0 && output == 0 {
            continue;
        }

        // Per-message cost (bit-exact Swift parity for tiered pricing).
        let cost_nanos = pricing::claude_cost_usd(&model, input, cache_read, cache_create, output)
            .map(|c| (c * COST_SCALE).round() as i64)
            .unwrap_or(0);

        let norm_model = pricing::normalize_claude_model(&model);
        let day_models = out.file_days.entry(day).or_default();
        let packed = day_models
            .entry(norm_model)
            .or_insert_with(|| vec![0, 0, 0, 0, 0, 0]);
        while packed.len() < 6 {
            packed.push(0);
        }
        packed[0] += input;
        packed[1] += cache_read;
        packed[2] += cache_create;
        packed[3] += output;
        packed[4] += cost_nanos;
        // slot 5 already bumped for this event via bump_msg? No — bump_msg
        // goes against the synthetic bucket. Per-model msg is a dedup-safe
        // counter that matches Swift's msgDelta=0 branch for token lines.
    }

    out.parsed_bytes = start_offset + bytes_seen;
    out
}

fn bump_msg(file_days: &mut HashMap<String, HashMap<String, Packed>>, day: &str) {
    let day_models = file_days.entry(day.to_string()).or_default();
    let packed = day_models
        .entry(CLAUDE_MSG_BUCKET_MODEL.to_string())
        .or_insert_with(|| vec![0, 0, 0, 0, 0, 0]);
    while packed.len() < 6 {
        packed.push(0);
    }
    packed[5] += 1;
}

// ========================================================================
// Emission — walk per-provider cache.days and build DailyEntries.
// ========================================================================

fn emit_entries(
    codex_cache: &CostUsageCache,
    claude_cache: &CostUsageCache,
    range: &DateRange,
) -> Vec<DailyEntry> {
    let mut out: Vec<DailyEntry> = Vec::new();

    // Codex: [input, cached, output, cost_nanos] — cost summed per request
    // during parse. A model without rates has no cost (None), not $0.
    for (day, models) in &codex_cache.days {
        if !in_range(day, range) {
            continue;
        }
        for (model, packed) in models {
            let input = packed.first().copied().unwrap_or(0);
            let cached = packed.get(1).copied().unwrap_or(0);
            let output = packed.get(2).copied().unwrap_or(0);
            let cost_nanos = packed.get(3).copied().unwrap_or(0);
            if input == 0 && cached == 0 && output == 0 {
                continue;
            }
            let cost = if pricing::codex_model_is_priced(model) {
                Some(cost_nanos as f64 / COST_SCALE)
            } else {
                None
            };
            out.push(DailyEntry {
                date: day.clone(),
                provider: "Codex".into(),
                model: model.clone(),
                input_tokens: input,
                cached_tokens: cached,
                output_tokens: output,
                cost_usd: cost,
                message_count: 0,
            });
        }
    }

    // Claude: [input, cache_read, cache_create, output, cost_nanos, msgs]
    for (day, models) in &claude_cache.days {
        if !in_range(day, range) {
            continue;
        }
        for (model, packed) in models {
            let input = packed.first().copied().unwrap_or(0);
            let cache_read = packed.get(1).copied().unwrap_or(0);
            let cache_create = packed.get(2).copied().unwrap_or(0);
            let output = packed.get(3).copied().unwrap_or(0);
            let cost_nanos = packed.get(4).copied().unwrap_or(0);
            let msgs = packed.get(5).copied().unwrap_or(0);
            // Emit when there's any real token activity OR the synthetic
            // bucket has a non-zero msg count (per v1.9.4 invariant).
            if input == 0 && cache_read == 0 && cache_create == 0 && output == 0 && msgs == 0 {
                continue;
            }
            let cost = if model == CLAUDE_MSG_BUCKET_MODEL {
                None
            } else if cost_nanos > 0 {
                Some(cost_nanos as f64 / COST_SCALE)
            } else {
                pricing::claude_cost_usd(model, input, cache_read, cache_create, output)
            };
            out.push(DailyEntry {
                date: day.clone(),
                provider: "Claude".into(),
                model: model.clone(),
                input_tokens: input,
                cached_tokens: cache_read + cache_create,
                output_tokens: output,
                cost_usd: cost,
                message_count: msgs,
            });
        }
    }

    out.sort_by(|a, b| {
        a.date
            .cmp(&b.date)
            .then(a.provider.cmp(&b.provider))
            .then(a.model.cmp(&b.model))
    });
    out
}

// ========================================================================
// JSON helpers
// ========================================================================

fn json_i64(v: &serde_json::Value, key: &str) -> i64 {
    v.get(key).and_then(|x| x.as_i64()).unwrap_or(0)
}

fn json_i64_or(v: &serde_json::Value, keys: &[&str]) -> i64 {
    for k in keys {
        if let Some(val) = v.get(*k).and_then(|x| x.as_i64()) {
            return val;
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression test for the timezone bug Codex caught post v0.2.1:
    /// `today` was anchored to `Utc::now()` while `today_key` and the
    /// per-event day classification used `Local::now()`. With the fix,
    /// today_key reflects whatever anchor we pin via `today_override`
    /// (in production: `chrono::Local::now()`). This is the FAST unit
    /// test — see `tests/scanner_integration.rs` for the full
    /// fixture-based regression that asserts the event survives the
    /// range filter.
    #[test]
    fn today_key_matches_today_override() {
        let tmp = std::env::temp_dir().join(format!(
            "cli-pulse-tz-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0),
        ));
        let pinned = chrono::NaiveDate::from_ymd_opt(2026, 1, 15).unwrap();
        let opts = ScanOptions {
            days: 1,
            force_rescan: true,
            cache_dir: Some(tmp.clone()),
            codex_roots_override: Some(vec![tmp.join("codex_empty")]),
            claude_roots_override: Some(vec![tmp.join("claude_empty")]),
            today_override: Some(pinned),
        };
        let result = scan_with_options(opts).expect("scan should succeed");
        assert_eq!(result.today_key, "2026-01-15");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn parse_day_key_local_handles_rfc3339() {
        // 2026-04-25T01:00:00Z: in JST (+09) → 2026-04-25 10:00 JST → "2026-04-25"
        // in PST (−07) → 2026-04-24 18:00 PST → "2026-04-24"
        // We can't pin the test TZ portably, but we can assert the function
        // returns *some* valid YYYY-MM-DD for a well-formed RFC3339 input.
        let day = parse_day_key_local("2026-04-25T01:00:00Z");
        assert!(day.is_some());
        let key = day.unwrap();
        assert_eq!(key.len(), 10);
        assert_eq!(&key[4..5], "-");
        assert_eq!(&key[7..8], "-");
    }

    #[test]
    fn parse_day_key_local_falls_back_to_prefix() {
        let day = parse_day_key_local("2026-04-25");
        assert_eq!(day.as_deref(), Some("2026-04-25"));
    }

    #[test]
    fn in_range_inclusive() {
        let r = DateRange {
            since_key: "2026-04-20".into(),
            until_key: "2026-04-25".into(),
        };
        assert!(in_range("2026-04-20", &r));
        assert!(in_range("2026-04-22", &r));
        assert!(in_range("2026-04-25", &r));
        assert!(!in_range("2026-04-19", &r));
        assert!(!in_range("2026-04-26", &r));
    }

    fn file_entry_with(day: &str, model: &str, packed: Vec<i64>) -> FileEntry {
        let mut models = HashMap::new();
        models.insert(model.to_string(), packed);
        let mut days = HashMap::new();
        days.insert(day.to_string(), models);
        FileEntry {
            days,
            ..Default::default()
        }
    }

    // ---- CodexCounter: cumulative snapshots → per-request deltas ----

    fn tot(input: i64, cached: i64, output: i64) -> CodexTotals {
        CodexTotals {
            input,
            cached,
            output,
        }
    }

    /// Sum of the input deltas `CodexCounter` produces for these snapshots.
    fn counted_input(snapshots: &[i64]) -> i64 {
        let mut c = CodexCounter::default();
        snapshots
            .iter()
            .map(|&i| c.observe_total(tot(i, 0, 0)).input)
            .sum()
    }

    /// The rule this replaced: baseline follows every snapshot.
    fn old_rule_input(snapshots: &[i64]) -> i64 {
        let mut prev: Option<i64> = None;
        let mut sum = 0;
        for &i in snapshots {
            sum += prev.map_or(i, |p| (i - p).max(0));
            prev = Some(i);
        }
        sum
    }

    /// A pure high-water mark: only growth above the highest snapshot.
    fn high_water_input(snapshots: &[i64]) -> i64 {
        let mut peak: Option<i64> = None;
        let mut sum = 0;
        for &i in snapshots {
            sum += peak.map_or(i, |p| (i - p).max(0));
            peak = Some(peak.map_or(i, |p| p.max(i)));
        }
        sum
    }

    #[test]
    fn counter_monotone_counts_plain_differences() {
        let mut c = CodexCounter::default();
        assert_eq!(c.observe_total(tot(100, 0, 10)), tot(100, 0, 10));
        assert_eq!(c.observe_total(tot(250, 100, 30)), tot(150, 100, 20));
        assert_eq!(c.observe_total(tot(400, 150, 70)), tot(150, 50, 40));
    }

    #[test]
    fn counter_repeated_snapshot_counts_zero() {
        let mut c = CodexCounter::default();
        c.observe_total(tot(500, 100, 50));
        assert_eq!(c.observe_total(tot(500, 100, 50)), tot(0, 0, 0));
        assert_eq!(c.observe_total(tot(500, 100, 50)), tot(0, 0, 0));
    }

    #[test]
    fn counter_interleaved_series_never_recount_the_gap() {
        // A high series (1000 → 1300) and a low one (100 → 130) written into
        // one file, alternating. Every jump back up to A is only A's growth.
        let snapshots = [1000, 100, 1100, 110, 1200, 120, 1300, 130];
        assert_eq!(counted_input(&snapshots), 1300);
        // The old rule re-counts the ~1000 gap on every jump back up.
        assert_eq!(old_rule_input(&snapshots), 1000 + 1000 + 1090 + 1180);
    }

    #[test]
    fn counter_restart_keeps_counting() {
        // The counter reaches 2000, restarts at 100 and climbs to 900. The
        // requests after the restart are real: 300 + 500 more.
        let snapshots = [1000, 2000, 100, 400, 900];
        assert_eq!(counted_input(&snapshots), 2800);
        // A pure high-water mark would drop all of it.
        assert_eq!(high_water_input(&snapshots), 2000);
    }

    #[test]
    fn counter_restart_passing_the_old_peak_counts_from_the_peak() {
        // After the restart the counter climbs past the old peak: the request
        // that crosses it is counted from the peak (the documented undercount),
        // and growth after that is counted normally.
        let snapshots = [2000, 100, 1500, 2300, 2600];
        assert_eq!(counted_input(&snapshots), 2000 + 1400 + 300 + 300);
    }

    #[test]
    fn counter_components_are_independent_and_never_negative() {
        let mut c = CodexCounter::default();
        c.observe_total(tot(1000, 800, 100));
        // Input and cached drop (restart), output grows past its peak.
        let d = c.observe_total(tot(50, 20, 130));
        assert_eq!(d, tot(0, 0, 30));
        let d = c.observe_total(tot(-5, -5, -5));
        assert_eq!(d, tot(0, 0, 0));
    }

    #[test]
    fn counter_last_only_event_is_not_counted_again_by_a_later_total() {
        let mut c = CodexCounter::default();
        assert_eq!(c.observe_total(tot(100, 0, 10)), tot(100, 0, 10));
        // An event with only last_token_usage (this request: 50 / 5).
        assert_eq!(c.observe_last(tot(50, 0, 5)), tot(50, 0, 5));
        // The next cumulative snapshot already includes those 50 / 5.
        assert_eq!(c.observe_total(tot(180, 0, 20)), tot(30, 0, 5));
    }

    #[test]
    fn counter_resumes_from_saved_state_like_a_full_pass() {
        let snapshots = [1000, 100, 1100, 150, 1200];
        let full = counted_input(&snapshots);
        // Parse the first two, save, resume for the rest.
        let mut c = CodexCounter::default();
        let first: i64 = snapshots[..2]
            .iter()
            .map(|&i| c.observe_total(tot(i, 0, 0)).input)
            .sum();
        let mut resumed = CodexCounter {
            prev: c.prev,
            peak: c.peak,
        };
        let rest: i64 = snapshots[2..]
            .iter()
            .map(|&i| resumed.observe_total(tot(i, 0, 0)).input)
            .sum();
        assert_eq!(first + rest, full);
        // Resuming without the peak (only the previous snapshot, as the cache
        // used to store) would re-count the gap.
        let mut peakless = CodexCounter {
            prev: c.prev,
            peak: None,
        };
        let rest_peakless: i64 = snapshots[2..]
            .iter()
            .map(|&i| peakless.observe_total(tot(i, 0, 0)).input)
            .sum();
        assert_ne!(first + rest_peakless, full);
    }

    // ---- codex_duplicate_copies ----

    fn rollout(id: &str, first: i64, last: i64, events: i64, final_input: i64) -> FileEntry {
        FileEntry {
            rollout_id: Some(id.to_string()),
            first_event_ms: Some(first),
            last_event_ms: Some(last),
            event_count: Some(events),
            peak_totals: Some(tot(final_input, 0, 0)),
            ..Default::default()
        }
    }

    fn cache_of(files: &[(&str, FileEntry)]) -> CostUsageCache {
        let mut cache = CostUsageCache::default();
        for (path, entry) in files {
            cache.files.insert(path.to_string(), entry.clone());
        }
        cache
    }

    #[test]
    fn identical_copies_of_one_rollout_count_once() {
        let copy = rollout("r1", 1_000, 5_000, 12, 9_000);
        let cache = cache_of(&[
            (
                "/h/.codex/sessions/2026/09/10/rollout-r1.jsonl",
                copy.clone(),
            ),
            ("/h/.codex/archived_sessions/rollout-r1.jsonl", copy),
        ]);
        let copies = codex_duplicate_copies(&cache);
        assert_eq!(copies.len(), 1);
        // Equal completeness: the first path in sort order is kept.
        assert!(copies.contains("/h/.codex/sessions/2026/09/10/rollout-r1.jsonl"));
    }

    #[test]
    fn the_more_complete_copy_is_kept() {
        // A stale copy taken mid-session (fewer events) and the full file.
        let cache = cache_of(&[
            ("/a/rollout-r1.jsonl", rollout("r1", 1_000, 3_000, 6, 4_000)),
            (
                "/b/rollout-r1.jsonl",
                rollout("r1", 1_000, 5_000, 12, 9_000),
            ),
        ]);
        let copies = codex_duplicate_copies(&cache);
        assert_eq!(
            copies.into_iter().collect::<Vec<_>>(),
            vec!["/a/rollout-r1.jsonl"]
        );
    }

    #[test]
    fn a_continuation_under_the_same_id_is_not_a_copy() {
        // Same rollout id, no overlap in time: a rollout continued in a second
        // file. Both halves are real usage.
        let cache = cache_of(&[
            ("/a/rollout-r1.jsonl", rollout("r1", 1_000, 3_000, 6, 4_000)),
            (
                "/b/rollout-r1b.jsonl",
                rollout("r1", 3_001, 9_000, 9, 5_000),
            ),
        ]);
        assert!(codex_duplicate_copies(&cache).is_empty());
    }

    #[test]
    fn different_rollouts_are_never_copies() {
        // Overlapping in time (a parent and its sub-agent, say) but with
        // different rollout ids.
        let cache = cache_of(&[
            (
                "/a/parent.jsonl",
                rollout("parent", 1_000, 9_000, 20, 50_000),
            ),
            ("/a/sub.jsonl", rollout("sub", 2_000, 4_000, 5, 3_000)),
        ]);
        assert!(codex_duplicate_copies(&cache).is_empty());
    }

    #[test]
    fn files_without_token_events_or_ids_are_left_alone() {
        let mut no_events = rollout("r1", 0, 0, 0, 0);
        no_events.first_event_ms = None;
        no_events.last_event_ms = None;
        let mut no_id = rollout("r1", 1_000, 5_000, 12, 9_000);
        no_id.rollout_id = None;
        let cache = cache_of(&[
            ("/a/shell.jsonl", no_events),
            ("/a/noid.jsonl", no_id),
            ("/a/real.jsonl", rollout("r1", 1_000, 5_000, 12, 9_000)),
        ]);
        assert!(codex_duplicate_copies(&cache).is_empty());
    }

    #[test]
    fn rebuilt_aggregate_leaves_out_copies() {
        let mut copy = rollout("r1", 1_000, 5_000, 12, 9_000);
        copy.days = HashMap::from([(
            "2026-09-10".to_string(),
            HashMap::from([("gpt-5.5".to_string(), vec![9_000, 1_000, 300, 42])]),
        )]);
        let mut cache = cache_of(&[("/a/r1.jsonl", copy.clone()), ("/b/r1.jsonl", copy)]);
        let copies = codex_duplicate_copies(&cache);
        rebuild_days_from_files(&mut cache, &copies);
        assert_eq!(
            cache.days["2026-09-10"]["gpt-5.5"],
            vec![9_000, 1_000, 300, 42]
        );
        // Without leaving the copy out the day would be doubled.
        rebuild_days_from_files(&mut cache, &HashSet::new());
        assert_eq!(
            cache.days["2026-09-10"]["gpt-5.5"],
            vec![18_000, 2_000, 600, 84]
        );
    }

    #[test]
    fn origin_usage_splits_native_and_wsl() {
        let range = DateRange {
            since_key: "2026-04-01".into(),
            until_key: "2026-04-30".into(),
        };
        let mut codex = CostUsageCache::default();
        // Native Codex [input, cached, output] → I/O = input+output = 100+30 = 130
        // (cached 20 EXCLUDED, matching the app-wide "tokens" definition).
        codex.files.insert(
            r"C:\Users\jason\.codex\sessions\a.jsonl".into(),
            file_entry_with("2026-04-10", "gpt-5", vec![100, 20, 30]),
        );
        // WSL Codex (Ubuntu): I/O = 10 + 5 = 15.
        codex.files.insert(
            r"\\wsl.localhost\Ubuntu\home\jason\.codex\sessions\b.jsonl".into(),
            file_entry_with("2026-04-11", "gpt-5", vec![10, 0, 5]),
        );

        let mut claude = CostUsageCache::default();
        // WSL Claude (Ubuntu): [input, cache_read, cache_create, output, cost_nanos,
        // msgs] → I/O = input+output = 40 + 20 = 60 (cache/cost/msgs excluded).
        claude.files.insert(
            r"\\wsl.localhost\Ubuntu\home\jason\.claude\projects\p\c.jsonl".into(),
            file_entry_with("2026-04-12", "sonnet", vec![40, 10, 5, 20, 999, 3]),
        );
        // Native Claude synthetic message bucket: no I/O tokens → excluded entirely.
        claude.files.insert(
            r"C:\Users\jason\.claude\projects\p\d.jsonl".into(),
            file_entry_with(
                "2026-04-13",
                CLAUDE_MSG_BUCKET_MODEL,
                vec![0, 0, 0, 0, 0, 7],
            ),
        );

        let out = origin_usage(&codex, &claude, &range);
        // Native (130 I/O tokens, 1 file) sorts first; WSL Ubuntu = 15 + 60 = 75 (2 files).
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].kind, "native");
        assert_eq!(out[0].distro, None);
        assert_eq!(out[0].tokens, 130);
        assert_eq!(out[0].files, 1);
        assert_eq!(out[1].kind, "wsl");
        assert_eq!(out[1].distro.as_deref(), Some("Ubuntu"));
        assert_eq!(out[1].tokens, 75);
        assert_eq!(out[1].files, 2);
    }

    #[test]
    fn origin_usage_excludes_out_of_range_and_empty() {
        let range = DateRange {
            since_key: "2026-04-01".into(),
            until_key: "2026-04-30".into(),
        };
        let mut codex = CostUsageCache::default();
        // Out-of-range day → contributes nothing.
        codex.files.insert(
            r"C:\x\a.jsonl".into(),
            file_entry_with("2026-01-01", "gpt-5", vec![100, 0, 0]),
        );
        let out = origin_usage(&codex, &CostUsageCache::default(), &range);
        assert!(out.is_empty());
    }
}
