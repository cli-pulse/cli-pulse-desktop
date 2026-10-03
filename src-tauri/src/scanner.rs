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

use crate::cache::{
    self, CodexChildMeta, CodexCopiedPrefix, CodexTotals, CostUsageCache, FileAction, FileEntry,
    Packed,
};
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
        // Debug, not info: a copy stays on disk, so this would repeat on
        // every scan.
        log::debug!(
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
/// (`session_meta.payload.id`) AND one file's token events lie within the time
/// span of the other's: a copy is the whole file or an earlier state of it.
/// The id alone is not enough: an editor can start a rollout in one file and
/// continue it in another under the same id, and both are real usage. Nor is
/// an overlap in time: two files of one rollout whose spans overlap without
/// one containing the other each hold events the other lacks. Files with no
/// token events are never copies of anything.
///
/// Files are taken most complete first (most token events, then the larger
/// final input + output, then the first path in sort order), and a file whose
/// span lies within one already taken is a copy. Same rule as the macOS app's
/// `CodexCopyResolver`.
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

    fn completeness(e: &FileEntry) -> (i64, i64) {
        let totals = e.last_totals.unwrap_or_default();
        (
            e.event_count.unwrap_or(0),
            totals.input.saturating_add(totals.output),
        )
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
            if kept.iter().any(|&(s, e)| s <= start && end <= e) {
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

/// Turns a rollout's `token_count` events into the tokens they add, the same
/// way the macOS app counts them (`CodexTokenAccountant` in CLIPulseCore), so
/// the two report the same Codex usage from the same logs.
///
/// Each event carries `total_token_usage`, a cumulative counter, and usually
/// `last_token_usage`, the request just made. What an event adds is the growth
/// of the cumulative counter over a baseline: the totals counted so far.
///
/// 1. **The baseline only rises.** A snapshot below it in any component is
///    skipped and leaves it where it is. Moving the baseline down to it, as
///    the scanner did before, counts the climb back up a second time: a file
///    whose counter flips between two series re-counted the gap on every flip.
///    Growth above the baseline is counted in full, even when it is more than
///    the event's own request: the counter also covers requests that wrote no
///    token event of their own (an aborted turn's). CodexBar counts the
///    request there instead; the shared case `gap_over_last_counts_the_growth`
///    pins the Mac's and the desktop's choice.
/// 2. **A counter carried over is not counted again.** When a file's first
///    event reports a cumulative total larger than its own request
///    (`last_token_usage`), the difference was counted before this file
///    began: by the rollout a fork continues, or by an earlier file of the
///    same rollout that this one picks up. It becomes the baseline. A fresh
///    counter's first total equals its first request, so this changes nothing
///    for it. Without `last_token_usage` the first total counts in full,
///    unless the file names a parent: then all of it is taken as carried.
///
/// These are the Mac's rules 1 and 3. Its rule 2, a sub-agent's or fork's
/// copy of its parent's history, is applied before an event reaches the
/// counter (`CodexResume::receive`): a copied event is neither counted nor
/// moves the baseline.
///
/// Rule 1 follows the monotonic watermark of CodexBar's `CodexTotalsTracker`
/// (MIT; see the notice in pricing.rs). Not verbatim: upstream latches an
/// "interleaved lineage" mode after a drop and then counts the smaller of the
/// request and the growth; this keeps the plain rule and skips the dropped
/// event outright, as the Mac does. Its cost: when a counter restarts inside
/// a file, the requests after the restart are counted only once the total
/// passes the old high. The Mac and CodexBar make the same choice, and the
/// shared case `counter_restart_counts_only_above_the_old_high` pins it, so
/// changing it is a decision for both apps at once.
///
/// An event with only `last_token_usage` counts as reported and moves the
/// baseline up by it, so a later cumulative total that includes it does not
/// count it again. (The Mac leaves its baseline alone here; the two differ
/// only for a file that mixes events with and without a cumulative total.)
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct CodexCounter {
    /// The totals counted so far; `None` before the file's first event.
    pub baseline: Option<CodexTotals>,
    /// The rollout names a parent (a sub-agent or a fork). Not saved: it is
    /// restored from the file's `CodexChildMeta`.
    pub child: bool,
}

impl CodexCounter {
    /// A `token_count` event with a `total_token_usage` snapshot and, when
    /// present, its `last_token_usage`. Returns the tokens it adds.
    pub fn observe_total(&mut self, total: CodexTotals, last: Option<CodexTotals>) -> CodexTotals {
        let total = clamp_totals(total);
        // Rule 2: the file's first event, continuing a counter from elsewhere.
        if self.baseline.is_none() {
            let carried = match last {
                Some(last) => saturating_sub(total, clamp_totals(last)),
                // Without `last` nothing shows how much of a child's first
                // total is its own, so all of it is taken as the parent's: at
                // worst one request is missed, never a parent's history
                // counted. A file that names no parent counts it.
                None if self.child => total,
                None => CodexTotals::default(),
            };
            if !is_zero(&carried) {
                self.baseline = Some(carried);
            }
        }
        // Rule 1: the baseline only rises.
        match self.baseline {
            Some(base)
                if total.input < base.input
                    || total.cached < base.cached
                    || total.output < base.output =>
            {
                CodexTotals::default()
            }
            Some(base) => {
                self.baseline = Some(total);
                saturating_sub(total, base)
            }
            None => {
                self.baseline = Some(total);
                total
            }
        }
    }

    /// An event with only `last_token_usage` (the request's own usage).
    pub fn observe_last(&mut self, last: CodexTotals) -> CodexTotals {
        let delta = clamp_totals(last);
        let base = self.baseline.unwrap_or_default();
        self.baseline = Some(CodexTotals {
            input: base.input.saturating_add(delta.input),
            cached: base.cached.saturating_add(delta.cached),
            output: base.output.saturating_add(delta.output),
        });
        delta
    }
}

fn saturating_sub(a: CodexTotals, b: CodexTotals) -> CodexTotals {
    CodexTotals {
        input: (a.input - b.input).max(0),
        cached: (a.cached - b.cached).max(0),
        output: (a.output - b.output).max(0),
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

/// One `token_count` event, with what is needed to file it once it counts.
/// An event can be held (`CodexResume::receive`) and counted later in the
/// same parse, so it carries the model in effect when it was written.
#[derive(Debug, Clone)]
struct CodexEvent {
    /// The local day it falls on.
    day: String,
    at_ms: Option<i64>,
    /// The line's own number (`ordinal`), when Codex wrote one.
    ordinal: Option<i64>,
    /// `total_token_usage`
    total: Option<CodexTotals>,
    /// `last_token_usage`: the request this event reports.
    last: Option<CodexTotals>,
    model: String,
}

/// What a parse carries over from the part of a file already parsed.
#[derive(Debug, Clone, Default)]
struct CodexResume {
    model: Option<String>,
    counter: CodexCounter,
    span: EventSpan,
    /// The file's first line has been read for its identity, whether or not
    /// it held a readable `session_meta`. Any later `session_meta` is an
    /// ancestor's, copied in with its history, and never replaces it.
    saw_meta: bool,
    session_id: Option<String>,
    rollout_id: Option<String>,
    /// Set when the rollout names a parent.
    child: Option<CodexChildMeta>,
    /// Events held until the copied part of the file is known. Never saved:
    /// every parse ends with `finish`, so none is left held.
    pending: Vec<CodexEvent>,
}

impl CodexResume {
    fn from_entry(e: &FileEntry) -> Self {
        Self {
            model: e.last_model.clone(),
            counter: CodexCounter {
                baseline: e.last_totals,
                child: e.codex_child.is_some(),
            },
            span: EventSpan {
                first_ms: e.first_event_ms,
                last_ms: e.last_event_ms,
                count: e.event_count.unwrap_or(0),
            },
            // A parse that got anywhere has read the first line.
            saw_meta: e.parsed_bytes.unwrap_or(0) > 0
                || e.session_id.is_some()
                || e.rollout_id.is_some()
                || e.codex_child.is_some(),
            session_id: e.session_id.clone(),
            rollout_id: e.rollout_id.clone(),
            child: e.codex_child,
            pending: Vec::new(),
        }
    }

    /// The file's first line, a `session_meta`: the rollout's own. Codex
    /// writes it there; a later one is an ancestor's (`observe_copied_session_meta`).
    fn observe_session_meta(&mut self, line: &serde_json::Value) {
        if self.saw_meta {
            return;
        }
        self.saw_meta = true;
        let Some(payload) = line.get("payload") else {
            return;
        };
        let text = |key: &str| payload.get(key).and_then(|v| v.as_str());
        self.session_id = text("session_id")
            .or_else(|| text("sessionId"))
            .or_else(|| text("id"))
            .map(String::from);
        self.rollout_id = text("id").map(String::from);
        let names_parent = text("parent_thread_id").is_some_and(|s| !s.is_empty())
            || text("forked_from_id").is_some_and(|s| !s.is_empty())
            || payload
                .get("source")
                .and_then(|s| s.get("subagent"))
                .is_some();
        if names_parent {
            let meta_ts =
                text("timestamp").or_else(|| line.get("timestamp").and_then(|v| v.as_str()));
            self.child = Some(CodexChildMeta {
                meta_ms: meta_ts.and_then(parse_unix_ms),
                history_start_ordinal: payload
                    .get("subagent_history_start_ordinal")
                    .and_then(|v| v.as_i64()),
                copied_prefix: None,
            });
            self.counter.child = true;
        }
    }

    /// The file's first line is not a readable `session_meta` (something
    /// else, or too long). Its identity stays unknown, and must: taking a
    /// later `session_meta` would take a parent's copied metadata for the
    /// file's own, and match a child against its parent as if it were a copy.
    fn observe_unreadable_first_line(&mut self) {
        self.saw_meta = true;
    }

    /// A child with a history boundary whose copied part is not known yet.
    fn awaits_copied_prefix_marker(&self) -> bool {
        matches!(
            self.child,
            Some(CodexChildMeta {
                history_start_ordinal: Some(_),
                copied_prefix: None,
                ..
            })
        )
    }

    /// A `session_meta` line after the first: an ancestor's, copied in with
    /// its history. Numbered before the boundary, it shows that the lines
    /// before the boundary are copied; held events are dropped.
    fn observe_copied_session_meta(&mut self, ordinal: Option<i64>) {
        self.observe_prefix_marker(ordinal, CodexCopiedPrefix::AncestorMetadata);
    }

    /// An `inter_agent_communication_metadata` line: a message from another
    /// agent. The first one before the boundary of a child with no copied
    /// `session_meta` ends the parent's replayed tail; held events are
    /// dropped.
    fn observe_inter_agent_message(&mut self, ordinal: Option<i64>) {
        self.observe_prefix_marker(ordinal, CodexCopiedPrefix::InterAgentMessage);
    }

    fn observe_prefix_marker(&mut self, ordinal: Option<i64>, marker: CodexCopiedPrefix) {
        if !self.awaits_copied_prefix_marker() {
            return;
        }
        let Some(child) = self.child.as_mut() else {
            return;
        };
        let Some(start) = child.history_start_ordinal else {
            return;
        };
        if ordinal.is_some_and(|line| line >= start) {
            return;
        }
        child.copied_prefix = Some(marker);
        self.pending.clear();
    }

    /// The events a `token_count` event makes count now, in log order, with
    /// what each adds: none while it is held, and the held ones first when it
    /// is the first past the boundary with no marker ahead of it.
    ///
    /// The Mac's rule 2. A sub-agent or fork rollout can begin with its
    /// parent's history copied in, token events included. Codex stamps the
    /// copied lines when it writes them, after the file's own `session_meta`,
    /// so their times say nothing; what marks them depends on which of two
    /// shapes the file has (`CodexCopiedPrefix`). The `session_meta` names the
    /// first line of the file's own history (`subagent_history_start_ordinal`)
    /// and every line carries its number (`ordinal`):
    /// - when an ancestor's `session_meta` was copied in ahead of that
    ///   boundary (a current rollout), an event numbered before it is the
    ///   parent's;
    /// - when none was (Codex's migration of older sub-agent rollouts moves
    ///   the boundary to the end of the file and drops the copied
    ///   `session_meta` lines), the boundary marks nothing: the events before
    ///   the parent's first inter-agent message are its replayed last
    ///   requests, and the rest is the sub-agent's own. Until one of the two
    ///   markers is seen, the events numbered before the boundary are held;
    ///   if neither comes before the end of the parse or a line past the
    ///   boundary, they count (`finish`).
    ///
    /// An event stamped earlier than the child's own `session_meta` is not
    /// its own either (`is_copied_history`).
    fn receive(&mut self, event: CodexEvent) -> Vec<(CodexEvent, CodexTotals)> {
        let mut counted = Vec::new();
        if self.awaits_copied_prefix_marker() {
            let start = self.child.and_then(|c| c.history_start_ordinal);
            if let (Some(start), Some(line)) = (start, event.ordinal) {
                if line < start {
                    self.pending.push(event);
                    return counted;
                }
                counted = self.finish();
            }
        }
        if let Some(delta) = self.count(&event) {
            counted.push((event, delta));
        }
        counted
    }

    /// No marker came for the held events: they are the file's own and
    /// count, in order. Called at the end of every parse, so a parse never
    /// leaves events held.
    fn finish(&mut self) -> Vec<(CodexEvent, CodexTotals)> {
        if self.pending.is_empty() {
            return Vec::new();
        }
        if let Some(child) = self.child.as_mut() {
            child.copied_prefix = Some(CodexCopiedPrefix::NoMarker);
        }
        let held = std::mem::take(&mut self.pending);
        held.into_iter()
            .filter_map(|event| self.count(&event).map(|delta| (event, delta)))
            .collect()
    }

    /// What one event adds, judged now, or `None` when it is copied history
    /// or no event at all. A copied event does not move the counter or the
    /// file's event span.
    fn count(&mut self, event: &CodexEvent) -> Option<CodexTotals> {
        if self.is_copied_history(event.ordinal, event.at_ms) {
            return None;
        }
        // The counter advances on EVERY event, in the scan window or not. If
        // it only advanced on in-window events, the first in-window event of a
        // rollout that began before the window would be measured from zero,
        // and everything the rollout used before the window would land on the
        // window's first day.
        let delta = match (event.total, event.last) {
            (Some(total), last) => self.counter.observe_total(total, last),
            (None, Some(last)) => self.counter.observe_last(last),
            (None, None) => return None,
        };
        self.span.observe(event.at_ms);
        Some(delta)
    }

    /// Whether a token event is part of the parent's history that a
    /// sub-agent or fork file begins with: numbered before the boundary once
    /// an ancestor's `session_meta` has shown the lines before it are copied,
    /// or stamped before the file's own `session_meta` (logs without the
    /// ordinal).
    fn is_copied_history(&self, ordinal: Option<i64>, at_ms: Option<i64>) -> bool {
        let Some(child) = self.child else {
            return false;
        };
        let before_start = child.copied_prefix == Some(CodexCopiedPrefix::AncestorMetadata)
            && matches!(
                (child.history_start_ordinal, ordinal),
                (Some(start), Some(line)) if line < start
            );
        let before_meta = matches!(
            (child.meta_ms, at_ms),
            (Some(meta), Some(at)) if at < meta
        );
        before_start || before_meta
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
            last_totals: s.counter.baseline,
            session_id: s.session_id,
            rollout_id: s.rollout_id,
            codex_child: s.child,
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

/// Reads the next line of a JSONL log into `buf` and returns how many bytes it
/// took, terminator included, or `None` when there is no complete line left.
///
/// A last line without a newline is returned only when it already parses: a
/// log's final line written without one. Anything else there is a line the
/// CLI is still writing. It is left unread and its bytes are not counted, so
/// the saved offset stays at its first byte and the next scan reads the whole
/// line once it is complete. Counting it here made the next incremental scan
/// start in the middle of that line, which never parses, and its usage was
/// lost for good. Same rule as the macOS app and CodexBar (#2168).
fn read_jsonl_line<R: BufRead>(reader: &mut R, buf: &mut Vec<u8>) -> Option<usize> {
    buf.clear();
    loop {
        match reader.read_until(b'\n', buf) {
            Ok(0) if buf.is_empty() => return None,
            Ok(_) if buf.last() == Some(&b'\n') => return Some(buf.len()),
            Ok(_) => {
                let complete =
                    serde_json::from_slice::<serde::de::IgnoredAny>(buf.trim_ascii_end()).is_ok();
                return complete.then_some(buf.len());
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(_) => return None,
        }
    }
}

/// The longest first line read for a rollout's identity. A longer one leaves
/// the identity unknown (the Mac's `codexFirstLineMaxBytes`).
const CODEX_FIRST_LINE_MAX_BYTES: usize = 1 << 20;
/// Lines longer than this are never decoded: only their head is looked at, to
/// recognise a copied `session_meta` or an inter-agent message. Token events
/// and turn contexts are far shorter. The Mac's limit (`prefixBytes`).
const CODEX_DECODED_LINE_MAX_BYTES: usize = 32 * 1024;
/// How much of a longer line is looked at (the Mac's `jsonlTruncatedHeadBytes`).
const CODEX_LONG_LINE_HEAD_BYTES: usize = 4096;

fn bytes_contain(haystack: &[u8], needle: &[u8]) -> bool {
    memchr::memmem::find(haystack, needle).is_some()
}

/// A JSONL line's own number (`"ordinal":N`), read from its first 512 bytes
/// without decoding the line: the lines it is needed for, an ancestor's
/// copied `session_meta`, can be far too long to decode, and Codex writes the
/// number near the start. The Mac's `codexLineOrdinal`.
fn codex_line_ordinal(line: &[u8]) -> Option<i64> {
    let head = &line[..line.len().min(512)];
    let key = b"\"ordinal\":";
    let found = head.windows(key.len()).position(|w| w == key)?;
    let mut rest = head[found + key.len()..]
        .iter()
        .copied()
        .skip_while(|b| matches!(b, b' ' | b'\t' | b'\r'))
        .peekable();
    let negative = rest.next_if_eq(&b'-').is_some();
    let mut value: i64 = 0;
    let mut digits = 0;
    while let Some(b) = rest.next_if(|b| b.is_ascii_digit()) {
        if digits == 18 {
            break;
        }
        value = value * 10 + i64::from(b - b'0');
        digits += 1;
    }
    (digits > 0).then_some(if negative { -value } else { value })
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

    let mut bytes_seen: i64 = 0;
    let mut buf: Vec<u8> = Vec::with_capacity(4096);
    let mut first_line = start_offset == 0;

    // IMPORTANT: don't use `reader.lines()` here — it strips `\r\n` AND `\n`
    // but doesn't tell us how many bytes were actually consumed. On Windows
    // CRLF JSONLs that under-counted by 1 byte per line, so the cached
    // `parsed_bytes` drifted and the next incremental scan would seek into
    // the middle of a line. `read_jsonl_line` returns the exact byte count
    // including the terminator, which we strip ourselves, and leaves a line
    // that is still being written for the next scan.
    while let Some(n) = read_jsonl_line(&mut reader, &mut buf) {
        bytes_seen += n as i64;
        if buf.last() == Some(&b'\n') {
            buf.pop();
        }
        // The line's length as the Mac measures it: everything before the
        // newline.
        let line_len = buf.len();
        while buf.last() == Some(&b'\r') {
            buf.pop();
        }

        // The file's identity comes from its first line and nowhere else.
        // When that line is something other than a session_meta, it is read
        // like any other line; the Mac skips it, but Codex always writes the
        // session_meta first, so the two differ only on a file Codex did not
        // write.
        if std::mem::take(&mut first_line) {
            let own_meta = (line_len <= CODEX_FIRST_LINE_MAX_BYTES
                && bytes_contain(&buf, b"\"type\":\"session_meta\""))
            .then(|| serde_json::from_slice::<serde_json::Value>(&buf).ok())
            .flatten()
            .filter(|obj| obj.get("type").and_then(|v| v.as_str()) == Some("session_meta"));
            match own_meta {
                Some(obj) => {
                    out.state.observe_session_meta(&obj);
                    continue;
                }
                None => out.state.observe_unreadable_first_line(),
            }
        }
        if buf.is_empty() {
            continue;
        }

        let long = line_len > CODEX_DECODED_LINE_MAX_BYTES;
        let head = if long {
            &buf[..buf.len().min(CODEX_LONG_LINE_HEAD_BYTES)]
        } else {
            &buf[..]
        };
        // A later session_meta is an ancestor's, copied in with its history.
        // It never gives the file its identity and is never decoded; where it
        // sits is what marks copied history.
        if bytes_contain(head, b"\"type\":\"session_meta\"") {
            out.state
                .observe_copied_session_meta(codex_line_ordinal(head));
            continue;
        }
        if out.state.awaits_copied_prefix_marker()
            && bytes_contain(head, b"\"type\":\"inter_agent_communication_metadata\"")
        {
            out.state
                .observe_inter_agent_message(codex_line_ordinal(head));
            continue;
        }
        if long {
            continue;
        }
        let is_event = bytes_contain(&buf, b"\"type\":\"event_msg\"");
        if is_event && !bytes_contain(&buf, b"\"token_count\"") {
            continue;
        }
        if !is_event && !bytes_contain(&buf, b"\"type\":\"turn_context\"") {
            continue;
        }
        let obj: serde_json::Value = match serde_json::from_slice(&buf) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let ty = obj.get("type").and_then(|v| v.as_str()).unwrap_or("");
        // A line without a timestamp is skipped, a turn context included.
        let ts = match obj.get("timestamp").and_then(|v| v.as_str()) {
            Some(t) => t,
            None => continue,
        };
        let day = match parse_day_key_local(ts) {
            Some(d) => d,
            None => continue,
        };

        if ty == "turn_context" {
            if let Some(payload) = obj.get("payload") {
                if let Some(m) = payload.get("model").and_then(|v| v.as_str()) {
                    out.state.model = Some(m.to_string());
                } else if let Some(info) = payload.get("info") {
                    if let Some(m) = info.get("model").and_then(|v| v.as_str()) {
                        out.state.model = Some(m.to_string());
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

        let info = payload.get("info");
        let usage = |key: &str| info.and_then(|i| i.get(key)).filter(|v| v.is_object());
        let total = usage("total_token_usage");
        let last = usage("last_token_usage");
        if total.is_none() && last.is_none() {
            continue;
        }
        let model = info
            .and_then(|i| i.get("model").and_then(|v| v.as_str()))
            .or_else(|| info.and_then(|i| i.get("model_name").and_then(|v| v.as_str())))
            .or_else(|| payload.get("model").and_then(|v| v.as_str()))
            .or_else(|| obj.get("model").and_then(|v| v.as_str()))
            .map(String::from)
            .or_else(|| out.state.model.clone())
            .unwrap_or_else(|| "gpt-5".to_string());
        let event = CodexEvent {
            day,
            at_ms: parse_unix_ms(ts),
            ordinal: obj.get("ordinal").and_then(|v| v.as_i64()),
            total: total.map(codex_totals),
            last: last.map(codex_totals),
            model,
        };
        for (event, delta) in out.state.receive(event) {
            file_codex_event(&mut out.file_days, range, &event, delta);
        }
    }
    // Events still held at the end of the parse count.
    for (event, delta) in out.state.finish() {
        file_codex_event(&mut out.file_days, range, &event, delta);
    }

    out.parsed_bytes = start_offset + bytes_seen;
    out
}

/// Add what one counted event adds to its day and model, priced at the
/// rates in force at its own time, with its own request deciding the
/// long-context tier (`pricing::codex_event_cost_usd`).
fn file_codex_event(
    file_days: &mut HashMap<String, HashMap<String, Packed>>,
    range: &DateRange,
    event: &CodexEvent,
    delta: CodexTotals,
) {
    if !in_range(&event.day, range) || is_zero(&delta) {
        return;
    }
    let cost_nanos = pricing::codex_event_cost_usd(
        &event.model,
        delta,
        event.last.map(clamp_totals),
        event.at_ms,
    )
    .map(|c| (c * COST_SCALE).round() as i64)
    .unwrap_or(0);

    let norm_model = pricing::normalize_codex_model(&event.model);
    let day_models = file_days.entry(event.day.clone()).or_default();
    let packed = day_models
        .entry(norm_model)
        .or_insert_with(|| vec![0, 0, 0, 0]);
    while packed.len() < 4 {
        packed.push(0);
    }
    // Saturating, so an absurd count in a corrupt log cannot overflow.
    packed[0] = packed[0].saturating_add(delta.input);
    packed[1] = packed[1].saturating_add(delta.cached.min(delta.input));
    packed[2] = packed[2].saturating_add(delta.output);
    packed[3] = packed[3].saturating_add(cost_nanos);
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
    while let Some(n) = read_jsonl_line(&mut reader, &mut buf) {
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

    // ---- CodexCounter: cumulative snapshots → counted tokens ----

    fn tot(input: i64, cached: i64, output: i64) -> CodexTotals {
        CodexTotals {
            input,
            cached,
            output,
        }
    }

    /// Sum of the input `CodexCounter` counts for these snapshots (no
    /// `last_token_usage`).
    fn counted_input(snapshots: &[i64]) -> i64 {
        let mut c = CodexCounter::default();
        snapshots
            .iter()
            .map(|&i| c.observe_total(tot(i, 0, 0), None).input)
            .sum()
    }

    /// The pre-1.56 rule: the baseline follows every snapshot.
    fn old_rule_input(snapshots: &[i64]) -> i64 {
        let mut prev: Option<i64> = None;
        let mut sum = 0;
        for &i in snapshots {
            sum += prev.map_or(i, |p| (i - p).max(0));
            prev = Some(i);
        }
        sum
    }

    /// A restart-aware alternative that neither the macOS app nor CodexBar
    /// uses: growth above the peak, else growth since the previous snapshot.
    fn restart_aware_input(snapshots: &[i64]) -> i64 {
        let (mut prev, mut peak): (Option<i64>, Option<i64>) = (None, None);
        let mut sum = 0;
        for &i in snapshots {
            sum += match (prev, peak) {
                (Some(p), Some(k)) if i < k => (i - p).max(0),
                (_, Some(k)) => i - k,
                _ => i,
            };
            prev = Some(i);
            peak = Some(peak.map_or(i, |k| k.max(i)));
        }
        sum
    }

    #[test]
    fn counter_monotone_counts_plain_differences() {
        let mut c = CodexCounter::default();
        assert_eq!(c.observe_total(tot(100, 0, 10), None), tot(100, 0, 10));
        assert_eq!(c.observe_total(tot(250, 100, 30), None), tot(150, 100, 20));
        assert_eq!(c.observe_total(tot(400, 150, 70), None), tot(150, 50, 40));
    }

    #[test]
    fn counter_repeated_snapshot_counts_zero() {
        let mut c = CodexCounter::default();
        c.observe_total(tot(500, 100, 50), None);
        assert_eq!(c.observe_total(tot(500, 100, 50), None), tot(0, 0, 0));
        assert_eq!(c.observe_total(tot(500, 100, 50), None), tot(0, 0, 0));
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
    fn counter_going_down_and_back_counts_the_climb_once() {
        // 100, 200, then down to 150 and up to 250 and 300: 300 in all.
        let snapshots = [100, 200, 150, 250, 300];
        assert_eq!(counted_input(&snapshots), 300);
        assert_eq!(old_rule_input(&snapshots), 350);
    }

    #[test]
    fn counter_restart_counts_only_above_the_old_high() {
        // The counter reaches 2000, restarts at 100 and climbs to 900: none
        // of it passes 2000, so nothing after the restart counts. Same as the
        // macOS app and CodexBar; the undercount is documented on the type.
        let snapshots = [1000, 2000, 100, 400, 900];
        assert_eq!(counted_input(&snapshots), 2000);
        // Past the old high, growth above it counts again.
        let snapshots = [2000, 100, 1500, 2300, 2600];
        assert_eq!(counted_input(&snapshots), 2000 + 300 + 300);
        // The restart-aware alternative counts the restarted requests, so
        // these shapes tell the two apart.
        assert_eq!(restart_aware_input(&[1000, 2000, 100, 400, 900]), 2800);
        assert_eq!(
            restart_aware_input(&[2000, 100, 1500, 2300, 2600]),
            2000 + 1400 + 300 + 300
        );
    }

    #[test]
    fn counter_skips_a_snapshot_below_the_baseline_in_any_component() {
        let mut c = CodexCounter::default();
        c.observe_total(tot(1000, 800, 100), None);
        // Input and cached drop, output grows: the whole event is skipped and
        // the baseline stays where it was.
        assert_eq!(c.observe_total(tot(50, 20, 130), None), tot(0, 0, 0));
        assert_eq!(c.baseline, Some(tot(1000, 800, 100)));
        // The next snapshot above it counts everything since the baseline,
        // including the output growth the skipped event carried.
        assert_eq!(
            c.observe_total(tot(1100, 900, 140), None),
            tot(100, 100, 40)
        );
        // A negative snapshot never counts and never moves the baseline down.
        assert_eq!(c.observe_total(tot(-5, -5, -5), None), tot(0, 0, 0));
        assert_eq!(c.baseline, Some(tot(1100, 900, 140)));
    }

    #[test]
    fn counter_first_event_carrying_a_counter_counts_only_its_own_request() {
        // A fork or a continuation file: its first total, 5700, carries the
        // 5000 counted elsewhere; its own request is the 700 in `last`.
        let mut c = CodexCounter::default();
        assert_eq!(
            c.observe_total(tot(5700, 4500, 340), Some(tot(700, 500, 40))),
            tot(700, 500, 40)
        );
        assert_eq!(
            c.observe_total(tot(6500, 5100, 380), Some(tot(800, 600, 40))),
            tot(800, 600, 40)
        );
        // A fresh counter's first total equals its request: nothing changes.
        let mut fresh = CodexCounter::default();
        assert_eq!(
            fresh.observe_total(tot(2000, 1600, 100), Some(tot(2000, 1600, 100))),
            tot(2000, 1600, 100)
        );
        // Without `last` there is nothing to tell the carried part from the
        // request, and the first total counts in full.
        let mut no_last = CodexCounter::default();
        assert_eq!(
            no_last.observe_total(tot(5700, 4500, 340), None),
            tot(5700, 4500, 340)
        );
        // Only the file's first event is checked: later a total above `last`
        // is the normal case.
        assert_eq!(
            no_last.observe_total(tot(6500, 5100, 380), Some(tot(800, 600, 40))),
            tot(800, 600, 40)
        );
    }

    #[test]
    fn counter_child_first_total_without_last_is_taken_as_carried() {
        // A sub-agent or fork whose first event has no `last_token_usage`:
        // nothing shows which part of the total is its own, so none of it is
        // counted, and the next event counts from there.
        let mut child = CodexCounter {
            child: true,
            ..Default::default()
        };
        assert_eq!(
            child.observe_total(tot(5000, 4000, 300), None),
            tot(0, 0, 0)
        );
        assert_eq!(
            child.observe_total(tot(5600, 4400, 330), None),
            tot(600, 400, 30)
        );
        // With `last`, a child is treated like any file.
        let mut with_last = CodexCounter {
            child: true,
            ..Default::default()
        };
        assert_eq!(
            with_last.observe_total(tot(5700, 4500, 340), Some(tot(700, 500, 40))),
            tot(700, 500, 40)
        );
    }

    // ---- Copied history (the Mac's rule 2) ----

    fn child_meta(start: Option<i64>, prefix: Option<CodexCopiedPrefix>) -> CodexResume {
        CodexResume {
            saw_meta: true,
            counter: CodexCounter {
                child: true,
                ..Default::default()
            },
            child: Some(CodexChildMeta {
                meta_ms: Some(10_000),
                history_start_ordinal: start,
                copied_prefix: prefix,
            }),
            ..Default::default()
        }
    }

    /// A token event at `at_ms` with a cumulative `total` input and its own
    /// request `last` (cached and output zero).
    fn ev(ordinal: Option<i64>, at_ms: i64, total: i64, last: i64) -> CodexEvent {
        CodexEvent {
            day: "2026-09-29".into(),
            at_ms: Some(at_ms),
            ordinal,
            total: Some(tot(total, 0, 0)),
            last: Some(tot(last, 0, 0)),
            model: "gpt-5.5".into(),
        }
    }

    fn inputs(counted: Vec<(CodexEvent, CodexTotals)>) -> Vec<(Option<i64>, i64)> {
        counted
            .into_iter()
            .map(|(e, d)| (e.ordinal, d.input))
            .collect()
    }

    #[test]
    fn copied_history_is_told_by_line_number_once_marked_or_by_time() {
        // An ancestor's session_meta ahead of the boundary marks the lines
        // before it as copied. Their time says nothing: Codex stamps copied
        // lines after the file's own session_meta.
        let marked = child_meta(Some(5), Some(CodexCopiedPrefix::AncestorMetadata));
        assert!(marked.is_copied_history(Some(3), Some(20_000)));
        assert!(marked.is_copied_history(Some(4), None));
        assert!(!marked.is_copied_history(Some(5), Some(20_000)));
        assert!(!marked.is_copied_history(None, Some(20_000)));
        // Without that marker the boundary marks nothing by itself: a
        // migrated sub-agent numbers all of its own work before it.
        for prefix in [
            None,
            Some(CodexCopiedPrefix::InterAgentMessage),
            Some(CodexCopiedPrefix::NoMarker),
        ] {
            let state = child_meta(Some(5), prefix);
            assert!(
                !state.is_copied_history(Some(3), Some(20_000)),
                "{prefix:?}"
            );
        }
        // An event stamped before the file's own session_meta is copied,
        // whatever its number.
        let unmarked = child_meta(None, None);
        assert!(unmarked.is_copied_history(Some(3), Some(9_999)));
        assert!(!unmarked.is_copied_history(Some(3), Some(10_000)));
        assert!(!unmarked.is_copied_history(Some(3), None));
        // A file that names no parent has no copied history.
        let root = CodexResume::default();
        assert!(!root.is_copied_history(Some(0), Some(0)));
    }

    #[test]
    fn an_ancestors_session_meta_drops_the_held_events() {
        let mut s = child_meta(Some(10), None);
        assert!(s.awaits_copied_prefix_marker());
        // Lines numbered before the boundary wait for a marker.
        assert!(s.receive(ev(Some(2), 20_000, 4000, 4000)).is_empty());
        assert_eq!(s.pending.len(), 1);
        // A copied session_meta numbered at or past the boundary is not one.
        s.observe_copied_session_meta(Some(10));
        assert!(s.awaits_copied_prefix_marker());
        // Numbered before it, it is: the held event was the parent's.
        s.observe_copied_session_meta(Some(1));
        assert_eq!(
            s.child.unwrap().copied_prefix,
            Some(CodexCopiedPrefix::AncestorMetadata)
        );
        assert!(s.pending.is_empty());
        // Later copied lines are dropped as they come; the file's own count,
        // its first one measured against what it carries over.
        assert!(s.receive(ev(Some(5), 20_100, 4500, 500)).is_empty());
        assert_eq!(
            inputs(s.receive(ev(Some(11), 20_200, 5200, 700))),
            vec![(Some(11), 700)]
        );
        assert!(s.finish().is_empty());
        assert_eq!(s.span.count, 1);
        // A later inter-agent message changes nothing once it is decided.
        s.observe_inter_agent_message(Some(3));
        assert_eq!(
            s.child.unwrap().copied_prefix,
            Some(CodexCopiedPrefix::AncestorMetadata)
        );
    }

    #[test]
    fn an_inter_agent_message_ends_a_migrated_sub_agents_replayed_tail() {
        // A migrated sub-agent: boundary at the end of the file, no copied
        // session_meta. It starts by replaying the parent's last request.
        let mut s = child_meta(Some(8), None);
        assert!(s.receive(ev(Some(2), 20_000, 4500, 2500)).is_empty());
        s.observe_inter_agent_message(Some(3));
        assert_eq!(
            s.child.unwrap().copied_prefix,
            Some(CodexCopiedPrefix::InterAgentMessage)
        );
        assert!(s.pending.is_empty(), "the replayed event was not dropped");
        // Everything after the message counts, though it is numbered before
        // the boundary; the first own event counts only its own request.
        assert_eq!(
            inputs(s.receive(ev(Some(6), 21_000, 5200, 700))),
            vec![(Some(6), 700)]
        );
        assert_eq!(
            inputs(s.receive(ev(Some(7), 22_000, 5500, 300))),
            vec![(Some(7), 300)]
        );
        // A marker without a number counts as before the boundary.
        let mut unnumbered = child_meta(Some(8), None);
        unnumbered.receive(ev(Some(2), 20_000, 4500, 2500));
        unnumbered.observe_inter_agent_message(None);
        assert!(unnumbered.pending.is_empty());
        // One numbered past the boundary is not a marker.
        let mut past = child_meta(Some(8), None);
        past.receive(ev(Some(2), 20_000, 4500, 2500));
        past.observe_inter_agent_message(Some(8));
        assert!(past.awaits_copied_prefix_marker());
        assert_eq!(past.pending.len(), 1);
    }

    #[test]
    fn held_events_count_in_order_when_no_marker_comes() {
        // No marker before a line past the boundary: the held events are the
        // file's own, and count first, in log order. Counting the later event
        // first would take its total for a carried counter and skip the rest.
        let mut s = child_meta(Some(5), None);
        assert!(s.receive(ev(Some(1), 20_000, 1000, 1000)).is_empty());
        assert!(s.receive(ev(Some(2), 20_100, 1500, 500)).is_empty());
        assert_eq!(
            inputs(s.receive(ev(Some(5), 20_200, 1800, 300))),
            vec![(Some(1), 1000), (Some(2), 500), (Some(5), 300)]
        );
        assert_eq!(
            s.child.unwrap().copied_prefix,
            Some(CodexCopiedPrefix::NoMarker)
        );
        // Decided: an event before the boundary now counts as it comes.
        assert_eq!(
            inputs(s.receive(ev(Some(3), 20_300, 2000, 200))),
            vec![(Some(3), 200)]
        );

        // Or at the end of the parse.
        let mut ended = child_meta(Some(9), None);
        ended.receive(ev(Some(1), 20_000, 1000, 1000));
        ended.receive(ev(Some(2), 20_100, 1500, 500));
        assert_eq!(
            inputs(ended.finish()),
            vec![(Some(1), 1000), (Some(2), 500)]
        );
        assert_eq!(
            ended.child.unwrap().copied_prefix,
            Some(CodexCopiedPrefix::NoMarker)
        );
        // Nothing held: the end of a parse decides nothing.
        let mut empty = child_meta(Some(9), None);
        assert!(empty.finish().is_empty());
        assert!(empty.awaits_copied_prefix_marker());

        // An event without a number is judged as it comes, and the held ones
        // keep waiting.
        let mut unnumbered = child_meta(Some(9), None);
        unnumbered.receive(ev(Some(1), 20_000, 1000, 1000));
        assert_eq!(
            inputs(unnumbered.receive(ev(None, 20_100, 1500, 500))),
            vec![(None, 500)]
        );
        assert_eq!(unnumbered.pending.len(), 1);

        // Without a boundary nothing is held.
        let mut no_boundary = child_meta(None, None);
        assert!(!no_boundary.awaits_copied_prefix_marker());
        assert_eq!(
            inputs(no_boundary.receive(ev(Some(1), 20_000, 1000, 1000))),
            vec![(Some(1), 1000)]
        );
        let mut root = CodexResume::default();
        assert!(!root.awaits_copied_prefix_marker());
        assert_eq!(
            inputs(root.receive(ev(Some(1), 20_000, 1000, 1000))),
            vec![(Some(1), 1000)]
        );
    }

    /// Write `lines` to a fresh file and return its path (in a fresh temp dir).
    fn temp_rollout(name: &str, lines: &[&str]) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "cli-pulse-scanner-unit-{name}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("rollout.jsonl");
        let mut body = String::new();
        for l in lines {
            body.push_str(l);
            body.push('\n');
        }
        std::fs::write(&path, body).unwrap();
        path
    }

    fn wide_range() -> DateRange {
        DateRange {
            since_key: "2026-01-01".into(),
            until_key: "2026-12-31".into(),
        }
    }

    const CHILD_META: &str = r#"{"timestamp":"2026-09-29T11:10:00Z","type":"session_meta","ordinal":0,"payload":{"id":"child","session_id":"parent","timestamp":"2026-09-29T11:10:00Z","source":{"subagent":{"thread_spawn":{"parent_thread_id":"parent"}}},"subagent_history_start_ordinal":5}}"#;
    const PARENT_META: &str = r#"{"timestamp":"2026-09-29T11:10:00.100Z","type":"session_meta","ordinal":1,"payload":{"id":"parent","source":"cli"}}"#;

    fn token_line(ordinal: i64, minute: u32, total: i64, last: i64) -> String {
        format!(
            r#"{{"timestamp":"2026-09-29T11:{minute:02}:30Z","type":"event_msg","ordinal":{ordinal},"payload":{{"type":"token_count","info":{{"total_token_usage":{{"input_tokens":{total},"cached_input_tokens":0,"output_tokens":0}},"last_token_usage":{{"input_tokens":{last},"cached_input_tokens":0,"output_tokens":0}}}}}}}}"#
        )
    }

    #[test]
    fn only_the_first_line_can_be_the_files_own_session_meta() {
        // The child's own session_meta, then its parent's, copied in.
        let path = temp_rollout("own-meta", &[CHILD_META, PARENT_META]);
        let parsed = parse_codex_file(&path, &wide_range(), 0, CodexResume::default());
        let state = &parsed.state;
        assert_eq!(state.rollout_id.as_deref(), Some("child"));
        assert_eq!(state.session_id.as_deref(), Some("parent"));
        assert!(state.counter.child);
        assert_eq!(
            state.child,
            Some(CodexChildMeta {
                meta_ms: parse_unix_ms("2026-09-29T11:10:00Z"),
                history_start_ordinal: Some(5),
                copied_prefix: Some(CodexCopiedPrefix::AncestorMetadata),
            })
        );
        // Resuming keeps all of it.
        let entry = parsed.into_entry(0, 0, HashMap::new());
        let resumed = CodexResume::from_entry(&entry);
        assert!(resumed.saw_meta && resumed.counter.child);
        assert_eq!(resumed.child, entry.codex_child);
        assert!(!resumed.awaits_copied_prefix_marker());

        // A first line that is not a session_meta leaves the identity unknown:
        // a later one is an ancestor's, and taking it would make the file a
        // copy of its parent. The first line itself is read as usual.
        let path = temp_rollout(
            "no-own-meta",
            &[
                &token_line(0, 11, 900, 900),
                PARENT_META,
                &token_line(2, 12, 1500, 600),
            ],
        );
        let parsed = parse_codex_file(&path, &wide_range(), 0, CodexResume::default());
        assert!(parsed.state.saw_meta);
        assert_eq!(parsed.state.rollout_id, None);
        assert_eq!(parsed.state.session_id, None);
        assert_eq!(parsed.state.child, None);
        assert_eq!(parsed.file_days["2026-09-29"]["gpt-5"][0], 1500);
        // Resumed past its first line, a file stays without one.
        let mut entry = parsed.into_entry(0, 0, HashMap::new());
        assert!(entry.rollout_id.is_none() && entry.codex_child.is_none());
        assert!(CodexResume::from_entry(&entry).saw_meta);
        // A file not parsed at all has not read it.
        entry.parsed_bytes = Some(0);
        assert!(!CodexResume::from_entry(&entry).saw_meta);
    }

    #[test]
    fn the_copied_part_decided_in_one_parse_holds_in_the_next() {
        // A current child rollout read in two parts: its own session_meta,
        // the ancestor's, a copied event; then more copied history and its
        // own work. The second parse must still leave the copy out.
        let first = [CHILD_META, PARENT_META, &token_line(2, 11, 4000, 4000)];
        let path = temp_rollout("resume-ancestor", &first);
        let parsed = parse_codex_file(&path, &wide_range(), 0, CodexResume::default());
        assert!(parsed.file_days.is_empty());
        let offset = parsed.parsed_bytes;
        let entry = parsed.into_entry(0, offset, HashMap::new());
        let rest = [token_line(3, 12, 4500, 500), token_line(5, 13, 5200, 700)];
        let mut body = std::fs::read_to_string(&path).unwrap();
        for l in &rest {
            body.push_str(l);
            body.push('\n');
        }
        std::fs::write(&path, body).unwrap();
        let resumed = parse_codex_file(
            &path,
            &wide_range(),
            offset,
            CodexResume::from_entry(&entry),
        );
        assert_eq!(resumed.file_days["2026-09-29"]["gpt-5"][0], 700);

        // A migrated sub-agent read in two parts, the replayed event at the
        // end of the first: the first parse cannot wait for a marker that is
        // not written yet, so it counts the event, and the second parse goes
        // on counting. The Mac does the same (it ends every read with
        // `finish`). Codex writes a migrated rollout in one go, so a scan
        // seldom sees it half written.
        let migrated_meta = CHILD_META.replace(
            "\"subagent_history_start_ordinal\":5",
            "\"subagent_history_start_ordinal\":9",
        );
        let first = [migrated_meta.as_str(), &token_line(2, 11, 4500, 2500)];
        let path = temp_rollout("resume-migrated", &first);
        let parsed = parse_codex_file(&path, &wide_range(), 0, CodexResume::default());
        assert_eq!(parsed.file_days["2026-09-29"]["gpt-5"][0], 2500);
        assert_eq!(
            parsed.state.child.unwrap().copied_prefix,
            Some(CodexCopiedPrefix::NoMarker)
        );
    }

    #[test]
    fn a_copied_session_meta_too_long_to_decode_still_marks_the_copy() {
        // Copied session_meta lines are often far over the 32 KB decode
        // limit; their head says what they are and where they sit.
        let long_parent_meta = format!(
            r#"{{"timestamp":"2026-09-29T11:10:00.100Z","type":"session_meta","ordinal":1,"payload":{{"id":"parent","base_instructions":{{"text":"{}"}}}}}}"#,
            "x".repeat(40 * 1024)
        );
        let path = temp_rollout(
            "long-meta",
            &[
                CHILD_META,
                &long_parent_meta,
                &token_line(2, 11, 4000, 4000),
                &token_line(5, 12, 4600, 600),
            ],
        );
        let parsed = parse_codex_file(&path, &wide_range(), 0, CodexResume::default());
        assert_eq!(
            parsed.state.child.unwrap().copied_prefix,
            Some(CodexCopiedPrefix::AncestorMetadata)
        );
        assert_eq!(parsed.file_days["2026-09-29"]["gpt-5"][0], 600);
        assert_eq!(parsed.state.rollout_id.as_deref(), Some("child"));
    }

    #[test]
    fn the_line_number_is_read_without_decoding_the_line() {
        assert_eq!(
            codex_line_ordinal(br#"{"type":"session_meta","ordinal":12,"payload":{}}"#),
            Some(12)
        );
        assert_eq!(codex_line_ordinal(br#"{"ordinal": -3,"x":1}"#), Some(-3));
        assert_eq!(codex_line_ordinal(br#"{"ordinal":"x"}"#), None);
        assert_eq!(codex_line_ordinal(br#"{"type":"session_meta"}"#), None);
        // Only the first 512 bytes are looked at.
        let far = format!(r#"{{"pad":"{}","ordinal":7}}"#, "x".repeat(600));
        assert_eq!(codex_line_ordinal(far.as_bytes()), None);
    }

    #[test]
    fn counter_last_only_event_is_not_counted_again_by_a_later_total() {
        let mut c = CodexCounter::default();
        assert_eq!(c.observe_total(tot(100, 0, 10), None), tot(100, 0, 10));
        // An event with only last_token_usage (this request: 50 / 5).
        assert_eq!(c.observe_last(tot(50, 0, 5)), tot(50, 0, 5));
        // The next cumulative snapshot already includes those 50 / 5.
        assert_eq!(c.observe_total(tot(180, 0, 20), None), tot(30, 0, 5));
    }

    #[test]
    fn counter_resumes_from_saved_state_like_a_full_pass() {
        let snapshots = [1000, 100, 1100, 150, 1200];
        let full = counted_input(&snapshots);
        assert_eq!(full, 1200);
        // Parse the first two, save the baseline, resume for the rest.
        let mut c = CodexCounter::default();
        let first: i64 = snapshots[..2]
            .iter()
            .map(|&i| c.observe_total(tot(i, 0, 0), None).input)
            .sum();
        let mut resumed = CodexCounter {
            baseline: c.baseline,
            child: false,
        };
        let rest: i64 = snapshots[2..]
            .iter()
            .map(|&i| resumed.observe_total(tot(i, 0, 0), None).input)
            .sum();
        assert_eq!(first + rest, full);
        // Resuming without it would count the next snapshot from zero.
        let mut forgotten = CodexCounter::default();
        let rest_forgotten: i64 = snapshots[2..]
            .iter()
            .map(|&i| forgotten.observe_total(tot(i, 0, 0), None).input)
            .sum();
        assert_ne!(first + rest_forgotten, full);
    }

    // ---- codex_duplicate_copies ----

    fn rollout(id: &str, first: i64, last: i64, events: i64, final_input: i64) -> FileEntry {
        FileEntry {
            rollout_id: Some(id.to_string()),
            first_event_ms: Some(first),
            last_event_ms: Some(last),
            event_count: Some(events),
            last_totals: Some(tot(final_input, 0, 0)),
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
    fn a_partial_overlap_is_not_a_copy() {
        // Same id, spans overlapping without one containing the other: each
        // file has events the other lacks, so both count.
        let cache = cache_of(&[
            (
                "/a/rollout-r1.jsonl",
                rollout("r1", 1_000, 20_000, 3, 3_000),
            ),
            (
                "/b/rollout-r1b.jsonl",
                rollout("r1", 15_000, 40_000, 2, 1_000),
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
