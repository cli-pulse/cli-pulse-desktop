//! End-to-end integration tests for the scanner against synthetic
//! JSONL fixtures.
//!
//! These tests deliberately avoid touching the user's real
//! `~/.claude/projects/` or `~/.codex/sessions/` — every fixture is
//! built fresh into a temp directory, scanned via
//! `ScanOptions::{codex,claude}_roots_override`, and asserted in
//! detail. Tests run in CI on all four platforms (Win+Linux × x64+ARM)
//! so any platform-specific JSONL parsing regression gets caught.
//!
//! Highest-value coverage in this file:
//! - Bit-exact Claude per-message cost summation (the v0.1.3 invariant)
//! - Codex cumulative `total_token_usage` delta math
//! - Token dedup by `(message.id, requestId)` for streaming chunks
//! - The `__claude_msg__` synthetic-bucket counts both user + assistant
//!   events (incl. dedup'd streaming chunks)
//! - Date-range filtering with `today_override` (would have caught the
//!   v0.2.2 timezone bug)
//! - Codex cost priced per request (272K long-context tier, dated rates)
//! - Codex cumulative-counter drops and copies of one rollout in two places
//! - The Codex accounting cases shared with the macOS app
//!   (`fixtures/codex-accounting-cases.json`)
//! - A line still being written is read once it is complete

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};

use chrono::NaiveDate;
use cli_pulse_desktop_lib::scanner::{self, DailyEntry, ScanOptions, CLAUDE_MSG_BUCKET_MODEL};

/// Build an isolated scratch dir under tmp/ that's removed when the
/// returned `TempEnv` goes out of scope. Each test gets its own.
struct TempEnv {
    pub root: PathBuf,
    pub codex_root: PathBuf,
    pub claude_root: PathBuf,
    pub cache_dir: PathBuf,
}

impl TempEnv {
    fn new(name: &str) -> Self {
        let root = std::env::temp_dir().join(format!(
            "cli-pulse-int-{}-{}-{}",
            name,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let codex_root = root.join("codex");
        let claude_root = root.join("claude");
        let cache_dir = root.join("cache");
        fs::create_dir_all(&codex_root).unwrap();
        fs::create_dir_all(&claude_root).unwrap();
        fs::create_dir_all(&cache_dir).unwrap();
        Self {
            root,
            codex_root,
            claude_root,
            cache_dir,
        }
    }

    fn write_codex(&self, year: &str, month: &str, day: &str, name: &str, body: &str) -> PathBuf {
        let dir = self.codex_root.join(year).join(month).join(day);
        fs::create_dir_all(&dir).unwrap();
        let p = dir.join(name);
        fs::write(&p, body).unwrap();
        p
    }

    /// Write a Codex rollout under `root.join(rel_dir)` (any layout, e.g. an
    /// `archived_sessions` folder with no date directories).
    fn write_at(&self, dir: &std::path::Path, name: &str, body: &str) -> PathBuf {
        fs::create_dir_all(dir).unwrap();
        let p = dir.join(name);
        fs::write(&p, body).unwrap();
        p
    }

    fn write_claude(&self, project: &str, name: &str, body: &str) -> PathBuf {
        let dir = self.claude_root.join(project);
        fs::create_dir_all(&dir).unwrap();
        let p = dir.join(name);
        fs::write(&p, body).unwrap();
        p
    }

    fn options(&self, days: u32, today: Option<NaiveDate>) -> ScanOptions {
        ScanOptions {
            days,
            force_rescan: true,
            cache_dir: Some(self.cache_dir.clone()),
            codex_roots_override: Some(vec![self.codex_root.clone()]),
            claude_roots_override: Some(vec![self.claude_root.clone()]),
            today_override: today,
        }
    }
}

impl Drop for TempEnv {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

/// Convenience: pick the entry matching (date, provider, model) or
/// fail loudly with a useful message.
fn pick<'a>(entries: &'a [DailyEntry], date: &str, provider: &str, model: &str) -> &'a DailyEntry {
    entries
        .iter()
        .find(|e| e.date == date && e.provider == provider && e.model == model)
        .unwrap_or_else(|| {
            let listing: Vec<String> = entries
                .iter()
                .map(|e| format!("({}, {}, {})", e.date, e.provider, e.model))
                .collect();
            panic!(
                "no entry for ({date}, {provider}, {model}). Got: {:?}",
                listing
            )
        })
}

// ========================================================================
// Codex — cumulative total_token_usage delta math
// ========================================================================

const CODEX_THREE_TURNS: &str = r#"{"type":"session_meta","timestamp":"2026-04-25T10:00:00Z","payload":{"session_id":"sess-1"}}
{"type":"turn_context","timestamp":"2026-04-25T10:00:01Z","payload":{"model":"gpt-5","info":{}}}
{"type":"event_msg","timestamp":"2026-04-25T10:00:02Z","payload":{"type":"token_count","info":{"total_token_usage":{"input_tokens":1000,"cached_input_tokens":0,"output_tokens":500},"model":"gpt-5"}}}
{"type":"event_msg","timestamp":"2026-04-25T10:00:10Z","payload":{"type":"token_count","info":{"total_token_usage":{"input_tokens":1800,"cached_input_tokens":600,"output_tokens":900},"model":"gpt-5"}}}
{"type":"event_msg","timestamp":"2026-04-25T10:00:20Z","payload":{"type":"token_count","info":{"total_token_usage":{"input_tokens":3000,"cached_input_tokens":1200,"output_tokens":1500},"model":"gpt-5"}}}
"#;

#[test]
fn codex_three_turns_yields_cumulative_totals() {
    let env = TempEnv::new("codex_three");
    env.write_codex("2026", "04", "25", "session.jsonl", CODEX_THREE_TURNS);

    let today = NaiveDate::from_ymd_opt(2026, 4, 25).unwrap();
    let result = scanner::scan_with_options(env.options(1, Some(today))).unwrap();

    let e = pick(&result.entries, "2026-04-25", "Codex", "gpt-5");
    // 1st turn: +1000 input, +0 cached, +500 output (no prior baseline)
    // 2nd turn: total goes 1000→1800 input (+800), 0→600 cached (+600), 500→900 output (+400)
    // 3rd turn: total 1800→3000 (+1200), 600→1200 (+600), 900→1500 (+600)
    // Sum: input 1000+800+1200 = 3000, cached 0+600+600 = 1200, output 500+400+600 = 1500
    assert_eq!(e.input_tokens, 3000);
    assert_eq!(e.output_tokens, 1500);
    // `cached` capped at min(cached_delta, input_delta) per slot — all three
    // hold (cached_delta <= input_delta), so sum is 1200.
    assert_eq!(e.cached_tokens, 1200);
}

#[test]
fn codex_gpt5_cost_is_the_sum_of_request_costs() {
    let env = TempEnv::new("codex_pricing");
    env.write_codex("2026", "04", "25", "s.jsonl", CODEX_THREE_TURNS);
    let today = NaiveDate::from_ymd_opt(2026, 4, 25).unwrap();
    let result = scanner::scan_with_options(env.options(1, Some(today))).unwrap();

    let e = pick(&result.entries, "2026-04-25", "Codex", "gpt-5");
    // gpt-5 rates: input $1.25/M, cached $0.125/M, output $10/M
    // (3000 - 1200) non-cached × 1.25e-6 = $0.00225
    //  1200 cached × 0.125e-6 = $0.00015
    //  1500 output × 10e-6 = $0.015
    // total = 0.0174
    let cost = e.cost_usd.expect("Codex gpt-5 has a price");
    assert!(
        (cost - 0.0174).abs() < 1e-9,
        "expected $0.0174, got ${cost}"
    );
}

// ========================================================================
// Claude — per-message cost, msg bucket, dedup
// ========================================================================

const CLAUDE_TIERED_BIG_MSG: &str = r#"{"type":"user","timestamp":"2026-04-25T11:00:00Z"}
{"type":"assistant","timestamp":"2026-04-25T11:00:05Z","requestId":"r1","message":{"id":"m1","model":"claude-sonnet-4-6","usage":{"input_tokens":250000,"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"output_tokens":1000}}}
"#;

#[test]
fn claude_per_message_cost_under_aggregation_threshold() {
    let env = TempEnv::new("claude_tiered");
    env.write_claude("proj", "session.jsonl", CLAUDE_TIERED_BIG_MSG);
    let today = NaiveDate::from_ymd_opt(2026, 4, 25).unwrap();
    let result = scanner::scan_with_options(env.options(1, Some(today))).unwrap();

    let e = pick(&result.entries, "2026-04-25", "Claude", "claude-sonnet-4-6");
    assert_eq!(e.input_tokens, 250000);
    assert_eq!(e.output_tokens, 1000);

    // 250K input on sonnet-4-6: 200K @ $3/M base + 50K @ $6/M tiered = $0.6 + $0.3 = $0.9
    // 1000 output: 1000 @ $15/M = $0.015
    // total = 0.915
    let cost = e.cost_usd.expect("sonnet-4-6 priced");
    assert!(
        (cost - 0.915).abs() < 1e-6,
        "expected $0.915 (per-message tiered), got ${cost}"
    );
}

const CLAUDE_TWO_SMALL_MSGS_NO_TIER: &str = r#"{"type":"user","timestamp":"2026-04-25T12:00:00Z"}
{"type":"assistant","timestamp":"2026-04-25T12:00:02Z","requestId":"r1","message":{"id":"m1","model":"claude-sonnet-4-6","usage":{"input_tokens":150000,"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"output_tokens":500}}}
{"type":"user","timestamp":"2026-04-25T12:01:00Z"}
{"type":"assistant","timestamp":"2026-04-25T12:01:02Z","requestId":"r2","message":{"id":"m2","model":"claude-sonnet-4-6","usage":{"input_tokens":150000,"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"output_tokens":500}}}
"#;

#[test]
fn claude_two_small_messages_stay_under_tier_threshold() {
    // The v0.1.3 invariant: cost is sum of per-message cost. Two 150K
    // messages each price at $3/M ($0.45) — total $0.90 input cost.
    // BUG (would have shipped without per-message accumulation): sum
    // tokens first → 300K, then price tiered → 200K @ $3/M + 100K @ $6/M
    // = $1.20 input cost. Wrong by 33%.
    let env = TempEnv::new("claude_two_msgs");
    env.write_claude("proj", "session.jsonl", CLAUDE_TWO_SMALL_MSGS_NO_TIER);
    let today = NaiveDate::from_ymd_opt(2026, 4, 25).unwrap();
    let result = scanner::scan_with_options(env.options(1, Some(today))).unwrap();

    let e = pick(&result.entries, "2026-04-25", "Claude", "claude-sonnet-4-6");
    assert_eq!(e.input_tokens, 300000);
    assert_eq!(e.output_tokens, 1000);

    // input: 0.90 (two flat-rate messages), output: 1000 @ $15/M = $0.015
    let cost = e.cost_usd.expect("sonnet-4-6 priced");
    assert!(
        (cost - 0.915).abs() < 1e-6,
        "per-message tier semantic broken: expected $0.915, got ${cost}"
    );
}

const CLAUDE_STREAMING_DEDUP: &str = r#"{"type":"user","timestamp":"2026-04-25T13:00:00Z"}
{"type":"assistant","timestamp":"2026-04-25T13:00:01Z","requestId":"r1","message":{"id":"m1","model":"claude-haiku-4-5","usage":{"input_tokens":100,"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"output_tokens":50}}}
{"type":"assistant","timestamp":"2026-04-25T13:00:02Z","requestId":"r1","message":{"id":"m1","model":"claude-haiku-4-5","usage":{"input_tokens":100,"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"output_tokens":50}}}
{"type":"assistant","timestamp":"2026-04-25T13:00:03Z","requestId":"r1","message":{"id":"m1","model":"claude-haiku-4-5","usage":{"input_tokens":100,"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"output_tokens":50}}}
"#;

#[test]
fn claude_streaming_chunks_deduped_for_tokens_but_counted_for_msgs() {
    // Three streaming chunks of the same (message.id=m1, requestId=r1) —
    // tokens should count exactly once; msg bucket should count all 4
    // events (1 user + 3 assistant chunks).
    let env = TempEnv::new("claude_dedup");
    env.write_claude("proj", "session.jsonl", CLAUDE_STREAMING_DEDUP);
    let today = NaiveDate::from_ymd_opt(2026, 4, 25).unwrap();
    let result = scanner::scan_with_options(env.options(1, Some(today))).unwrap();

    let e = pick(&result.entries, "2026-04-25", "Claude", "claude-haiku-4-5");
    assert_eq!(e.input_tokens, 100, "tokens NOT deduped");
    assert_eq!(e.output_tokens, 50, "tokens NOT deduped");

    let m = pick(
        &result.entries,
        "2026-04-25",
        "Claude",
        CLAUDE_MSG_BUCKET_MODEL,
    );
    // 1 user + 3 assistant streaming events = 4 msgs against synthetic bucket
    assert_eq!(m.message_count, 4);
}

// ========================================================================
// Date range — TIMEZONE bug regression (v0.2.2)
// ========================================================================

// TZ-stable timestamp — mid-day UTC stays on 2026-04-25 in every
// reasonable timezone (UTC-12 to UTC+14). The previous version used
// `2026-04-25T05:00:00+09:00` which resolves to 2026-04-25 in JST but
// 2026-04-24 on a UTC CI runner — the test passed on my dev Mac (JST)
// and failed on all four CI platforms (UTC). Lesson: never let a
// test's outcome depend on the host's local timezone.
const CLAUDE_TZ_STABLE: &str = r#"{"type":"user","timestamp":"2026-04-25T12:00:00Z"}
{"type":"assistant","timestamp":"2026-04-25T12:00:01Z","requestId":"r1","message":{"id":"m1","model":"claude-haiku-4-5","usage":{"input_tokens":100,"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"output_tokens":50}}}
"#;

#[test]
fn timezone_anchor_uses_today_override_consistently() {
    // Regression for v0.2.2: prior to the fix, `today` was anchored on
    // Utc::now() while parse_day_key_local converted timestamps to local.
    // Now both share `today_override`. Pinning today=2026-04-25 with a
    // mid-day-UTC event stamp guarantees the event lands on 2026-04-25
    // regardless of the test host's TZ — what we actually want to
    // verify is consistency between today_key and the range filter.
    let env = TempEnv::new("tz_anchor");
    env.write_claude("proj", "session.jsonl", CLAUDE_TZ_STABLE);
    let today = NaiveDate::from_ymd_opt(2026, 4, 25).unwrap();
    let result = scanner::scan_with_options(env.options(1, Some(today))).unwrap();

    assert_eq!(result.today_key, "2026-04-25");
    let _e = pick(&result.entries, "2026-04-25", "Claude", "claude-haiku-4-5");
}

// ========================================================================
// Aggregate sanity
// ========================================================================

// ========================================================================
// CRLF byte-offset regression — flagged by Codex deep review of v0.2.8 as
// the highest-value correctness risk in the shipped product. Pre-fix,
// `parsed_bytes` was incremented as `line.len() + 1` after stripping
// the line terminator with `BufRead::lines()`, but on Windows JSONLs the
// terminator is `\r\n` (2 bytes). One byte/line drift compounded over
// 2700+ files; the next incremental scan would seek mid-line and drop
// the first event of every grown file. This test writes a CRLF fixture
// and verifies the cumulative-delta math doesn't lose any token.
// ========================================================================

const CODEX_CRLF: &str = "{\"type\":\"event_msg\",\"timestamp\":\"2026-04-25T10:00:00Z\",\"payload\":{\"type\":\"token_count\",\"info\":{\"total_token_usage\":{\"input_tokens\":1000,\"cached_input_tokens\":0,\"output_tokens\":500},\"model\":\"gpt-5\"}}}\r\n{\"type\":\"event_msg\",\"timestamp\":\"2026-04-25T10:00:10Z\",\"payload\":{\"type\":\"token_count\",\"info\":{\"total_token_usage\":{\"input_tokens\":2500,\"cached_input_tokens\":0,\"output_tokens\":1100},\"model\":\"gpt-5\"}}}\r\n";

#[test]
fn crlf_codex_jsonl_parses_identically_to_lf() {
    // Build the same fixture twice — once with CRLF, once with LF — and
    // assert the scanner produces identical (input_tokens, output_tokens).
    // If the byte-offset bookkeeping is wrong for CRLF the warm-cache
    // case below would diverge from the LF version.
    let crlf = TempEnv::new("crlf");
    crlf.write_codex("2026", "04", "25", "s.jsonl", CODEX_CRLF);
    let today = NaiveDate::from_ymd_opt(2026, 4, 25).unwrap();
    let r_crlf = scanner::scan_with_options(crlf.options(1, Some(today))).unwrap();

    let lf = TempEnv::new("lf");
    lf.write_codex(
        "2026",
        "04",
        "25",
        "s.jsonl",
        &CODEX_CRLF.replace("\r\n", "\n"),
    );
    let r_lf = scanner::scan_with_options(lf.options(1, Some(today))).unwrap();

    let crlf_e = pick(&r_crlf.entries, "2026-04-25", "Codex", "gpt-5");
    let lf_e = pick(&r_lf.entries, "2026-04-25", "Codex", "gpt-5");

    assert_eq!(crlf_e.input_tokens, lf_e.input_tokens);
    assert_eq!(crlf_e.output_tokens, lf_e.output_tokens);
    // 1st turn baseline 1000/500, 2nd turn delta 1500/600 → totals 2500/1100
    assert_eq!(crlf_e.input_tokens, 2500);
    assert_eq!(crlf_e.output_tokens, 1100);
}

#[test]
fn crlf_incremental_resume_does_not_drop_lines() {
    // Simulate a CRLF file growing between two scans. Pre-fix, the cached
    // `parsed_bytes` would be N less than the true byte count after N
    // lines, and the second scan would seek into the middle of line N+1,
    // dropping the FIRST event of the appended tail.
    let env = TempEnv::new("crlf_grow");
    let today = NaiveDate::from_ymd_opt(2026, 4, 25).unwrap();

    // Initial state: one line.
    let first = "{\"type\":\"event_msg\",\"timestamp\":\"2026-04-25T10:00:00Z\",\"payload\":{\"type\":\"token_count\",\"info\":{\"total_token_usage\":{\"input_tokens\":1000,\"cached_input_tokens\":0,\"output_tokens\":500},\"model\":\"gpt-5\"}}}\r\n";
    env.write_codex("2026", "04", "25", "s.jsonl", first);

    // Cold scan, populates cache. Use force_rescan=false so the cache
    // path is the one we actually exercise in production.
    let mut opts = env.options(1, Some(today));
    opts.force_rescan = false;
    let r1 = scanner::scan_with_options(opts.clone()).unwrap();
    let r1_entry = pick(&r1.entries, "2026-04-25", "Codex", "gpt-5");
    assert_eq!(r1_entry.input_tokens, 1000);
    assert_eq!(r1_entry.output_tokens, 500);

    // Append one more line. The bug would cause this delta to land off
    // by one byte; the new "line N+1" would lose its first character
    // and fail to parse.
    let appended = format!(
        "{first}{}",
        "{\"type\":\"event_msg\",\"timestamp\":\"2026-04-25T10:00:10Z\",\"payload\":{\"type\":\"token_count\",\"info\":{\"total_token_usage\":{\"input_tokens\":2500,\"cached_input_tokens\":0,\"output_tokens\":1100},\"model\":\"gpt-5\"}}}\r\n"
    );
    env.write_codex("2026", "04", "25", "s.jsonl", &appended);

    let r2 = scanner::scan_with_options(opts).unwrap();
    let r2_entry = pick(&r2.entries, "2026-04-25", "Codex", "gpt-5");
    // After incremental parse: same totals as the LF-equivalent scan.
    assert_eq!(r2_entry.input_tokens, 2500);
    assert_eq!(r2_entry.output_tokens, 1100);
}

#[test]
fn empty_roots_yields_empty_result_no_panics() {
    let env = TempEnv::new("empty");
    let today = NaiveDate::from_ymd_opt(2026, 4, 25).unwrap();
    let result = scanner::scan_with_options(env.options(7, Some(today))).unwrap();
    assert_eq!(result.entries.len(), 0);
    assert_eq!(result.total_cost_usd, 0.0);
    assert_eq!(result.total_tokens, 0);
    assert_eq!(result.files_scanned, 0);
    assert_eq!(result.files_cached, 0);
}

#[test]
fn out_of_range_files_excluded() {
    // File dated 2026-01-01 with today=2026-04-25 days=7 → out of range.
    let env = TempEnv::new("out_of_range");
    env.write_codex(
        "2026",
        "01",
        "01",
        "old.jsonl",
        // New Year content shouldn't appear in an end-of-April scan.
        r#"{"type":"event_msg","timestamp":"2026-01-01T12:00:00Z","payload":{"type":"token_count","info":{"total_token_usage":{"input_tokens":1000,"cached_input_tokens":0,"output_tokens":500},"model":"gpt-5"}}}"#,
    );
    let today = NaiveDate::from_ymd_opt(2026, 4, 25).unwrap();
    let result = scanner::scan_with_options(env.options(7, Some(today))).unwrap();
    let codex_entries: Vec<&DailyEntry> = result
        .entries
        .iter()
        .filter(|e| e.provider == "Codex")
        .collect();
    assert!(
        codex_entries.is_empty(),
        "out-of-range file leaked through: {codex_entries:?}"
    );
}

#[test]
fn cache_makes_repeat_scans_idempotent() {
    let env = TempEnv::new("cache_idempotent");
    env.write_claude("proj", "s.jsonl", CLAUDE_TWO_SMALL_MSGS_NO_TIER);
    let today = NaiveDate::from_ymd_opt(2026, 4, 25).unwrap();

    // First scan — cold (force_rescan=true so we ignore any stale cache).
    let r1 = scanner::scan_with_options(env.options(1, Some(today))).unwrap();
    // Second scan — warm. NB: env.options uses force_rescan=true; flip it
    // to false so the cache actually gets reused.
    let mut warm_opts = env.options(1, Some(today));
    warm_opts.force_rescan = false;
    let r2 = scanner::scan_with_options(warm_opts).unwrap();

    assert_eq!(
        r1.total_cost_usd, r2.total_cost_usd,
        "warm scan must not change totals"
    );
    assert_eq!(r1.total_tokens, r2.total_tokens);
    // Warm scan: file cached = 1, scanned = 0
    assert_eq!(r2.files_cached, 1);
    assert_eq!(r2.files_scanned, 0);
}

#[test]
fn codex_event_grouped_by_local_date_in_user_tz() {
    // Ensure that a single file with events on TWO local days produces
    // entries for both days.
    let env = TempEnv::new("two_days");
    env.write_codex(
        "2026",
        "04",
        "24",
        "day1.jsonl",
        r#"{"type":"event_msg","timestamp":"2026-04-24T10:00:00Z","payload":{"type":"token_count","info":{"total_token_usage":{"input_tokens":100,"cached_input_tokens":0,"output_tokens":50},"model":"gpt-5"}}}
"#,
    );
    env.write_codex(
        "2026",
        "04",
        "25",
        "day2.jsonl",
        r#"{"type":"event_msg","timestamp":"2026-04-25T10:00:00Z","payload":{"type":"token_count","info":{"total_token_usage":{"input_tokens":200,"cached_input_tokens":0,"output_tokens":100},"model":"gpt-5"}}}
"#,
    );
    let today = NaiveDate::from_ymd_opt(2026, 4, 25).unwrap();
    let result = scanner::scan_with_options(env.options(7, Some(today))).unwrap();

    let _d1 = pick(&result.entries, "2026-04-24", "Codex", "gpt-5");
    let _d2 = pick(&result.entries, "2026-04-25", "Codex", "gpt-5");

    // Verify the two-day total = 100+200 input, 50+100 output across the matching entries.
    let by_day: HashMap<String, &DailyEntry> = result
        .entries
        .iter()
        .filter(|e| e.provider == "Codex")
        .map(|e| (e.date.clone(), e))
        .collect();
    assert_eq!(by_day["2026-04-24"].input_tokens, 100);
    assert_eq!(by_day["2026-04-25"].input_tokens, 200);
}

// ========================================================================
// Codex — per-request pricing, dated rates, counter drops, rollout copies
// ========================================================================

/// `session_meta` line. `id` is the rollout's own id; `session_id` (when
/// given) is what a sub-agent's rollout carries: its parent's id.
fn codex_meta(ts: &str, id: &str, session_id: Option<&str>) -> String {
    match session_id {
        Some(sid) => format!(
            r#"{{"type":"session_meta","timestamp":"{ts}","payload":{{"id":"{id}","session_id":"{sid}"}}}}"#
        ),
        None => {
            format!(r#"{{"type":"session_meta","timestamp":"{ts}","payload":{{"id":"{id}"}}}}"#)
        }
    }
}

/// `token_count` line with a cumulative `total_token_usage` snapshot.
fn codex_total(ts: &str, model: &str, input: i64, cached: i64, output: i64) -> String {
    format!(
        r#"{{"type":"event_msg","timestamp":"{ts}","payload":{{"type":"token_count","info":{{"total_token_usage":{{"input_tokens":{input},"cached_input_tokens":{cached},"output_tokens":{output}}},"model":"{model}"}}}}}}"#
    )
}

fn lines(ls: &[String]) -> String {
    let mut body = ls.join("\n");
    body.push('\n');
    body
}

fn codex_sum(entries: &[DailyEntry], model: &str) -> (i64, i64, i64, f64) {
    entries
        .iter()
        .filter(|e| e.provider == "Codex" && e.model == model)
        .fold((0, 0, 0, 0.0), |(i, c, o, cost), e| {
            (
                i + e.input_tokens,
                c + e.cached_tokens,
                o + e.output_tokens,
                cost + e.cost_usd.expect("priced model has a cost"),
            )
        })
}

fn assert_usd(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-6,
        "expected ${expected}, got ${actual}"
    );
}

#[test]
fn codex_long_context_tier_is_decided_per_request_not_per_day() {
    // Two gpt-5.5 requests of 200K input each on one day. Each is under the
    // 272K line, so both are standard: 400K @ $5/M = $2.00. Pricing the day's
    // 400K total would cross the line and bill all of it at $10/M = $4.00.
    let env = TempEnv::new("codex_per_request");
    env.write_codex(
        "2026",
        "09",
        "10",
        "r.jsonl",
        &lines(&[
            codex_total("2026-09-10T12:00:00Z", "gpt-5.5", 200_000, 0, 0),
            codex_total("2026-09-10T12:05:00Z", "gpt-5.5", 400_000, 0, 0),
        ]),
    );
    let today = NaiveDate::from_ymd_opt(2026, 9, 10).unwrap();
    let result = scanner::scan_with_options(env.options(1, Some(today))).unwrap();
    let (input, _, _, cost) = codex_sum(&result.entries, "gpt-5.5");
    assert_eq!(input, 400_000);
    assert_usd(cost, 2.00);
}

#[test]
fn codex_long_context_request_bills_the_whole_request_higher() {
    // One gpt-6-astra request of 300K input and 1K output: over the line, so
    // 300K @ $20/M + 1K @ $75/M = $6.075.
    let env = TempEnv::new("codex_long_context");
    env.write_codex(
        "2026",
        "09",
        "10",
        "r.jsonl",
        &lines(&[codex_total(
            "2026-09-10T12:00:00Z",
            "gpt-6-astra",
            300_000,
            0,
            1_000,
        )]),
    );
    let today = NaiveDate::from_ymd_opt(2026, 9, 10).unwrap();
    let result = scanner::scan_with_options(env.options(1, Some(today))).unwrap();
    assert_usd(codex_sum(&result.entries, "gpt-6-astra").3, 6.075);
}

#[test]
fn codex_rates_follow_the_request_time() {
    // gpt-5.6-sol was repriced from $5/M to $4/M input on 2026-08-21. One
    // 100K request the day before and one the day after: $0.50 + $0.40.
    // The model is logged as the alias `gpt-5.6`, which is billed as Sol.
    let env = TempEnv::new("codex_dated");
    env.write_at(
        &env.codex_root,
        "r.jsonl",
        &lines(&[
            codex_total("2026-08-20T12:00:00Z", "gpt-5.6", 100_000, 0, 0),
            codex_total("2026-08-22T12:00:00Z", "gpt-5.6", 200_000, 0, 0),
        ]),
    );
    let today = NaiveDate::from_ymd_opt(2026, 8, 22).unwrap();
    let result = scanner::scan_with_options(env.options(7, Some(today))).unwrap();
    let (input, _, _, cost) = codex_sum(&result.entries, "gpt-5.6");
    assert_eq!(input, 200_000);
    assert_usd(cost, 0.90);
    // Stored and uploaded under the name Codex logged, as on the Mac: a
    // renamed row would sit beside the rows already uploaded under the old
    // name, and the day would count twice.
    assert!(
        result.entries.iter().all(|e| e.model != "gpt-5.6-sol"),
        "the alias was renamed to the model it is billed as"
    );
}

#[test]
fn codex_unpriced_model_has_no_cost_and_zero_priced_model_has_zero() {
    let env = TempEnv::new("codex_unpriced");
    env.write_codex(
        "2026",
        "09",
        "10",
        "a.jsonl",
        &lines(&[codex_total(
            "2026-09-10T12:00:00Z",
            "gpt-42-unicorn",
            1_000,
            0,
            10,
        )]),
    );
    env.write_codex(
        "2026",
        "09",
        "10",
        "b.jsonl",
        &lines(&[codex_total(
            "2026-09-10T12:00:00Z",
            "gpt-5.3-codex-spark",
            1_000,
            0,
            10,
        )]),
    );
    let today = NaiveDate::from_ymd_opt(2026, 9, 10).unwrap();
    let result = scanner::scan_with_options(env.options(1, Some(today))).unwrap();
    assert_eq!(
        pick(&result.entries, "2026-09-10", "Codex", "gpt-42-unicorn").cost_usd,
        None
    );
    assert_eq!(
        pick(
            &result.entries,
            "2026-09-10",
            "Codex",
            "gpt-5.3-codex-spark"
        )
        .cost_usd,
        Some(0.0)
    );
}

#[test]
fn codex_counter_jumping_between_two_series_is_not_recounted() {
    // Two cumulative series interleaved in one rollout. The old rule counted
    // 1000 + 1000 + 1090 + 1180 = 4270 here; the tokens are 1300.
    let env = TempEnv::new("codex_interleaved");
    let snaps = [1000, 100, 1100, 110, 1200, 120, 1300, 130];
    let body: Vec<String> = snaps
        .iter()
        .enumerate()
        .map(|(i, &n)| codex_total(&format!("2026-09-10T12:{i:02}:00Z"), "gpt-5.5", n, 0, 0))
        .collect();
    env.write_codex("2026", "09", "10", "r.jsonl", &lines(&body));
    let today = NaiveDate::from_ymd_opt(2026, 9, 10).unwrap();
    let result = scanner::scan_with_options(env.options(1, Some(today))).unwrap();
    assert_eq!(codex_sum(&result.entries, "gpt-5.5").0, 1300);
}

#[test]
fn codex_counter_restart_counts_only_above_the_old_high() {
    // The counter reaches 2000, restarts and climbs to 900. The baseline only
    // rises, so nothing after the restart passes 2000 and only 2000 counts:
    // the undercount the macOS app and CodexBar accept too.
    let env = TempEnv::new("codex_restart");
    let snaps = [1000, 2000, 100, 400, 900];
    let body: Vec<String> = snaps
        .iter()
        .enumerate()
        .map(|(i, &n)| codex_total(&format!("2026-09-10T12:{i:02}:00Z"), "gpt-5.5", n, 0, 0))
        .collect();
    env.write_codex("2026", "09", "10", "r.jsonl", &lines(&body));
    let today = NaiveDate::from_ymd_opt(2026, 9, 10).unwrap();
    let result = scanner::scan_with_options(env.options(1, Some(today))).unwrap();
    assert_eq!(codex_sum(&result.entries, "gpt-5.5").0, 2000);
}

#[test]
fn codex_usage_before_the_window_is_not_put_on_its_first_day() {
    // A rollout that started before the window: 5000 input on 09-07, then
    // one more request of 600 on 09-10. A 1-day window (09-09..09-10) holds
    // 600, not the 5600 the rollout used since it started.
    let env = TempEnv::new("codex_straddle");
    env.write_at(
        &env.codex_root,
        "r.jsonl",
        &lines(&[
            codex_total("2026-09-07T12:00:00Z", "gpt-5.5", 5_000, 0, 0),
            codex_total("2026-09-10T12:00:00Z", "gpt-5.5", 5_600, 0, 0),
        ]),
    );
    let today = NaiveDate::from_ymd_opt(2026, 9, 10).unwrap();
    let result = scanner::scan_with_options(env.options(1, Some(today))).unwrap();
    assert_eq!(codex_sum(&result.entries, "gpt-5.5").0, 600);
}

const ROLLOUT_NAME: &str = "rollout-2026-09-10T12-00-00-11111111-2222-3333-4444-555555555555.jsonl";

fn one_rollout(id: &str, session_id: Option<&str>, minute0: u32, snaps: &[i64]) -> String {
    let mut ls = vec![codex_meta(
        &format!("2026-09-10T12:{minute0:02}:00Z"),
        id,
        session_id,
    )];
    for (i, &n) in snaps.iter().enumerate() {
        ls.push(codex_total(
            &format!("2026-09-10T12:{:02}:30Z", minute0 as usize + i),
            "gpt-5.5",
            n,
            n / 2,
            n / 10,
        ));
    }
    lines(&ls)
}

#[test]
fn codex_rollout_in_both_sessions_and_archive_is_counted_once() {
    let env = TempEnv::new("codex_archived_copy");
    let body = one_rollout("rollout-a", None, 0, &[1_000, 3_000, 6_000]);
    env.write_codex("2026", "09", "10", ROLLOUT_NAME, &body);
    let archived = env.root.join("archived_sessions");
    env.write_at(&archived, ROLLOUT_NAME, &body);

    let today = NaiveDate::from_ymd_opt(2026, 9, 10).unwrap();
    let mut opts = env.options(1, Some(today));
    opts.codex_roots_override = Some(vec![env.codex_root.clone(), archived]);
    let result = scanner::scan_with_options(opts).unwrap();

    let (input, cached, output, cost) = codex_sum(&result.entries, "gpt-5.5");
    assert_eq!((input, cached, output), (6_000, 3_000, 600));
    // 3000 uncached @ $5/M + 3000 cached @ $0.50/M + 600 output @ $30/M.
    assert_usd(cost, 0.015 + 0.0015 + 0.018);
    // The per-origin split agrees with the totals: one file, 6000 + 600.
    let native: Vec<_> = result
        .origin_usage
        .iter()
        .filter(|o| o.kind == "native")
        .collect();
    assert_eq!(native.len(), 1);
    assert_eq!((native[0].tokens, native[0].files), (6_600, 1));
}

#[test]
fn codex_stale_copy_loses_to_the_full_rollout() {
    // The archive holds a copy taken part-way through; the live file went on.
    let env = TempEnv::new("codex_stale_copy");
    env.write_codex(
        "2026",
        "09",
        "10",
        ROLLOUT_NAME,
        &one_rollout("rollout-a", None, 0, &[1_000, 3_000, 6_000]),
    );
    let archived = env.root.join("archived_sessions");
    env.write_at(
        &archived,
        ROLLOUT_NAME,
        &one_rollout("rollout-a", None, 0, &[1_000, 3_000]),
    );
    let today = NaiveDate::from_ymd_opt(2026, 9, 10).unwrap();
    let mut opts = env.options(1, Some(today));
    opts.codex_roots_override = Some(vec![env.codex_root.clone(), archived]);
    let result = scanner::scan_with_options(opts).unwrap();
    assert_eq!(codex_sum(&result.entries, "gpt-5.5").0, 6_000);
}

#[test]
fn codex_rollout_continued_in_a_second_file_counts_both_halves() {
    // Same rollout id, but the second file starts after the first one ends
    // and shares no event with it: both are real usage.
    let env = TempEnv::new("codex_continuation");
    env.write_codex(
        "2026",
        "09",
        "10",
        "rollout-first.jsonl",
        &one_rollout("rollout-a", None, 0, &[1_000, 2_000]),
    );
    env.write_codex(
        "2026",
        "09",
        "10",
        "rollout-second.jsonl",
        &one_rollout("rollout-a", None, 30, &[500, 1_500]),
    );
    let today = NaiveDate::from_ymd_opt(2026, 9, 10).unwrap();
    let result = scanner::scan_with_options(env.options(1, Some(today))).unwrap();
    assert_eq!(codex_sum(&result.entries, "gpt-5.5").0, 3_500);
}

#[test]
fn codex_sub_agent_rollouts_are_all_counted() {
    // A parent and two sub-agents running at the same time. The sub-agents'
    // session_id is the parent's, but each has its own rollout id.
    let env = TempEnv::new("codex_subagents");
    env.write_codex(
        "2026",
        "09",
        "10",
        "rollout-parent.jsonl",
        &one_rollout("parent", None, 0, &[1_000, 4_000]),
    );
    env.write_codex(
        "2026",
        "09",
        "10",
        "rollout-sub1.jsonl",
        &one_rollout("sub-1", Some("parent"), 0, &[700]),
    );
    env.write_codex(
        "2026",
        "09",
        "10",
        "rollout-sub2.jsonl",
        &one_rollout("sub-2", Some("parent"), 1, &[300]),
    );
    let today = NaiveDate::from_ymd_opt(2026, 9, 10).unwrap();
    let result = scanner::scan_with_options(env.options(1, Some(today))).unwrap();
    assert_eq!(codex_sum(&result.entries, "gpt-5.5").0, 5_000);
}

#[test]
fn codex_warm_scan_reuses_the_cache() {
    // A cache saved without the current rules version would be thrown away on
    // every load, silently re-parsing every file on every scan.
    let env = TempEnv::new("codex_warm");
    env.write_codex(
        "2026",
        "09",
        "10",
        "r.jsonl",
        &one_rollout("rollout-a", None, 0, &[1_000, 3_000]),
    );
    let today = NaiveDate::from_ymd_opt(2026, 9, 10).unwrap();
    let mut opts = env.options(1, Some(today));
    opts.force_rescan = false;
    let cold = scanner::scan_with_options(opts.clone()).unwrap();
    let warm = scanner::scan_with_options(opts).unwrap();
    assert_eq!(warm.files_scanned, 0, "warm scan re-parsed the Codex file");
    assert_eq!(warm.files_cached, 1);
    assert_eq!(
        codex_sum(&cold.entries, "gpt-5.5"),
        codex_sum(&warm.entries, "gpt-5.5")
    );
}

#[test]
fn codex_incremental_scan_matches_a_full_rescan_across_a_counter_drop() {
    // First scan sees the high series and a drop; the appended snapshot
    // returns to the high series. Resuming must start from the saved
    // baseline (1000), not the last snapshot seen (100), or the jump back up
    // re-counts the gap (1100 - 100 = 1000 instead of 100).
    let env = TempEnv::new("codex_incremental_drop");
    let today = NaiveDate::from_ymd_opt(2026, 9, 10).unwrap();
    let head = vec![
        codex_meta("2026-09-10T12:00:00Z", "rollout-a", None),
        codex_total("2026-09-10T12:01:00Z", "gpt-5.5", 1_000, 0, 0),
        codex_total("2026-09-10T12:02:00Z", "gpt-5.5", 100, 0, 0),
    ];
    env.write_codex("2026", "09", "10", "r.jsonl", &lines(&head));
    let mut warm = env.options(1, Some(today));
    warm.force_rescan = false;
    let first = scanner::scan_with_options(warm.clone()).unwrap();
    assert_eq!(codex_sum(&first.entries, "gpt-5.5").0, 1_000);

    let mut all = head;
    all.push(codex_total("2026-09-10T12:03:00Z", "gpt-5.5", 1_100, 0, 0));
    env.write_codex("2026", "09", "10", "r.jsonl", &lines(&all));
    let incremental = scanner::scan_with_options(warm).unwrap();
    assert_eq!(incremental.files_scanned, 1);
    let full = scanner::scan_with_options(env.options(1, Some(today))).unwrap();

    assert_eq!(codex_sum(&incremental.entries, "gpt-5.5").0, 1_100);
    assert_eq!(
        codex_sum(&incremental.entries, "gpt-5.5"),
        codex_sum(&full.entries, "gpt-5.5")
    );
}

#[test]
fn codex_copy_is_still_recognised_after_the_live_file_grows() {
    // The case the copy check exists for: a rollout still being written in
    // `sessions/` while a copy taken earlier sits in `archived_sessions/`.
    // The live file is parsed incrementally from then on, so the rollout id
    // and event span it was matched on must survive the resume.
    let env = TempEnv::new("codex_copy_incremental");
    let archived = env.root.join("archived_sessions");
    let early = one_rollout("rollout-a", None, 0, &[1_000, 3_000]);
    env.write_codex("2026", "09", "10", ROLLOUT_NAME, &early);
    env.write_at(&archived, ROLLOUT_NAME, &early);

    let today = NaiveDate::from_ymd_opt(2026, 9, 10).unwrap();
    let mut warm = env.options(1, Some(today));
    warm.force_rescan = false;
    warm.codex_roots_override = Some(vec![env.codex_root.clone(), archived]);
    let first = scanner::scan_with_options(warm.clone()).unwrap();
    assert_eq!(codex_sum(&first.entries, "gpt-5.5").0, 3_000);

    // The live file goes on; the archived copy does not.
    env.write_codex(
        "2026",
        "09",
        "10",
        ROLLOUT_NAME,
        &one_rollout("rollout-a", None, 0, &[1_000, 3_000, 6_000]),
    );
    let grown = scanner::scan_with_options(warm.clone()).unwrap();
    assert_eq!(
        (grown.files_scanned, grown.files_cached),
        (1, 1),
        "the live file should be parsed incrementally, the copy reused"
    );
    let mut full = warm;
    full.force_rescan = true;
    let rescanned = scanner::scan_with_options(full).unwrap();

    assert_eq!(codex_sum(&grown.entries, "gpt-5.5").0, 6_000);
    assert_eq!(
        codex_sum(&grown.entries, "gpt-5.5"),
        codex_sum(&rescanned.entries, "gpt-5.5")
    );
    // The per-origin split leaves the copy out too: one file, 6000 + 600.
    let native: Vec<_> = grown
        .origin_usage
        .iter()
        .filter(|o| o.kind == "native")
        .collect();
    assert_eq!((native[0].tokens, native[0].files), (6_600, 1));
}

// ========================================================================
// A line still being written is read once it is complete
// ========================================================================

#[test]
fn codex_half_written_last_line_is_counted_once_it_is_complete() {
    // A scan that lands while Codex is writing a line sees only part of it.
    // That part must be left for the next scan, not skipped: skipping it moved
    // the saved offset into the middle of the line, and the next scan read the
    // rest as a line of its own, which never parses.
    let env = TempEnv::new("codex_partial_line");
    let today = NaiveDate::from_ymd_opt(2026, 9, 10).unwrap();
    let head = format!(
        "{}\n",
        codex_total("2026-09-10T12:00:00Z", "gpt-5.5", 1_000, 0, 100)
    );
    let second = codex_total("2026-09-10T12:01:00Z", "gpt-5.5", 2_500, 0, 300);
    let (written, rest) = second.split_at(second.len() / 2);
    env.write_codex("2026", "09", "10", "r.jsonl", &format!("{head}{written}"));

    let mut warm = env.options(1, Some(today));
    warm.force_rescan = false;
    let first = scanner::scan_with_options(warm.clone()).unwrap();
    assert_eq!(codex_sum(&first.entries, "gpt-5.5").0, 1_000);

    env.write_codex(
        "2026",
        "09",
        "10",
        "r.jsonl",
        &format!("{head}{written}{rest}\n"),
    );
    let completed = scanner::scan_with_options(warm).unwrap();
    assert_eq!(completed.files_scanned, 1);
    assert_eq!(
        codex_sum(&completed.entries, "gpt-5.5").0,
        2_500,
        "the completed line was lost"
    );
    let full = scanner::scan_with_options(env.options(1, Some(today))).unwrap();
    assert_eq!(
        codex_sum(&completed.entries, "gpt-5.5"),
        codex_sum(&full.entries, "gpt-5.5")
    );
}

#[test]
fn codex_complete_last_line_without_a_newline_is_counted() {
    // A final line that already parses is a finished line written without a
    // trailing newline, not one still being written.
    let env = TempEnv::new("codex_no_final_newline");
    let today = NaiveDate::from_ymd_opt(2026, 9, 10).unwrap();
    env.write_codex(
        "2026",
        "09",
        "10",
        "r.jsonl",
        &format!(
            "{}\n{}",
            codex_total("2026-09-10T12:00:00Z", "gpt-5.5", 1_000, 0, 100),
            codex_total("2026-09-10T12:01:00Z", "gpt-5.5", 2_500, 0, 300)
        ),
    );
    let result = scanner::scan_with_options(env.options(1, Some(today))).unwrap();
    assert_eq!(codex_sum(&result.entries, "gpt-5.5").0, 2_500);
}

#[test]
fn claude_half_written_last_line_is_counted_once_it_is_complete() {
    // Same rule on the Claude side, which reads its logs the same way.
    let env = TempEnv::new("claude_partial_line");
    let today = NaiveDate::from_ymd_opt(2026, 4, 25).unwrap();
    let (head, assistant) = CLAUDE_TZ_STABLE.split_once('\n').unwrap();
    let assistant = assistant.trim_end();
    let (written, rest) = assistant.split_at(assistant.len() / 2);
    env.write_claude("proj", "s.jsonl", &format!("{head}\n{written}"));

    let mut warm = env.options(1, Some(today));
    warm.force_rescan = false;
    let first = scanner::scan_with_options(warm.clone()).unwrap();
    assert!(first.entries.iter().all(|e| e.model != "claude-haiku-4-5"));

    env.write_claude("proj", "s.jsonl", &format!("{head}\n{written}{rest}\n"));
    let completed = scanner::scan_with_options(warm).unwrap();
    let e = pick(
        &completed.entries,
        "2026-04-25",
        "Claude",
        "claude-haiku-4-5",
    );
    assert_eq!((e.input_tokens, e.output_tokens), (100, 50));
    let m = pick(
        &completed.entries,
        "2026-04-25",
        "Claude",
        CLAUDE_MSG_BUCKET_MODEL,
    );
    assert_eq!(m.message_count, 2);
}

// ========================================================================
// Codex — the accounting cases shared with the macOS app
// ========================================================================

/// The macOS app's Codex accounting cases, copied unchanged from
/// `CLI Pulse Bar/CLIPulseCore/Tests/Fixtures/codex-accounting-cases.json` in
/// cli-pulse/cli-pulse-private (commit 4c3dc377). Both apps must report the
/// same tokens per day and model for them, so the two cannot drift apart
/// unnoticed. When the Mac's copy changes, copy it again.
const MAC_CODEX_CASES: &str = include_str!("fixtures/codex-accounting-cases.json");

/// Cases the desktop deliberately counts differently from the Mac, as
/// `(case, what the desktop reports)`. Each is asserted both ways: the
/// desktop's own number, and that it still differs from the Mac's, so a
/// change on either side has to update this list. Empty: the two agree on
/// every case.
const DESKTOP_DIFFERS_FROM_MAC: &[(&str, &str)] = &[];

type DayModelTokens = BTreeMap<String, BTreeMap<String, [i64; 3]>>;

fn tokens_from_json(v: &serde_json::Value) -> DayModelTokens {
    let mut out = DayModelTokens::new();
    for (day, models) in v.as_object().expect("day map") {
        for (model, t) in models.as_object().expect("model map") {
            let t: Vec<i64> = t
                .as_array()
                .expect("[input, cached, output]")
                .iter()
                .map(|n| n.as_i64().unwrap())
                .collect();
            out.entry(day.clone())
                .or_default()
                .insert(model.clone(), [t[0], t[1], t[2]]);
        }
    }
    out
}

fn codex_tokens(entries: &[DailyEntry]) -> DayModelTokens {
    let mut out = DayModelTokens::new();
    for e in entries.iter().filter(|e| e.provider == "Codex") {
        let slot = out
            .entry(e.date.clone())
            .or_default()
            .entry(e.model.clone())
            .or_insert([0, 0, 0]);
        slot[0] += e.input_tokens;
        slot[1] += e.cached_tokens;
        slot[2] += e.output_tokens;
    }
    out
}

/// Write each file of a case under `home`, keeping the first `keep(n)` of
/// its `n` lines. Lines are compact JSON, one per line, as Codex writes them.
fn write_case_files(home: &Path, case: &serde_json::Value, keep: impl Fn(usize) -> usize) {
    for file in case["files"].as_array().unwrap() {
        let path = home.join(file["path"].as_str().unwrap());
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        let lines = file["lines"].as_array().unwrap();
        let mut body = String::new();
        for line in &lines[..keep(lines.len())] {
            body.push_str(&serde_json::to_string(line).unwrap());
            body.push('\n');
        }
        fs::write(&path, body).unwrap();
    }
}

#[test]
fn codex_accounting_matches_the_mac_on_the_shared_cases() {
    let fixture: serde_json::Value = serde_json::from_str(MAC_CODEX_CASES).unwrap();
    let today = chrono::DateTime::parse_from_rfc3339(fixture["now_utc"].as_str().unwrap())
        .unwrap()
        .with_timezone(&chrono::Local)
        .date_naive();
    let days = fixture["days_to_scan"].as_u64().unwrap() as u32;
    let cases = fixture["cases"].as_array().unwrap();
    assert!(cases.len() >= 14, "fixture lost cases");
    for (name, _) in DESKTOP_DIFFERS_FROM_MAC {
        assert!(
            cases.iter().any(|c| c["name"] == *name),
            "{name} is listed as a known difference but is not in the fixture"
        );
    }

    let mut failures: Vec<String> = Vec::new();
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let mac = tokens_from_json(&case["expected"]);
        let known = DESKTOP_DIFFERS_FROM_MAC.iter().find(|(n, _)| *n == name);
        let want = match known {
            Some((_, desktop)) => {
                let desktop = tokens_from_json(&serde_json::from_str(desktop).unwrap());
                if desktop == mac {
                    failures.push(format!(
                        "{name}: listed as a known difference, but the Mac now expects the same"
                    ));
                }
                desktop
            }
            None => mac,
        };

        let env = TempEnv::new(&format!("mac_case_{name}"));
        let home = env.root.join("codex-home");
        let mut opts = env.options(days, Some(today));
        opts.codex_roots_override =
            Some(vec![home.join("sessions"), home.join("archived_sessions")]);
        let scan = |force: bool| {
            let mut o = opts.clone();
            o.force_rescan = force;
            codex_tokens(&scanner::scan_with_options(o).unwrap().entries)
        };

        // 1. A full scan.
        write_case_files(&home, case, |n| n);
        let full = scan(true);
        // 2. A warm scan that reuses the cache the full scan saved.
        let warm = scan(false);
        // 3. Incremental: every file half written and scanned, then completed
        //    and scanned again, so each resumes from its saved state.
        let _ = fs::remove_dir_all(&home);
        let _ = fs::remove_dir_all(&env.cache_dir);
        write_case_files(&home, case, |n| n / 2);
        let _ = scan(false);
        write_case_files(&home, case, |n| n);
        let incremental = scan(false);

        for (how, got) in [("full", full), ("warm", warm), ("incremental", incremental)] {
            if got != want {
                failures.push(format!("{name} ({how} scan): got {got:?}, want {want:?}"));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "desktop Codex accounting disagrees with the shared cases:\n{}",
        failures.join("\n")
    );
}
