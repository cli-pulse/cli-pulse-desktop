//! MiniMax coding / token plan quota — port of macOS `MiniMaxCollector.swift`
//! (itself derived from steipete/CodexBar, MIT).
//!
//! Endpoints, newest first: `GET https://api.minimax.io/v1/token_plan/remains`,
//! then `/v1/api/openplatform/coding_plan/remains`. Only when the global host
//! REJECTS the key (401/403, `base_resp` 1004, "invalid api key") are the same
//! two asked on `api.minimaxi.com` (China mainland keys only work there); the
//! region that answered is remembered for that key (a SHA-256 of it, in
//! memory). An outage never counts as a rejection and never sends the key to
//! the other region. Overrides: `MINIMAX_REMAINS_URL` (that https URL only) or
//! `MINIMAX_HOST` (that host only, both paths).
//! Auth: `Authorization: Bearer <apiKey>` — env `MINIMAX_API_KEY` or the
//! Settings-stored `minimax_api_key`. Redirects to another host are refused.
//!
//! Until this port the collector read a guessed flat `{ "model_remains": int,
//! "total": int }` from `/v1/coding_plan/remains`, so every real account got
//! nothing. MiniMax actually sends per-model lanes, at the top level or under
//! `data`, numbers sometimes as strings, times in epoch seconds or ms:
//!
//! ```json
//! {"model_remains": [{"model_name": "general",
//!   "start_time": 1780279200000, "end_time": 1780297200000, "remains_time": 14859830,
//!   "current_interval_total_count": 1500, "current_interval_usage_count": 1200,
//!   "current_interval_remaining_percent": 96, "current_interval_status": 1,
//!   "current_weekly_remaining_percent": 99, "weekly_end_time": 1780848000000}]}
//! ```
//!
//! `*_usage_count` is what is LEFT. Only the text lane (`general`, else a
//! MiniMax-M / M2.x model) is read; placeholder lanes (status 3, zero counts,
//! 100 % left) are skipped and an unlimited weekly window is dropped. Tiers are
//! percentages: "5-hour", "Daily" or "Coding Plan" for the interval, plus
//! "Weekly". The flat shape still parses in case anything returns it.
//!
//! Only the API-token path is ported: the Mac's cookie fallback is "less
//! reliable" by its own docs and the desktop credential model is key-based.

use std::future::Future;
use std::sync::Mutex;
use std::time::Duration;

use chrono::{DateTime, SecondsFormat, Utc};
use once_cell::sync::Lazy;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use super::{CollectorError, QuotaSnapshot, TierEntry};

const TIMEOUT: Duration = Duration::from_secs(15);

/// Newest first: the token plan, then the coding plan it replaced.
const API_PATHS: [&str; 2] = [
    "/v1/token_plan/remains",
    "/v1/api/openplatform/coding_plan/remains",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Region {
    Global,
    ChinaMainland,
}

impl Region {
    fn api_host(self) -> &'static str {
        match self {
            Region::Global => "api.minimax.io",
            Region::ChinaMainland => "api.minimaxi.com",
        }
    }

    fn other(self) -> Region {
        match self {
            Region::Global => Region::ChinaMainland,
            Region::ChinaMainland => Region::Global,
        }
    }
}

/// The region whose host last accepted the key, for that key only: a key not
/// seen answering starts on the global host, so a new global key is never sent
/// to China because the previous key lived there.
#[derive(Default)]
struct RegionMemory(Mutex<Option<([u8; 32], Region)>>);

impl RegionMemory {
    fn region(&self, key: &str) -> Region {
        let hash = key_hash(key);
        match *self.0.lock().unwrap_or_else(|e| e.into_inner()) {
            Some((h, region)) if h == hash => region,
            _ => Region::Global,
        }
    }

    fn remember(&self, region: Region, key: &str) {
        *self.0.lock().unwrap_or_else(|e| e.into_inner()) = Some((key_hash(key), region));
    }
}

fn key_hash(key: &str) -> [u8; 32] {
    Sha256::digest(key.as_bytes()).into()
}

static REMEMBERED_REGION: Lazy<RegionMemory> = Lazy::new(RegionMemory::default);

// ── Fetch ───────────────────────────────────────────────────────────

/// How one request ended.
#[derive(Debug, Clone, PartialEq)]
enum Attempt {
    Ok(Parsed),
    /// 401/403, `base_resp` 1004, "invalid api key".
    Rejected,
    /// 404/405: this host does not serve this path.
    Missing(u16),
    /// Anything else: another status, a network error, a refused redirect,
    /// an answer we cannot read. We do not know what the key is worth.
    Unavailable(String),
}

#[derive(Debug, Clone, PartialEq)]
enum Verdict {
    Ok(Parsed),
    Rejected,
    Unavailable(String),
}

/// A region rejected the key only if some path rejected it and every other
/// path rejected it too or does not exist there. One outage or unreadable
/// answer makes the region unavailable: a key must never be called rejected
/// because a server was down.
fn region_verdict(attempts: Vec<Attempt>) -> Verdict {
    let mut saw_rejection = false;
    let mut last_missing = None;
    let mut outage = None;
    for attempt in attempts {
        match attempt {
            Attempt::Ok(parsed) => return Verdict::Ok(parsed),
            Attempt::Rejected => saw_rejection = true,
            Attempt::Missing(status) => last_missing = Some(status),
            Attempt::Unavailable(e) => {
                outage.get_or_insert(e);
            }
        }
    }
    if let Some(e) = outage {
        return Verdict::Unavailable(e);
    }
    if saw_rejection {
        return Verdict::Rejected;
    }
    Verdict::Unavailable(format!("HTTP {}", last_missing.unwrap_or(0)))
}

fn settle(verdict: Verdict) -> Result<Parsed, CollectorError> {
    match verdict {
        Verdict::Ok(parsed) => Ok(parsed),
        Verdict::Rejected => Err(CollectorError::Http("API key rejected".to_string())),
        Verdict::Unavailable(e) => Err(CollectorError::Http(e)),
    }
}

/// `MINIMAX_REMAINS_URL` / `MINIMAX_HOST`, read once per collect.
#[derive(Debug, Default, Clone)]
struct Overrides {
    remains_url: Option<String>,
    host: Option<String>,
}

impl Overrides {
    fn from_env() -> Self {
        let var = |name: &str| {
            std::env::var(name)
                .ok()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
        };
        Overrides {
            remains_url: var("MINIMAX_REMAINS_URL"),
            host: var("MINIMAX_HOST"),
        }
    }
}

/// An https URL with a host, else None: an override cannot send the key in
/// clear text.
fn https_url(s: &str) -> Option<reqwest::Url> {
    let url = reqwest::Url::parse(s).ok()?;
    (url.scheme() == "https" && url.host_str().is_some_and(|h| !h.is_empty())).then_some(url)
}

/// Asks the paths of one host in order and adds up what they said.
async fn host_verdict<F, Fut>(host: &str, attempt: &F) -> Result<Verdict, CollectorError>
where
    F: Fn(reqwest::Url) -> Fut,
    Fut: Future<Output = Attempt>,
{
    let mut attempts = Vec::new();
    for path in API_PATHS {
        let url = https_url(&format!("https://{host}{path}"))
            .ok_or_else(|| CollectorError::Http("invalid MINIMAX_HOST".to_string()))?;
        let result = attempt(url).await;
        if matches!(result, Attempt::Ok(_)) {
            return Ok(region_verdict(vec![result]));
        }
        attempts.push(result);
    }
    Ok(region_verdict(attempts))
}

/// The region logic, with the request itself handed in (tests stub it).
async fn fetch_with_key<F, Fut>(
    token: &str,
    overrides: &Overrides,
    memory: &RegionMemory,
    attempt: F,
) -> Result<Parsed, CollectorError>
where
    F: Fn(reqwest::Url) -> Fut,
    Fut: Future<Output = Attempt>,
{
    if let Some(raw) = &overrides.remains_url {
        let url = https_url(raw)
            .ok_or_else(|| CollectorError::Http("invalid MINIMAX_REMAINS_URL".to_string()))?;
        return settle(region_verdict(vec![attempt(url).await]));
    }
    if let Some(host) = &overrides.host {
        return settle(host_verdict(host, &attempt).await?);
    }

    let first = memory.region(token);
    match host_verdict(first.api_host(), &attempt).await? {
        Verdict::Ok(parsed) => Ok(parsed),
        Verdict::Unavailable(e) => Err(CollectorError::Http(e)),
        Verdict::Rejected => {
            // A key from the other region is rejected here: ask there, once.
            let second = first.other();
            let verdict = host_verdict(second.api_host(), &attempt).await?;
            if matches!(verdict, Verdict::Ok(_)) {
                memory.remember(second, token);
            }
            settle(verdict)
        }
    }
}

/// A redirect is followed only to the same host over https, so the
/// `Authorization` header cannot be carried to another one.
fn redirect_allowed(from: Option<&reqwest::Url>, to: &reqwest::Url) -> bool {
    match from {
        Some(from) => to.scheme() == "https" && from.host_str() == to.host_str(),
        None => false,
    }
}

async fn request(client: &reqwest::Client, url: reqwest::Url, token: &str) -> Attempt {
    let resp = match client
        .get(url)
        .header("Authorization", format!("Bearer {token}"))
        .header("Accept", "application/json")
        // Mirror the Mac's source tag verbatim — a server-side client
        // attribution header.
        .header("MM-API-Source", "CLIPulseBar")
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => return Attempt::Unavailable(format!("request: {}", e.without_url())),
    };
    let status = resp.status().as_u16();
    match status {
        401 | 403 => return Attempt::Rejected,
        404 | 405 => return Attempt::Missing(status),
        200..=299 => {}
        _ => return Attempt::Unavailable(format!("HTTP {status}")),
    }
    let body = match resp.text().await {
        Ok(b) => b,
        Err(e) => return Attempt::Unavailable(format!("body: {}", e.without_url())),
    };
    match parse_remains(&body, Utc::now()) {
        Ok(parsed) => Attempt::Ok(parsed),
        Err(ParseError::Credential) => Attempt::Rejected,
        Err(ParseError::Failed(e)) => Attempt::Unavailable(e),
    }
}

pub async fn collect() -> Result<Option<QuotaSnapshot>, CollectorError> {
    let token = match resolve_key() {
        Some(k) => k,
        None => {
            log::debug!("[MiniMax] no API key (env or Settings UI) — skipping");
            return Ok(None);
        }
    };
    let client = reqwest::Client::builder()
        .timeout(TIMEOUT)
        .redirect(reqwest::redirect::Policy::custom(|attempt| {
            if attempt.previous().len() > 5 {
                attempt.stop()
            } else if redirect_allowed(attempt.previous().last(), attempt.url()) {
                attempt.follow()
            } else {
                // The 3xx itself comes back, which reads as unavailable.
                attempt.stop()
            }
        }))
        .build()
        .map_err(|e| CollectorError::Http(format!("client build: {e}")))?;
    let parsed = fetch_with_key(&token, &Overrides::from_env(), &REMEMBERED_REGION, |url| {
        request(&client, url, &token)
    })
    .await?;
    Ok(Some(map_to_snapshot(&parsed)))
}

fn resolve_key() -> Option<String> {
    if let Ok(k) = std::env::var("MINIMAX_API_KEY") {
        let k = k.trim().to_string();
        if !k.is_empty() {
            return Some(k);
        }
    }
    crate::provider_creds::load()
        .ok()
        .and_then(|c| c.minimax_api_key)
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

// ── Parsing ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
enum ParseError {
    /// The answer refused the credential (`base_resp` 1004, "invalid api key").
    Credential,
    /// The detail never carries MiniMax's own message: only a code.
    Failed(String),
}

/// One quota window, as a percentage left.
#[derive(Debug, Clone, PartialEq)]
struct Window {
    name: &'static str,
    remaining_percent: i64,
    resets_at: Option<DateTime<Utc>>,
    window_minutes: Option<i64>,
}

#[derive(Debug, Clone, PartialEq)]
enum Parsed {
    Windows {
        plan: Option<String>,
        interval: Option<Window>,
        weekly: Option<Window>,
    },
    /// The flat shape the collector used to expect. Never observed.
    Legacy {
        model_remains: i64,
        total: i64,
        end_time: Option<String>,
    },
}

fn parse_remains(text: &str, now: DateTime<Utc>) -> Result<Parsed, ParseError> {
    let json: Map<String, Value> = serde_json::from_str(text)
        .map_err(|_| ParseError::Failed("MiniMax: invalid JSON".to_string()))?;
    let body = json.get("data").and_then(Value::as_object).unwrap_or(&json);

    let base = body
        .get("base_resp")
        .or_else(|| json.get("base_resp"))
        .and_then(Value::as_object);
    if let Some(base) = base {
        if let Some(code) = number(base.get("status_code")).filter(|c| *c != 0.0) {
            let message = base
                .get("status_msg")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_lowercase();
            // Only an explicit refusal. A message that merely mentions
            // logging in ("login service unavailable") is an outage.
            if code == 1004.0 || message.contains("invalid api key") {
                return Err(ParseError::Credential);
            }
            let code = to_i64(code).map_or_else(|| "?".to_string(), |c| c.to_string());
            return Err(ParseError::Failed(format!("MiniMax: status {code}")));
        }
    }

    if let Some(remains) = number(json.get("model_remains")) {
        let total = number(json.get("total")).unwrap_or(0.0);
        let (Some(model_remains), Some(total)) = (to_i64(remains), to_i64(total)) else {
            return Err(ParseError::Failed(
                "MiniMax: count out of range".to_string(),
            ));
        };
        let text_field = |k: &str| {
            json.get(k)
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
        };
        let end_time = text_field("end_time").or_else(|| text_field("remains_time"));
        return Ok(Parsed::Legacy {
            model_remains,
            total,
            end_time,
        });
    }

    let no_quota = || ParseError::Failed("MiniMax: no coding plan quota".to_string());
    let lanes: Vec<&Map<String, Value>> = body
        .get("model_remains")
        .or_else(|| json.get("model_remains"))
        .and_then(Value::as_array)
        .ok_or_else(no_quota)?
        .iter()
        .filter_map(Value::as_object)
        .collect();
    let lane_name = |lane: &Map<String, Value>| {
        lane.get("model_name")
            .and_then(Value::as_str)
            .map(|s| s.trim().to_lowercase())
    };
    // `general` first, then any text model in the order MiniMax sent them.
    let text: Vec<_> = lanes
        .into_iter()
        .filter(|l| is_text_lane(lane_name(l).as_deref()))
        .collect();
    let ordered = text
        .iter()
        .filter(|l| lane_name(l).as_deref() == Some("general"))
        .chain(
            text.iter()
                .filter(|l| lane_name(l).as_deref() != Some("general")),
        );
    for lane in ordered {
        let interval = interval_window(lane, now);
        let weekly = weekly_window(lane, now);
        if interval.is_some() || weekly.is_some() {
            return Ok(Parsed::Windows {
                plan: plan_name(body),
                interval,
                weekly,
            });
        }
    }
    Err(no_quota())
}

/// `general` (the token plan's text bucket) or a text model by name.
fn is_text_lane(lower: Option<&str>) -> bool {
    match lower {
        Some(n) => n == "general" || n.contains("minimax-m") || n.starts_with("m2."),
        None => false,
    }
}

fn interval_window(lane: &Map<String, Value>, now: DateTime<Utc>) -> Option<Window> {
    let start = date(lane.get("start_time"));
    let end = date(lane.get("end_time"));
    let remaining_percent = remaining_percent(
        lane.get("current_interval_remaining_percent"),
        lane.get("current_interval_total_count"),
        lane.get("current_interval_usage_count"),
        lane.get("current_interval_status"),
    )?;
    let window_minutes = window_minutes(start, end);
    let name = match window_minutes {
        Some(240..=360) => "5-hour",
        Some(1380..=1500) => "Daily",
        _ => "Coding Plan",
    };
    Some(Window {
        name,
        remaining_percent,
        resets_at: resets_at(end, lane.get("remains_time"), now),
        window_minutes,
    })
}

fn weekly_window(lane: &Map<String, Value>, now: DateTime<Utc>) -> Option<Window> {
    let status = number(lane.get("current_weekly_status"));
    let percent = number(lane.get("current_weekly_remaining_percent"));
    // Upstream's "no weekly limit" signature.
    if status == Some(3.0) && percent.is_some_and(|p| p >= 100.0) {
        return None;
    }
    let remaining_percent = remaining_percent(
        lane.get("current_weekly_remaining_percent"),
        lane.get("current_weekly_total_count"),
        lane.get("current_weekly_usage_count"),
        lane.get("current_weekly_status"),
    )?;
    let end = date(lane.get("weekly_end_time"));
    Some(Window {
        name: "Weekly",
        remaining_percent,
        resets_at: resets_at(end, lane.get("weekly_remains_time"), now),
        window_minutes: Some(10080),
    })
}

/// Percent left, 0…100: the percent MiniMax sends, else remaining / total.
/// None for a window this plan does not have (status 3, zero counts, 100 %
/// left) or one with nothing to measure.
fn remaining_percent(
    percent: Option<&Value>,
    total: Option<&Value>,
    remaining: Option<&Value>,
    status: Option<&Value>,
) -> Option<i64> {
    let percent = number(percent);
    let total = number(total);
    let remaining = number(remaining);
    if number(status) == Some(3.0)
        && total.unwrap_or(0.0) == 0.0
        && remaining.unwrap_or(0.0) == 0.0
        && percent.unwrap_or(0.0) >= 100.0
    {
        return None;
    }
    let value = match (percent, total, remaining) {
        (Some(p), _, _) => p,
        (None, Some(t), Some(r)) if t > 0.0 => 100.0 * r / t,
        _ => return None,
    };
    Some(value.clamp(0.0, 100.0).round() as i64)
}

fn plan_name(body: &Map<String, Value>) -> Option<String> {
    let card = body
        .get("current_combo_card")
        .and_then(Value::as_object)
        .and_then(|c| c.get("title"));
    [
        body.get("current_subscribe_title"),
        body.get("plan_name"),
        body.get("combo_title"),
        body.get("current_plan_title"),
        card,
    ]
    .into_iter()
    .flatten()
    .filter_map(Value::as_str)
    .map(str::trim)
    .find(|s| !s.is_empty())
    .map(str::to_string)
}

/// A finite number from a JSON number or a numeric string (never a bool).
fn number(value: Option<&Value>) -> Option<f64> {
    let n = match value? {
        Value::Number(n) => n.as_f64()?,
        Value::String(s) => s.trim().parse::<f64>().ok()?,
        _ => return None,
    };
    n.is_finite().then_some(n)
}

/// `value` as an i64 when it is a size an i64 holds without rounding
/// (`"1e100"` is finite, and `as` would silently saturate it).
fn to_i64(value: f64) -> Option<i64> {
    (value.is_finite() && value.abs() < 1e15).then_some(value as i64)
}

/// Epoch milliseconds or seconds before 2100; anything else is not a time.
fn date(value: Option<&Value>) -> Option<DateTime<Utc>> {
    let raw = number(value)?;
    let seconds = if raw > 1_000_000_000_000.0 {
        raw / 1000.0
    } else if raw > 1_000_000_000.0 {
        raw
    } else {
        return None;
    };
    if seconds >= 4_102_444_800.0 {
        return None;
    }
    DateTime::from_timestamp_millis((seconds * 1000.0).round() as i64)
}

fn window_minutes(start: Option<DateTime<Utc>>, end: Option<DateTime<Utc>>) -> Option<i64> {
    let minutes = ((end? - start?).num_milliseconds() as f64 / 60_000.0).round() as i64;
    (minutes > 0).then_some(minutes)
}

/// The window's end if it is still ahead, else now plus the time MiniMax says
/// is left (milliseconds above a million, else seconds).
fn resets_at(
    end: Option<DateTime<Utc>>,
    remains: Option<&Value>,
    now: DateTime<Utc>,
) -> Option<DateTime<Utc>> {
    if let Some(end) = end.filter(|e| *e > now) {
        return Some(end);
    }
    let left = number(remains).filter(|r| *r > 0.0)?;
    let seconds = if left > 1_000_000.0 {
        left / 1000.0
    } else {
        left
    };
    // A century ahead is not a reset (and would overflow the date).
    if seconds > 3_155_760_000.0 {
        return None;
    }
    now.checked_add_signed(chrono::Duration::milliseconds(
        (seconds * 1000.0).round() as i64
    ))
}

// ── Result ──────────────────────────────────────────────────────────

fn iso(d: DateTime<Utc>) -> String {
    d.to_rfc3339_opts(SecondsFormat::Secs, true)
}

fn map_to_snapshot(parsed: &Parsed) -> QuotaSnapshot {
    match parsed {
        Parsed::Windows {
            plan,
            interval,
            weekly,
        } => {
            let tiers: Vec<TierEntry> = [interval, weekly]
                .into_iter()
                .flatten()
                .map(|w| TierEntry {
                    name: w.name.to_string(),
                    quota: 100,
                    remaining: w.remaining_percent,
                    reset_time: w.resets_at.map(iso),
                })
                .collect();
            let headline = tiers.first();
            QuotaSnapshot {
                status_text: None,
                plan_type: plan.clone().unwrap_or_else(|| "Coding Plan".to_string()),
                remaining: headline.map_or(0, |t| t.remaining),
                quota: 100,
                session_reset: headline.and_then(|t| t.reset_time.clone()),
                tiers,
            }
        }
        Parsed::Legacy {
            model_remains,
            total,
            end_time,
        } => {
            let total = (*total).max(0);
            let remaining = (*model_remains).max(0);
            let mut tiers = Vec::new();
            if total > 0 {
                tiers.push(TierEntry {
                    name: "Coding Plan".to_string(),
                    quota: total,
                    remaining,
                    reset_time: end_time.clone(),
                });
            }
            QuotaSnapshot {
                status_text: None,
                plan_type: "Coding Plan".to_string(),
                remaining,
                quota: total,
                session_reset: end_time.clone(),
                tiers,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    //! Mirrors `MiniMaxCollectorTests.swift`. The response shapes are
    //! upstream CodexBar's fixtures, which follow what MiniMax sends.
    use super::*;
    use std::sync::Arc;

    /// 2026-06-01 02:52:20 UTC, inside every window below.
    fn now() -> DateTime<Utc> {
        DateTime::from_timestamp(1_780_282_340, 0).unwrap()
    }

    fn at(seconds: i64) -> DateTime<Utc> {
        DateTime::from_timestamp(seconds, 0).unwrap()
    }

    fn parse(json: &str) -> Result<Parsed, ParseError> {
        parse_remains(json, now())
    }

    fn windows(parsed: Parsed) -> (Option<String>, Option<Window>, Option<Window>) {
        match parsed {
            Parsed::Windows {
                plan,
                interval,
                weekly,
            } => (plan, interval, weekly),
            other => panic!("expected windows, got {other:?}"),
        }
    }

    fn v(json: &str) -> Value {
        serde_json::from_str(json).unwrap()
    }

    fn pct(p: &str, t: &str, r: &str, s: &str) -> Option<i64> {
        let (p, t, r, s) = (v(p), v(t), v(r), v(s));
        remaining_percent(Some(&p), Some(&t), Some(&r), Some(&s))
    }

    /// The token plan as MiniMax sends it to a Plus account: a `general`
    /// lane with a 5-hour and a weekly window, and a video lane the plan does
    /// not include (status 3, zero counts, 100 % left).
    const TOKEN_PLAN: &str = r#"{
      "base_resp": { "status_code": 0, "status_msg": "success" },
      "model_remains": [
        {
          "model_name": "video",
          "start_time": 1780243200000, "end_time": 1780329600000,
          "current_interval_total_count": 0, "current_interval_usage_count": 0,
          "current_interval_status": 3, "current_interval_remaining_percent": 100,
          "current_weekly_status": 3, "current_weekly_remaining_percent": 100
        },
        {
          "model_name": "general",
          "start_time": 1780279200000, "end_time": 1780297200000, "remains_time": 14859830,
          "current_interval_total_count": 0, "current_interval_usage_count": 0,
          "current_interval_status": 1, "current_interval_remaining_percent": 96,
          "current_weekly_total_count": 0, "current_weekly_usage_count": 0,
          "weekly_start_time": 1780243200000, "weekly_end_time": 1780848000000,
          "current_weekly_status": 1, "current_weekly_remaining_percent": 99
        }
      ]
    }"#;

    #[test]
    fn token_plan_reads_the_general_lane_and_skips_the_video_placeholder() {
        let (plan, interval, weekly) = windows(parse(TOKEN_PLAN).unwrap());
        assert_eq!(plan, None);
        assert_eq!(
            interval,
            Some(Window {
                name: "5-hour",
                remaining_percent: 96,
                resets_at: Some(at(1_780_297_200)),
                window_minutes: Some(300),
            })
        );
        assert_eq!(
            weekly,
            Some(Window {
                name: "Weekly",
                remaining_percent: 99,
                resets_at: Some(at(1_780_848_000)),
                window_minutes: Some(10080),
            })
        );
    }

    #[test]
    fn token_plan_builds_two_percent_tiers_with_the_five_hour_one_as_headline() {
        let snap = map_to_snapshot(&parse(TOKEN_PLAN).unwrap());
        let names: Vec<_> = snap.tiers.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, ["5-hour", "Weekly"]);
        let remaining: Vec<_> = snap.tiers.iter().map(|t| t.remaining).collect();
        assert_eq!(remaining, [96, 99]);
        assert!(snap.tiers.iter().all(|t| t.quota == 100));
        let resets: Vec<_> = snap.tiers.iter().map(|t| t.reset_time.as_deref()).collect();
        assert_eq!(
            resets,
            [Some("2026-06-01T07:00:00Z"), Some("2026-06-07T16:00:00Z")]
        );
        assert_eq!(snap.quota, 100);
        assert_eq!(snap.remaining, 96);
        assert_eq!(snap.session_reset.as_deref(), Some("2026-06-01T07:00:00Z"));
        assert_eq!(snap.plan_type, "Coding Plan");
        assert_eq!(snap.status_text, None, "a real gauge needs no status line");
    }

    #[test]
    fn data_wrapped_string_numbers_parse_the_same_and_name_the_plan() {
        let (plan, interval, weekly) = windows(
            parse(
                r#"{
          "base_resp": { "status_code": "0" },
          "data": {
            "current_subscribe_title": "Token Plan · TokenPlanPlus-年度会员",
            "model_remains": [{
              "model_name": "general",
              "current_interval_total_count": 0, "current_interval_usage_count": 0,
              "current_interval_remaining_percent": "96",
              "start_time": "1780279200000", "end_time": 1780297200000,
              "current_weekly_remaining_percent": "99",
              "weekly_start_time": 1780243200000, "weekly_end_time": 1780848000000
            }]
          }
        }"#,
            )
            .unwrap(),
        );
        assert_eq!(plan.as_deref(), Some("Token Plan · TokenPlanPlus-年度会员"));
        let interval = interval.unwrap();
        assert_eq!(interval.remaining_percent, 96);
        assert_eq!(interval.name, "5-hour");
        assert_eq!(weekly.unwrap().remaining_percent, 99);
    }

    /// The coding plan before the token plan: counts per text model, where
    /// `current_interval_usage_count` is what is LEFT, not what was used.
    #[test]
    fn count_based_coding_plan_usage_count_is_the_remainder() {
        let (_, interval, weekly) = windows(
            parse(
                r#"{"model_remains": [{
          "model_name": "MiniMax-M2",
          "start_time": 1780279200, "end_time": 1780297200,
          "current_interval_total_count": 1500, "current_interval_usage_count": 1200
        }]}"#,
            )
            .unwrap(),
        );
        let interval = interval.unwrap();
        assert_eq!(interval.remaining_percent, 80);
        assert_eq!(
            interval.name, "5-hour",
            "epoch seconds work like milliseconds"
        );
        assert_eq!(weekly, None);
    }

    #[test]
    fn a_media_lane_first_is_never_read() {
        let (_, interval, _) = windows(
            parse(
                r#"{"model_remains": [
          {"model_name": "video", "current_interval_total_count": 100, "current_interval_usage_count": 30,
           "start_time": 1780243200000, "end_time": 1780329600000},
          {"model_name": "speech-hd", "current_interval_total_count": 100, "current_interval_usage_count": 10},
          {"model_name": "M2.5", "current_interval_total_count": 200, "current_interval_usage_count": 150,
           "start_time": 1780243200000, "end_time": 1780329600000}
        ]}"#,
            )
            .unwrap(),
        );
        let interval = interval.unwrap();
        assert_eq!(interval.remaining_percent, 75);
        assert_eq!(interval.name, "Daily");
        assert_eq!(interval.window_minutes, Some(1440));
    }

    /// A `general` placeholder must not hide a text model that has a quota.
    #[test]
    fn a_general_placeholder_falls_through_to_a_usable_text_lane() {
        let (_, interval, _) = windows(
            parse(
                r#"{"model_remains": [
          {"model_name": "general", "current_interval_status": 3,
           "current_interval_total_count": 0, "current_interval_usage_count": 0,
           "current_interval_remaining_percent": 100},
          {"model_name": "MiniMax-M2", "current_interval_total_count": 100, "current_interval_usage_count": 40}
        ]}"#,
            )
            .unwrap(),
        );
        let interval = interval.unwrap();
        assert_eq!(interval.remaining_percent, 40);
        assert_eq!(
            interval.name, "Coding Plan",
            "no start and end: the length is unknown"
        );
        assert_eq!(interval.window_minutes, None);
    }

    #[test]
    fn an_unlimited_weekly_window_is_dropped() {
        let (_, interval, weekly) = windows(
            parse(
                r#"{"model_remains": [{
          "model_name": "general", "start_time": 1780279200000, "end_time": 1780297200000,
          "current_interval_remaining_percent": 99,
          "current_weekly_total_count": 0, "current_weekly_usage_count": 0,
          "current_weekly_status": 3, "current_weekly_remaining_percent": 100
        }]}"#,
            )
            .unwrap(),
        );
        assert_eq!(interval.unwrap().remaining_percent, 99);
        assert_eq!(weekly, None);
    }

    #[test]
    fn weekly_only_is_the_headline() {
        let parsed = parse(
            r#"{"model_remains": [{
          "model_name": "general",
          "current_weekly_remaining_percent": 35,
          "weekly_start_time": 1780243200000, "weekly_end_time": 1780848000000
        }]}"#,
        )
        .unwrap();
        let snap = map_to_snapshot(&parsed);
        assert_eq!(windows(parsed).1, None);
        let names: Vec<_> = snap.tiers.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, ["Weekly"]);
        assert_eq!(snap.remaining, 35);
        assert_eq!(snap.session_reset.as_deref(), Some("2026-06-07T16:00:00Z"));
    }

    #[test]
    fn percentages_are_clamped_and_rounded() {
        assert_eq!(pct("120", "null", "null", "null"), Some(100));
        assert_eq!(pct("-5", "null", "null", "null"), Some(0));
        assert_eq!(pct(r#""42.5""#, "null", "null", "null"), Some(43));
        assert_eq!(pct("null", "3", "2", "null"), Some(67));
        assert_eq!(pct("null", "100", "250", "null"), Some(100));
        assert_eq!(pct("null", "0", "0", "null"), None, "zero total");
        assert_eq!(pct("null", "100", "null", "null"), None, "no count");
        assert_eq!(pct(r#""n/a""#, "null", "null", "null"), None);
        assert_eq!(
            pct("true", "null", "null", "null"),
            None,
            "a bool is no number"
        );
        assert_eq!(pct("100", "0", "0", "3"), None, "not in this plan");
        assert_eq!(pct("100", "0", "0", "1"), Some(100));
    }

    #[test]
    fn resets_prefer_an_end_ahead_else_the_time_left() {
        let n = now();
        let ahead = n + chrono::Duration::seconds(600);
        assert_eq!(resets_at(Some(ahead), Some(&v("5000000")), n), Some(ahead));
        let past = n - chrono::Duration::seconds(600);
        assert_eq!(
            resets_at(Some(past), Some(&v("5000000")), n),
            Some(n + chrono::Duration::seconds(5_000)),
            "a stale end falls back to the milliseconds left"
        );
        assert_eq!(
            resets_at(None, Some(&v("900")), n),
            Some(n + chrono::Duration::seconds(900)),
            "seconds"
        );
        assert_eq!(resets_at(Some(past), Some(&v("0")), n), None);
        assert_eq!(resets_at(None, Some(&v("1e300")), n), None, "no overflow");
        assert_eq!(date(Some(&v("12345"))), None, "not a time");
        assert_eq!(date(Some(&v("1e300"))), None, "not a time either");
        assert_eq!(
            date(Some(&v("1780297200"))),
            date(Some(&v("1780297200000")))
        );
    }

    #[test]
    fn base_resp_1004_and_invalid_api_key_are_a_rejected_credential() {
        assert_eq!(
            parse(
                r#"{"base_resp": {"status_code": 1004, "status_msg": "cookie is missing, log in again"}}"#
            ),
            Err(ParseError::Credential)
        );
        assert_eq!(
            parse(r#"{"base_resp": {"status_code": 2049, "status_msg": "invalid api key"}}"#),
            Err(ParseError::Credential)
        );
    }

    #[test]
    fn another_status_fails_with_the_code_only() {
        assert_eq!(
            parse(
                r#"{"base_resp": {"status_code": 1002, "status_msg": "rate limit for user 12345"}}"#
            ),
            Err(ParseError::Failed("MiniMax: status 1002".to_string()))
        );
    }

    /// A message that only mentions logging in is not a refusal.
    #[test]
    fn a_login_service_outage_is_not_a_rejection() {
        let r = parse(
            r#"{"base_resp": {"status_code": 1001, "status_msg": "login service unavailable"}}"#,
        );
        assert!(matches!(r, Err(ParseError::Failed(_))), "{r:?}");
    }

    /// Finite but too big for an integer: a parse failure or no window.
    #[test]
    fn huge_numbers_do_not_saturate_into_values() {
        assert!(matches!(
            parse(r#"{"base_resp": {"status_code": "1e100"}}"#),
            Err(ParseError::Failed(_))
        ));
        assert!(matches!(
            parse(r#"{"model_remains": "1e100", "total": 5}"#),
            Err(ParseError::Failed(_))
        ));
        let (_, interval, _) = windows(
            parse(r#"{"model_remains": [{"model_name": "general", "current_interval_remaining_percent": 50, "start_time": 1780279200, "end_time": "1e300", "remains_time": 1e300}]}"#)
                .unwrap(),
        );
        let interval = interval.unwrap();
        assert_eq!(interval.remaining_percent, 50);
        assert_eq!(interval.window_minutes, None);
        assert_eq!(interval.resets_at, None);
        assert_eq!(to_i64(1e100), None);
        assert_eq!(to_i64(42.0), Some(42));
    }

    #[test]
    fn no_usable_lane_is_a_parse_failure() {
        for body in [
            r#"{"model_remains": [{"model_name": "video", "current_interval_total_count": 10, "current_interval_usage_count": 5}]}"#,
            r#"{"model_remains": []}"#,
            r#"{"base_resp": {"status_code": 0}}"#,
            r#"{}"#,
            "bad",
        ] {
            assert!(matches!(parse(body), Err(ParseError::Failed(_))), "{body}");
        }
    }

    /// The flat shape the collector used to expect still parses.
    #[test]
    fn the_old_flat_shape_still_parses() {
        let parsed =
            parse(r#"{"model_remains":700,"total":1000,"end_time":"2026-07-10T00:00:00Z"}"#)
                .unwrap();
        let snap = map_to_snapshot(&parsed);
        assert_eq!(snap.plan_type, "Coding Plan");
        assert_eq!((snap.quota, snap.remaining), (1000, 700));
        assert_eq!(snap.session_reset.as_deref(), Some("2026-07-10T00:00:00Z"));
        assert_eq!(snap.tiers.len(), 1);
        assert_eq!(snap.tiers[0].name, "Coding Plan");

        // end_time wins; an empty one falls back to remains_time.
        let reset = |j: &str| map_to_snapshot(&parse(j).unwrap()).session_reset;
        assert_eq!(
            reset(r#"{"total":10,"model_remains":5,"end_time":"E","remains_time":"R"}"#).as_deref(),
            Some("E")
        );
        assert_eq!(
            reset(r#"{"total":10,"model_remains":5,"end_time":"","remains_time":"R"}"#).as_deref(),
            Some("R")
        );

        // No total, no tier; negative counts floor at 0.
        let snap = map_to_snapshot(&parse(r#"{"model_remains":0,"total":0}"#).unwrap());
        assert!(snap.tiers.is_empty());
        let snap = map_to_snapshot(&parse(r#"{"model_remains":-9,"total":-3}"#).unwrap());
        assert_eq!((snap.quota, snap.remaining), (0, 0));
    }

    // ── Fetching ────────────────────────────────────────────────────

    const OK: &str =
        r#"{"model_remains":[{"model_name":"general","current_interval_remaining_percent":70}]}"#;

    /// A stub server: `reply(url, token)` answers, and every URL asked is
    /// recorded as "host/path" with the key it carried.
    struct Stub {
        asked: Arc<Mutex<Vec<(String, String)>>>,
    }

    impl Stub {
        fn new() -> Self {
            Stub {
                asked: Arc::new(Mutex::new(Vec::new())),
            }
        }

        fn asked(&self) -> Vec<String> {
            self.asked
                .lock()
                .unwrap()
                .iter()
                .map(|(u, _)| u.clone())
                .collect()
        }

        fn clear(&self) {
            self.asked.lock().unwrap().clear();
        }

        async fn fetch(
            &self,
            token: &str,
            overrides: &Overrides,
            memory: &RegionMemory,
            reply: impl Fn(&reqwest::Url, &str) -> Attempt,
        ) -> Result<Parsed, CollectorError> {
            let asked = self.asked.clone();
            fetch_with_key(token, overrides, memory, |url| {
                let key = token.to_string();
                asked.lock().unwrap().push((
                    format!("{}{}", url.host_str().unwrap_or(""), url.path()),
                    key.clone(),
                ));
                std::future::ready(reply(&url, &key))
            })
            .await
        }
    }

    fn ok() -> Attempt {
        Attempt::Ok(parse(OK).unwrap())
    }

    fn china(url: &reqwest::Url) -> bool {
        url.host_str() == Some("api.minimaxi.com")
    }

    fn err_message(r: Result<Parsed, CollectorError>) -> String {
        r.expect_err("expected an error").message().to_string()
    }

    #[tokio::test]
    async fn a_global_key_is_answered_by_the_first_path() {
        let stub = Stub::new();
        let parsed = stub
            .fetch(
                "k",
                &Overrides::default(),
                &RegionMemory::default(),
                |_, _| ok(),
            )
            .await
            .unwrap();
        assert_eq!(map_to_snapshot(&parsed).remaining, 70);
        assert_eq!(stub.asked(), ["api.minimax.io/v1/token_plan/remains"]);
    }

    #[tokio::test]
    async fn a_china_key_is_asked_in_china_after_global_rejects_it_and_china_is_remembered() {
        let stub = Stub::new();
        let memory = RegionMemory::default();
        let reply = |u: &reqwest::Url, _: &str| if china(u) { ok() } else { Attempt::Rejected };
        stub.fetch("k", &Overrides::default(), &memory, reply)
            .await
            .unwrap();
        assert_eq!(
            stub.asked(),
            [
                "api.minimax.io/v1/token_plan/remains",
                "api.minimax.io/v1/api/openplatform/coding_plan/remains",
                "api.minimaxi.com/v1/token_plan/remains",
            ]
        );
        stub.clear();
        stub.fetch("k", &Overrides::default(), &memory, reply)
            .await
            .unwrap();
        assert_eq!(
            stub.asked(),
            ["api.minimaxi.com/v1/token_plan/remains"],
            "the region that answered goes first"
        );
    }

    /// The region belongs to the key that answered there: a new key starts
    /// on the global host, so a global key is never sent to China first.
    #[tokio::test]
    async fn another_key_starts_on_the_global_host_again() {
        let stub = Stub::new();
        let memory = RegionMemory::default();
        let reply = |u: &reqwest::Url, key: &str| {
            let good = if china(u) { "china" } else { "global" };
            if key == good {
                ok()
            } else {
                Attempt::Rejected
            }
        };
        stub.fetch("china", &Overrides::default(), &memory, reply)
            .await
            .unwrap();
        stub.clear();
        stub.fetch("global", &Overrides::default(), &memory, reply)
            .await
            .unwrap();
        assert_eq!(stub.asked(), ["api.minimax.io/v1/token_plan/remains"]);
        stub.clear();
        stub.fetch("china", &Overrides::default(), &memory, reply)
            .await
            .unwrap();
        assert_eq!(
            stub.asked(),
            ["api.minimaxi.com/v1/token_plan/remains"],
            "the china key still goes to China first"
        );
    }

    /// Upstream's case: the token plan refuses the key, the old path does
    /// not exist on that host, and the key works in China.
    #[tokio::test]
    async fn rejected_then_missing_still_counts_as_rejected() {
        let stub = Stub::new();
        stub.fetch(
            "k",
            &Overrides::default(),
            &RegionMemory::default(),
            |u, _| {
                if china(u) {
                    ok()
                } else if u.path() == "/v1/token_plan/remains" {
                    Attempt::Rejected
                } else {
                    Attempt::Missing(404)
                }
            },
        )
        .await
        .unwrap();
        assert_eq!(
            stub.asked().last().map(String::as_str),
            Some("api.minimaxi.com/v1/token_plan/remains")
        );
    }

    #[tokio::test]
    async fn a_global_outage_does_not_send_the_key_to_china() {
        let stub = Stub::new();
        let r = stub
            .fetch(
                "k",
                &Overrides::default(),
                &RegionMemory::default(),
                |_, _| Attempt::Unavailable("HTTP 503".to_string()),
            )
            .await;
        assert_eq!(err_message(r), "HTTP 503");
        assert!(!stub
            .asked()
            .iter()
            .any(|u| u.starts_with("api.minimaxi.com")));
    }

    #[tokio::test]
    async fn a_rejection_beside_an_outage_is_not_called_rejected() {
        let stub = Stub::new();
        let r = stub
            .fetch(
                "k",
                &Overrides::default(),
                &RegionMemory::default(),
                |u, _| {
                    if u.path() == "/v1/token_plan/remains" {
                        Attempt::Rejected
                    } else {
                        Attempt::Unavailable("request: timed out".to_string())
                    }
                },
            )
            .await;
        assert_eq!(err_message(r), "request: timed out");
        assert!(!stub
            .asked()
            .iter()
            .any(|u| u.starts_with("api.minimaxi.com")));
    }

    #[tokio::test]
    async fn a_china_outage_after_a_global_rejection_is_an_outage_not_a_rejection() {
        let stub = Stub::new();
        let r = stub
            .fetch(
                "k",
                &Overrides::default(),
                &RegionMemory::default(),
                |u, _| {
                    if china(u) {
                        Attempt::Unavailable("HTTP 502".to_string())
                    } else {
                        Attempt::Rejected
                    }
                },
            )
            .await;
        assert_eq!(err_message(r), "HTTP 502");
    }

    #[tokio::test]
    async fn both_regions_rejecting_is_a_rejected_key() {
        let stub = Stub::new();
        let r = stub
            .fetch(
                "k",
                &Overrides::default(),
                &RegionMemory::default(),
                |_, _| Attempt::Rejected,
            )
            .await;
        assert_eq!(err_message(r), "API key rejected");
        assert_eq!(stub.asked().len(), 4);
    }

    #[tokio::test]
    async fn a_missing_token_plan_path_falls_back_to_the_coding_plan_path() {
        let stub = Stub::new();
        stub.fetch(
            "k",
            &Overrides::default(),
            &RegionMemory::default(),
            |u, _| {
                if u.path() == "/v1/token_plan/remains" {
                    Attempt::Missing(404)
                } else {
                    ok()
                }
            },
        )
        .await
        .unwrap();
        assert_eq!(
            stub.asked(),
            [
                "api.minimax.io/v1/token_plan/remains",
                "api.minimax.io/v1/api/openplatform/coding_plan/remains",
            ]
        );
    }

    #[tokio::test]
    async fn the_host_override_is_the_only_host_asked() {
        let stub = Stub::new();
        let overrides = Overrides {
            host: Some("mm.example.test".to_string()),
            ..Overrides::default()
        };
        let r = stub
            .fetch("k", &overrides, &RegionMemory::default(), |_, _| {
                Attempt::Rejected
            })
            .await;
        assert_eq!(err_message(r), "API key rejected");
        assert_eq!(stub.asked().len(), 2);
        assert!(stub
            .asked()
            .iter()
            .all(|u| u.starts_with("mm.example.test/")));
    }

    #[tokio::test]
    async fn the_url_override_is_the_only_url_asked_and_must_be_https() {
        let stub = Stub::new();
        let overrides = Overrides {
            remains_url: Some("https://mm.example.test/v1/coding_plan/remains".to_string()),
            ..Overrides::default()
        };
        stub.fetch("k", &overrides, &RegionMemory::default(), |_, _| ok())
            .await
            .unwrap();
        assert_eq!(stub.asked(), ["mm.example.test/v1/coding_plan/remains"]);

        stub.clear();
        let overrides = Overrides {
            remains_url: Some("http://mm.example.test/x".to_string()),
            ..Overrides::default()
        };
        let r = stub
            .fetch("k", &overrides, &RegionMemory::default(), |_, _| ok())
            .await;
        assert_eq!(err_message(r), "invalid MINIMAX_REMAINS_URL");
        assert!(
            stub.asked().is_empty(),
            "the key is never sent in clear text"
        );
    }

    #[test]
    fn a_redirect_is_followed_only_to_the_same_host_over_https() {
        let url = |s: &str| reqwest::Url::parse(s).unwrap();
        let from = url("https://api.minimax.io/v1/token_plan/remains");
        assert!(redirect_allowed(
            Some(&from),
            &url("https://api.minimax.io/v2/x")
        ));
        assert!(!redirect_allowed(
            Some(&from),
            &url("https://evil.example.test/steal")
        ));
        assert!(!redirect_allowed(
            Some(&from),
            &url("http://api.minimax.io/v1/x")
        ));
        assert!(!redirect_allowed(None, &url("https://api.minimax.io/")));
    }
}
