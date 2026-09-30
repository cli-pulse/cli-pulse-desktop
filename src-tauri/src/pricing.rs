//! Per-token pricing for Codex (OpenAI) and Claude (Anthropic) models.
//!
//! Ported from Swift `CostUsageScanner.Pricing` in the macOS app.
//! Rates are USD per token (not per million).
//!
//! Claude sonnet-4-5 / sonnet-4-6 / sonnet-4-20250514 use tiered pricing:
//! first 200K input tokens at base rate, above threshold at 2x rate.
//!
//! Codex rates follow the table bundled with steipete/CodexBar (notice below),
//! which cites OpenAI's published API prices. Three rules come with it:
//!
//! - **Long context is decided per request.** When one request sends more than
//!   272K input tokens, every token of that request is billed at the model's
//!   long-context rates. A day's total crosses 272K almost always, so Codex cost
//!   must be summed from per-request costs (`scanner.rs` does this while
//!   parsing), never computed from a day's tokens.
//! - **Rates are dated.** A model that was repriced keeps its old rate for
//!   requests made before the change (`CODEX_EARLIER_RATES`). Pass the request's
//!   time; `None` means today's rates.
//! - **Aliases.** A few names OpenAI routes to a priced model (`gpt-5.6` is Sol)
//!   are resolved by `normalize_codex_model`.
//!
// Codex rate rows, the dated rates, the aliases and the long-context rule are
// derived from steipete/CodexBar
// Sources/CodexBarCore/Vendored/CostUsage/CostUsagePricing.swift, taken at
// upstream commit 25bba9b7 (2026-09-28) (https://github.com/steipete/CodexBar).
// The numbers and rules are upstream's; the code is a Rust restatement, and
// differs from upstream in these ways:
//
// - Cache-write rates are left out. Codex CLI logs report no cache writes, so
//   upstream bills every uncached input token at the input rate for them too.
// - API Fast (priority) multipliers and the models.dev live catalogue are not
//   used; the bundled Standard rates are.
// - `gpt-5.5-codex`, `gpt-5.5-mini` and `gpt-5.5-nano` are not in upstream's
//   table. They are this file's own rows (see the comments on them).
//
// ─── MIT License (full notice required by upstream) ───────────────
//
// MIT License
//
// Copyright (c) 2026 Peter Steinberger
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// ──────────────────────────────────────────────────────────────────

use once_cell::sync::Lazy;
use regex::Regex;
use std::collections::HashMap;

/// One set of Codex rates, USD per token.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CodexRates {
    pub input: f64,
    /// Rate for cached input. `None` bills cached input at the full input rate
    /// (the `-pro` models publish no cached rate).
    pub cache_read: Option<f64>,
    pub output: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CodexModel {
    pub standard: CodexRates,
    /// Rates for a request whose input exceeds `CODEX_LONG_CONTEXT_INPUT_TOKENS`.
    /// They apply to the WHOLE request, not only to the tokens above the line.
    pub long_context: Option<CodexRates>,
}

/// A request with more input tokens than this is a long-context request.
pub const CODEX_LONG_CONTEXT_INPUT_TOKENS: i64 = 272_000;

fn rates(input: f64, cache_read: Option<f64>, output: f64) -> CodexRates {
    CodexRates {
        input,
        cache_read,
        output,
    }
}

/// A model with one set of rates, whatever the request size.
fn flat(input: f64, cache_read: Option<f64>, output: f64) -> CodexModel {
    CodexModel {
        standard: rates(input, cache_read, output),
        long_context: None,
    }
}

/// A model whose long-context requests are billed at `long_context`.
fn tiered(standard: CodexRates, long_context: CodexRates) -> CodexModel {
    CodexModel {
        standard,
        long_context: Some(long_context),
    }
}

static CODEX_MODELS: Lazy<HashMap<&'static str, CodexModel>> = Lazy::new(|| {
    let mut m = HashMap::new();
    m.insert("gpt-5", flat(1.25e-6, Some(1.25e-7), 1e-5));
    m.insert("gpt-5-codex", flat(1.25e-6, Some(1.25e-7), 1e-5));
    m.insert("gpt-5-mini", flat(2.5e-7, Some(2.5e-8), 2e-6));
    m.insert("gpt-5-nano", flat(5e-8, Some(5e-9), 4e-7));
    m.insert("gpt-5-pro", flat(1.5e-5, None, 1.2e-4));
    m.insert("gpt-5.1", flat(1.25e-6, Some(1.25e-7), 1e-5));
    m.insert("gpt-5.1-codex", flat(1.25e-6, Some(1.25e-7), 1e-5));
    m.insert("gpt-5.1-codex-max", flat(1.25e-6, Some(1.25e-7), 1e-5));
    m.insert("gpt-5.1-codex-mini", flat(2.5e-7, Some(2.5e-8), 2e-6));
    m.insert("gpt-5.2", flat(1.75e-6, Some(1.75e-7), 1.4e-5));
    m.insert("gpt-5.2-codex", flat(1.75e-6, Some(1.75e-7), 1.4e-5));
    m.insert("gpt-5.2-pro", flat(2.1e-5, None, 1.68e-4));
    m.insert("gpt-5.3-codex", flat(1.75e-6, Some(1.75e-7), 1.4e-5));
    // Research preview, not billed.
    m.insert("gpt-5.3-codex-spark", flat(0.0, Some(0.0), 0.0));
    m.insert(
        "gpt-5.4",
        tiered(
            rates(2.5e-6, Some(2.5e-7), 1.5e-5),
            rates(5e-6, Some(5e-7), 2.25e-5),
        ),
    );
    m.insert("gpt-5.4-mini", flat(7.5e-7, Some(7.5e-8), 4.5e-6));
    m.insert("gpt-5.4-nano", flat(2e-7, Some(2e-8), 1.25e-6));
    m.insert("gpt-5.4-pro", flat(3e-5, None, 1.8e-4));
    // gpt-5.5 now has published prices ($5 / $30 per 1M, with a long-context
    // tier). Until they existed this row mirrored gpt-5.4 ($2.50 / $15) as a
    // placeholder, which priced gpt-5.5 at half its real rate.
    let gpt_5_5 = tiered(
        rates(5e-6, Some(5e-7), 3e-5),
        rates(1e-5, Some(1e-6), 4.5e-5),
    );
    m.insert("gpt-5.5", gpt_5_5);
    // Not in OpenAI's published list. Every `-codex` row above costs the same
    // as its base model, so this one follows gpt-5.5 instead of keeping the old
    // gpt-5.4 placeholder, which would now make it half the price of gpt-5.5.
    m.insert("gpt-5.5-codex", gpt_5_5);
    // Not in OpenAI's published list either. Kept at the gpt-5.4-mini /
    // gpt-5.4-nano rates they were given when gpt-5.5 first appeared, so a log
    // that names them is not priced at nothing.
    m.insert("gpt-5.5-mini", flat(7.5e-7, Some(7.5e-8), 4.5e-6));
    m.insert("gpt-5.5-nano", flat(2e-7, Some(2e-8), 1.25e-6));
    m.insert("gpt-5.5-pro", flat(3e-5, None, 1.8e-4));
    // Cyber models publish no long-context tier.
    m.insert("gpt-5.5-cyber", flat(1.25e-5, Some(1.25e-6), 7.5e-5));
    m.insert("gpt-5.6-cyber", flat(1.25e-5, Some(1.25e-6), 7.5e-5));
    // GPT-5.6: long context is 2x input and 1.5x output for the whole request.
    // Sol was repriced from $5 / $30 to $4 / $20 on 2026-08-21, and Terra and
    // Luna changed on 2026-07-30; the earlier rates are in CODEX_EARLIER_RATES.
    m.insert(
        "gpt-5.6-sol",
        tiered(rates(4e-6, Some(4e-7), 2e-5), rates(8e-6, Some(8e-7), 3e-5)),
    );
    m.insert(
        "gpt-5.6-terra",
        tiered(
            rates(2e-6, Some(2e-7), 1.2e-5),
            rates(4e-6, Some(4e-7), 1.8e-5),
        ),
    );
    m.insert(
        "gpt-5.6-luna",
        tiered(
            rates(2e-7, Some(2e-8), 1.2e-6),
            rates(4e-7, Some(4e-8), 1.8e-6),
        ),
    );
    m.insert(
        "gpt-6-astra",
        tiered(
            rates(1e-5, Some(1e-6), 5e-5),
            rates(2e-5, Some(2e-6), 7.5e-5),
        ),
    );
    m
});

/// 2026-07-30T00:00:00Z, when GPT-5.6 Terra and Luna took today's rates.
pub const CODEX_TERRA_LUNA_REPRICED_UNIX_MS: i64 = 1_785_369_600_000;
/// 2026-08-21T00:00:00Z, when GPT-5.6 Sol took today's rates (OpenAI's API
/// changelog dates the change August 21).
pub const CODEX_SOL_REPRICED_UNIX_MS: i64 = 1_787_270_400_000;

/// Rates a model had before it was repriced: `model -> (repriced_at, rates)`.
/// A request made strictly before `repriced_at` is billed at these.
static CODEX_EARLIER_RATES: Lazy<HashMap<&'static str, (i64, CodexModel)>> = Lazy::new(|| {
    let mut m = HashMap::new();
    m.insert(
        "gpt-5.6-sol",
        (
            CODEX_SOL_REPRICED_UNIX_MS,
            tiered(
                rates(5e-6, Some(5e-7), 3e-5),
                rates(1e-5, Some(1e-6), 4.5e-5),
            ),
        ),
    );
    m.insert(
        "gpt-5.6-terra",
        (
            CODEX_TERRA_LUNA_REPRICED_UNIX_MS,
            tiered(
                rates(2.5e-6, Some(2.5e-7), 1.5e-5),
                rates(5e-6, Some(5e-7), 2.25e-5),
            ),
        ),
    );
    m.insert(
        "gpt-5.6-luna",
        (
            CODEX_TERRA_LUNA_REPRICED_UNIX_MS,
            tiered(rates(1e-6, Some(1e-7), 6e-6), rates(2e-6, Some(2e-7), 9e-6)),
        ),
    );
    m
});

#[derive(Debug, Clone, Copy)]
pub struct ClaudeModel {
    pub input: f64,
    pub output: f64,
    pub cache_creation: f64,
    pub cache_read: f64,
    pub threshold: Option<i64>,
    pub input_above: Option<f64>,
    pub output_above: Option<f64>,
    pub cache_creation_above: Option<f64>,
    pub cache_read_above: Option<f64>,
}

static CLAUDE_MODELS: Lazy<HashMap<&'static str, ClaudeModel>> = Lazy::new(|| {
    let mut m = HashMap::new();
    // Haiku 4.5
    let haiku_4_5 = ClaudeModel {
        input: 1e-6,
        output: 5e-6,
        cache_creation: 1.25e-6,
        cache_read: 1e-7,
        threshold: None,
        input_above: None,
        output_above: None,
        cache_creation_above: None,
        cache_read_above: None,
    };
    m.insert("claude-haiku-4-5", haiku_4_5);
    m.insert("claude-haiku-4-5-20251001", haiku_4_5);

    // Opus 4.5 / 4.6
    let opus_45_46 = ClaudeModel {
        input: 5e-6,
        output: 2.5e-5,
        cache_creation: 6.25e-6,
        cache_read: 5e-7,
        threshold: None,
        input_above: None,
        output_above: None,
        cache_creation_above: None,
        cache_read_above: None,
    };
    m.insert("claude-opus-4-5", opus_45_46);
    m.insert("claude-opus-4-5-20251101", opus_45_46);
    m.insert("claude-opus-4-6", opus_45_46);
    m.insert("claude-opus-4-6-20260205", opus_45_46);
    // Opus 4.7 — same pricing tier as 4.5 / 4.6
    m.insert("claude-opus-4-7", opus_45_46);

    // Sonnet 4.5 / 4.6 — tiered above 200K
    let sonnet_tiered = ClaudeModel {
        input: 3e-6,
        output: 1.5e-5,
        cache_creation: 3.75e-6,
        cache_read: 3e-7,
        threshold: Some(200_000),
        input_above: Some(6e-6),
        output_above: Some(2.25e-5),
        cache_creation_above: Some(7.5e-6),
        cache_read_above: Some(6e-7),
    };
    m.insert("claude-sonnet-4-5", sonnet_tiered);
    m.insert("claude-sonnet-4-5-20250929", sonnet_tiered);
    m.insert("claude-sonnet-4-6", sonnet_tiered);
    m.insert("claude-sonnet-4-20250514", sonnet_tiered);

    // Opus 4 / 4.1 (legacy)
    let opus_4 = ClaudeModel {
        input: 1.5e-5,
        output: 7.5e-5,
        cache_creation: 1.875e-5,
        cache_read: 1.5e-6,
        threshold: None,
        input_above: None,
        output_above: None,
        cache_creation_above: None,
        cache_read_above: None,
    };
    m.insert("claude-opus-4-20250514", opus_4);
    m.insert("claude-opus-4-1", opus_4);

    m
});

static CODEX_DATED_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"-\d{4}-\d{2}-\d{2}$").unwrap());
static CLAUDE_DATED_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"-\d{8}$").unwrap());
static CLAUDE_BEDROCK_VER_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"-v\d+:\d+$").unwrap());
// Major+minor family form Claude Code emits for a released model
// (dates already stripped by the time we test this). Drives the
// family fallback so a not-yet-tabled minor doesn't price to $0.
static CLAUDE_FAMILY_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^claude-(opus|sonnet|haiku)-\d+-\d+$").unwrap());

/// Names OpenAI routes to a priced model.
fn codex_alias(model: &str) -> Option<&'static str> {
    match model {
        // The unsuffixed gpt-5.6 is Sol.
        "gpt-5.6" => Some("gpt-5.6-sol"),
        // Codex's name for the Luna Reserve quota bucket.
        "gpt-reserve" => Some("gpt-5.6-luna"),
        // The Daybreak aliases point to Sol (blue) and Cyber (red).
        "gpt-daybreak-blue-latest" => Some("gpt-5.6-sol"),
        "gpt-daybreak-red-latest" => Some("gpt-5.6-cyber"),
        _ => None,
    }
}

pub fn normalize_codex_model(raw: &str) -> String {
    let trimmed = raw.trim();
    let trimmed = trimmed.strip_prefix("openai/").unwrap_or(trimmed);
    if let Some(target) = codex_alias(trimmed) {
        return target.to_string();
    }
    if CODEX_MODELS.contains_key(trimmed) {
        return trimmed.to_string();
    }
    if let Some(m) = CODEX_DATED_RE.find(trimmed) {
        let base = &trimmed[..m.start()];
        if CODEX_MODELS.contains_key(base) {
            return base.to_string();
        }
    }
    trimmed.to_string()
}

pub fn normalize_claude_model(raw: &str) -> String {
    let mut trimmed = raw.trim().to_string();
    if let Some(rest) = trimmed.strip_prefix("anthropic.") {
        trimmed = rest.to_string();
    }
    // Strip Bedrock vendor prefixes like "us.anthropic.claude-sonnet-4-5"
    if let Some(last_dot) = trimmed.rfind('.') {
        if trimmed.contains("claude-") {
            let tail = &trimmed[last_dot + 1..];
            if tail.starts_with("claude-") {
                trimmed = tail.to_string();
            }
        }
    }
    // Strip Bedrock version suffix: "-v1:0"
    if let Some(m) = CLAUDE_BEDROCK_VER_RE.find(&trimmed) {
        trimmed = trimmed[..m.start()].to_string();
    }
    // If dated form ("-YYYYMMDD") matches a base in the table, prefer base
    if let Some(m) = CLAUDE_DATED_RE.find(&trimmed) {
        let base = &trimmed[..m.start()];
        if CLAUDE_MODELS.contains_key(base) {
            return base.to_string();
        }
    }
    // Exact match MUST precede the family fallback so a real key is never
    // shadowed by a sibling.
    if CLAUDE_MODELS.contains_key(trimmed.as_str()) {
        return trimmed;
    }
    // Claude family fallback (Swift parity —
    // CostUsageScanner.Pricing.familyFallback). When a freshly-released
    // minor (e.g. claude-opus-4-8) isn't priced yet, resolve to the
    // highest-numbered priced sibling in the same
    // claude-(opus|sonnet|haiku)-N-M family so Today/Week cost + charts
    // don't silently regress to $0 the day the model ships. This is
    // exactly how the missing claude-opus-4-7 entry was caught upstream.
    if let Some(fallback) = claude_family_fallback(&trimmed) {
        return fallback;
    }
    trimmed
}

/// Resolve an unknown `claude-(opus|sonnet|haiku)-N-Y` model to the
/// highest-numbered priced sibling `claude-(opus|sonnet|haiku)-N-X` in the
/// same family. Returns `None` if the stem doesn't parse or no sibling is
/// priced. Mirrors Swift `CostUsageScanner.Pricing.familyFallback` so the
/// desktop and Mac agree byte-for-byte on any not-yet-tabled minor —
/// both tables carry identical family keys.
///
/// The `minor < 100` cap keeps a legacy dated row like
/// `claude-opus-4-20250514` (where `20250514` is a date masquerading as a
/// minor) from beating real minors like 5 / 6 / 7.
fn claude_family_fallback(model: &str) -> Option<String> {
    if !CLAUDE_FAMILY_RE.is_match(model) {
        return None;
    }
    // Regex guarantees the `claude-<fam>-<gen>-<minor>` shape → 4 parts.
    let parts: Vec<&str> = model.split('-').collect();
    if parts.len() != 4 {
        return None;
    }
    let family = format!("claude-{}-{}-", parts[1], parts[2]);
    let mut best_key: Option<&'static str> = None;
    let mut best_minor: i64 = -1;
    for &key in CLAUDE_MODELS.keys() {
        let Some(tail) = key.strip_prefix(family.as_str()) else {
            continue;
        };
        // Only bare numeric minors are siblings; dated rows
        // (`5-20251101`) fail this parse and are skipped.
        let Ok(minor) = tail.parse::<i64>() else {
            continue;
        };
        if minor >= 100 {
            continue; // date-masquerade guard
        }
        if minor > best_minor {
            best_minor = minor;
            best_key = Some(key);
        }
    }
    best_key.map(|k| k.to_string())
}

/// Whether `model` has Codex rates. A model without them has no cost at all
/// (shown as unknown), which is different from a model priced at zero.
pub fn codex_model_is_priced(model: &str) -> bool {
    CODEX_MODELS.contains_key(normalize_codex_model(model).as_str())
}

/// The rates `model` had at `at_unix_ms` (`None` = today's rates).
pub fn codex_rates_at(model: &str, at_unix_ms: Option<i64>) -> Option<CodexModel> {
    let key = normalize_codex_model(model);
    if let (Some(at), Some((repriced_at, earlier))) =
        (at_unix_ms, CODEX_EARLIER_RATES.get(key.as_str()))
    {
        if at < *repriced_at {
            return Some(*earlier);
        }
    }
    CODEX_MODELS.get(key.as_str()).copied()
}

/// USD cost of ONE Codex request, at the rates in force at `at_unix_ms`
/// (`None` = today's rates). `None` when the model has no rates.
///
/// OpenAI reports `input_tokens` as the whole prompt, with cached reads a
/// subset of it, so cached tokens are clamped to the input and billed at the
/// cached rate while the rest is billed at the input rate. When the request's
/// input exceeds `CODEX_LONG_CONTEXT_INPUT_TOKENS` and the model has a
/// long-context tier, the whole request uses the long-context rates.
///
/// Call this per request. Summing a day's tokens first and pricing the total
/// would put almost every busy day into the long-context tier.
pub fn codex_cost_usd(
    model: &str,
    input_tokens: i64,
    cached_input_tokens: i64,
    output_tokens: i64,
    at_unix_ms: Option<i64>,
) -> Option<f64> {
    let priced = codex_rates_at(model, at_unix_ms)?;
    let total_input = input_tokens.max(0);
    let cached = cached_input_tokens.max(0).min(total_input);
    let non_cached = total_input - cached;
    let r = match priced.long_context {
        Some(long) if total_input > CODEX_LONG_CONTEXT_INPUT_TOKENS => CodexRates {
            input: long.input,
            cache_read: long.cache_read.or(priced.standard.cache_read),
            output: long.output,
        },
        _ => priced.standard,
    };
    let cached_rate = r.cache_read.unwrap_or(r.input);
    Some(
        non_cached as f64 * r.input
            + cached as f64 * cached_rate
            + output_tokens.max(0) as f64 * r.output,
    )
}

pub fn claude_cost_usd(
    model: &str,
    input_tokens: i64,
    cache_read_input_tokens: i64,
    cache_creation_input_tokens: i64,
    output_tokens: i64,
) -> Option<f64> {
    let key = normalize_claude_model(model);
    let p = CLAUDE_MODELS.get(key.as_str())?;

    fn tiered(tokens: i64, base: f64, above: Option<f64>, threshold: Option<i64>) -> f64 {
        let tokens = tokens.max(0);
        match (threshold, above) {
            (Some(t), Some(a)) => {
                let below = tokens.min(t);
                let over = (tokens - t).max(0);
                below as f64 * base + over as f64 * a
            }
            _ => tokens as f64 * base,
        }
    }

    Some(
        tiered(input_tokens, p.input, p.input_above, p.threshold)
            + tiered(
                cache_read_input_tokens,
                p.cache_read,
                p.cache_read_above,
                p.threshold,
            )
            + tiered(
                cache_creation_input_tokens,
                p.cache_creation,
                p.cache_creation_above,
                p.threshold,
            )
            + tiered(output_tokens, p.output, p.output_above, p.threshold),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_codex_dated_suffix_strips_to_base() {
        assert_eq!(
            normalize_codex_model("gpt-5-codex-2025-11-15"),
            "gpt-5-codex"
        );
        assert_eq!(normalize_codex_model("openai/gpt-5.4"), "gpt-5.4");
        assert_eq!(normalize_codex_model("unknown-model"), "unknown-model");
    }

    #[test]
    fn normalize_claude_bedrock_prefix_strips() {
        assert_eq!(
            normalize_claude_model("us.anthropic.claude-sonnet-4-5-v1:0"),
            "claude-sonnet-4-5"
        );
        assert_eq!(
            normalize_claude_model("claude-sonnet-4-5-20250929"),
            "claude-sonnet-4-5"
        );
    }

    #[test]
    fn codex_cost_basic_gpt5() {
        // 1M input @ $1.25/M, 0 cache, 100K output @ $10/M  = $1.25 + $1.00 = $2.25
        let c = codex_cost_usd("gpt-5", 1_000_000, 0, 100_000, None).unwrap();
        assert!((c - 2.25).abs() < 1e-9, "expected 2.25, got {}", c);
    }

    #[test]
    fn codex_cost_cached_input_discounted() {
        // 1M input, 500K of it cached (10x cheaper), 0 output
        // 500K @ $1.25/M + 500K @ $0.125/M = $0.625 + $0.0625 = $0.6875
        let c = codex_cost_usd("gpt-5", 1_000_000, 500_000, 0, None).unwrap();
        assert!((c - 0.6875).abs() < 1e-9, "expected 0.6875, got {}", c);
    }

    #[test]
    fn claude_cost_sonnet_tier_boundary() {
        // 200K input (at threshold) @ $3/M = $0.60
        let c = claude_cost_usd("claude-sonnet-4-6", 200_000, 0, 0, 0).unwrap();
        assert!((c - 0.60).abs() < 1e-9, "expected 0.60, got {}", c);

        // 300K input: 200K @ $3/M + 100K @ $6/M = $0.60 + $0.60 = $1.20
        let c = claude_cost_usd("claude-sonnet-4-6", 300_000, 0, 0, 0).unwrap();
        assert!((c - 1.20).abs() < 1e-9, "expected 1.20, got {}", c);
    }

    #[test]
    fn claude_cost_haiku_no_tier() {
        // Haiku has no threshold — 1M input @ $1/M = $1.00
        let c = claude_cost_usd("claude-haiku-4-5", 1_000_000, 0, 0, 0).unwrap();
        assert!((c - 1.00).abs() < 1e-9, "expected 1.00, got {}", c);
    }

    #[test]
    fn claude_cost_opus_4_7_priced_like_4_5_4_6() {
        // Opus 4.7 was missing from the table in v0.2.11, causing per-row Cost
        // to render "—" and the 7-day chart / Provider quota bar to collapse
        // to zero. Same per-token rates as Opus 4.5 / 4.6.
        // 1M input @ $5/M + 100K output @ $25/M = $5.00 + $2.50 = $7.50
        let c = claude_cost_usd("claude-opus-4-7", 1_000_000, 0, 0, 100_000).unwrap();
        assert!((c - 7.50).abs() < 1e-9, "expected 7.50, got {}", c);
    }

    #[test]
    fn unknown_model_returns_none() {
        assert!(codex_cost_usd("gpt-42-unicorn", 1000, 0, 0, None).is_none());
        assert!(claude_cost_usd("claude-opus-99", 1000, 0, 0, 0).is_none());
    }

    // ---- Codex rates (CodexBar table, dated rates, long context) ----

    fn close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-9,
            "expected {expected}, got {actual}"
        );
    }

    /// 2026-08-20T12:00:00Z and 2026-08-22T12:00:00Z: a day either side of the
    /// Sol repricing.
    const AUG_20_NOON_MS: i64 = 1_787_227_200_000;
    const AUG_22_NOON_MS: i64 = 1_787_400_000_000;

    #[test]
    fn gpt_5_5_uses_published_rates_not_the_gpt_5_4_placeholder() {
        // 100K input @ $5/M + 10K output @ $30/M = $0.50 + $0.30 = $0.80
        // (the placeholder gave $0.25 + $0.15).
        close(
            codex_cost_usd("gpt-5.5", 100_000, 0, 10_000, None).unwrap(),
            0.80,
        );
        // 200K input, 150K of it cached: 50K @ $5/M + 150K @ $0.50/M = $0.325.
        close(
            codex_cost_usd("gpt-5.5", 200_000, 150_000, 0, None).unwrap(),
            0.325,
        );
    }

    #[test]
    fn new_models_are_priced() {
        // 100K input + 10K output each, today's rates.
        close(
            codex_cost_usd("gpt-6-astra", 100_000, 0, 10_000, None).unwrap(),
            100_000.0 * 1e-5 + 10_000.0 * 5e-5,
        );
        close(
            codex_cost_usd("gpt-5.6-sol", 100_000, 0, 10_000, None).unwrap(),
            100_000.0 * 4e-6 + 10_000.0 * 2e-5,
        );
        close(
            codex_cost_usd("gpt-5.6-terra", 100_000, 0, 10_000, None).unwrap(),
            100_000.0 * 2e-6 + 10_000.0 * 1.2e-5,
        );
        close(
            codex_cost_usd("gpt-5.6-luna", 100_000, 0, 10_000, None).unwrap(),
            100_000.0 * 2e-7 + 10_000.0 * 1.2e-6,
        );
        close(
            codex_cost_usd("gpt-5.6-cyber", 100_000, 0, 10_000, None).unwrap(),
            100_000.0 * 1.25e-5 + 10_000.0 * 7.5e-5,
        );
        close(
            codex_cost_usd("gpt-5.5-cyber", 100_000, 0, 10_000, None).unwrap(),
            100_000.0 * 1.25e-5 + 10_000.0 * 7.5e-5,
        );
    }

    #[test]
    fn long_context_switches_the_whole_request_above_272k_input() {
        // Exactly 272K is still standard: 272K @ $10/M = $2.72.
        close(
            codex_cost_usd("gpt-6-astra", 272_000, 0, 0, None).unwrap(),
            2.72,
        );
        // One token more and EVERY token is at the long-context rate
        // ($20/M), not just the one above the line: 272,001 @ $20/M.
        close(
            codex_cost_usd("gpt-6-astra", 272_001, 0, 0, None).unwrap(),
            272_001.0 * 2e-5,
        );
        // Output and cached input switch too: 300K input with 200K cached and
        // 10K output = 100K @ $20/M + 200K @ $2/M + 10K @ $75/M
        // = $2.00 + $0.40 + $0.75.
        close(
            codex_cost_usd("gpt-6-astra", 300_000, 200_000, 10_000, None).unwrap(),
            3.15,
        );
        // gpt-5.4 gained the same tier: 300K @ $5/M = $1.50.
        close(
            codex_cost_usd("gpt-5.4", 300_000, 0, 0, None).unwrap(),
            1.50,
        );
        // A flat model never switches: 300K @ $1.25/M = $0.375.
        close(codex_cost_usd("gpt-5", 300_000, 0, 0, None).unwrap(), 0.375);
    }

    #[test]
    fn sol_is_billed_at_the_rate_of_the_request_day() {
        // 100K input (under the 272K line). Before 2026-08-21: $5/M = $0.50.
        // From 2026-08-21: $4/M = $0.40.
        let before = codex_cost_usd("gpt-5.6-sol", 100_000, 0, 0, Some(AUG_20_NOON_MS));
        let after = codex_cost_usd("gpt-5.6-sol", 100_000, 0, 0, Some(AUG_22_NOON_MS));
        close(before.unwrap(), 0.50);
        close(after.unwrap(), 0.40);
        // No time = today's rate.
        close(
            codex_cost_usd("gpt-5.6-sol", 100_000, 0, 0, None).unwrap(),
            0.40,
        );
        // The earlier rates keep their own long-context tier: 300K @ $10/M.
        close(
            codex_cost_usd("gpt-5.6-sol", 300_000, 0, 0, Some(AUG_20_NOON_MS)).unwrap(),
            3.0,
        );
        // Today's long-context rate for comparison: 300K @ $8/M.
        close(
            codex_cost_usd("gpt-5.6-sol", 300_000, 0, 0, Some(AUG_22_NOON_MS)).unwrap(),
            2.4,
        );
    }

    #[test]
    fn repricing_boundaries_are_exact() {
        // The last millisecond before the change is billed at the old rate,
        // the first millisecond of the change day at the new one. 100K input
        // each, under the long-context line.
        let cost = |model: &str, at: i64| codex_cost_usd(model, 100_000, 0, 0, Some(at)).unwrap();
        let sol = CODEX_SOL_REPRICED_UNIX_MS;
        close(cost("gpt-5.6-sol", sol - 1), 0.50);
        close(cost("gpt-5.6-sol", sol), 0.40);
        let tl = CODEX_TERRA_LUNA_REPRICED_UNIX_MS;
        close(cost("gpt-5.6-terra", tl - 1), 0.25);
        close(cost("gpt-5.6-terra", tl), 0.20);
        // Luna was five times today's price before the change.
        close(cost("gpt-5.6-luna", tl - 1), 0.10);
        close(cost("gpt-5.6-luna", tl), 0.02);
        // The dates are the ones the constants claim.
        let as_utc = |ms: i64| {
            chrono::DateTime::<chrono::Utc>::from_timestamp_millis(ms)
                .unwrap()
                .format("%Y-%m-%dT%H:%M:%S")
                .to_string()
        };
        assert_eq!(as_utc(sol), "2026-08-21T00:00:00");
        assert_eq!(as_utc(tl), "2026-07-30T00:00:00");
        assert_eq!(as_utc(AUG_20_NOON_MS), "2026-08-20T12:00:00");
        assert_eq!(as_utc(AUG_22_NOON_MS), "2026-08-22T12:00:00");
    }

    #[test]
    fn models_without_a_repricing_ignore_the_date() {
        let early = Some(1_700_000_000_000); // 2023
        assert_eq!(
            codex_cost_usd("gpt-6-astra", 1_000, 0, 0, early),
            codex_cost_usd("gpt-6-astra", 1_000, 0, 0, None)
        );
    }

    #[test]
    fn codex_aliases_resolve_to_priced_models() {
        assert_eq!(normalize_codex_model("gpt-5.6"), "gpt-5.6-sol");
        assert_eq!(normalize_codex_model("openai/gpt-5.6"), "gpt-5.6-sol");
        assert_eq!(normalize_codex_model("gpt-reserve"), "gpt-5.6-luna");
        assert_eq!(
            normalize_codex_model("gpt-daybreak-blue-latest"),
            "gpt-5.6-sol"
        );
        assert_eq!(
            normalize_codex_model("gpt-daybreak-red-latest"),
            "gpt-5.6-cyber"
        );
        assert_eq!(
            normalize_codex_model("gpt-6-astra-2026-09-01"),
            "gpt-6-astra"
        );
        // An alias is priced exactly like its target, dated rates included.
        assert_eq!(
            codex_cost_usd("gpt-5.6", 1_000, 0, 0, Some(AUG_20_NOON_MS)),
            codex_cost_usd("gpt-5.6-sol", 1_000, 0, 0, Some(AUG_20_NOON_MS))
        );
    }

    #[test]
    fn gpt_5_5_codex_costs_the_same_as_gpt_5_5() {
        for (input, cached, output) in [(1_000, 0, 10), (300_000, 100_000, 5_000)] {
            assert_eq!(
                codex_cost_usd("gpt-5.5-codex", input, cached, output, None),
                codex_cost_usd("gpt-5.5", input, cached, output, None)
            );
        }
    }

    #[test]
    fn priced_means_has_rates_even_at_zero() {
        assert!(codex_model_is_priced("gpt-6-astra"));
        assert!(codex_model_is_priced("gpt-5.6"));
        // Spark is priced, at $0: a known zero, not an unknown.
        assert!(codex_model_is_priced("gpt-5.3-codex-spark"));
        assert_eq!(
            codex_cost_usd("gpt-5.3-codex-spark", 500_000, 0, 1_000, None),
            Some(0.0)
        );
        assert!(!codex_model_is_priced("gpt-42-unicorn"));
    }

    #[test]
    fn codex_table_rows_are_internally_consistent() {
        // Catches a transposed argument in a row: cached input never costs
        // more than input, output never less than input, and a long-context
        // tier is never cheaper than the standard rates it replaces.
        let all = CODEX_MODELS
            .iter()
            .map(|(k, v)| (format!("{k} (today)"), *v))
            .chain(
                CODEX_EARLIER_RATES
                    .iter()
                    .map(|(k, (_, v))| (format!("{k} (earlier)"), *v)),
            );
        for (name, model) in all {
            let s = model.standard;
            if let Some(c) = s.cache_read {
                assert!(c <= s.input, "{name}: cached rate above input rate");
            }
            assert!(s.output >= s.input, "{name}: output rate below input rate");
            if let Some(l) = model.long_context {
                assert!(l.input >= s.input, "{name}: long-context input cheaper");
                assert!(l.output >= s.output, "{name}: long-context output cheaper");
                if let (Some(lc), Some(sc)) = (l.cache_read, s.cache_read) {
                    assert!(lc >= sc, "{name}: long-context cached cheaper");
                    assert!(lc <= l.input, "{name}: long-context cached above input");
                }
            }
        }
    }

    #[test]
    fn earlier_rates_only_cover_priced_models() {
        for key in CODEX_EARLIER_RATES.keys() {
            assert!(
                CODEX_MODELS.contains_key(key),
                "{key} has earlier rates but no current row"
            );
        }
    }

    // ---- Claude family fallback (Swift-parity; guards $0-on-new-minor) ----

    #[test]
    fn family_fallback_unknown_minor_resolves_to_latest_sibling() {
        // claude-opus-4-8 isn't tabled → highest priced opus-4-* sibling is 4-7.
        assert_eq!(normalize_claude_model("claude-opus-4-8"), "claude-opus-4-7");
        // Still 4-7 for an even-higher unseen minor (7 is the max known).
        assert_eq!(normalize_claude_model("claude-opus-4-9"), "claude-opus-4-7");
    }

    #[test]
    fn family_fallback_prices_new_minor_like_sibling_not_zero() {
        // THE regression this fixes: a fresh minor must NOT collapse to $0.
        // 1M input @ $5/M + 100K output @ $25/M = $7.50 (same as opus 4.7).
        let c = claude_cost_usd("claude-opus-4-8", 1_000_000, 0, 0, 100_000)
            .expect("unknown opus minor must price via family fallback, not None");
        assert!((c - 7.50).abs() < 1e-9, "expected 7.50, got {c}");
        // Byte-identical to the resolved sibling (the wire invariant).
        let sib = claude_cost_usd("claude-opus-4-7", 1_000_000, 0, 0, 100_000).unwrap();
        assert!((c - sib).abs() < 1e-12);
    }

    #[test]
    fn family_fallback_sonnet_carries_tiered_fields() {
        // Highest bare-minor sonnet-4-* sibling is 4-6 (tiered above 200K).
        assert_eq!(
            normalize_claude_model("claude-sonnet-4-9"),
            "claude-sonnet-4-6"
        );
        // 300K input: 200K @ $3/M + 100K @ $6/M = $1.20 — proves the sibling's
        // threshold/above fields carried through the fallback.
        let c = claude_cost_usd("claude-sonnet-4-9", 300_000, 0, 0, 0).unwrap();
        assert!((c - 1.20).abs() < 1e-9, "expected 1.20 (tiered), got {c}");
    }

    #[test]
    fn family_fallback_exact_match_takes_precedence() {
        // A real key must resolve to itself, never a higher sibling.
        assert_eq!(normalize_claude_model("claude-opus-4-6"), "claude-opus-4-6");
        assert_eq!(normalize_claude_model("claude-opus-4-5"), "claude-opus-4-5");
    }

    #[test]
    fn family_fallback_new_generation_has_no_sibling() {
        // A brand-new GENERATION (no opus-5-* priced) can't fall back →
        // returns itself → still None cost (matches Mac; acceptable).
        assert_eq!(normalize_claude_model("claude-opus-5-0"), "claude-opus-5-0");
        assert!(claude_cost_usd("claude-opus-5-0", 1000, 0, 0, 0).is_none());
    }

    #[test]
    fn family_fallback_ignores_date_masquerade_sibling() {
        // The minor<100 cap: claude-opus-4-8 must resolve to 4-7, NOT the
        // dated claude-opus-4-20250514 (where 20250514 parses as a huge minor).
        assert_ne!(
            normalize_claude_model("claude-opus-4-8"),
            "claude-opus-4-20250514"
        );
        // And a single-number form (not major-minor) never fallback-matches.
        assert_eq!(normalize_claude_model("claude-opus-99"), "claude-opus-99");
    }
}
