import type { TFunction } from "i18next";

// Tauri commands fail with English prose (`Err(String)` in src-tauri), and the UI used
// to render it with String(e). This recognizes the messages the Rust side actually
// writes and says them in the UI language, keeping any technical detail (an HTTP body,
// a keychain backend message) verbatim after the sentence. Anything unrecognized is
// shown as it arrived.
//
// The Rust strings stay English on purpose: looks_like_auth_failure,
// looks_like_transient_5xx, ALREADY_DECIDED and the updater categorizer match them.
// commandErrors.test.ts reads the Rust sources and fails if a message this list
// depends on is reworded there.

type Rule = { match: RegExp; render: (m: RegExpMatchArray, t: TFunction) => string };

export const COMMAND_ERROR_RULES: Rule[] = [
  { match: /^Sign in required to /, render: (_, t) => t("errors.sign_in_required") },
  {
    match: /^(Session expired — sign in again|Your sign-in expired\. Please sign in again\.)/,
    render: (_, t) => t("errors.session_expired"),
  },
  { match: /^Too many tries — please wait a minute and try again\.$/, render: (_, t) => t("errors.rate_limited") },
  { match: /^Invalid or expired code\.$/, render: (_, t) => t("errors.invalid_code") },
  { match: /^Network error: ([\s\S]*)$/, render: (m, t) => t("errors.network", { detail: m[1] }) },
  { match: /^Email is empty$/, render: (_, t) => t("errors.email_empty") },
  { match: /^Email or code is empty$/, render: (_, t) => t("errors.email_or_code_empty") },
  { match: /^Pairing code is empty$/, render: (_, t) => t("errors.pairing_code_empty") },
  { match: /^Prompt command requires non-empty payload\.$/, render: (_, t) => t("errors.prompt_empty") },
  { match: /^Device not paired — pair first, then set budgets$/, render: (_, t) => t("errors.not_paired_budgets") },
  { match: /^Device not paired( yet)?$/, render: (_, t) => t("errors.not_paired") },
  { match: /^OS keychain not available\. On Linux/, render: (_, t) => t("errors.keychain_unavailable") },
  { match: /^Keychain (?:error|unavailable): ([\s\S]*)$/, render: (m, t) => t("errors.keychain", { detail: m[1] }) },
  {
    match: /^Supabase HTTP (\d+): ([\s\S]*)$/,
    render: (m, t) => t("errors.server_http", { status: m[1], detail: m[2] }),
  },
  {
    match: /^Auth error \(HTTP (\d+)\): ([\s\S]*)$/,
    render: (m, t) => t("errors.auth_http", { status: m[1], detail: m[2] }),
  },
  { match: /^too many local terminals open \(max (\d+)\)$/, render: (m, t) => t("errors.too_many_terminals", { max: m[1] }) },
  { match: /^no Downloads directory available$/, render: (_, t) => t("errors.no_downloads_dir") },
  { match: /^invalid export filename$/, render: (_, t) => t("errors.invalid_export_filename") },
  { match: /^Failed to wipe scan cache: ([\s\S]*)$/, render: (m, t) => t("errors.wipe_cache_failed", { detail: m[1] }) },
  { match: /^could not resolve home directory$/, render: (_, t) => t("errors.no_home_dir") },
];

/** A command failure as a sentence in the UI language; the raw text when unrecognized. */
export function describeError(error: unknown, t: TFunction): string {
  const raw = typeof error === "string" ? error : error instanceof Error ? error.message : String(error);
  for (const rule of COMMAND_ERROR_RULES) {
    const m = raw.match(rule.match);
    if (m) return rule.render(m, t);
  }
  return raw;
}
