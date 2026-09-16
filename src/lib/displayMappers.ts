import type { TFunction } from "i18next";
import manifest from "./quotaTierNames.json";

// Render-time labels for values the collectors, the server and other devices store in
// English. Ported from Apple's CLIPulseCore (L10n.quotaTier.localized,
// L10n.providers.planDisplay / localizedStatusText, AlertPresentation.swift) so a row
// reads the same on the desktop as on the Mac and iPhone.
//
// Only the RENDERING changes. The stored English keeps doing its other jobs: React
// keys, windowMinutesForTier's parse, the helper_sync upload, alert dedupe. Every
// mapper passes anything it does not recognize through unchanged, so a value from a
// newer client reads as itself rather than as a wrong translation.

// ---- quota tier names ----------------------------------------------------------
//
// quotaTierNames.json is vendored from cli-pulse scripts/quota_tier_names.json
// (at 212e29c6), the decision record every platform shares: generic words the apps
// composed TRANSLATE; vendor products, plans, models, coined units and currency codes
// PASSTHROUGH, so the row still matches the vendor's billing page.

type ManifestEntry = { name: string; display: string; l10n_key?: string };

const TIER_KEYS = new Map<string, string>(
  (manifest.entries as ManifestEntry[])
    .filter((e) => e.display === "TRANSLATE" && e.l10n_key)
    .map((e) => [e.name.toLowerCase(), e.l10n_key!.replace(/^quota_tier\./, "")]),
);

/** The locale key suffix under `quota_tier.` for a tier name, or null to show it raw. */
export function quotaTierKey(name: string): string | null {
  return TIER_KEYS.get(name.toLowerCase()) ?? null;
}

export function quotaTierLabel(name: string, t: TFunction): string {
  const suffix = quotaTierKey(name);
  return suffix ? t(`quota_tier.${suffix}`) : name;
}

// ---- plan type and collector status ----------------------------------------------

/**
 * Plan badge text. Only the value the SERVER composes is translated
 * ("Multiple accounts", app_rpc.sql). Vendor plans ("Max 20x", "Pro") pass through,
 * and so do the generic labels collectors upload ("API key", "Credits"): translating
 * those on one platform only would make the same row read differently per device.
 */
export function planLabel(plan: string, t: TFunction): string {
  return plan === "Multiple accounts" ? t("providers.plan_multiple_accounts") : plan;
}

/** The collector status line: sentinels translate; composed strings ("$12.34 balance") do not. */
export function collectorStatusLabel(text: string, t: TFunction): string {
  switch (text.toLowerCase()) {
    case "connected":
      return t("collector_status.connected");
    case "unknown":
      return t("collector_status.unknown");
    default:
      return text;
  }
}

// ---- alerts ---------------------------------------------------------------------
//
// Alerts carry no kind or parameters column, and rows arrive from three producers
// with their own wording: the macOS AlertGenerator, the Python helper, and this app
// (src-tauri/src/alerts.rs). The patterns are copied verbatim from Apple's
// AlertPresentation.swift; alertPresentation.test.ts pins them to this app's own
// alerts.rs templates. The server RPC evaluate_budget_alerts is not a producer: its
// production body inserts nothing.

type AlertLike = {
  id: string;
  type: string;
  title: string;
  message: string | null;
  related_provider?: string | null;
};

export const ALERT_PATTERNS = {
  deviceCpu: /^helper sampled CPU usage at (\d+(?:\.\d+)?)%\.$/,
  sessionCpuSystem: /^Using ~(\d+)% of total system CPU \((\d+) cores\) for (.+)\.$/,
  sessionCpuProcessInProject: /^Process CPU is (\d+(?:\.\d+)?)% for (.+) in (.+)\.$/,
  sessionCpuProcess: /^Process CPU is (\d+(?:\.\d+)?)% for (.+)\.$/,
  quota: /^Quota window '(.+)' is (\d+)% used \((\d+)% remaining\)(?: \(resets (.+)\))?\.$/,
  budgetDailyTitle: /^Daily budget exceeded — \$(.+)$/,
  budgetDailyMessage: /^Today's spend of \$(.+) is above your daily budget of \$(.+)\.$/,
  budgetWeeklyTitle: /^Weekly budget exceeded — \$(.+)$/,
  budgetWeeklyMessage: /^Last 7 days of spend totals \$(.+), above your weekly budget of \$(.+)\.$/,
} as const;

const stripSuffix = (s: string, suffix: string) => (s.endsWith(suffix) ? s.slice(0, -suffix.length) : s);

export type AlertText = { title: string; message: string; recognized: boolean };

/** Localized title and message for an alert row; the stored English when not recognized. */
export function presentAlert(a: AlertLike, t: TFunction): AlertText {
  const message = a.message ?? "";
  const hit = (title: string, body: string): AlertText => ({ title, message: body, recognized: true });

  if (a.type === "Usage Spike" && a.id.startsWith("cpu-spike-") && a.title === "Device CPU usage is elevated") {
    const m = ALERT_PATTERNS.deviceCpu.exec(message);
    if (m) return hit(t("alert_kind.device_cpu_title"), t("alert_kind.device_cpu_message", { percent: m[1] }));
  }

  if (a.type === "Usage Spike" && a.id.startsWith("session-spike-")) {
    const session = stripSuffix(a.title, " is consuming high CPU");
    const title = t("alert_kind.session_cpu_title", { session });
    let m = ALERT_PATTERNS.sessionCpuSystem.exec(message);
    if (m) {
      return hit(title, t("alert_kind.session_cpu_message_system", { percent: m[1], cores: m[2], provider: m[3] }));
    }
    // This app's own form, tried before the Python helper's: that pattern's greedy
    // `(.+)` would otherwise read "Claude in acme" as the provider.
    m = ALERT_PATTERNS.sessionCpuProcessInProject.exec(message);
    if (m) {
      return hit(title, t("alert_kind.session_cpu_message_process_in_project", { percent: m[1], provider: m[2], project: m[3] }));
    }
    m = ALERT_PATTERNS.sessionCpuProcess.exec(message);
    if (m) return hit(title, t("alert_kind.session_cpu_message_process", { percent: m[1], provider: m[2] }));
  }

  if (
    a.type === "Session Too Long" &&
    a.id.startsWith("session-long-") &&
    message === "Long-running local agent session detected by helper."
  ) {
    const session = stripSuffix(a.title, " has been running for a long time");
    return hit(t("alert_kind.session_long_title", { session }), t("alert_kind.session_long_message"));
  }

  if (a.type === "Quota Warning" && a.id.startsWith("quota-")) {
    const m = ALERT_PATTERNS.quota.exec(message);
    if (m) {
      const [, rawTier, used, remaining, reset] = m;
      const tier = quotaTierLabel(rawTier, t);
      const provider = a.related_provider ?? stripSuffix(a.title, ` ${rawTier} at ${used}%`);
      const body = reset
        ? t("alert_kind.quota_message_reset", { tier, used, remaining, reset })
        : t("alert_kind.quota_message", { tier, used, remaining });
      return hit(t("alert_kind.quota_title", { provider, tier, used }), body);
    }
  }

  if (a.type === "Daily Budget Exceeded" && a.id.startsWith("budget-daily-")) {
    const tm = ALERT_PATTERNS.budgetDailyTitle.exec(a.title);
    const mm = ALERT_PATTERNS.budgetDailyMessage.exec(message);
    if (tm && mm) {
      return hit(
        t("alert_kind.budget_daily_title", { amount: tm[1] }),
        t("alert_kind.budget_daily_message", { spend: mm[1], budget: mm[2] }),
      );
    }
  }

  if (a.type === "Weekly Budget Exceeded" && a.id.startsWith("budget-weekly-")) {
    const tm = ALERT_PATTERNS.budgetWeeklyTitle.exec(a.title);
    const mm = ALERT_PATTERNS.budgetWeeklyMessage.exec(message);
    if (tm && mm) {
      return hit(
        t("alert_kind.budget_weekly_title", { amount: tm[1] }),
        t("alert_kind.budget_weekly_message", { spend: mm[1], budget: mm[2] }),
      );
    }
  }

  return { title: a.title, message, recognized: false };
}
