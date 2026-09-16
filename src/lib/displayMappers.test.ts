import { describe, it, expect } from "vitest";
import i18n from "../i18n";
import manifest from "./quotaTierNames.json";
import { collectorStatusLabel, planLabel, presentAlert, quotaTierKey, quotaTierLabel } from "./displayMappers";

const LOCALES = import.meta.glob<Record<string, Record<string, string>>>("../locales/*.json", {
  eager: true,
  import: "default",
});
// This app's own alert producer. Its templates are what presentAlert must recognize.
const ALERTS_RS = import.meta.glob<string>("../../src-tauri/src/alerts.rs", {
  eager: true,
  query: "?raw",
  import: "default",
});

const en = i18n.getFixedT("en");
const ja = i18n.getFixedT("ja");

describe("quota tier names", () => {
  type Entry = { name: string; display: string; l10n_key?: string };
  const entries = manifest.entries as Entry[];

  it("every TRANSLATE name has its key in every locale and maps case-insensitively", () => {
    const translate = entries.filter((e) => e.display === "TRANSLATE");
    expect(translate.length).toBeGreaterThan(30);
    for (const e of translate) {
      const suffix = e.l10n_key!.replace(/^quota_tier\./, "");
      expect(quotaTierKey(e.name), e.name).toBe(suffix);
      expect(quotaTierKey(e.name.toUpperCase()), e.name).toBe(suffix);
      for (const [path, catalogue] of Object.entries(LOCALES)) {
        expect(catalogue.quota_tier?.[suffix], `${path} quota_tier.${suffix}`).toBeTruthy();
      }
    }
  });

  it("vendor terms and unknown names pass through", () => {
    for (const e of entries.filter((x) => x.display === "PASSTHROUGH")) {
      expect(quotaTierKey(e.name), e.name).toBeNull();
      expect(quotaTierLabel(e.name, ja)).toBe(e.name);
    }
    expect(quotaTierLabel("Some New Bucket", ja)).toBe("Some New Bucket");
    expect(quotaTierLabel("Weekly", ja)).not.toBe("Weekly");
  });
});

describe("plan and collector status", () => {
  it("translates only the server-composed plan and the status sentinels", () => {
    expect(planLabel("Multiple accounts", ja)).not.toBe("Multiple accounts");
    expect(planLabel("Max 20x", ja)).toBe("Max 20x");
    expect(planLabel("API key", ja)).toBe("API key");
    expect(collectorStatusLabel("Connected", ja)).not.toBe("Connected");
    expect(collectorStatusLabel("$12.34 balance", ja)).toBe("$12.34 balance");
  });
});

describe("alert presentation", () => {
  const alert = (id: string, type: string, title: string, message: string, related_provider: string | null = null) => ({
    id, type, title, message, related_provider,
  });

  it("recognizes each producer's wording and renders it in the UI language", () => {
    const rows = [
      alert("cpu-spike-mac-1", "Usage Spike", "Device CPU usage is elevated", "helper sampled CPU usage at 91%."),
      alert("session-spike-s1", "Usage Spike", "api-gateway is consuming high CPU", "Using ~37% of total system CPU (10 cores) for Claude."),
      alert("session-spike-s1", "Usage Spike", "api-gateway is consuming high CPU", "Process CPU is 184.5% for Claude."),
      alert("session-long-s1", "Session Too Long", "api-gateway has been running for a long time", "Long-running local agent session detected by helper."),
      alert("quota-Claude-Weekly-80", "Quota Warning", "Claude Weekly at 85%", "Quota window 'Weekly' is 85% used (15% remaining).", "Claude"),
      alert("quota-Codex-5h Window-95", "Quota Warning", "Codex 5h Window at 96%", "Quota window '5h Window' is 96% used (4% remaining) (resets 2026-09-16T14:00:00Z)."),
    ];
    for (const row of rows) {
      const shown = presentAlert(row, ja);
      expect(shown.recognized, row.message).toBe(true);
      expect(shown.title, row.title).not.toBe(row.title);
      expect(shown.message).not.toContain("{{");
    }
    expect(presentAlert(rows[4], en).title).toBe("Claude Weekly at 85%");
    expect(presentAlert(rows[5], en).title).toBe("Codex 5h Window at 96%");
  });

  // Asserted in Japanese: in English both templates render the identical sentence, so
  // an English assertion passes whichever matcher ran first.
  it("keeps this app's project out of the provider (matcher order)", () => {
    const shown = presentAlert(
      alert("session-spike-s1", "Usage Spike", "api-gateway is consuming high CPU", "Process CPU is 184.5% for Claude in acme."),
      ja,
    );
    expect(shown.message).toBe(ja("alert_kind.session_cpu_message_process_in_project", {
      percent: "184.5", provider: "Claude", project: "acme",
    }));
    expect(shown.message).not.toContain("Claude in acme");
  });

  it("shows unrecognized rows as stored", () => {
    const rows = [
      alert("a1", "Quota Critical", "Claude quota almost gone", "You have used 96% of your weekly quota."),
      alert("cpu-spike-mac-1", "Usage Spike", "Device CPU usage is elevated", "helper measured CPU at 91 percent."),
      alert("x-1", "Usage Spike", "Device CPU usage is elevated", "helper sampled CPU usage at 91%."),
    ];
    for (const row of rows) {
      expect(presentAlert(row, ja)).toEqual({ title: row.title, message: row.message, recognized: false });
    }
  });

  // If alerts.rs changes a template, this fails here instead of the desktop quietly
  // rendering its own alerts in English.
  it("recognizes the templates src-tauri/src/alerts.rs actually writes", () => {
    const source = Object.values(ALERTS_RS)[0];
    expect(source, "alerts.rs not found").toBeTruthy();
    const templates = [
      'title: format!("Daily budget exceeded — ${today_cost:.2}")',
      '"Today\'s spend of ${today_cost:.2} is above your daily budget of ${daily_limit:.2}."',
      'title: format!("Weekly budget exceeded — ${week_cost:.2}")',
      '"Last 7 days of spend totals ${week_cost:.2}, above your weekly budget of ${weekly_limit:.2}."',
      'title: format!("{} is consuming high CPU", short_name(&s.name))',
      '"Process CPU is {:.1}% for {} in {}."',
    ];
    for (const template of templates) expect(source, template).toContain(template);

    const daily = presentAlert(
      alert("budget-daily-2026-09-16", "Daily Budget Exceeded", "Daily budget exceeded — $12.34",
        "Today's spend of $12.34 is above your daily budget of $10.00."),
      ja,
    );
    const weekly = presentAlert(
      alert("budget-weekly-2026-W38", "Weekly Budget Exceeded", "Weekly budget exceeded — $88.20",
        "Last 7 days of spend totals $88.20, above your weekly budget of $50.00."),
      ja,
    );
    const spike = presentAlert(
      alert("session-spike-p1", "Usage Spike", "claude is consuming high CPU", "Process CPU is 184.5% for Claude in acme."),
      ja,
    );
    for (const shown of [daily, weekly, spike]) expect(shown.recognized).toBe(true);
    expect(daily.message).toContain("$12.34");
    expect(spike.message).toContain("acme");
  });
});
