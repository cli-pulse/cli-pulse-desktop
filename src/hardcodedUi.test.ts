import { describe, it, expect } from "vitest";
import { scanSource } from "./test/hardcodedUiScanner";
import baseline from "./hardcodedUi.baseline.json";

// English written straight into the UI: the half of localization the locale
// catalogue test cannot see, because a string that never became a key has no
// key to be missing. A RATCHET: a literal not in the baseline fails, and a
// baseline entry that no longer matches fails too, so the allowlist can only
// shrink. Every entry says why it may stay English (brand names, units, examples).

const SOURCES = import.meta.glob<string>(["./**/*.tsx", "!./**/*.test.tsx"], {
  eager: true,
  query: "?raw",
  import: "default",
});

type Entry = { file: string; text: string; count: number; reason: string };

describe("hardcoded UI copy", () => {
  it("every literal a user reads comes from t() or is allowlisted with a reason", () => {
    expect(Object.keys(SOURCES)).toContain("./App.tsx");
    const found = new Map<string, { file: string; text: string; lines: number[] }>();
    for (const [path, source] of Object.entries(SOURCES)) {
      const file = path.replace(/^\.\//, "src/");
      for (const hit of scanSource(file, source)) {
        const key = JSON.stringify([file, hit.text]);
        const entry = found.get(key) ?? { file, text: hit.text, lines: [] };
        entry.lines.push(hit.line);
        found.set(key, entry);
      }
    }
    const allowed = new Map<string, Entry>();
    for (const e of baseline.entries as Entry[]) {
      expect(e.reason.trim().length, `${e.file} ${e.text} needs a reason`).toBeGreaterThan(0);
      allowed.set(JSON.stringify([e.file, e.text]), e);
    }
    const problems: string[] = [];
    for (const [key, f] of found) {
      const extra = f.lines.length - (allowed.get(key)?.count ?? 0);
      if (extra > 0) problems.push(`NEW   ${f.file}:${f.lines[f.lines.length - 1]}  ${JSON.stringify(f.text)}`);
    }
    for (const [key, e] of allowed) {
      const present = found.get(key)?.lines.length ?? 0;
      if (present < e.count) problems.push(`STALE ${e.file}  ${JSON.stringify(e.text)} (baseline ${e.count}, found ${present})`);
    }
    expect(problems, "route copy through t() in every locale, or allowlist it with a reason").toEqual([]);
  });
});

describe("hardcoded UI scanner (negative controls)", () => {
  const texts = (src: string) => scanSource("x.tsx", src).map((h) => h.text);

  it("flags JSX text, display attributes and literal children", () => {
    expect(texts(`const A = () => <th className="px-3">Provider</th>;`)).toEqual(["Provider"]);
    expect(texts(`const A = () => <svg role="img" aria-label="7-day cost trend" />;`)).toEqual(["7-day cost trend"]);
    expect(texts(`const A = ({ ok }) => <span>{ok ? "Saved" : "Failed"}</span>;`)).toEqual(["Saved", "Failed"]);
    expect(texts("const A = ({ m }) => <b title={`${m} min left`}>x</b>;")).toEqual(["{} min left"]);
    expect(texts(`const A = ({ n }) => <span>{n} MB</span>;`)).toEqual(["MB"]);
  });

  it("does not flag keys, comparisons, call arguments, classes or statement blocks", () => {
    // Capitalized tokens, so only the rule under test can skip them (a lowercase
    // identifier would also pass the class-list rule and prove nothing).
    expect(texts(`const A = ({ t }) => <span title={t("x.tooltip")}>{t("x.label", { defaultValue: "Some English" })}</span>;`)).toEqual([]);
    expect(texts(`const A = ({ s }) => <span>{s === "Running" ? "▶" : "■"}</span>;`)).toEqual([]);
    expect(texts(`const A = ({ f, t }) => <div>{f("Open", t("alerts.filter_open"))}</div>;`)).toEqual([]);
    expect(texts(`const A = ({ on }) => <div>{on ? "px-2 bg-red-900/60" : "px-2"}</div>;`)).toEqual([]);
    expect(texts(`const A = () => <div>{(() => { const c = "Some English"; return c; })()}</div>;`)).toEqual([]);
    expect(texts(`const A = () => <div className="text-xs font-mono">42</div>;`)).toEqual([]);
  });
});
