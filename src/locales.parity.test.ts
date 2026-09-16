import { describe, it, expect } from "vitest";
import i18n, { SUPPORTED_LANGS } from "./i18n";

// Locale catalogue gate, read from the RAW JSON.
//
// The older "required keys" test resolves through `t()`, and `fallbackLng` is
// "en", so a key missing from ja comes back as the English string and passes.
// Even `i18n.exists(key, { fallbackLng: false })` answers true for it. Nothing
// else compared the catalogues, so nothing could fail for a non-English locale.
//
// What this checks, per language in SUPPORTED_LANGS:
//   * the file exists, i18n registers it, and no unregistered file sits in locales/
//   * the same keys as en (missing and orphan), no empty values
//   * plural families carry exactly the CLDR categories that language selects,
//     from Intl.PluralRules, with a bare key accepted as `other`. A missing
//     `_many` in es is not cosmetic: 1000000 then renders the ENGLISH string.
//   * the i18next v3 `_plural` suffix, which v4+ ignores (en rendered
//     "3 active alert" for every count)
//   * the same {{placeholder}} and single-brace {placeholder} names as en
// And across the code:
//   * every literal t("key") exists in en
//   * every en key is used: literally, or under a declared dynamic prefix

// Loaded through Vite so the files are the ones the bundle sees, and so this
// needs no Node typings (tsc checks test files as part of `npm run build`).
const LOCALE_FILES = import.meta.glob<Record<string, unknown>>("./locales/*.json", {
  eager: true,
  import: "default",
});
const SOURCE_FILES = import.meta.glob<string>(
  ["./**/*.ts", "./**/*.tsx", "!./**/*.test.ts", "!./**/*.test.tsx", "!./locales/**", "!./test/**"],
  { eager: true, query: "?raw", import: "default" },
);
const CATEGORIES = ["zero", "one", "two", "few", "many", "other"] as const;
const SUFFIX = new RegExp(`_(${CATEGORIES.join("|")}|plural)$`);

// Keys built at runtime with a template literal. Each prefix must still be
// used that way, so an entry cannot outlive the code that needed it.
const DYNAMIC_PREFIXES = [
  "time.unit_",
  "providers.status_",
  "settings.export_fmt_",
  "sessions.confidence_",
  "machine.batt_",
  "remote.session_status_",
];

type Flat = Record<string, string>;

function flatten(obj: Record<string, unknown>, prefix = ""): Flat {
  const out: Flat = {};
  for (const [k, v] of Object.entries(obj)) {
    const key = prefix ? `${prefix}.${k}` : k;
    if (v !== null && typeof v === "object") Object.assign(out, flatten(v as Record<string, unknown>, key));
    else out[key] = String(v);
  }
  return out;
}

function load(code: string): Flat {
  const json = LOCALE_FILES[`./locales/${code}.json`];
  if (!json) throw new Error(`src/locales/${code}.json does not exist`);
  return flatten(json);
}

function placeholders(value: string): string[] {
  const double = [...value.matchAll(/\{\{\s*([\w.]+)(?:\s*,[^}]*)?\}\}/g)].map((m) => `{{${m[1]}}}`);
  const single = [...value.replace(/\{\{[^}]*\}\}/g, "").matchAll(/\{(\w+)\}/g)].map((m) => `{${m[1]}}`);
  return [...double, ...single].sort();
}

/** Plural family base → the categories en declares for it. */
function pluralFamilies(cat: Flat): Map<string, Set<string>> {
  const families = new Map<string, Set<string>>();
  for (const key of Object.keys(cat)) {
    const m = key.match(SUFFIX);
    if (!m || m[1] === "plural") continue;
    const base = key.slice(0, -m[0].length);
    if (!families.has(base)) families.set(base, new Set());
    families.get(base)!.add(m[1]);
  }
  return families;
}

const en = load("en");
const enFamilies = pluralFamilies(en);
const codes = SUPPORTED_LANGS.map((l) => l.code as string);

/** en's value for a key as another locale names it (bare key = the `other` form). */
function englishFor(key: string): string | undefined {
  return en[key] ?? (enFamilies.has(key) ? en[`${key}_other`] : undefined);
}

describe("locale catalogues", () => {
  it("every supported language has a file, is registered, and nothing else is in locales/", () => {
    const files = Object.keys(LOCALE_FILES).map((f) => f.replace("./locales/", "").replace(/\.json$/, ""));
    expect(files.sort()).toEqual([...codes].sort());
    const registered = Object.keys((i18n.options.resources ?? {}) as Record<string, unknown>);
    expect(registered.sort()).toEqual([...codes].sort());
  });

  it("no key uses the i18next v3 `_plural` suffix, which v4+ ignores", () => {
    for (const code of codes) {
      const stale = Object.keys(load(code)).filter((k) => k.endsWith("_plural"));
      expect(stale, `${code}: use _one/_other (the CLDR categories), not _plural`).toEqual([]);
    }
  });

  it.each(codes)("%s: plural families carry exactly the CLDR categories the language selects", (code) => {
    const cat = load(code);
    const required = new Intl.PluralRules(code).resolvedOptions().pluralCategories as string[];
    const problems: string[] = [];
    for (const base of enFamilies.keys()) {
      for (const c of required) {
        const present = `${base}_${c}` in cat || (c === "other" && base in cat);
        if (!present) problems.push(`${base}: missing _${c}`);
      }
      for (const c of CATEGORIES) {
        if (!required.includes(c) && `${base}_${c}` in cat) problems.push(`${base}: _${c} is never selected in ${code}`);
      }
    }
    expect(problems).toEqual([]);
  });

  it.each(codes.filter((c) => c !== "en"))("%s: same keys as en, none empty", (code) => {
    const cat = load(code);
    const strip = (k: string) => (enFamilies.has(k.replace(SUFFIX, "")) ? k.replace(SUFFIX, "") : k);
    const enKeys = new Set(Object.keys(en).map(strip));
    const keys = new Set(Object.keys(cat).map(strip));
    expect([...enKeys].filter((k) => !keys.has(k)), `${code} is missing`).toEqual([]);
    expect([...keys].filter((k) => !enKeys.has(k)), `${code} has keys en does not`).toEqual([]);
    expect(Object.entries(cat).filter(([, v]) => v.trim() === "").map(([k]) => k), `${code} empty values`).toEqual([]);
  });

  it.each(codes.filter((c) => c !== "en"))("%s: same placeholders as en", (code) => {
    const mismatched: string[] = [];
    for (const [key, value] of Object.entries(load(code))) {
      const base = key.replace(SUFFIX, "");
      const english = en[key] ?? englishFor(enFamilies.has(base) ? base : key);
      if (english === undefined) continue;
      const want = placeholders(english);
      const got = placeholders(value);
      if (JSON.stringify(got) !== JSON.stringify(want)) mismatched.push(`${key}: ${got.join(" ")} vs en ${want.join(" ")}`);
    }
    expect(mismatched).toEqual([]);
  });

  it("every literal t() key exists in en, and every en key is used", () => {
    const code = Object.values(SOURCE_FILES).join("\n");
    expect(Object.keys(SOURCE_FILES)).toContain("./App.tsx");
    const literal = new Set<string>();
    for (const m of code.matchAll(/\bt\(\s*["']([\w.-]+)["']/g)) literal.add(m[1]);
    const known = (k: string) => k in en || enFamilies.has(k);
    expect([...literal].filter((k) => !known(k)), "t() keys that are in no catalogue").toEqual([]);

    const bases = new Set(Object.keys(en).map((k) => (enFamilies.has(k.replace(SUFFIX, "")) ? k.replace(SUFFIX, "") : k)));
    const unused = [...bases].filter(
      (k) => !code.includes(`"${k}"`) && !code.includes(`'${k}'`) && !DYNAMIC_PREFIXES.some((p) => k.startsWith(p)),
    );
    expect(unused, "en keys nothing references (delete them from every locale)").toEqual([]);

    const stalePrefixes = DYNAMIC_PREFIXES.filter((p) => !code.includes("`" + p + "${"));
    expect(stalePrefixes, "dynamic prefixes no template literal builds any more").toEqual([]);
  });
});
