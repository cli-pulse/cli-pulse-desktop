import i18n from "i18next";
import { initReactI18next } from "react-i18next";

import en from "./locales/en.json";
import zhCN from "./locales/zh-CN.json";
import zhTW from "./locales/zh-TW.json";
import ja from "./locales/ja.json";
import ko from "./locales/ko.json";
import es from "./locales/es.json";

/**
 * Supported UI languages. Keep in sync with the `locales/` directory
 * and the `<select>` in Settings → Language.
 */
export const SUPPORTED_LANGS = [
  { code: "en", label: "English" },
  { code: "zh-CN", label: "简体中文" },
  { code: "zh-TW", label: "繁體中文" },
  { code: "ja", label: "日本語" },
  { code: "ko", label: "한국어" },
  { code: "es", label: "Español" },
] as const;

export type LangCode = (typeof SUPPORTED_LANGS)[number]["code"];

const STORAGE_KEY = "cli-pulse.lang";

/**
 * An OS or browser language tag -> the UI language to show, or null.
 *
 * Case- and separator-insensitive (`zh_cn`, `ZH-CN`, `zh-Hans-CN`). Chinese goes by
 * script or region, not by the first entry that happens to share the `zh` subtag:
 * Traditional tags (Hant, TW, HK, MO) get zh-TW; every other Chinese tag gets zh-CN. Other languages match on the language subtag,
 * so en-GB is en and ja-JP is ja.
 */
export function resolveLanguage(tag: string | null | undefined): LangCode | null {
  if (!tag) return null;
  const parts = tag.replace(/_/g, "-").toLowerCase().split("-").filter(Boolean);
  if (parts.length === 0) return null;
  const codes: string[] = SUPPORTED_LANGS.map((l) => l.code);
  const exact = codes.find((c) => c.toLowerCase() === parts.join("-"));
  if (exact) return exact as LangCode;
  const [language, ...rest] = parts;
  if (language === "zh") {
    const traditional = rest.includes("hant") || rest.some((p) => p === "tw" || p === "hk" || p === "mo");
    const target = traditional && codes.includes("zh-TW") ? "zh-TW" : "zh-CN";
    return codes.includes(target) ? (target as LangCode) : null;
  }
  const byLanguage = codes.find((c) => c.toLowerCase() === language);
  return (byLanguage as LangCode | undefined) ?? null;
}

function detectInitialLang(): LangCode {
  // 1. User choice stashed in localStorage (set via Settings)
  const stored = (globalThis as any).localStorage?.getItem(STORAGE_KEY);
  if (stored && SUPPORTED_LANGS.some((l) => l.code === stored)) {
    return stored as LangCode;
  }
  // 2. The OS/browser preference list, in order, then the single preferred tag.
  const nav = (globalThis as any).navigator;
  const preferred: string[] = [...(nav?.languages ?? []), nav?.language].filter(Boolean);
  for (const tag of preferred) {
    const lang = resolveLanguage(tag);
    if (lang) return lang;
  }
  return "en";
}

/** The active UI language, for date and number formatting that should follow it. */
export function uiLocale(): string {
  return i18n.language || "en";
}

i18n.use(initReactI18next).init({
  resources: {
    en: { translation: en },
    "zh-CN": { translation: zhCN },
    "zh-TW": { translation: zhTW },
    ja: { translation: ja },
    ko: { translation: ko },
    es: { translation: es },
  },
  lng: detectInitialLang(),
  supportedLngs: SUPPORTED_LANGS.map((l) => l.code),
  fallbackLng: "en",
  interpolation: {
    escapeValue: false,
    // v0.4.6 — `{{n, number}}` formatter routes through Intl.NumberFormat
    // with the active language so 2782 renders as "2,782" in en/zh-CN/zh-TW/ja/ko
    // (all use the comma per CLDR; es groups with a period, and only from five
    // digits: 2782, 12.345). VM 2026-05-04 flagged that
    // numbers were being interpolated as raw `String(n)` ("2782") under
    // v0.4.5's plural-aware {{count}} interpolation, since by default
    // i18next doesn't run numbers through toLocaleString.
    format: (value, fmt, lng) => {
      if (fmt === "number" && typeof value === "number") {
        return value.toLocaleString(lng ?? "en-US");
      }
      return String(value);
    },
  },
  returnNull: false,
});

// <html lang> decides which CJK glyphs the WebView falls back to (Han
// unification: the same code point is drawn differently for zh-CN, zh-TW and ja)
// and how screen readers pronounce the page. index.html ships "en"; keep it true.
function syncDocumentLang(lng: string): void {
  if (typeof document !== "undefined" && document.documentElement) {
    document.documentElement.lang = lng;
  }
}
syncDocumentLang(i18n.language);
i18n.on("languageChanged", syncDocumentLang);

/**
 * Switch the active UI language. `i18next.changeLanguage` returns a
 * Promise that resolves once resources for `code` are loaded, but
 * because every locale is bundled at build time (statically
 * imported above), resolution is effectively synchronous in practice.
 * We still track the Promise so any future resource-loading error
 * surfaces as a console warning instead of an unhandled rejection.
 *
 * Caller can `await` if it cares about completion (Settings panel
 * doesn't — it triggers a re-render via React state change anyway).
 */
export function setLang(code: LangCode): Promise<void> {
  // Persist BEFORE switching — if changeLanguage somehow throws, we
  // still want the choice remembered for the next launch.
  try {
    (globalThis as any).localStorage?.setItem(STORAGE_KEY, code);
  } catch {
    /* localStorage can be unavailable in weird contexts — ignore */
  }
  return Promise.resolve(i18n.changeLanguage(code))
    .then(() => undefined)
    .catch((err) => {
      // Don't propagate — language switch failures shouldn't crash the
      // app. Log so they're not silently swallowed.
      console.warn("setLang(", code, ") failed:", err);
    });
}

export default i18n;
