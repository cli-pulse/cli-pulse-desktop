import { describe, it, expect } from "vitest";

// zh-TW terminology, read from the RAW JSON (the same way locales.parity.test.ts does).
//
// Decided in 2026-09 for every CLI Pulse platform, with the evidence that settled it:
//
//   alert   = 警示, never 告警. Apple's macOS zh_TW tables use 警示 34 times and 告警 0.
//   session = 工作階段, never 會話. Microsoft and Google zh-TW both say 工作階段 (Chrome's
//             "工作階段 Cookie"), and Apple's own zh_TW never uses 會話: 0 times in 5,114
//             tables, which say 工作階段 / 作業階段 / 階段 instead (「等待遠端工作階段」).
//             會話 is the mainland word; a Taiwan reader sees it as Simplified-Chinese copy.
//
// Two terms deliberately DIFFER from the Mac/iPhone zh-Hant catalogue. This app runs on
// Windows, so it follows Microsoft's zh-TW glossary, the words its users already read in
// Task Manager and Settings. Do not "align" them with the Apple catalogue:
//
//   process = 處理程序 (Microsoft). Apple says 程序 (Apple zh_TW: 程序 390 / 處理程序 0).
//   account = 帳戶     (Microsoft). Apple and Android say 帳號.
//
// "transcripts" is 對話記錄, not 工作階段記錄: it means the conversation text.

const LOCALE_FILES = import.meta.glob<Record<string, unknown>>("./locales/zh-TW.json", {
  eager: true,
  import: "default",
});

const FORBIDDEN: Record<string, string> = {
  告警: "alert is 警示 in zh-TW",
  會話: "session is 工作階段 in zh-TW",
};

function flatten(obj: Record<string, unknown>, prefix = ""): Record<string, string> {
  const out: Record<string, string> = {};
  for (const [k, v] of Object.entries(obj)) {
    const key = prefix ? `${prefix}.${k}` : k;
    if (v !== null && typeof v === "object") Object.assign(out, flatten(v as Record<string, unknown>, key));
    else out[key] = String(v);
  }
  return out;
}

describe("zh-TW terminology", () => {
  const json = LOCALE_FILES["./locales/zh-TW.json"];
  const cat = json ? flatten(json) : {};

  it("reads the real catalogue", () => {
    // An empty or missing file would make the scan below pass with nothing to scan.
    expect(json, "src/locales/zh-TW.json not found").toBeTruthy();
    expect(Object.keys(cat).length).toBeGreaterThan(300);
    expect(cat["tab.sessions"]).toBe("工作階段");
    expect(cat["tab.alerts"]).toBe("警示");
  });

  it.each(Object.keys(FORBIDDEN))("never uses %s", (term) => {
    const hits = Object.entries(cat)
      .filter(([, value]) => value.includes(term))
      .map(([key, value]) => `${key}: ${value}`);
    expect(hits, FORBIDDEN[term]).toEqual([]);
  });
});
