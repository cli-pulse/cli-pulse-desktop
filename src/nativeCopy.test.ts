import { describe, it, expect } from "vitest";
import { formatMoney, moneyDisplayMeta } from "./lib/money";

const APP = import.meta.glob<string>("./App.tsx", { eager: true, query: "?raw", import: "default" });
const LIB_RS = import.meta.glob<string>("../src-tauri/src/lib.rs", { eager: true, query: "?raw", import: "default" });

const snake = (camel: string) => camel.replace(/[A-Z]/g, (c) => `_${c.toLowerCase()}`);

/** Field names of a Rust struct, and which of them are Option<…>. */
function rustFields(source: string, name: string): { all: string[]; required: string[] } {
  const body = new RegExp(`struct ${name} \\{([^}]*)\\}`).exec(source)?.[1] ?? "";
  const fields = [...body.matchAll(/^\s*(\w+):\s*([^,\n]+),/gm)].map((m) => ({ name: m[1], type: m[2] }));
  return { all: fields.map((f) => f.name), required: fields.filter((f) => !f.type.startsWith("Option<")).map((f) => f.name) };
}

/** camelCase keys of the object literal assigned to `label:` inside pushTrayCopyFromI18n. */
function sentKeys(source: string, label: string): string[] {
  const fn = source.slice(source.indexOf("function pushTrayCopyFromI18n"));
  const start = fn.indexOf(`${label}: {`);
  const block = fn.slice(start, fn.indexOf("}", start));
  return [...block.matchAll(/^\s*(\w+)(?::|,)/gm)].map((m) => m[1]).filter((k) => k !== label);
}

// serde ignores unknown fields and the newer ones are Option<…>, so a misspelt key on
// either side does not fail: that string silently stays English. Pin the contract.
describe("tray and notification copy pushed to Rust", () => {
  const app = Object.values(APP)[0];
  const lib = Object.values(LIB_RS)[0];

  it.each([
    ["copy", "TrayCopyPayload"],
    ["notification", "NotificationCopyPayload"],
  ])("%s keys match %s", (label, struct) => {
    expect(app && lib, "sources not found").toBeTruthy();
    const sent = sentKeys(app, label).map(snake);
    const fields = rustFields(lib, struct);
    expect(fields.all.length).toBeGreaterThan(8);
    expect(sent.filter((k) => !fields.all.includes(k)), "sent but no such field").toEqual([]);
    expect(fields.all.filter((f) => !sent.includes(f)), "field never sent").toEqual([]);
  });
});

describe("moneyDisplayMeta", () => {
  it("falls back to dollars exactly when formatMoney does", () => {
    expect(moneyDisplayMeta("USD", { CNY: 7.1 })).toBeNull();
    expect(moneyDisplayMeta("CNY", null)).toBeNull();
    expect(moneyDisplayMeta("CNY", { CNY: Number.NaN })).toBeNull();
    expect(moneyDisplayMeta("XXX", { XXX: 2 })).toBeNull();
    expect(formatMoney(12.5, "CNY", null)).toBe("$12.50");
  });

  it("carries what formatMoney uses for a real currency", () => {
    expect(moneyDisplayMeta("JPY", { JPY: 147 })).toEqual({ symbol: "JP¥", rate: 147, decimals: 0 });
    expect(formatMoney(12.5, "JPY", { JPY: 147 })).toBe("JP¥1,838");
  });
});
