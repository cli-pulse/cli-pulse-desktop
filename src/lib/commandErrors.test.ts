import { describe, it, expect } from "vitest";
import i18n from "../i18n";
import { describeError } from "./commandErrors";

const RUST = import.meta.glob<string>(
  ["../../src-tauri/src/lib.rs", "../../src-tauri/src/terminal.rs"],
  { eager: true, query: "?raw", import: "default" },
);
const rust = Object.values(RUST).join("\n");
const ja = i18n.getFixedT("ja");
const en = i18n.getFixedT("en");

describe("describeError", () => {
  // The messages exactly as the Rust commands write them. If src-tauri rewords one,
  // the source check below fails instead of the UI quietly falling back to English.
  const cases: [raw: string, rustSource: string][] = [
    ["Sign in required to view dashboard data.", '"Sign in required to view dashboard data."'],
    ["Sign in required to change Remote Control.", '"Sign in required to change Remote Control."'],
    ["Session expired — sign in again to view dashboard.", '"Session expired — sign in again to view dashboard."'],
    ["Your sign-in expired. Please sign in again.", '"Your sign-in expired. Please sign in again."'],
    ["Too many tries — please wait a minute and try again.", '"Too many tries — please wait a minute and try again."'],
    ["Invalid or expired code.", '"Invalid or expired code."'],
    ["Network error: dns error", 'format!("Network error: {err}")'],
    ["Email is empty", '"Email is empty"'],
    ["Email or code is empty", '"Email or code is empty"'],
    ["Pairing code is empty", '"Pairing code is empty"'],
    ["Prompt command requires non-empty payload.", '"Prompt command requires non-empty payload."'],
    ["Device not paired — pair first, then set budgets", '"Device not paired — pair first, then set budgets"'],
    ["Device not paired", '"Device not paired"'],
    ["Device not paired yet", '"Device not paired yet"'],
    ["OS keychain not available. On Linux, install libsecret", '"OS keychain not available. On Linux, install libsecret'],
    ["Keychain error: locked", 'format!("Keychain error: {msg}")'],
    ["Keychain unavailable: NotAvailable", 'format!("Keychain unavailable: {e:?}")'],
    ["Supabase HTTP 500: upstream", 'format!("Supabase HTTP {status}: {snippet}")'],
    ["Auth error (HTTP 422): bad", 'format!("Auth error (HTTP {status}): {body}")'],
    ["too many local terminals open (max 8)", '"too many local terminals open (max {})"'],
    ["no Downloads directory available", '"no Downloads directory available"'],
    ["invalid export filename", '"invalid export filename"'],
    ["Failed to wipe scan cache: denied", 'format!("Failed to wipe scan cache: {e}")'],
    ["could not resolve home directory", '"could not resolve home directory"'],
  ];

  it("recognizes every message the Rust commands actually write", () => {
    expect(rust.length, "Rust sources not found").toBeGreaterThan(1000);
    for (const [raw, source] of cases) {
      expect(rust, `src-tauri no longer writes ${source}`).toContain(source);
      const shown = describeError(raw, ja);
      expect(shown, raw).not.toBe(raw);
      expect(shown, raw).not.toContain("{{");
    }
  });

  it("keeps the technical detail verbatim after the sentence", () => {
    expect(describeError("Supabase HTTP 503: upstream connect error", ja)).toContain("503");
    expect(describeError("Supabase HTTP 503: upstream connect error", ja)).toContain("upstream connect error");
    expect(describeError(new Error("Network error: dns error"), en)).toBe("Network error: dns error");
  });

  it("shows anything it does not know as it arrived", () => {
    expect(describeError("ALREADY_DECIDED", ja)).toBe("ALREADY_DECIDED");
    expect(describeError("Some future failure", ja)).toBe("Some future failure");
    expect(describeError(42, ja)).toBe("42");
  });
});
