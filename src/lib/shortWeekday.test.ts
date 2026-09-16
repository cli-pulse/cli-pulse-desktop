import { describe, it, expect, vi } from "vitest";
import { shortWeekday } from "./format";

// The cost chart built its days at UTC midnight and formatted the weekday in the
// local zone, so every label west of UTC named the day before. This must hold in
// ANY process time zone; run with TZ=America/Mexico_City to see the old bug.
describe("shortWeekday", () => {
  it("names the calendar day itself, in the given language", () => {
    expect(shortWeekday("2026-09-17", "en")).toBe("Thu");
    expect(shortWeekday("2026-09-17", "ja")).toBe("木");
    expect(shortWeekday("2026-09-17", "zh-CN")).toBe("周四");
    expect(shortWeekday("2026-09-17", "es")).toBe("jue");
  });

  // CI runs in UTC, where the local-zone bug is invisible, so pin the cause too.
  it("formats in UTC whatever zone the process runs in", () => {
    const spy = vi.spyOn(Date.prototype, "toLocaleDateString");
    shortWeekday("2026-09-17", "en");
    expect(spy.mock.calls[0]?.[1]).toMatchObject({ timeZone: "UTC" });
    spy.mockRestore();
  });

  it("does not move across midnight at either end of the year", () => {
    expect(shortWeekday("2026-01-01", "en")).toBe("Thu");
    expect(shortWeekday("2026-12-31", "en")).toBe("Thu");
  });
});
