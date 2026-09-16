//! Native desktop notifications via `tauri-plugin-notification`.
//!
//! Used sparingly — the macOS app's in-process heuristic is "notify on
//! state change that the user cares about, never on every tick." Sprint
//! 2.5 wires two triggers:
//!
//!   1. Pair success (one-time reassurance)
//!   2. Sync failure streak ≥ 3 (so the user doesn't silently miss
//!      uploads — the iOS/macOS apps show stale data otherwise)
//!
//! Permission model: on macOS/Linux the plugin handles permission
//! requests implicitly; on Windows the first notify call triggers the
//! OS-level permission dialog. We don't block on permission grant —
//! failures are logged but never surfaced to the user.

use std::sync::RwLock;

use once_cell::sync::Lazy;
use tauri::{AppHandle, Runtime};
use tauri_plugin_notification::NotificationExt;

/// Notification text in the UI language, pushed by the frontend alongside the tray
/// copy (`force_tray_menu_refresh`). The webview lives as long as the app, so the push
/// always precedes the first sync tick (20 s) that can raise a notification.
/// Defaults are the English these notifications always used.
///
/// Placeholders are single-brace, filled with `String::replace`: `{device}`,
/// `{count}`, `{amount}`, `{spend}`, `{limit}`.
#[derive(Debug, Clone, PartialEq)]
pub struct NotificationCopy {
    pub pair_title: String,
    pub pair_body: String,
    pub sync_paused_title: String,
    pub sync_paused_lead: String,
    pub signed_out_title: String,
    pub signed_out_device_missing: String,
    pub signed_out_account_missing: String,
    pub signed_out_expired: String,
    pub budget_daily_title: String,
    pub budget_daily_body: String,
    pub budget_weekly_title: String,
    pub budget_weekly_body: String,
}

impl Default for NotificationCopy {
    fn default() -> Self {
        Self {
            pair_title: "CLI Pulse — Paired".into(),
            pair_body: "Device “{device}” is now syncing with your phone.".into(),
            sync_paused_title: "CLI Pulse — Sync paused".into(),
            sync_paused_lead: "{count} consecutive sync failures.".into(),
            signed_out_title: "CLI Pulse — Sign in again".into(),
            signed_out_device_missing:
                "This device was removed from your account. Sign in again to re-pair.".into(),
            signed_out_account_missing:
                "Your CLI Pulse account is no longer accessible. Sign in again to continue.".into(),
            signed_out_expired: "Your sign-in expired. Please sign in again to keep syncing."
                .into(),
            budget_daily_title: "Daily budget exceeded — {amount}".into(),
            budget_daily_body: "Today's spend of {spend} is above your daily budget of {limit}."
                .into(),
            budget_weekly_title: "Weekly budget exceeded — {amount}".into(),
            budget_weekly_body:
                "Last 7 days of spend totals {spend}, above your weekly budget of {limit}.".into(),
        }
    }
}

static COPY: Lazy<RwLock<NotificationCopy>> =
    Lazy::new(|| RwLock::new(NotificationCopy::default()));

pub fn set_copy(copy: NotificationCopy) {
    if let Ok(mut g) = COPY.write() {
        *g = copy;
    }
}

fn copy() -> NotificationCopy {
    COPY.read().map(|g| g.clone()).unwrap_or_default()
}

pub fn pair_success<R: Runtime>(app: &AppHandle<R>, device_name: &str) {
    let c = copy();
    send(
        app,
        &c.pair_title,
        &c.pair_body.replace("{device}", device_name),
    );
}

pub fn sync_failure_streak<R: Runtime>(app: &AppHandle<R>, consecutive: u32, err: &str) {
    let c = copy();
    send(
        app,
        &c.sync_paused_title,
        &sync_paused_body(&c, consecutive, err),
    );
}

/// A localized lead, then the raw error as a detail line: the error text comes from
/// the server or the network stack and is what a bug report needs verbatim.
fn sync_paused_body(c: &NotificationCopy, consecutive: u32, err: &str) -> String {
    let short: String = err.chars().take(140).collect();
    format!(
        "{}\n{short}",
        c.sync_paused_lead
            .replace("{count}", &consecutive.to_string())
    )
}

/// v0.3.0 — emitted once when the helper_sync error classifier
/// determines the device/account is gone server-side. Pairs with a
/// local-state clear so the next user-facing surface is the sign-in
/// screen instead of an unrecoverable loop of 401s.
pub fn session_expired<R: Runtime>(app: &AppHandle<R>, kind: &str) {
    let c = copy();
    let body = match kind {
        "device_missing" => &c.signed_out_device_missing,
        "account_missing" => &c.signed_out_account_missing,
        _ => &c.signed_out_expired,
    };
    send(app, &c.signed_out_title, body);
}

/// Fired once per (day, budget kind) — see `maybe_notify_budget_breach` in lib.rs for
/// the de-dup logic. The alert itself is uploaded and deduplicated as it is, so its
/// English title and message are never changed; the notification is rebuilt from the
/// amounts in it, in the UI language.
pub fn budget_breach<R: Runtime>(app: &AppHandle<R>, alert: &crate::alerts::Alert) {
    let (title, body) = budget_text(&copy(), alert);
    // Truncate extremely long messages so Windows Action Center / macOS
    // Notification Center don't refuse the payload.
    let body: String = body.chars().take(280).collect();
    send(app, &title, &body);
}

/// Title and body for a budget alert. Falls back to the alert's own English when its
/// id or message is not the budget template `alerts::compute` writes.
pub(crate) fn budget_text(c: &NotificationCopy, alert: &crate::alerts::Alert) -> (String, String) {
    static AMOUNTS: Lazy<regex::Regex> =
        Lazy::new(|| regex::Regex::new(r"\$([0-9][0-9,]*(?:\.[0-9]+)?)").expect("static regex"));
    let amounts: Vec<String> = AMOUNTS
        .captures_iter(&alert.message)
        .map(|m| format!("${}", &m[1]))
        .collect();
    let templates = if alert.id.starts_with("budget-daily-") {
        Some((&c.budget_daily_title, &c.budget_daily_body))
    } else if alert.id.starts_with("budget-weekly-") {
        Some((&c.budget_weekly_title, &c.budget_weekly_body))
    } else {
        None
    };
    match (templates, amounts.as_slice()) {
        (Some((title, body)), [spend, limit]) => (
            title.replace("{amount}", spend),
            body.replace("{spend}", spend).replace("{limit}", limit),
        ),
        _ => (alert.title.clone(), alert.message.clone()),
    }
}

fn send<R: Runtime>(app: &AppHandle<R>, title: &str, body: &str) {
    let result = app.notification().builder().title(title).body(body).show();
    if let Err(e) = result {
        log::warn!("notification send failed: {e}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn budget_alert(id: &str, title: &str, message: &str) -> crate::alerts::Alert {
        crate::alerts::Alert {
            id: id.into(),
            alert_type: "Daily Budget Exceeded".into(),
            severity: "Warning".into(),
            title: title.into(),
            message: message.into(),
            created_at: "2026-09-17T00:00:00Z".into(),
            related_project_id: None,
            related_project_name: None,
            related_session_id: None,
            related_session_name: None,
            related_provider: None,
            related_device_name: None,
            source_kind: Some("budget".into()),
            source_id: None,
            grouping_key: None,
            suppression_key: None,
        }
    }

    /// Default copy must render exactly the English these notifications always sent.
    #[test]
    fn default_copy_is_the_previous_english() {
        let c = NotificationCopy::default();
        assert_eq!(
            c.pair_body.replace("{device}", "box"),
            "Device “box” is now syncing with your phone."
        );
        assert_eq!(
            sync_paused_body(&c, 3, "HTTP 500"),
            "3 consecutive sync failures.\nHTTP 500"
        );
        let a = budget_alert(
            "budget-daily-2026-09-17",
            "Daily budget exceeded — $12.34",
            "Today's spend of $12.34 is above your daily budget of $10.00.",
        );
        assert_eq!(budget_text(&c, &a), (a.title.clone(), a.message.clone()));
    }

    #[test]
    fn budget_notifications_render_in_the_pushed_language() {
        let c = NotificationCopy {
            budget_daily_title: "每日预算已超出 — {amount}".into(),
            budget_daily_body: "今日支出 {spend} 已超过每日预算 {limit}。".into(),
            budget_weekly_title: "每周预算已超出 — {amount}".into(),
            budget_weekly_body: "最近 7 天支出共计 {spend}，已超过每周预算 {limit}。".into(),
            ..NotificationCopy::default()
        };
        let daily = budget_alert(
            "budget-daily-2026-09-17",
            "Daily budget exceeded — $12.34",
            "Today's spend of $12.34 is above your daily budget of $10.00.",
        );
        assert_eq!(
            budget_text(&c, &daily),
            (
                "每日预算已超出 — $12.34".into(),
                "今日支出 $12.34 已超过每日预算 $10.00。".into()
            )
        );
        let weekly = budget_alert(
            "budget-weekly-2026-W38",
            "Weekly budget exceeded — $1,088.20",
            "Last 7 days of spend totals $1,088.20, above your weekly budget of $50.00.",
        );
        assert_eq!(
            budget_text(&c, &weekly).1,
            "最近 7 天支出共计 $1,088.20，已超过每周预算 $50.00。"
        );
    }

    #[test]
    fn unknown_alerts_keep_their_own_text() {
        let c = NotificationCopy::default();
        let a = budget_alert("something-else", "Title", "Message with $1 only.");
        assert_eq!(
            budget_text(&c, &a),
            ("Title".into(), "Message with $1 only.".into())
        );
    }
}
