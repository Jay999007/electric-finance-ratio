from __future__ import annotations

import argparse
import html
import json
import logging
import math
import re
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.json"
DATA_DIR = ROOT / "data"
DOCS_DIR = ROOT / "docs"
DATA_CSV = DATA_DIR / "electronics_finance_ratio.csv"
WEB_CSV = DOCS_DIR / "data.csv"
SIGNALS_CSV = DOCS_DIR / "signals.csv"
PNG_PATH = DOCS_DIR / "latest.png"
HTML_PATH = DOCS_DIR / "index.html"

TAIPEI = ZoneInfo("Asia/Taipei")

# 官方 TWSE 每日收盤行情端點。先試新版 RWD，失敗再試相容舊端點。
TWSE_ENDPOINTS = (
    "https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX",
    "https://www.twse.com.tw/exchangeReport/MI_INDEX",
)

ELECTRONICS_NAMES = {
    "電子工業類指數",
    "電子類指數",
    "電子工業類股價指數",
    "電子類股價指數",
    "Electronics",
    "Electronics Index",
}
FINANCE_NAMES = {
    "金融保險類指數",
    "金融保險類股價指數",
    "Finance and Insurance",
    "Finance and Insurance Index",
}

LOGGER = logging.getLogger("electric_finance_ratio")


@dataclass(frozen=True)
class AppConfig:
    backfill_days: int = 500
    refresh_days: int = 10
    chart_days: int = 260
    moving_averages: tuple[int, ...] = (20, 120)
    buffer_pct: float = 0.0
    request_interval_seconds: float = 0.25


@dataclass(frozen=True)
class DailyIndexRow:
    day: date
    electronics_index: float
    finance_index: float

    @property
    def ratio(self) -> float:
        return self.electronics_index / self.finance_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="抓取 TWSE 電子類與金融保險類指數，建立電金比圖表。"
    )
    parser.add_argument(
        "--backfill-days",
        type=int,
        default=None,
        help="第一次執行時向前回補的日曆天數，預設讀取 config.json。",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="指定抓取起日，格式 YYYY-MM-DD。",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="指定抓取迄日，格式 YYYY-MM-DD；預設為台北今天。",
    )
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="不連線抓資料，只用既有 CSV 重畫圖表。",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="建立示範資料與圖表，不連線抓 TWSE。",
    )
    return parser.parse_args()


def load_config() -> AppConfig:
    raw: dict[str, Any] = {}
    if CONFIG_PATH.exists():
        raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    mas = tuple(int(v) for v in raw.get("moving_averages", [20, 120]))
    if not mas or any(v <= 1 for v in mas):
        raise ValueError("moving_averages 必須是大於 1 的整數陣列。")

    return AppConfig(
        backfill_days=int(raw.get("backfill_days", 500)),
        refresh_days=int(raw.get("refresh_days", 10)),
        chart_days=int(raw.get("chart_days", 260)),
        moving_averages=mas,
        buffer_pct=float(raw.get("buffer_pct", 0.0)),
        request_interval_seconds=float(raw.get("request_interval_seconds", 0.25)),
    )


def build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        status=3,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/130 Safari/537.36 "
                "electric-finance-ratio/1.0"
            ),
            "Accept": "application/json,text/plain,*/*",
            "Referer": "https://www.twse.com.tw/zh/indices/taiex/eftri-hist.html",
        }
    )
    return session


def normalize_label(value: Any) -> str:
    text = str(value or "").strip()
    text = text.replace("\u3000", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def parse_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None

    text = str(value).strip()
    if not text or text in {"--", "---", "N/A", "nan", "None"}:
        return None
    text = (
        text.replace(",", "")
        .replace("＋", "+")
        .replace("－", "-")
        .replace("−", "-")
        .replace("%", "")
    )
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def iter_candidate_tables(payload: dict[str, Any]) -> Iterable[tuple[list[Any], list[Any]]]:
    tables = payload.get("tables")
    if isinstance(tables, list):
        for table in tables:
            if not isinstance(table, dict):
                continue
            fields = table.get("fields") or table.get("field") or []
            data = table.get("data") or []
            if isinstance(fields, list) and isinstance(data, list):
                yield fields, data

    # 舊格式常見 fields1/data1、fields2/data2……
    for key, fields in payload.items():
        if not isinstance(key, str) or not key.startswith("fields"):
            continue
        suffix = key[len("fields") :]
        data = payload.get(f"data{suffix}")
        if isinstance(fields, list) and isinstance(data, list):
            yield fields, data

    # 少數回應直接使用 fields/data。
    fields = payload.get("fields")
    data = payload.get("data")
    if isinstance(fields, list) and isinstance(data, list):
        yield fields, data


def field_index(fields: list[Any], keywords: tuple[str, ...], fallback: int) -> int:
    normalized = [normalize_label(field) for field in fields]
    for index, label in enumerate(normalized):
        if any(keyword in label for keyword in keywords):
            return index
    return fallback


def classify_index_name(name: str) -> str | None:
    compact = re.sub(r"\s+", "", name)

    for candidate in ELECTRONICS_NAMES:
        if compact == re.sub(r"\s+", "", candidate):
            return "electronics"

    for candidate in FINANCE_NAMES:
        if compact == re.sub(r"\s+", "", candidate):
            return "finance"

    return None


def extract_indices(payload: dict[str, Any]) -> tuple[float, float] | None:
    found: dict[str, float] = {}

    for fields, rows in iter_candidate_tables(payload):
        if not rows:
            continue
        name_idx = field_index(fields, ("指數名稱", "指數", "名稱", "Index"), 0)
        close_idx = field_index(fields, ("收盤指數", "收盤", "Closing Index", "Close"), 1)

        for row in rows:
            if not isinstance(row, (list, tuple)):
                continue
            if len(row) <= max(name_idx, close_idx):
                continue
            kind = classify_index_name(normalize_label(row[name_idx]))
            if kind is None:
                continue
            number = parse_number(row[close_idx])
            if number is not None and number > 0:
                found[kind] = number

        if "electronics" in found and "finance" in found:
            return found["electronics"], found["finance"]

    return None


def fetch_one_day(session: requests.Session, target_day: date) -> DailyIndexRow | None:
    date_text = target_day.strftime("%Y%m%d")
    attempts: list[str] = []

    # IND 通常只取指數表；ALL 是資料格式異動時的保底。
    for endpoint in TWSE_ENDPOINTS:
        for report_type in ("IND", "ALL"):
            params = {"date": date_text, "type": report_type, "response": "json"}
            try:
                response = session.get(endpoint, params=params, timeout=(10, 35))
            except requests.RequestException as exc:
                attempts.append(f"{endpoint} {report_type}: {exc}")
                continue

            if response.status_code != 200:
                attempts.append(
                    f"{endpoint} {report_type}: HTTP {response.status_code}"
                )
                continue

            try:
                payload = response.json()
            except ValueError:
                attempts.append(f"{endpoint} {report_type}: 非 JSON 回應")
                continue

            stat = normalize_label(payload.get("stat", ""))
            result = extract_indices(payload)
            if result is not None:
                electronics, finance = result
                return DailyIndexRow(target_day, electronics, finance)

            # 休市日、尚未發布或查無資料，換另一種 type；全部都沒有就回傳 None。
            attempts.append(f"{endpoint} {report_type}: stat={stat or 'unknown'} 無目標資料")

    LOGGER.debug("%s 無資料：%s", target_day, " | ".join(attempts))
    return None


def read_existing_data() -> pd.DataFrame:
    columns = ["date", "electronics_index", "finance_index", "ratio"]
    if not DATA_CSV.exists():
        return pd.DataFrame(columns=columns)

    frame = pd.read_csv(DATA_CSV, parse_dates=["date"])
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"既有 CSV 缺少欄位：{sorted(missing)}")

    frame = frame[columns].copy()
    for column in ["electronics_index", "finance_index", "ratio"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=columns).sort_values("date")
    frame = frame.drop_duplicates(subset=["date"], keep="last")
    return frame.reset_index(drop=True)


def resolve_fetch_range(
    frame: pd.DataFrame,
    config: AppConfig,
    args: argparse.Namespace,
) -> tuple[date, date]:
    end_day = (
        datetime.strptime(args.end, "%Y-%m-%d").date()
        if args.end
        else datetime.now(TAIPEI).date()
    )

    if args.start:
        start_day = datetime.strptime(args.start, "%Y-%m-%d").date()
    elif frame.empty:
        days = args.backfill_days or config.backfill_days
        start_day = end_day - timedelta(days=days)
    else:
        latest = pd.Timestamp(frame["date"].max()).date()
        # 重抓最近數日，可補上先前因發布延遲或資料修正造成的缺口。
        start_day = min(latest + timedelta(days=1), end_day - timedelta(days=config.refresh_days))

    if start_day > end_day:
        start_day = end_day
    return start_day, end_day


def date_range_weekdays(start_day: date, end_day: date) -> list[date]:
    days: list[date] = []
    current = start_day
    while current <= end_day:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return days


def update_data(
    frame: pd.DataFrame,
    config: AppConfig,
    args: argparse.Namespace,
) -> pd.DataFrame:
    start_day, end_day = resolve_fetch_range(frame, config, args)
    weekdays = date_range_weekdays(start_day, end_day)
    if not weekdays:
        LOGGER.info("指定區間沒有平日，略過抓取。")
        return frame

    LOGGER.info("抓取 TWSE：%s 至 %s，共 %d 個平日。", start_day, end_day, len(weekdays))
    session = build_session()
    rows: list[dict[str, Any]] = []

    for index, target_day in enumerate(weekdays, start=1):
        row = fetch_one_day(session, target_day)
        if row is not None:
            rows.append(
                {
                    "date": pd.Timestamp(row.day),
                    "electronics_index": row.electronics_index,
                    "finance_index": row.finance_index,
                    "ratio": row.ratio,
                }
            )
            LOGGER.info(
                "[%d/%d] %s 電子 %.2f／金融 %.2f／電金比 %.6f",
                index,
                len(weekdays),
                target_day,
                row.electronics_index,
                row.finance_index,
                row.ratio,
            )
        else:
            LOGGER.info("[%d/%d] %s 無交易資料。", index, len(weekdays), target_day)

        if index < len(weekdays):
            time.sleep(max(config.request_interval_seconds, 0.0))

    if not rows:
        LOGGER.warning("本次沒有取得新資料，沿用既有 CSV。")
        return frame

    incoming = pd.DataFrame(rows)
    combined = pd.concat([frame, incoming], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    combined = combined.sort_values("date").drop_duplicates("date", keep="last")
    combined = combined.dropna(
        subset=["date", "electronics_index", "finance_index", "ratio"]
    )
    return combined.reset_index(drop=True)


def make_demo_data(config: AppConfig) -> pd.DataFrame:
    end_day = datetime.now(TAIPEI).date()
    dates = pd.bdate_range(end=end_day, periods=max(300, config.chart_days))
    values: list[float] = []
    ratio = 0.78
    for i, _ in enumerate(dates):
        wave = math.sin(i / 22.0) * 0.0035 + math.sin(i / 63.0) * 0.002
        drift = 0.0012 if i < len(dates) * 0.6 else -0.0008
        ratio = max(0.62, ratio + drift + wave)
        values.append(ratio)
    finance = pd.Series([1600 + i * 0.7 for i in range(len(dates))], dtype=float)
    electronics = finance * pd.Series(values)
    return pd.DataFrame(
        {
            "date": dates,
            "electronics_index": electronics,
            "finance_index": finance,
            "ratio": values,
        }
    )


def compute_signals(frame: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
    result = frame.copy().sort_values("date").reset_index(drop=True)
    result["ratio"] = result["electronics_index"] / result["finance_index"]

    for window in config.moving_averages:
        ma_col = f"ma{window}"
        slope_col = f"slope{window}"
        bull_col = f"bull_turn{window}"
        bear_col = f"bear_turn{window}"
        state_col = f"state{window}"
        cross_on_col = f"cross_on{window}"
        cross_off_col = f"cross_off{window}"

        result[ma_col] = result["ratio"].rolling(window=window, min_periods=window).mean()
        result[slope_col] = result[ma_col].diff()
        result[bull_col] = (result[slope_col] > 0) & (result[slope_col].shift(1) <= 0)
        result[bear_col] = (result[slope_col] < 0) & (result[slope_col].shift(1) >= 0)

        upper = result[ma_col] * (1 + config.buffer_pct)
        lower = result[ma_col] * (1 - config.buffer_pct)
        raw_state = pd.Series(pd.NA, index=result.index, dtype="object")
        raw_state.loc[result["ratio"] > upper] = "Risk On"
        raw_state.loc[result["ratio"] < lower] = "Risk Off"
        result[state_col] = raw_state.ffill().fillna("Neutral")

        prior_ratio = result["ratio"].shift(1)
        prior_ma = result[ma_col].shift(1)
        result[cross_on_col] = (result["ratio"] > upper) & (
            prior_ratio <= prior_ma * (1 + config.buffer_pct)
        )
        result[cross_off_col] = (result["ratio"] < lower) & (
            prior_ratio >= prior_ma * (1 - config.buffer_pct)
        )

    return result


def choose_font() -> str | None:
    preferred = [
        "Microsoft JhengHei",
        "Noto Sans CJK TC",
        "Noto Sans CJK JP",
        "PingFang TC",
        "Arial Unicode MS",
    ]
    installed = {font.name for font in fm.fontManager.ttflist}
    for font_name in preferred:
        if font_name in installed:
            plt.rcParams["font.family"] = font_name
            plt.rcParams["axes.unicode_minus"] = False
            return font_name
    return None


def last_signal_date(frame: pd.DataFrame, column: str) -> str:
    matched = frame.loc[frame[column].fillna(False), "date"]
    if matched.empty:
        return "—"
    return pd.Timestamp(matched.iloc[-1]).strftime("%Y-%m-%d")


def make_chart(frame: pd.DataFrame, config: AppConfig) -> None:
    choose_font()
    windows = list(config.moving_averages)
    shown = frame.tail(config.chart_days).copy()
    latest = frame.iloc[-1]

    fig, axes = plt.subplots(
        nrows=len(windows),
        ncols=1,
        figsize=(16, 9),
        sharex=True,
        gridspec_kw={"hspace": 0.08},
    )
    if len(windows) == 1:
        axes = [axes]

    fig.patch.set_facecolor("#050505")

    for panel_index, (axis, window) in enumerate(zip(axes, windows)):
        ma_col = f"ma{window}"
        slope_col = f"slope{window}"
        bull_col = f"bull_turn{window}"
        bear_col = f"bear_turn{window}"
        state_col = f"state{window}"

        axis.set_facecolor("#050505")
        bar_color = "#9b641f" if panel_index == 0 else "#262626"
        edge_color = "#d48a31" if panel_index == 0 else "#666666"
        axis.bar(
            shown["date"],
            shown["ratio"],
            width=0.78,
            color=bar_color,
            edgecolor=edge_color,
            linewidth=0.45,
            alpha=0.95,
            label="電金比",
        )
        axis.plot(
            shown["date"],
            shown[ma_col],
            color="#f1f1f1",
            linewidth=1.35,
            label=f"MA{window}",
            zorder=4,
        )

        bull = shown[shown[bull_col].fillna(False)]
        bear = shown[shown[bear_col].fillna(False)]
        axis.scatter(
            bull["date"],
            bull[ma_col],
            s=125,
            color="#ff2c55",
            edgecolor="#ff8aa1",
            linewidth=0.8,
            zorder=6,
            label="均線斜率由負轉正",
        )
        axis.scatter(
            bear["date"],
            bear[ma_col],
            s=125,
            color="#00df45",
            edgecolor="#9affb5",
            linewidth=0.8,
            zorder=6,
            label="均線斜率由正轉負",
        )

        axis.grid(True, color="#3a3a3a", linewidth=0.5, alpha=0.7)
        axis.tick_params(axis="both", colors="#d7d7d7", labelsize=9)
        for spine in axis.spines.values():
            spine.set_color("#555555")

        latest_ma = latest.get(ma_col)
        latest_slope = latest.get(slope_col)
        latest_state = latest.get(state_col, "—")
        ma_text = "—" if pd.isna(latest_ma) else f"{latest_ma:.4f}"
        slope_text = "—" if pd.isna(latest_slope) else f"{latest_slope:+.5f}"
        axis.set_title(
            (
                f"{window}日趨勢｜電金比 {latest['ratio']:.4f}｜MA{window} {ma_text}｜"
                f"斜率 {slope_text}｜狀態 {latest_state}"
            ),
            loc="left",
            color="#f5f5f5",
            fontsize=11,
            pad=8,
        )
        axis.yaxis.tick_right()
        axis.yaxis.set_label_position("right")

        ymin = min(shown["ratio"].min(), shown[ma_col].min(skipna=True))
        ymax = max(shown["ratio"].max(), shown[ma_col].max(skipna=True))
        if pd.notna(ymin) and pd.notna(ymax):
            padding = max((ymax - ymin) * 0.08, 0.01)
            axis.set_ylim(ymin - padding, ymax + padding)

    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(axes[-1].get_xticklabels(), rotation=0, ha="center")

    fig.suptitle(
        "台灣電子指數 ÷ 金融保險指數｜市場風險偏好",
        color="#ffffff",
        fontsize=15,
        x=0.01,
        ha="left",
        y=0.985,
    )
    fig.text(
        0.01,
        0.012,
        (
            "紅點：均線斜率由負轉正；綠點：均線斜率由正轉負。"
            "電金比高於均線＝Risk On，低於均線＝Risk Off。資料來源：TWSE。"
        ),
        color="#bfbfbf",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.04, right=0.965, top=0.93, bottom=0.075)
    fig.savefig(PNG_PATH, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)


def format_value(value: Any, digits: int = 4, signed: bool = False) -> str:
    if value is None or pd.isna(value):
        return "—"
    number = float(value)
    return f"{number:+.{digits}f}" if signed else f"{number:.{digits}f}"


def make_html(frame: pd.DataFrame, config: AppConfig) -> None:
    latest = frame.iloc[-1]
    latest_date = pd.Timestamp(latest["date"]).strftime("%Y-%m-%d")
    generated_at = datetime.now(TAIPEI).strftime("%Y-%m-%d %H:%M:%S Asia/Taipei")

    cards: list[str] = []
    signal_rows: list[str] = []
    for window in config.moving_averages:
        state = html.escape(str(latest.get(f"state{window}", "—")))
        state_class = "on" if state == "Risk On" else "off" if state == "Risk Off" else "neutral"
        cards.append(
            f"""
            <section class="metric-card">
              <div class="metric-title">{window} 日趨勢</div>
              <div class="state {state_class}">{state}</div>
              <dl>
                <div><dt>MA{window}</dt><dd>{format_value(latest.get(f'ma{window}'))}</dd></div>
                <div><dt>斜率</dt><dd>{format_value(latest.get(f'slope{window}'), 5, True)}</dd></div>
                <div><dt>最近多頭反轉</dt><dd>{last_signal_date(frame, f'bull_turn{window}')}</dd></div>
                <div><dt>最近空頭反轉</dt><dd>{last_signal_date(frame, f'bear_turn{window}')}</dd></div>
              </dl>
            </section>
            """
        )

        subset = frame.loc[
            frame[f"bull_turn{window}"].fillna(False)
            | frame[f"bear_turn{window}"].fillna(False),
            ["date", "ratio", f"ma{window}", f"bull_turn{window}", f"bear_turn{window}"],
        ].tail(8)
        for _, row in subset.iterrows():
            kind = "多頭反轉" if bool(row[f"bull_turn{window}"]) else "空頭反轉"
            css_class = "bull" if kind == "多頭反轉" else "bear"
            signal_rows.append(
                "<tr>"
                f"<td>{pd.Timestamp(row['date']).strftime('%Y-%m-%d')}</td>"
                f"<td>MA{window}</td>"
                f"<td class='{css_class}'>{kind}</td>"
                f"<td>{row['ratio']:.4f}</td>"
                f"<td>{row[f'ma{window}']:.4f}</td>"
                "</tr>"
            )

    buffer_text = f"{config.buffer_pct * 100:.2f}%"
    html_text = f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="dark">
  <title>台灣電金比風險偏好指標</title>
  <style>
    :root {{ color-scheme: dark; --bg:#080808; --panel:#151515; --line:#343434; --text:#f0f0f0; --muted:#aaa; }}
    * {{ box-sizing: border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--text); font-family:"Noto Sans TC","Microsoft JhengHei",system-ui,sans-serif; }}
    main {{ max-width:1500px; margin:auto; padding:28px 22px 60px; }}
    h1 {{ margin:0 0 6px; font-size:clamp(1.45rem,3vw,2.2rem); }}
    .sub {{ color:var(--muted); margin-bottom:22px; }}
    .summary {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:14px; margin-bottom:18px; }}
    .metric-card,.ratio-card,.section {{ background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:18px; }}
    .ratio-card .value {{ font-size:2rem; font-weight:750; margin:8px 0; }}
    .metric-title {{ font-weight:700; }}
    .state {{ display:inline-block; margin:9px 0 12px; padding:5px 10px; border-radius:999px; font-weight:750; }}
    .state.on {{ background:#4a1020; color:#ff7995; }}
    .state.off {{ background:#073c18; color:#76f79a; }}
    .state.neutral {{ background:#333; color:#ddd; }}
    dl {{ margin:0; }} dl div {{ display:flex; justify-content:space-between; gap:12px; border-top:1px solid #292929; padding:7px 0; }}
    dt {{ color:var(--muted); }} dd {{ margin:0; font-variant-numeric:tabular-nums; }}
    .chart {{ width:100%; display:block; border:1px solid var(--line); border-radius:14px; background:#050505; }}
    .links {{ display:flex; flex-wrap:wrap; gap:10px; margin:16px 0 26px; }}
    .links a {{ color:#ffd37a; text-decoration:none; background:#222; border:1px solid #444; border-radius:9px; padding:8px 12px; }}
    .section {{ margin-top:18px; overflow-x:auto; }}
    table {{ width:100%; border-collapse:collapse; min-width:650px; }}
    th,td {{ text-align:left; padding:9px 10px; border-bottom:1px solid #303030; font-variant-numeric:tabular-nums; }}
    th {{ color:#bbb; }} .bull {{ color:#ff6687; font-weight:700; }} .bear {{ color:#58ee83; font-weight:700; }}
    code {{ color:#ffd37a; }}
    .note {{ color:#bdbdbd; line-height:1.75; }}
  </style>
</head>
<body>
<main>
  <h1>台灣電金比風險偏好指標</h1>
  <div class="sub">資料日：{latest_date}｜網站更新：{generated_at}</div>

  <div class="summary">
    <section class="ratio-card">
      <div>電子指數 ÷ 金融保險指數</div>
      <div class="value">{latest['ratio']:.4f}</div>
      <dl>
        <div><dt>電子類指數</dt><dd>{latest['electronics_index']:,.2f}</dd></div>
        <div><dt>金融保險類指數</dt><dd>{latest['finance_index']:,.2f}</dd></div>
      </dl>
    </section>
    {''.join(cards)}
  </div>

  <img class="chart" src="latest.png?v={latest_date}" alt="台灣電金比趨勢圖">
  <div class="links">
    <a href="data.csv">下載完整每日資料 CSV</a>
    <a href="signals.csv">下載反轉訊號 CSV</a>
  </div>

  <section class="section note">
    <h2>判讀規則</h2>
    <p><strong>電金比＝電子類指數 ÷ 金融保險類指數。</strong>比值上升代表電子相對金融強，通常視為市場風險偏好提高；比值下降則代表金融相對電子強，通常視為風險趨避提高。</p>
    <p>電金比高於移動平均線時標示為 <strong>Risk On</strong>；低於移動平均線時標示為 <strong>Risk Off</strong>。目前緩衝區設定為 <code>{buffer_text}</code>。紅點代表均線斜率由負轉正，綠點代表均線斜率由正轉負。</p>
    <p>這是相對強弱與市場風格指標，不等同加權指數必然上漲或下跌，也不構成投資建議。</p>
  </section>

  <section class="section">
    <h2>近期均線斜率反轉</h2>
    <table>
      <thead><tr><th>日期</th><th>均線</th><th>訊號</th><th>電金比</th><th>均線值</th></tr></thead>
      <tbody>{''.join(reversed(signal_rows)) or '<tr><td colspan="5">目前尚無足夠資料形成反轉訊號。</td></tr>'}</tbody>
    </table>
  </section>
</main>
</body>
</html>
"""
    HTML_PATH.write_text(html_text, encoding="utf-8")


def save_outputs(frame: pd.DataFrame, config: AppConfig) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    base_columns = ["date", "electronics_index", "finance_index", "ratio"]
    frame[base_columns].to_csv(DATA_CSV, index=False, date_format="%Y-%m-%d")
    frame.to_csv(WEB_CSV, index=False, date_format="%Y-%m-%d", encoding="utf-8-sig")

    signal_parts: list[pd.DataFrame] = []
    for window in config.moving_averages:
        mask = frame[f"bull_turn{window}"].fillna(False) | frame[f"bear_turn{window}"].fillna(False)
        part = frame.loc[mask, ["date", "ratio", f"ma{window}", f"bull_turn{window}"]].copy()
        if part.empty:
            continue
        part["window"] = window
        part["signal"] = part[f"bull_turn{window}"].map({True: "Bull Turn", False: "Bear Turn"})
        part = part.rename(columns={f"ma{window}": "moving_average"})
        signal_parts.append(part[["date", "window", "signal", "ratio", "moving_average"]])

    if signal_parts:
        signals = pd.concat(signal_parts, ignore_index=True).sort_values(["date", "window"])
    else:
        signals = pd.DataFrame(columns=["date", "window", "signal", "ratio", "moving_average"])
    signals.to_csv(SIGNALS_CSV, index=False, date_format="%Y-%m-%d", encoding="utf-8-sig")

    make_chart(frame, config)
    make_html(frame, config)
    (DOCS_DIR / ".nojekyll").touch()


def validate_minimum_data(frame: pd.DataFrame, config: AppConfig) -> None:
    minimum = min(config.moving_averages)
    if len(frame) < minimum:
        raise RuntimeError(
            f"目前只有 {len(frame)} 筆交易日資料，至少需要 {minimum} 筆才能計算最短均線。"
        )


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    config = load_config()

    if args.demo:
        frame = make_demo_data(config)
    else:
        frame = read_existing_data()
        if not args.no_fetch:
            frame = update_data(frame, config, args)

    if frame.empty:
        LOGGER.error("沒有任何資料可供繪圖。請確認網路、TWSE 回應或既有 CSV。")
        return 2

    frame = compute_signals(frame, config)
    validate_minimum_data(frame, config)
    save_outputs(frame, config)

    latest = frame.iloc[-1]
    LOGGER.info(
        "完成：%s，電金比 %.6f；輸出 %s",
        pd.Timestamp(latest["date"]).strftime("%Y-%m-%d"),
        latest["ratio"],
        HTML_PATH,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
