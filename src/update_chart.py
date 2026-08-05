from __future__ import annotations

import argparse
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
LOGGER = logging.getLogger("electric_finance_ratio")
REQUIRED_MAS = (20, 60, 120, 240)

TWSE_ENDPOINTS = (
    "https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX",
    "https://www.twse.com.tw/exchangeReport/MI_INDEX",
)
ELECTRONICS_NAMES = {
    "電子工業類指數", "電子類指數", "電子工業類股價指數",
    "電子類股價指數", "Electronics", "Electronics Index",
}
FINANCE_NAMES = {
    "金融保險類指數", "金融保險類股價指數",
    "Finance and Insurance", "Finance and Insurance Index",
}
WEIGHTED_NAMES = {
    "發行量加權股價指數", "臺灣加權股價指數",
    "加權股價指數", "TAIEX",
}


@dataclass(frozen=True)
class AppConfig:
    backfill_days: int = 500
    history_start_date: str = "2006-01-01"
    refresh_days: int = 10
    chart_days: int = 260
    default_chart_range_years: int = 1
    moving_averages: tuple[int, ...] = REQUIRED_MAS
    buffer_pct: float = 0.0
    request_interval_seconds: float = 0.35
    history_retry_rounds: int = 3


@dataclass(frozen=True)
class DailyIndexRow:
    day: date
    electronics_index: float
    finance_index: float
    weighted_index: float

    @property
    def ratio(self) -> float:
        return self.electronics_index / self.finance_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="抓取 TWSE 指數，建立台灣電金比互動網站。"
    )
    parser.add_argument("--backfill-days", type=int, default=None)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--no-fetch", action="store_true")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--data-only", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--retry-rounds", type=int, default=None)
    return parser.parse_args()


def load_config() -> AppConfig:
    raw: dict[str, Any] = {}
    if CONFIG_PATH.exists():
        raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    configured = tuple(int(value) for value in raw.get("moving_averages", []))
    if any(value <= 1 for value in configured):
        raise ValueError("moving_averages 必須是大於 1 的整數陣列。")
    windows = tuple(dict.fromkeys(REQUIRED_MAS + configured))

    return AppConfig(
        backfill_days=max(1, int(raw.get("backfill_days", 500))),
        history_start_date=str(raw.get("history_start_date", "2006-01-01")),
        refresh_days=max(1, int(raw.get("refresh_days", 10))),
        chart_days=max(60, int(raw.get("chart_days", 260))),
        default_chart_range_years=max(
            1, int(raw.get("default_chart_range_years", 1))
        ),
        moving_averages=windows,
        buffer_pct=max(0.0, float(raw.get("buffer_pct", 0.0))),
        request_interval_seconds=max(
            0.0, float(raw.get("request_interval_seconds", 0.35))
        ),
        history_retry_rounds=max(
            1, int(raw.get("history_retry_rounds", 3))
        ),
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
                "electric-finance-ratio/2.0"
            ),
            "Accept": "application/json,text/plain,*/*",
            "Referer": "https://www.twse.com.tw/",
        }
    )
    return session


def normalize_label(value: Any) -> str:
    return re.sub(
        r"\s+", " ", str(value or "").replace("\u3000", " ").strip()
    )


def parse_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None

    text = (
        str(value).strip().replace(",", "")
        .replace("＋", "+").replace("－", "-").replace("−", "-")
        .replace("%", "")
    )
    if not text or text in {"--", "---", "N/A", "nan", "None"}:
        return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def iter_candidate_tables(
    payload: dict[str, Any],
) -> Iterable[tuple[list[Any], list[Any]]]:
    tables = payload.get("tables")
    if isinstance(tables, list):
        for table in tables:
            if not isinstance(table, dict):
                continue
            fields = table.get("fields") or table.get("field") or []
            data = table.get("data") or []
            if isinstance(fields, list) and isinstance(data, list):
                yield fields, data

    for key, fields in payload.items():
        if not isinstance(key, str) or not key.startswith("fields"):
            continue
        suffix = key[len("fields") :]
        data = payload.get(f"data{suffix}")
        if isinstance(fields, list) and isinstance(data, list):
            yield fields, data

    fields = payload.get("fields")
    data = payload.get("data")
    if isinstance(fields, list) and isinstance(data, list):
        yield fields, data


def field_index(
    fields: list[Any], keywords: tuple[str, ...], fallback: int
) -> int:
    normalized = [normalize_label(field) for field in fields]
    for index, label in enumerate(normalized):
        if any(keyword in label for keyword in keywords):
            return index
    return fallback


def classify_index_name(name: str) -> str | None:
    compact = re.sub(r"\s+", "", name)
    if any(compact == re.sub(r"\s+", "", item) for item in ELECTRONICS_NAMES):
        return "electronics"
    if any(compact == re.sub(r"\s+", "", item) for item in FINANCE_NAMES):
        return "finance"
    if any(compact == re.sub(r"\s+", "", item) for item in WEIGHTED_NAMES):
        return "weighted"
    return None


def extract_indices(
    payload: dict[str, Any],
) -> tuple[float, float, float] | None:
    found: dict[str, float] = {}
    for fields, rows in iter_candidate_tables(payload):
        if not rows:
            continue
        name_index = field_index(
            fields, ("指數名稱", "指數", "名稱", "Index"), 0
        )
        close_index = field_index(
            fields, ("收盤指數", "收盤", "Closing Index", "Close"), 1
        )

        for row in rows:
            if not isinstance(row, (list, tuple)):
                continue
            if len(row) <= max(name_index, close_index):
                continue
            kind = classify_index_name(normalize_label(row[name_index]))
            number = parse_number(row[close_index])
            if kind and number is not None and number > 0:
                found[kind] = number

        if {"electronics", "finance", "weighted"} <= found.keys():
            return (
                found["electronics"],
                found["finance"],
                found["weighted"],
            )
    return None


def fetch_one_day(
    session: requests.Session, target_day: date
) -> DailyIndexRow | None:
    date_text = target_day.strftime("%Y%m%d")
    for endpoint in TWSE_ENDPOINTS:
        for report_type in ("IND", "ALL"):
            try:
                response = session.get(
                    endpoint,
                    params={
                        "date": date_text,
                        "type": report_type,
                        "response": "json",
                    },
                    timeout=(10, 35),
                )
            except requests.RequestException:
                continue
            if response.status_code != 200:
                continue
            try:
                payload = response.json()
            except ValueError:
                continue

            result = extract_indices(payload)
            if result is not None:
                electronics, finance, weighted = result
                return DailyIndexRow(
                    target_day, electronics, finance, weighted
                )
    return None


def read_existing_data() -> pd.DataFrame:
    columns = [
        "date", "electronics_index", "finance_index",
        "weighted_index", "ratio",
    ]
    if not DATA_CSV.exists():
        return pd.DataFrame(columns=columns)

    frame = pd.read_csv(DATA_CSV, parse_dates=["date"])
    required = {"date", "electronics_index", "finance_index", "ratio"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"既有資料缺少欄位：{sorted(missing)}")
    if "weighted_index" not in frame.columns:
        frame["weighted_index"] = pd.NA

    frame = frame[columns].copy()
    for column in columns[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return (
        frame.dropna(
            subset=["date", "electronics_index", "finance_index", "ratio"]
        )
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )


def resolve_fetch_range(
    frame: pd.DataFrame, config: AppConfig, args: argparse.Namespace
) -> tuple[date, date]:
    end_day = (
        datetime.strptime(args.end, "%Y-%m-%d").date()
        if args.end
        else datetime.now(TAIPEI).date()
    )
    if args.start:
        start_day = datetime.strptime(args.start, "%Y-%m-%d").date()
    elif frame.empty:
        start_day = end_day - timedelta(
            days=args.backfill_days or config.backfill_days
        )
    elif frame["weighted_index"].isna().any():
        start_day = end_day - timedelta(
            days=args.backfill_days or config.backfill_days
        )
    else:
        latest = pd.Timestamp(frame["date"].max()).date()
        start_day = min(
            latest + timedelta(days=1),
            end_day - timedelta(days=config.refresh_days),
        )
    return min(start_day, end_day), end_day


def weekdays_between(start_day: date, end_day: date) -> list[date]:
    result: list[date] = []
    current = start_day
    while current <= end_day:
        if current.weekday() < 5:
            result.append(current)
        current += timedelta(days=1)
    return result


def update_data(
    frame: pd.DataFrame, config: AppConfig, args: argparse.Namespace
) -> pd.DataFrame:
    start_day, end_day = resolve_fetch_range(frame, config, args)
    pending = weekdays_between(start_day, end_day)

    if args.skip_existing and not frame.empty:
        existing_dates = {
            pd.Timestamp(value).date()
            for value in pd.to_datetime(
                frame.loc[frame["weighted_index"].notna(), "date"],
                errors="coerce",
            ).dropna()
        }
        pending = [day for day in pending if day not in existing_dates]

    if not pending:
        LOGGER.info("指定區間沒有待抓取日期。")
        return frame

    retry_rounds = max(
        1, args.retry_rounds or config.history_retry_rounds
    )
    session = build_session()
    collected: dict[date, dict[str, Any]] = {}

    for round_index in range(1, retry_rounds + 1):
        if not pending:
            break
        LOGGER.info(
            "TWSE 第 %d/%d 輪，待查 %d 個平日。",
            round_index, retry_rounds, len(pending),
        )
        unresolved: list[date] = []

        for index, target_day in enumerate(pending, start=1):
            row = fetch_one_day(session, target_day)
            if row is None:
                unresolved.append(target_day)
            else:
                collected[target_day] = {
                    "date": pd.Timestamp(row.day),
                    "electronics_index": row.electronics_index,
                    "finance_index": row.finance_index,
                    "weighted_index": row.weighted_index,
                    "ratio": row.ratio,
                }
                LOGGER.info(
                    "[%d/%d] %s 電金比 %.6f",
                    index, len(pending), target_day, row.ratio,
                )
            if index < len(pending):
                time.sleep(config.request_interval_seconds)

        pending = unresolved
        if pending and round_index < retry_rounds:
            time.sleep(min(5 * round_index, 15))

    if collected:
        frame = pd.concat(
            [frame, pd.DataFrame(collected.values())],
            ignore_index=True,
        )
    return (
        frame.sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )


def make_demo_data() -> pd.DataFrame:
    dates = pd.bdate_range(
        end=pd.Timestamp(datetime.now(TAIPEI).date()), periods=900
    )
    rows: list[dict[str, Any]] = []
    for index, current in enumerate(dates):
        electronics = 900 + index * 0.65 + 80 * math.sin(index / 34)
        finance = 980 + index * 0.42 + 38 * math.sin(index / 52 + 0.7)
        weighted = 15000 + index * 8.5 + 1100 * math.sin(index / 70)
        rows.append(
            {
                "date": current,
                "electronics_index": electronics,
                "finance_index": finance,
                "weighted_index": weighted,
                "ratio": electronics / finance,
            }
        )
    return pd.DataFrame(rows)


def compute_signals(
    frame: pd.DataFrame, config: AppConfig
) -> pd.DataFrame:
    result = (
        frame.copy()
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )

    for window in config.moving_averages:
        ma_column = f"ma{window}"
        slope_column = f"slope{window}"
        state_column = f"state{window}"
        on_column = f"bull_turn{window}"
        off_column = f"bear_turn{window}"

        result[ma_column] = result["ratio"].rolling(
            window, min_periods=window
        ).mean()
        result[slope_column] = result[ma_column].diff()
        result[f"ma{window}_slope"] = result[slope_column]

        states: list[str] = []
        on_signals: list[bool] = []
        off_signals: list[bool] = []
        current_state = "Neutral"

        for ratio_value, ma_value in zip(
            result["ratio"], result[ma_column]
        ):
            on_signal = False
            off_signal = False
            if pd.notna(ma_value):
                upper = float(ma_value) * (1 + config.buffer_pct)
                lower = float(ma_value) * (1 - config.buffer_pct)
                if (
                    float(ratio_value) > upper
                    and current_state != "Risk On"
                ):
                    current_state = "Risk On"
                    on_signal = True
                elif (
                    float(ratio_value) < lower
                    and current_state != "Risk Off"
                ):
                    current_state = "Risk Off"
                    off_signal = True

            states.append(current_state)
            on_signals.append(on_signal)
            off_signals.append(off_signal)

        result[state_column] = states
        result[on_column] = on_signals
        result[off_column] = off_signals
        result[f"cross_on{window}"] = on_signals
        result[f"cross_off{window}"] = off_signals

    return result


def json_number(value: Any, digits: int = 8) -> float | None:
    if value is None or pd.isna(value):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return round(number, digits)


def last_signal_date(frame: pd.DataFrame, column: str) -> str:
    matched = frame.loc[frame[column].fillna(False), "date"]
    if matched.empty:
        return "—"
    return pd.Timestamp(matched.iloc[-1]).strftime("%Y-%m-%d")


def slope_direction(value: Any) -> str:
    if value is None or pd.isna(value):
        return "—"
    number = float(value)
    signed = f"{number:+.5f}"
    if number > 0:
        return f"↑ 正／往上 ({signed})"
    if number < 0:
        return f"↓ 負／往下 ({signed})"
    return f"→ 持平 ({signed})"


def slope_css(value: Any) -> str:
    if value is None or pd.isna(value):
        return "slope-flat"
    if float(value) > 0:
        return "slope-up"
    if float(value) < 0:
        return "slope-down"
    return "slope-flat"


def build_payload(
    frame: pd.DataFrame, config: AppConfig
) -> dict[str, Any]:
    shown = frame.copy().reset_index(drop=True)
    shown["date_label"] = shown["date"].map(
        lambda value: pd.Timestamp(value).strftime("%Y-%m-%d")
    )

    windows_payload: dict[str, Any] = {}
    for window in REQUIRED_MAS:
        ma_column = f"ma{window}"
        slope_column = f"slope{window}"
        state_column = f"state{window}"
        on_column = f"bull_turn{window}"
        off_column = f"bear_turn{window}"

        signal = pd.Series("—", index=shown.index, dtype="object")
        signal.loc[shown[on_column].fillna(False)] = "Risk On"
        signal.loc[shown[off_column].fillna(False)] = "Risk Off"

        recent: list[dict[str, Any]] = []
        subset = shown.loc[
            shown[on_column].fillna(False)
            | shown[off_column].fillna(False)
        ].tail(10)
        for _, row in subset.iloc[::-1].iterrows():
            recent.append(
                {
                    "date": row["date_label"],
                    "signal": (
                        "Risk On" if bool(row[on_column]) else "Risk Off"
                    ),
                    "ratio": json_number(row["ratio"]),
                    "ma": json_number(row[ma_column]),
                }
            )

        windows_payload[str(window)] = {
            "ma": [json_number(value) for value in shown[ma_column]],
            "slope": [
                json_number(value) for value in shown[slope_column]
            ],
            "state": [
                str(value) if pd.notna(value) else "—"
                for value in shown[state_column]
            ],
            "signal": signal.tolist(),
            "riskOnX": shown.loc[
                shown[on_column].fillna(False), "date_label"
            ].tolist(),
            "riskOnY": [
                json_number(value)
                for value in shown.loc[
                    shown[on_column].fillna(False), "ratio"
                ]
            ],
            "riskOffX": shown.loc[
                shown[off_column].fillna(False), "date_label"
            ].tolist(),
            "riskOffY": [
                json_number(value)
                for value in shown.loc[
                    shown[off_column].fillna(False), "ratio"
                ]
            ],
            "latestMa": json_number(shown.iloc[-1][ma_column]),
            "latestSlope": json_number(
                shown.iloc[-1][slope_column]
            ),
            "latestState": str(
                shown.iloc[-1].get(state_column, "—")
            ),
            "lastOn": last_signal_date(shown, on_column),
            "lastOff": last_signal_date(shown, off_column),
            "signalRows": recent,
        }

    return {
        "dates": shown["date_label"].tolist(),
        "ratios": [json_number(value) for value in shown["ratio"]],
        "electronics": [
            json_number(value, 4)
            for value in shown["electronics_index"]
        ],
        "finance": [
            json_number(value, 4)
            for value in shown["finance_index"]
        ],
        "weighted": [
            json_number(value, 4)
            for value in shown["weighted_index"]
        ],
        "windows": windows_payload,
        "defaultWindow": "20",
        "defaultRangeYears": config.default_chart_range_years,
        "bufferPct": config.buffer_pct,
    }


def choose_font() -> None:
    preferred = (
        "Microsoft JhengHei", "Noto Sans CJK TC",
        "Noto Sans CJK JP", "PingFang TC", "Arial Unicode MS",
    )
    installed = {font.name for font in fm.fontManager.ttflist}
    for font_name in preferred:
        if font_name in installed:
            plt.rcParams["font.family"] = font_name
            plt.rcParams["axes.unicode_minus"] = False
            return


def make_static_chart(
    frame: pd.DataFrame, config: AppConfig
) -> None:
    choose_font()
    shown = frame.tail(config.chart_days).copy()
    fig, axis = plt.subplots(figsize=(16, 8))
    fig.patch.set_facecolor("#050505")
    axis.set_facecolor("#050505")
    axis.bar(
        shown["date"], shown["ratio"], width=0.8,
        color="#9b641f", edgecolor="#d48a31", linewidth=0.4,
        label="電金比",
    )
    axis.plot(
        shown["date"], shown["ma20"],
        color="#f0f0f0", linewidth=1.5, label="MA20",
    )
    on = shown[shown["bull_turn20"].fillna(False)]
    off = shown[shown["bear_turn20"].fillna(False)]
    axis.scatter(
        on["date"], on["ratio"], s=70,
        color="#ff2c55", label="Risk On", zorder=5,
    )
    axis.scatter(
        off["date"], off["ratio"], s=70,
        color="#00df45", label="Risk Off", zorder=5,
    )
    axis.grid(True, color="#333333", linewidth=0.5)
    axis.tick_params(colors="#dddddd")
    for spine in axis.spines.values():
        spine.set_color("#555555")
    axis.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axis.set_title(
        "台灣電子指數 ÷ 金融保險指數｜MA20",
        color="#ffffff", loc="left", fontsize=16,
    )
    legend = axis.legend(
        loc="upper left", facecolor="#111111", edgecolor="#555555"
    )
    for text in legend.get_texts():
        text.set_color("#eeeeee")
    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)


def make_html(frame: pd.DataFrame, config: AppConfig) -> None:
    payload_json = json.dumps(
        build_payload(frame, config),
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    latest = frame.iloc[-1]
    first_date = pd.Timestamp(frame.iloc[0]["date"]).strftime("%Y-%m-%d")
    latest_date = pd.Timestamp(latest["date"]).strftime("%Y-%m-%d")
    generated_at = datetime.now(TAIPEI).strftime(
        "%Y-%m-%d %H:%M:%S Asia/Taipei"
    )
    state = str(latest.get("state20", "—"))
    state_class = (
        "on" if state == "Risk On"
        else "off" if state == "Risk Off"
        else "neutral"
    )
    ma_value = latest.get("ma20")
    slope_value = latest.get("slope20")
    ma_text = "—" if pd.isna(ma_value) else f"{float(ma_value):.4f}"
    slope_text = slope_direction(slope_value)
    slope_class = slope_css(slope_value)
    version = datetime.now(TAIPEI).strftime("%Y%m%d%H%M%S")

    html_text = f'''<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="dark">
<title>台灣與美股電金比</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root{{--bg:#080808;--panel:#151515;--line:#343434;--text:#f0f0f0;--muted:#aaa}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--text);font-family:"Noto Sans TC","Microsoft JhengHei",system-ui,sans-serif}}
main{{width:min(1980px,100%);margin:auto;padding:20px 18px 60px}}
h1{{margin:0 0 6px;font-size:clamp(1.45rem,2.5vw,2.3rem)}}
.sub{{color:var(--muted);margin-bottom:16px}}
.summary{{display:grid;grid-template-columns:minmax(260px,1fr) minmax(300px,1fr);gap:14px;margin-bottom:14px}}
.metric-card,.ratio-card,.section{{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:18px}}
.ratio-card .value{{font-size:2.15rem;font-weight:760;margin:7px 0}}
.metric-title{{font-weight:700}}
.state{{display:inline-block;margin:8px 0 11px;padding:5px 10px;border-radius:999px;font-weight:750}}
.state.on{{background:#4a1020;color:#ff7995}}.state.off{{background:#073c18;color:#76f79a}}.state.neutral{{background:#333;color:#ddd}}
dl{{margin:0}}dl div{{display:flex;justify-content:space-between;gap:12px;border-top:1px solid #292929;padding:7px 0}}
dt{{color:var(--muted)}}dd{{margin:0;font-variant-numeric:tabular-nums}}
.chart-shell{{background:#050505;border:1px solid var(--line);border-radius:14px;overflow:hidden}}
.quote-panel{{display:grid;grid-template-columns:repeat(9,minmax(110px,1fr));gap:1px;background:#292929;border-bottom:1px solid #333}}
.quote-item{{background:#111;padding:8px 10px;min-height:55px}}
.quote-label{{color:#999;font-size:.78rem;margin-bottom:3px}}
.quote-value{{color:#f5f5f5;font-weight:700;font-variant-numeric:tabular-nums;white-space:nowrap}}
.slope-up{{color:#ff7f9b}}.slope-down{{color:#65ef8c}}.slope-flat{{color:#d4d4d4}}
.plot-heading{{display:flex;align-items:flex-end;justify-content:space-between;gap:14px;padding:13px 16px 0}}
.plot-title{{font-size:1.1rem;font-weight:750}}.plot-subtitle{{color:#bcbcbc;font-size:.92rem}}
.ma-selector{{display:flex;align-items:center;gap:8px;flex-wrap:wrap}}.ma-selector label{{color:#bbb;font-size:.9rem}}
.ma-select{{border:1px solid #555;background:#181818;color:#fff;border-radius:8px;padding:7px 34px 7px 10px;font:inherit;font-weight:700;cursor:pointer}}
.range-controls{{display:flex;flex-wrap:wrap;gap:7px;padding:10px 16px 2px}}
.range-button{{border:1px solid #444;background:#191919;color:#ddd;border-radius:8px;padding:6px 11px;cursor:pointer;font:inherit}}
.range-button.active{{background:#75501d;border-color:#c58a35;color:#fff;font-weight:700}}
#interactive-chart{{width:100%;height:clamp(580px,65vh,760px);min-height:580px}}
.chart-help{{color:#999;font-size:.86rem;padding:8px 12px 11px;background:#0b0b0b}}
.links{{display:flex;flex-wrap:wrap;gap:10px;margin:14px 0 24px}}
.links a{{color:#ffd37a;text-decoration:none;background:#222;border:1px solid #444;border-radius:9px;padding:8px 12px}}
.section{{margin-top:18px;overflow-x:auto}}table{{width:100%;border-collapse:collapse;min-width:620px}}
th,td{{text-align:left;padding:9px 10px;border-bottom:1px solid #303030;font-variant-numeric:tabular-nums}}
th{{color:#bbb}}.bull{{color:#ff6687;font-weight:700}}.bear{{color:#58ee83;font-weight:700}}
code{{color:#ffd37a}}.note{{color:#bdbdbd;line-height:1.75}}
@media(max-width:900px){{main{{padding:14px 8px 45px}}.summary{{grid-template-columns:1fr}}.quote-panel{{grid-template-columns:repeat(2,minmax(120px,1fr))}}#interactive-chart{{height:600px}}.plot-heading{{display:block;padding:12px 12px 2px}}.ma-selector{{margin-top:10px}}}}
</style>
</head>
<body>
<main>
<h1>台灣電金比風險偏好指標</h1>
<div class="sub">資料範圍：{first_date}～{latest_date}｜共 {len(frame):,} 個交易日｜網站更新：{generated_at}</div>

<div class="summary">
<section class="ratio-card">
<div>電子工業類指數 ÷ 金融保險類指數</div>
<div class="value">{float(latest["ratio"]):.4f}</div>
<dl>
<div><dt>電子工業類指數</dt><dd>{float(latest["electronics_index"]):,.2f}</dd></div>
<div><dt>金融保險類指數</dt><dd>{float(latest["finance_index"]):,.2f}</dd></div>
</dl>
</section>
<section class="metric-card">
<div class="metric-title" id="metric-title">20 日趨勢</div>
<div class="state {state_class}" id="metric-state">{state}</div>
<dl>
<div><dt id="metric-ma-label">MA20</dt><dd id="metric-ma-value">{ma_text}</dd></div>
<div><dt id="metric-slope-label">MA20 斜率</dt><dd class="{slope_class}" id="metric-slope-value">{slope_text}</dd></div>
<div><dt>最近 Risk On</dt><dd id="metric-last-on">{last_signal_date(frame, "bull_turn20")}</dd></div>
<div><dt>最近 Risk Off</dt><dd id="metric-last-off">{last_signal_date(frame, "bear_turn20")}</dd></div>
</dl>
</section>
</div>

<section class="chart-shell">
<div class="quote-panel">
<div class="quote-item"><div class="quote-label">查價日期</div><div class="quote-value" id="q-date">{latest_date}</div></div>
<div class="quote-item"><div class="quote-label">電金比</div><div class="quote-value" id="q-ratio">{float(latest["ratio"]):.4f}</div></div>
<div class="quote-item"><div class="quote-label" id="q-ma-label">MA20</div><div class="quote-value" id="q-ma">{ma_text}</div></div>
<div class="quote-item"><div class="quote-label" id="q-slope-label">MA20 斜率</div><div class="quote-value {slope_class}" id="q-slope">{slope_text}</div></div>
<div class="quote-item"><div class="quote-label">電子指數</div><div class="quote-value" id="q-electronics">{float(latest["electronics_index"]):,.2f}</div></div>
<div class="quote-item"><div class="quote-label">金融指數</div><div class="quote-value" id="q-finance">{float(latest["finance_index"]):,.2f}</div></div>
<div class="quote-item"><div class="quote-label">加權指數</div><div class="quote-value" id="q-weighted">{float(latest["weighted_index"]):,.2f}</div></div>
<div class="quote-item"><div class="quote-label">狀態</div><div class="quote-value" id="q-state">{state}</div></div>
<div class="quote-item"><div class="quote-label">當日訊號</div><div class="quote-value" id="q-signal">—</div></div>
</div>

<div class="plot-heading">
<div><div class="plot-title">電金比互動圖</div><div class="plot-subtitle" id="plot-subtitle">MA20｜{config.buffer_pct*100:.2f}% 緩衝訊號</div></div>
<div class="ma-selector"><label for="ma-window-select">判讀均線</label>
<select class="ma-select" id="ma-window-select">
<option value="20" selected>MA20</option><option value="60">MA60</option>
<option value="120">MA120</option><option value="240">MA240</option>
</select></div>
</div>
<div class="range-controls">
<button class="range-button" type="button" data-range="1">1年</button>
<button class="range-button" type="button" data-range="3">3年</button>
<button class="range-button" type="button" data-range="5">5年</button>
<button class="range-button" type="button" data-range="10">10年</button>
<button class="range-button" type="button" data-range="20">20年</button>
<button class="range-button" type="button" data-range="all">全部</button>
</div>
<div id="interactive-chart" aria-label="台灣電金比互動查價圖"></div>
<div class="chart-help">切換 MA20／60／120／240 後，程式會重新建立完整交易日分類順序，並依目前期間重新計算 X、Y 軸，不需要再手動按回首頁。</div>
</section>

<div class="links">
<a href="data.csv">下載完整每日資料 CSV</a>
<a href="signals.csv">下載反轉訊號 CSV</a>
<a href="latest.png?v={version}">開啟靜態圖表 PNG</a>
</div>

<section class="section note">
<h2>判讀規則</h2>
<p><strong>電金比＝電子工業類指數 ÷ 金融保險類指數。</strong></p>
<p id="rule-ma-text">緩衝區設定為 <code>{config.buffer_pct*100:.2f}%</code>。粉紅點代表突破 MA20 上方緩衝區並切換為 <strong>Risk On</strong>；綠點代表跌破下方緩衝區並切換為 <strong>Risk Off</strong>。</p>
</section>
<section class="section">
<h2 id="signals-heading">近期 MA20 Risk On／Risk Off 切換</h2>
<table><thead><tr><th>日期</th><th>訊號</th><th>電金比</th><th id="signals-ma-heading">MA20</th></tr></thead>
<tbody id="signals-body"></tbody></table>
</section>
</main>

<script>
(()=>{{
'use strict';
const chartData={payload_json};
const dates=chartData.dates;
const plot=document.getElementById('interactive-chart');
const buttons=Array.from(document.querySelectorAll('[data-range]'));
const windowSelect=document.getElementById('ma-window-select');
let selectedWindow=String(chartData.defaultWindow||'20');
let activeRange=String(chartData.defaultRangeYears||1);
let currentPointIndex=Math.max(0,dates.length-1);

const fmt=(value,digits=2)=>value===null||value===undefined||Number.isNaN(Number(value))
?'—':Number(value).toLocaleString('zh-TW',{{minimumFractionDigits:digits,maximumFractionDigits:digits}});

function slopeInfo(value){{
 if(value===null||value===undefined||Number.isNaN(Number(value)))return{{text:'—',className:'slope-flat'}};
 const n=Number(value),signed=(n>=0?'+':'')+n.toFixed(5);
 if(n>0)return{{text:'↑ 正／往上 ('+signed+')',className:'slope-up'}};
 if(n<0)return{{text:'↓ 負／往下 ('+signed+')',className:'slope-down'}};
 return{{text:'→ 持平 ('+signed+')',className:'slope-flat'}};
}}
function setSlopeElement(element,value){{
 const info=slopeInfo(value);element.textContent=info.text;
 element.classList.remove('slope-up','slope-down','slope-flat');element.classList.add(info.className);
}}
function stateClass(value){{return value==='Risk On'?'on':value==='Risk Off'?'off':'neutral'}}
function currentWindowData(){{return chartData.windows[selectedWindow]}}
function buildCustomData(){{
 const data=currentWindowData();
 return dates.map((d,i)=>[d,chartData.electronics[i],chartData.finance[i],chartData.weighted[i],data.ma[i],data.slope[i],data.state[i],data.signal[i]]);
}}
function hoverTemplate(){{
 return '<b>%{{customdata[0]}}</b><br>電金比：%{{y:.4f}}<br>MA'+selectedWindow+
 '：%{{customdata[4]:.4f}}<br>斜率：%{{customdata[5]:+.5f}}<br>狀態：%{{customdata[6]}}<br>訊號：%{{customdata[7]}}<extra></extra>';
}}
function startIndexForYears(rangeValue){{
 if(String(rangeValue)==='all')return 0;
 const years=Math.max(1,Number(rangeValue));
 const latest=new Date(dates[dates.length-1]+'T00:00:00'),cutoff=new Date(latest);
 cutoff.setFullYear(cutoff.getFullYear()-years);
 const text=cutoff.toISOString().slice(0,10),found=dates.findIndex(v=>v>=text);
 return found<0?0:found;
}}
function buildTicks(startIndex,rangeValue){{
 const vals=[],texts=[],years=String(rangeValue)==='all'?99:Number(rangeValue);
 const step=years<=1?1:years<=3?3:years<=5?6:12;let previous='';
 for(let i=startIndex;i<dates.length;i++){{
  const value=dates[i],year=Number(value.slice(0,4)),month=Number(value.slice(5,7)),key=year+'-'+month;
  if(key===previous)continue;previous=key;
  if(step===12){{if(month!==1)continue}}else if((month-1)%step!==0)continue;
  vals.push(value);texts.push(step===12?String(year):value.slice(0,7));
 }}
 return{{vals,texts}};
}}
function finiteRange(values,minPadding){{
 const numeric=values.filter(v=>v!==null&&Number.isFinite(Number(v))).map(Number);
 if(!numeric.length)return null;
 const min=Math.min(...numeric),max=Math.max(...numeric),pad=Math.max((max-min)*0.075,minPadding);
 return[min-pad,max+pad];
}}
async function applyRange(rangeValue){{
 activeRange=String(rangeValue);
 const start=startIndexForYears(activeRange),ticks=buildTicks(start,activeRange),data=currentWindowData();
 const ratioRange=finiteRange(chartData.ratios.slice(start).concat(data.ma.slice(start)),0.015);
 const weightedRange=finiteRange(chartData.weighted.slice(start),100);
 const changes={{
  'xaxis.type':'category','xaxis.categoryorder':'array','xaxis.categoryarray':dates,
  'xaxis.autorange':false,'xaxis.range':[start-0.5,dates.length-0.5],
  'xaxis.tickmode':'array','xaxis.tickvals':ticks.vals,'xaxis.ticktext':ticks.texts,
  'yaxis.autorange':false,'yaxis2.autorange':false
 }};
 if(ratioRange)changes['yaxis.range']=ratioRange;
 if(weightedRange)changes['yaxis2.range']=weightedRange;
 await Plotly.relayout(plot,changes);Plotly.Plots.resize(plot);
 buttons.forEach(b=>b.classList.toggle('active',b.dataset.range===activeRange));
}}
function updateQuote(index){{
 currentPointIndex=Math.max(0,Math.min(Number(index),dates.length-1));
 const data=currentWindowData();
 document.getElementById('q-date').textContent=dates[currentPointIndex];
 document.getElementById('q-ratio').textContent=fmt(chartData.ratios[currentPointIndex],4);
 document.getElementById('q-ma').textContent=fmt(data.ma[currentPointIndex],4);
 setSlopeElement(document.getElementById('q-slope'),data.slope[currentPointIndex]);
 document.getElementById('q-electronics').textContent=fmt(chartData.electronics[currentPointIndex],2);
 document.getElementById('q-finance').textContent=fmt(chartData.finance[currentPointIndex],2);
 document.getElementById('q-weighted').textContent=fmt(chartData.weighted[currentPointIndex],2);
 document.getElementById('q-state').textContent=data.state[currentPointIndex]||'—';
 document.getElementById('q-signal').textContent=data.signal[currentPointIndex]||'—';
}}
function renderSignalTable(){{
 const rows=currentWindowData().signalRows,body=document.getElementById('signals-body');
 if(!rows.length){{body.innerHTML='<tr><td colspan="4">目前尚無足夠資料形成切換訊號。</td></tr>';return}}
 body.innerHTML=rows.map(r=>'<tr><td>'+r.date+'</td><td class="'+(r.signal==='Risk On'?'bull':'bear')+'">'+r.signal+
 '</td><td>'+fmt(r.ratio,4)+'</td><td>'+fmt(r.ma,4)+'</td></tr>').join('');
}}
function updateWindowText(){{
 const data=currentWindowData(),stateElement=document.getElementById('metric-state');
 document.getElementById('metric-title').textContent=selectedWindow+' 日趨勢';
 stateElement.textContent=data.latestState||'—';stateElement.classList.remove('on','off','neutral');stateElement.classList.add(stateClass(data.latestState));
 document.getElementById('metric-ma-label').textContent='MA'+selectedWindow;
 document.getElementById('metric-ma-value').textContent=fmt(data.latestMa,4);
 document.getElementById('metric-slope-label').textContent='MA'+selectedWindow+' 斜率';
 setSlopeElement(document.getElementById('metric-slope-value'),data.latestSlope);
 document.getElementById('metric-last-on').textContent=data.lastOn||'—';
 document.getElementById('metric-last-off').textContent=data.lastOff||'—';
 document.getElementById('q-ma-label').textContent='MA'+selectedWindow;
 document.getElementById('q-slope-label').textContent='MA'+selectedWindow+' 斜率';
 document.getElementById('plot-subtitle').textContent='MA'+selectedWindow+'｜'+(chartData.bufferPct*100).toFixed(2)+'% 緩衝訊號';
 document.getElementById('signals-heading').textContent='近期 MA'+selectedWindow+' Risk On／Risk Off 切換';
 document.getElementById('signals-ma-heading').textContent='MA'+selectedWindow;
 document.getElementById('rule-ma-text').innerHTML='緩衝區設定為 <code>'+(chartData.bufferPct*100).toFixed(2)+'%</code>。粉紅點代表突破 MA'+selectedWindow+
 ' 上方緩衝區並切換為 <strong>Risk On</strong>；綠點代表跌破下方緩衝區並切換為 <strong>Risk Off</strong>。';
 renderSignalTable();updateQuote(currentPointIndex);
}}
async function switchWindow(value){{
 const requested=String(value);
 if(!chartData.windows[requested]||requested===selectedWindow)return;
 selectedWindow=requested;windowSelect.value=selectedWindow;windowSelect.disabled=true;
 const data=currentWindowData();
 try{{
  await Plotly.restyle(plot,{{y:[data.ma],name:'MA'+selectedWindow}},[1]);
  await Plotly.restyle(plot,{{x:[data.riskOnX],y:[data.riskOnY]}},[3]);
  await Plotly.restyle(plot,{{x:[data.riskOffX],y:[data.riskOffY]}},[4]);
  await Plotly.restyle(plot,{{customdata:[buildCustomData()],hovertemplate:hoverTemplate()}},[5]);
  updateWindowText();
  await applyRange(activeRange);
 }}catch(error){{console.error('切換判讀均線失敗：',error);await applyRange(activeRange)}}
 finally{{windowSelect.disabled=false;windowSelect.focus({{preventScroll:true}})}}
}}

const initial=currentWindowData();
const traces=[
 {{type:'bar',x:dates,y:chartData.ratios,name:'電金比',marker:{{color:'#9b641f',line:{{color:'#d48a31',width:.35}}}},hoverinfo:'skip'}},
 {{type:'scatter',mode:'lines',x:dates,y:initial.ma,name:'MA'+selectedWindow,line:{{color:'#f1f1f1',width:1.6}},hoverinfo:'skip'}},
 {{type:'scatter',mode:'lines',x:dates,y:chartData.weighted,name:'加權指數',yaxis:'y2',visible:'legendonly',line:{{color:'#58a6ff',width:1.3}},hoverinfo:'skip'}},
 {{type:'scatter',mode:'markers',x:initial.riskOnX,y:initial.riskOnY,name:'Risk On',marker:{{color:'#ff2c55',size:10,line:{{color:'#ff8aa1',width:1}}}},hoverinfo:'skip'}},
 {{type:'scatter',mode:'markers',x:initial.riskOffX,y:initial.riskOffY,name:'Risk Off',marker:{{color:'#00df45',size:10,line:{{color:'#9affb5',width:1}}}},hoverinfo:'skip'}},
 {{type:'scatter',mode:'markers',x:dates,y:chartData.ratios,name:'查價',marker:{{size:18,color:'rgba(0,0,0,0)'}},showlegend:false,customdata:buildCustomData(),hovertemplate:hoverTemplate()}}
];
const layout={{
 paper_bgcolor:'#050505',plot_bgcolor:'#050505',margin:{{l:52,r:65,t:20,b:55}},
 hovermode:'x unified',dragmode:'zoom',bargap:.12,
 legend:{{orientation:'h',x:0,y:1.08,font:{{color:'#ddd'}}}},font:{{color:'#ddd'}},
 xaxis:{{type:'category',categoryorder:'array',categoryarray:dates,autorange:false,gridcolor:'#252525',zeroline:false,fixedrange:false}},
 yaxis:{{title:'電金比',autorange:false,side:'right',gridcolor:'#303030',zeroline:false,fixedrange:false}},
 yaxis2:{{title:'加權指數',autorange:false,overlaying:'y',side:'left',showgrid:false,visible:false,fixedrange:false}},
 uirevision:'tw-electric-finance-ratio-v3'
}};
Plotly.newPlot(plot,traces,layout,{{responsive:true,scrollZoom:true,displaylogo:false,modeBarButtonsToRemove:['lasso2d','select2d','toggleSpikelines'],doubleClick:'reset'}})
.then(async()=>{{await applyRange(activeRange);updateWindowText();document.documentElement.dataset.chartReady='true'}});
buttons.forEach(b=>b.addEventListener('click',()=>applyRange(b.dataset.range)));
windowSelect.addEventListener('change',e=>switchWindow(e.target.value));
plot.on('plotly_legendclick',event=>{{
 if(event.curveNumber!==2)return;
 const current=plot.data[2].visible,hidden=current==='legendonly'||current===false;
 Plotly.restyle(plot,{{visible:hidden?true:'legendonly'}},[2]);Plotly.relayout(plot,{{'yaxis2.visible':hidden}});return false;
}});
plot.on('plotly_hover',event=>{{const point=event.points.find(item=>item.curveNumber===5);if(point)updateQuote(point.pointIndex)}});
}})();
</script>
</body>
</html>
'''
    HTML_PATH.write_text(html_text, encoding="utf-8")


def save_base_data(frame: pd.DataFrame) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    frame[
        ["date", "electronics_index", "finance_index", "weighted_index", "ratio"]
    ].to_csv(DATA_CSV, index=False, date_format="%Y-%m-%d")


def save_outputs(frame: pd.DataFrame, config: AppConfig) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    save_base_data(frame)
    frame.to_csv(
        WEB_CSV, index=False, date_format="%Y-%m-%d",
        encoding="utf-8-sig",
    )

    parts: list[pd.DataFrame] = []
    for window in config.moving_averages:
        on_column = f"bull_turn{window}"
        off_column = f"bear_turn{window}"
        mask = frame[on_column].fillna(False) | frame[off_column].fillna(False)
        part = frame.loc[
            mask,
            [
                "date", "ratio", f"ma{window}", f"slope{window}",
                f"state{window}", on_column,
            ],
        ].copy()
        if part.empty:
            continue
        part["window"] = window
        part["signal"] = part[on_column].map(
            {True: "Risk On", False: "Risk Off"}
        )
        part = part.rename(
            columns={
                f"ma{window}": "moving_average",
                f"slope{window}": "slope",
                f"state{window}": "state",
            }
        )
        parts.append(
            part[
                [
                    "date", "window", "signal", "ratio",
                    "moving_average", "slope", "state",
                ]
            ]
        )

    signals = (
        pd.concat(parts, ignore_index=True).sort_values(["date", "window"])
        if parts else
        pd.DataFrame(
            columns=[
                "date", "window", "signal", "ratio",
                "moving_average", "slope", "state",
            ]
        )
    )
    signals.to_csv(
        SIGNALS_CSV, index=False, date_format="%Y-%m-%d",
        encoding="utf-8-sig",
    )
    make_static_chart(frame, config)
    make_html(frame, config)
    (DOCS_DIR / ".nojekyll").touch()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    config = load_config()

    if args.demo:
        frame = make_demo_data()
    else:
        frame = read_existing_data()
        if not args.no_fetch:
            frame = update_data(frame, config, args)

    if frame.empty:
        LOGGER.error("沒有任何資料可供繪圖。")
        return 2

    if args.data_only:
        save_base_data(frame)
        LOGGER.info("資料回補完成，共 %d 筆交易日。", len(frame))
        return 0

    frame = compute_signals(frame, config)
    if len(frame) < min(config.moving_averages):
        LOGGER.error("資料筆數不足以計算最短均線。")
        return 2

    save_outputs(frame, config)
    latest = frame.iloc[-1]
    LOGGER.info(
        "台股電金比完成：%s，最新 %.6f。",
        pd.Timestamp(latest["date"]).strftime("%Y-%m-%d"),
        float(latest["ratio"]),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
