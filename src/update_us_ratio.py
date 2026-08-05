from __future__ import annotations

import argparse
import json
import logging
import math
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.json"
DATA_DIR = ROOT / "data"
DOCS_DIR = ROOT / "docs"
HTML_PATH = DOCS_DIR / "index.html"
US_DATA_CSV = DATA_DIR / "us_technology_finance_ratio.csv"
US_WEB_CSV = DOCS_DIR / "us_data.csv"
US_SIGNALS_CSV = DOCS_DIR / "us_signals.csv"

TAIPEI = ZoneInfo("Asia/Taipei")
LOGGER = logging.getLogger("us_technology_finance_ratio")

SECTION_START = "<!-- US_RATIO_SECTION_START -->"
SECTION_END = "<!-- US_RATIO_SECTION_END -->"
SCRIPT_START = "<!-- US_RATIO_SCRIPT_START -->"
SCRIPT_END = "<!-- US_RATIO_SCRIPT_END -->"

TICKERS = ("XLK", "XLF", "SPY")
START_DATE = "1998-01-01"
DEFAULT_WINDOWS = (20, 60, 120, 240)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="抓取 XLK、XLF、SPY 還原後價格，建立美股電金比。"
    )
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="使用既有美股 CSV 重建網頁。",
    )
    return parser.parse_args()


def load_display_config() -> tuple[tuple[int, ...], int, float, int]:
    raw: dict[str, Any] = {}
    if CONFIG_PATH.exists():
        try:
            raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception as exc:
            LOGGER.warning("config.json 讀取失敗，採用預設值：%s", exc)

    configured = tuple(int(value) for value in raw.get("moving_averages", []))
    if any(value <= 1 for value in configured):
        raise ValueError("moving_averages 必須是大於 1 的整數陣列。")

    windows = tuple(dict.fromkeys(DEFAULT_WINDOWS + configured))
    default_window = 20
    buffer_pct = max(0.0, float(raw.get("buffer_pct", 0.0)))
    default_range_years = max(
        1, int(raw.get("default_chart_range_years", 1))
    )
    return windows, default_window, buffer_pct, default_range_years


def extract_adjusted_close(
    downloaded: pd.DataFrame, ticker: str
) -> pd.Series:
    if downloaded.empty:
        raise RuntimeError("yfinance 回傳空資料。")

    if isinstance(downloaded.columns, pd.MultiIndex):
        candidates = (
            ("Adj Close", ticker),
            (ticker, "Adj Close"),
            ("Close", ticker),
            (ticker, "Close"),
        )
        for candidate in candidates:
            if candidate in downloaded.columns:
                return pd.to_numeric(
                    downloaded[candidate], errors="coerce"
                ).rename(ticker)
    else:
        for column in ("Adj Close", "Close"):
            if column in downloaded.columns:
                return pd.to_numeric(
                    downloaded[column], errors="coerce"
                ).rename(ticker)

    raise RuntimeError(
        f"找不到 {ticker} 還原後收盤價。欄位："
        f"{list(downloaded.columns)[:12]}"
    )


def download_adjusted_prices() -> pd.DataFrame:
    end_date = (
        datetime.now(TAIPEI).date() + timedelta(days=2)
    ).isoformat()
    last_error: Exception | None = None

    for attempt in range(1, 4):
        try:
            LOGGER.info(
                "下載美股資料，第 %d/3 次：%s 至 %s",
                attempt, START_DATE, end_date,
            )
            downloaded = yf.download(
                tickers=list(TICKERS),
                start=START_DATE,
                end=end_date,
                interval="1d",
                auto_adjust=False,
                actions=False,
                repair=True,
                progress=False,
                threads=False,
                group_by="column",
                timeout=45,
            )
            parts = [
                extract_adjusted_close(downloaded, ticker)
                for ticker in TICKERS
            ]
            prices = pd.concat(parts, axis=1)
            prices.columns = list(TICKERS)
            prices.index = pd.to_datetime(
                prices.index, errors="coerce"
            )
            if getattr(prices.index, "tz", None) is not None:
                prices.index = prices.index.tz_localize(None)

            prices = prices[~prices.index.isna()].sort_index()
            prices = prices[
                ~prices.index.duplicated(keep="last")
            ]
            prices = prices.dropna(subset=["XLK", "XLF"])
            prices = prices[
                (prices["XLK"] > 0) & (prices["XLF"] > 0)
            ]
            prices["SPY"] = pd.to_numeric(
                prices["SPY"], errors="coerce"
            ).ffill()
            prices = prices.dropna(subset=["SPY"])
            prices = prices[prices["SPY"] > 0]

            if len(prices) < 5000:
                raise RuntimeError(
                    f"有效共同交易日只有 {len(prices)} 筆。"
                )
            LOGGER.info(
                "美股資料完成：%s 至 %s，共 %d 筆。",
                prices.index.min().date(),
                prices.index.max().date(),
                len(prices),
            )
            return prices

        except Exception as exc:
            last_error = exc
            LOGGER.warning("第 %d 次下載失敗：%s", attempt, exc)
            if attempt < 3:
                time.sleep(8 * attempt)

    raise RuntimeError(f"美股資料下載失敗：{last_error}")


def add_signal_columns(
    frame: pd.DataFrame,
    windows: tuple[int, ...],
    buffer_pct: float,
) -> pd.DataFrame:
    result = (
        frame.copy()
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )

    for window in windows:
        ma_column = f"ma{window}"
        slope_column = f"slope{window}"
        state_column = f"state{window}"
        on_column = f"bull_turn{window}"
        off_column = f"bear_turn{window}"

        result[ma_column] = result["us_ratio"].rolling(
            window, min_periods=window
        ).mean()
        result[slope_column] = result[ma_column].diff()
        result[f"ma{window}_slope"] = result[slope_column]

        states: list[str] = []
        on_signals: list[bool] = []
        off_signals: list[bool] = []
        current_state = "Neutral"

        for ratio_value, ma_value in zip(
            result["us_ratio"], result[ma_column]
        ):
            on_signal = False
            off_signal = False
            if pd.notna(ma_value):
                upper = float(ma_value) * (1 + buffer_pct)
                lower = float(ma_value) * (1 - buffer_pct)
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


def compute_ratio(
    prices: pd.DataFrame,
    windows: tuple[int, ...],
    buffer_pct: float,
) -> pd.DataFrame:
    result = prices.copy()
    result.index.name = "date"
    result = result.rename(
        columns={
            "XLK": "xlk_adj_close",
            "XLF": "xlf_adj_close",
            "SPY": "spy_adj_close",
        }
    )

    first = result.iloc[0]
    result["xlk_normalized"] = (
        result["xlk_adj_close"]
        / float(first["xlk_adj_close"])
        * 100.0
    )
    result["xlf_normalized"] = (
        result["xlf_adj_close"]
        / float(first["xlf_adj_close"])
        * 100.0
    )
    result["spy_normalized"] = (
        result["spy_adj_close"]
        / float(first["spy_adj_close"])
        * 100.0
    )
    result["us_ratio"] = (
        result["xlk_normalized"]
        / result["xlf_normalized"]
    )

    if not math.isclose(
        float(result["us_ratio"].iloc[0]),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("共同起始日美股電金比不等於 1。")

    return add_signal_columns(
        result.reset_index(), windows, buffer_pct
    )


def read_existing(
    windows: tuple[int, ...],
    buffer_pct: float,
) -> pd.DataFrame:
    if not US_DATA_CSV.exists():
        raise FileNotFoundError(
            f"找不到既有美股資料：{US_DATA_CSV}"
        )

    frame = pd.read_csv(US_DATA_CSV, parse_dates=["date"])
    required = {
        "date",
        "xlk_adj_close",
        "xlf_adj_close",
        "spy_adj_close",
        "xlk_normalized",
        "xlf_normalized",
        "spy_normalized",
        "us_ratio",
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(
            f"既有美股 CSV 缺少欄位：{sorted(missing)}"
        )

    for column in required - {"date"}:
        frame[column] = pd.to_numeric(
            frame[column], errors="coerce"
        )
    frame = (
        frame.dropna(subset=list(required))
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )
    return add_signal_columns(frame, windows, buffer_pct)


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


def save_data(
    frame: pd.DataFrame, windows: tuple[int, ...]
) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    frame.to_csv(
        US_DATA_CSV, index=False, date_format="%Y-%m-%d"
    )
    frame.to_csv(
        US_WEB_CSV, index=False, date_format="%Y-%m-%d",
        encoding="utf-8-sig",
    )

    parts: list[pd.DataFrame] = []
    for window in windows:
        on_column = f"bull_turn{window}"
        off_column = f"bear_turn{window}"
        mask = frame[on_column].fillna(False) | frame[off_column].fillna(False)
        part = frame.loc[
            mask,
            [
                "date", "us_ratio", f"ma{window}", f"slope{window}",
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
                "us_ratio": "ratio",
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
        US_SIGNALS_CSV,
        index=False,
        date_format="%Y-%m-%d",
        encoding="utf-8-sig",
    )


def remove_marker_block(text: str, start: str, end: str) -> str:
    pattern = re.compile(
        re.escape(start) + r".*?" + re.escape(end) + r"\s*",
        flags=re.DOTALL,
    )
    return pattern.sub("", text)


def build_payload(
    frame: pd.DataFrame,
    windows: tuple[int, ...],
    default_window: int,
    buffer_pct: float,
    default_range_years: int,
) -> dict[str, Any]:
    shown = frame.copy().reset_index(drop=True)
    shown["date_label"] = shown["date"].map(
        lambda value: pd.Timestamp(value).strftime("%Y-%m-%d")
    )

    windows_payload: dict[str, Any] = {}
    for window in DEFAULT_WINDOWS:
        if window not in windows:
            continue

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
                    "ratio": json_number(row["us_ratio"]),
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
                    shown[on_column].fillna(False), "us_ratio"
                ]
            ],
            "riskOffX": shown.loc[
                shown[off_column].fillna(False), "date_label"
            ].tolist(),
            "riskOffY": [
                json_number(value)
                for value in shown.loc[
                    shown[off_column].fillna(False), "us_ratio"
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
        "ratios": [json_number(value) for value in shown["us_ratio"]],
        "xlkNorm": [
            json_number(value, 6)
            for value in shown["xlk_normalized"]
        ],
        "xlfNorm": [
            json_number(value, 6)
            for value in shown["xlf_normalized"]
        ],
        "spyNorm": [
            json_number(value, 6)
            for value in shown["spy_normalized"]
        ],
        "xlkClose": [
            json_number(value, 6)
            for value in shown["xlk_adj_close"]
        ],
        "xlfClose": [
            json_number(value, 6)
            for value in shown["xlf_adj_close"]
        ],
        "windows": windows_payload,
        "defaultWindow": str(default_window),
        "defaultRangeYears": default_range_years,
        "bufferPct": buffer_pct,
    }


def build_section(
    frame: pd.DataFrame,
    default_window: int,
    buffer_pct: float,
) -> str:
    latest = frame.iloc[-1]
    first_date = pd.Timestamp(frame.iloc[0]["date"]).strftime("%Y-%m-%d")
    latest_date = pd.Timestamp(latest["date"]).strftime("%Y-%m-%d")
    generated_at = datetime.now(TAIPEI).strftime(
        "%Y-%m-%d %H:%M:%S Asia/Taipei"
    )
    ma_value = latest.get(f"ma{default_window}")
    slope_value = latest.get(f"slope{default_window}")
    state = str(latest.get(f"state{default_window}", "—"))
    state_class = (
        "on" if state == "Risk On"
        else "off" if state == "Risk Off"
        else "neutral"
    )
    ma_text = "—" if pd.isna(ma_value) else f"{float(ma_value):.4f}"

    return f'''{SECTION_START}
<section class="section note" style="margin-top:48px">
<h2 style="margin-top:0;color:#f5f5f5">美股電金比風險偏好指標</h2>
<div class="sub">資料範圍：{first_date}～{latest_date}｜共 {len(frame):,} 個共同交易日｜美股資料更新：{generated_at}</div>
<p>美股電金比＝XLK 科技總報酬標準化指數 ÷ XLF 金融總報酬標準化指數。</p>
</section>

<div class="summary" style="margin-top:14px">
<section class="ratio-card">
<div>XLK 科技標準化指數 ÷ XLF 金融標準化指數</div>
<div class="value">{float(latest["us_ratio"]):.4f}</div>
<dl>
<div><dt>XLK 標準化指數</dt><dd>{float(latest["xlk_normalized"]):,.2f}</dd></div>
<div><dt>XLF 標準化指數</dt><dd>{float(latest["xlf_normalized"]):,.2f}</dd></div>
</dl>
</section>
<section class="metric-card">
<div class="metric-title" id="us-metric-title">MA{default_window} 趨勢</div>
<div class="state {state_class}" id="us-metric-state">{state}</div>
<dl>
<div><dt id="us-metric-ma-label">MA{default_window}</dt><dd id="us-metric-ma-value">{ma_text}</dd></div>
<div><dt id="us-metric-slope-label">MA{default_window} 斜率</dt><dd class="{slope_css(slope_value)}" id="us-metric-slope-value">{slope_direction(slope_value)}</dd></div>
<div><dt>最近 Risk On</dt><dd id="us-metric-last-on">{last_signal_date(frame, f"bull_turn{default_window}")}</dd></div>
<div><dt>最近 Risk Off</dt><dd id="us-metric-last-off">{last_signal_date(frame, f"bear_turn{default_window}")}</dd></div>
</dl>
</section>
</div>

<section class="chart-shell" id="us-ratio-section">
<div class="quote-panel">
<div class="quote-item"><div class="quote-label">查價日期</div><div class="quote-value" id="us-q-date">{latest_date}</div></div>
<div class="quote-item"><div class="quote-label">美股電金比</div><div class="quote-value" id="us-q-ratio">{float(latest["us_ratio"]):.4f}</div></div>
<div class="quote-item"><div class="quote-label" id="us-q-ma-label">MA{default_window}</div><div class="quote-value" id="us-q-ma">{ma_text}</div></div>
<div class="quote-item"><div class="quote-label" id="us-q-slope-label">MA{default_window} 斜率</div><div class="quote-value {slope_css(slope_value)}" id="us-q-slope">{slope_direction(slope_value)}</div></div>
<div class="quote-item"><div class="quote-label">XLK 標準化</div><div class="quote-value" id="us-q-xlk">{float(latest["xlk_normalized"]):,.2f}</div></div>
<div class="quote-item"><div class="quote-label">XLF 標準化</div><div class="quote-value" id="us-q-xlf">{float(latest["xlf_normalized"]):,.2f}</div></div>
<div class="quote-item"><div class="quote-label">SPY 標準化</div><div class="quote-value" id="us-q-spy">{float(latest["spy_normalized"]):,.2f}</div></div>
<div class="quote-item"><div class="quote-label">狀態</div><div class="quote-value" id="us-q-state">{state}</div></div>
<div class="quote-item"><div class="quote-label">當日訊號</div><div class="quote-value" id="us-q-signal">—</div></div>
</div>

<div class="plot-heading">
<div><div class="plot-title">美股電金比互動圖</div><div class="plot-subtitle" id="us-plot-subtitle">XLK／XLF 還原後總報酬資料｜MA{default_window}</div></div>
<div class="ma-selector"><label for="us-ma-window-select">判讀均線</label>
<select class="ma-select" id="us-ma-window-select">
<option value="20" selected>MA20</option><option value="60">MA60</option>
<option value="120">MA120</option><option value="240">MA240</option>
</select></div>
</div>
<div class="range-controls">
<button class="range-button" type="button" data-us-range="1">1年</button>
<button class="range-button" type="button" data-us-range="3">3年</button>
<button class="range-button" type="button" data-us-range="5">5年</button>
<button class="range-button" type="button" data-us-range="10">10年</button>
<button class="range-button" type="button" data-us-range="20">20年</button>
<button class="range-button" type="button" data-us-range="all">全部</button>
</div>
<div id="us-interactive-chart" style="width:100%;height:clamp(580px,65vh,760px);min-height:580px" aria-label="美股電金比互動查價圖"></div>
<div class="chart-help">切換 MA20／60／120／240 後，程式會重新建立完整交易日分類順序，並依目前期間重新計算 X、Y 軸，不需要再手動按回首頁。</div>
</section>

<div class="links">
<a href="us_data.csv">下載美股完整每日資料 CSV</a>
<a href="us_signals.csv">下載美股反轉訊號 CSV</a>
</div>

<section class="section note">
<h2>美股版判讀規則</h2>
<p id="us-rule-ma-text">緩衝區設定為 <code>{buffer_pct*100:.2f}%</code>。粉紅點代表突破 MA{default_window} 上方緩衝區並切換為 <strong>Risk On</strong>；綠點代表跌破下方緩衝區並切換為 <strong>Risk Off</strong>。</p>
</section>
<section class="section">
<h2 id="us-signals-heading">近期 MA{default_window} Risk On／Risk Off 切換</h2>
<table><thead><tr><th>日期</th><th>訊號</th><th>美股電金比</th><th id="us-signals-ma-heading">MA{default_window}</th></tr></thead>
<tbody id="us-signals-body"></tbody></table>
</section>
{SECTION_END}
'''


def build_script(
    frame: pd.DataFrame,
    windows: tuple[int, ...],
    default_window: int,
    buffer_pct: float,
    default_range_years: int,
) -> str:
    payload_json = json.dumps(
        build_payload(
            frame, windows, default_window,
            buffer_pct, default_range_years,
        ),
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )

    return f'''{SCRIPT_START}
<script>
(()=>{{
'use strict';
const usChartData={payload_json};
const usDates=usChartData.dates;
const usPlot=document.getElementById('us-interactive-chart');
const usButtons=Array.from(document.querySelectorAll('[data-us-range]'));
const usWindowSelect=document.getElementById('us-ma-window-select');
let usSelectedWindow=String(usChartData.defaultWindow||'20');
let usActiveRange=String(usChartData.defaultRangeYears||1);
let usCurrentPointIndex=Math.max(0,usDates.length-1);

const usFmt=(value,digits=2)=>value===null||value===undefined||Number.isNaN(Number(value))
?'—':Number(value).toLocaleString('zh-TW',{{minimumFractionDigits:digits,maximumFractionDigits:digits}});

function usSlopeInfo(value){{
 if(value===null||value===undefined||Number.isNaN(Number(value)))return{{text:'—',className:'slope-flat'}};
 const n=Number(value),signed=(n>=0?'+':'')+n.toFixed(5);
 if(n>0)return{{text:'↑ 正／往上 ('+signed+')',className:'slope-up'}};
 if(n<0)return{{text:'↓ 負／往下 ('+signed+')',className:'slope-down'}};
 return{{text:'→ 持平 ('+signed+')',className:'slope-flat'}};
}}
function usSetSlopeElement(element,value){{
 const info=usSlopeInfo(value);element.textContent=info.text;
 element.classList.remove('slope-up','slope-down','slope-flat');element.classList.add(info.className);
}}
function usStateClass(value){{return value==='Risk On'?'on':value==='Risk Off'?'off':'neutral'}}
function usCurrentWindowData(){{return usChartData.windows[usSelectedWindow]}}
function usBuildCustomData(){{
 const data=usCurrentWindowData();
 return usDates.map((d,i)=>[d,usChartData.xlkNorm[i],usChartData.xlfNorm[i],usChartData.spyNorm[i],data.ma[i],data.slope[i],data.state[i],data.signal[i],usChartData.xlkClose[i],usChartData.xlfClose[i]]);
}}
function usHoverTemplate(){{
 return '<b>%{{customdata[0]}}</b><br>美股電金比：%{{y:.4f}}<br>MA'+usSelectedWindow+
 '：%{{customdata[4]:.4f}}<br>斜率：%{{customdata[5]:+.5f}}<br>狀態：%{{customdata[6]}}<br>訊號：%{{customdata[7]}}<extra></extra>';
}}
function usStartIndexForYears(rangeValue){{
 if(String(rangeValue)==='all')return 0;
 const years=Math.max(1,Number(rangeValue));
 const latest=new Date(usDates[usDates.length-1]+'T00:00:00'),cutoff=new Date(latest);
 cutoff.setFullYear(cutoff.getFullYear()-years);
 const text=cutoff.toISOString().slice(0,10),found=usDates.findIndex(v=>v>=text);
 return found<0?0:found;
}}
function usBuildTicks(startIndex,rangeValue){{
 const vals=[],texts=[],years=String(rangeValue)==='all'?99:Number(rangeValue);
 const step=years<=1?1:years<=3?3:years<=5?6:12;let previous='';
 for(let i=startIndex;i<usDates.length;i++){{
  const value=usDates[i],year=Number(value.slice(0,4)),month=Number(value.slice(5,7)),key=year+'-'+month;
  if(key===previous)continue;previous=key;
  if(step===12){{if(month!==1)continue}}else if((month-1)%step!==0)continue;
  vals.push(value);texts.push(step===12?String(year):value.slice(0,7));
 }}
 return{{vals,texts}};
}}
function usFiniteRange(values,minPadding){{
 const numeric=values.filter(v=>v!==null&&Number.isFinite(Number(v))).map(Number);
 if(!numeric.length)return null;
 const min=Math.min(...numeric),max=Math.max(...numeric),pad=Math.max((max-min)*0.075,minPadding);
 return[min-pad,max+pad];
}}
async function usApplyRange(rangeValue){{
 usActiveRange=String(rangeValue);
 const start=usStartIndexForYears(usActiveRange),ticks=usBuildTicks(start,usActiveRange),data=usCurrentWindowData();
 const ratioRange=usFiniteRange(usChartData.ratios.slice(start).concat(data.ma.slice(start)),0.015);
 const spyRange=usFiniteRange(usChartData.spyNorm.slice(start),5);
 const changes={{
  'xaxis.type':'category','xaxis.categoryorder':'array','xaxis.categoryarray':usDates,
  'xaxis.autorange':false,'xaxis.range':[start-0.5,usDates.length-0.5],
  'xaxis.tickmode':'array','xaxis.tickvals':ticks.vals,'xaxis.ticktext':ticks.texts,
  'yaxis.autorange':false,'yaxis2.autorange':false
 }};
 if(ratioRange)changes['yaxis.range']=ratioRange;
 if(spyRange)changes['yaxis2.range']=spyRange;
 await Plotly.relayout(usPlot,changes);Plotly.Plots.resize(usPlot);
 usButtons.forEach(b=>b.classList.toggle('active',b.dataset.usRange===usActiveRange));
}}
function usUpdateQuote(index){{
 usCurrentPointIndex=Math.max(0,Math.min(Number(index),usDates.length-1));
 const data=usCurrentWindowData();
 document.getElementById('us-q-date').textContent=usDates[usCurrentPointIndex];
 document.getElementById('us-q-ratio').textContent=usFmt(usChartData.ratios[usCurrentPointIndex],4);
 document.getElementById('us-q-ma').textContent=usFmt(data.ma[usCurrentPointIndex],4);
 usSetSlopeElement(document.getElementById('us-q-slope'),data.slope[usCurrentPointIndex]);
 document.getElementById('us-q-xlk').textContent=usFmt(usChartData.xlkNorm[usCurrentPointIndex],2);
 document.getElementById('us-q-xlf').textContent=usFmt(usChartData.xlfNorm[usCurrentPointIndex],2);
 document.getElementById('us-q-spy').textContent=usFmt(usChartData.spyNorm[usCurrentPointIndex],2);
 document.getElementById('us-q-state').textContent=data.state[usCurrentPointIndex]||'—';
 document.getElementById('us-q-signal').textContent=data.signal[usCurrentPointIndex]||'—';
}}
function usRenderSignalTable(){{
 const rows=usCurrentWindowData().signalRows,body=document.getElementById('us-signals-body');
 if(!rows.length){{body.innerHTML='<tr><td colspan="4">目前尚無足夠資料形成切換訊號。</td></tr>';return}}
 body.innerHTML=rows.map(r=>'<tr><td>'+r.date+'</td><td class="'+(r.signal==='Risk On'?'bull':'bear')+'">'+r.signal+
 '</td><td>'+usFmt(r.ratio,4)+'</td><td>'+usFmt(r.ma,4)+'</td></tr>').join('');
}}
function usUpdateWindowText(){{
 const data=usCurrentWindowData(),stateElement=document.getElementById('us-metric-state');
 document.getElementById('us-metric-title').textContent='MA'+usSelectedWindow+' 趨勢';
 stateElement.textContent=data.latestState||'—';stateElement.classList.remove('on','off','neutral');stateElement.classList.add(usStateClass(data.latestState));
 document.getElementById('us-metric-ma-label').textContent='MA'+usSelectedWindow;
 document.getElementById('us-metric-ma-value').textContent=usFmt(data.latestMa,4);
 document.getElementById('us-metric-slope-label').textContent='MA'+usSelectedWindow+' 斜率';
 usSetSlopeElement(document.getElementById('us-metric-slope-value'),data.latestSlope);
 document.getElementById('us-metric-last-on').textContent=data.lastOn||'—';
 document.getElementById('us-metric-last-off').textContent=data.lastOff||'—';
 document.getElementById('us-q-ma-label').textContent='MA'+usSelectedWindow;
 document.getElementById('us-q-slope-label').textContent='MA'+usSelectedWindow+' 斜率';
 document.getElementById('us-plot-subtitle').textContent='XLK／XLF 還原後總報酬資料｜MA'+usSelectedWindow+'｜'+(usChartData.bufferPct*100).toFixed(2)+'% 緩衝訊號';
 document.getElementById('us-signals-heading').textContent='近期 MA'+usSelectedWindow+' Risk On／Risk Off 切換';
 document.getElementById('us-signals-ma-heading').textContent='MA'+usSelectedWindow;
 document.getElementById('us-rule-ma-text').innerHTML='緩衝區設定為 <code>'+(usChartData.bufferPct*100).toFixed(2)+'%</code>。粉紅點代表突破 MA'+usSelectedWindow+
 ' 上方緩衝區並切換為 <strong>Risk On</strong>；綠點代表跌破下方緩衝區並切換為 <strong>Risk Off</strong>。';
 usRenderSignalTable();usUpdateQuote(usCurrentPointIndex);
}}
async function usSwitchWindow(value){{
 const requested=String(value);
 if(!usChartData.windows[requested]||requested===usSelectedWindow)return;
 usSelectedWindow=requested;usWindowSelect.value=usSelectedWindow;usWindowSelect.disabled=true;
 const data=usCurrentWindowData();
 try{{
  await Plotly.restyle(usPlot,{{y:[data.ma],name:'MA'+usSelectedWindow}},[1]);
  await Plotly.restyle(usPlot,{{x:[data.riskOnX],y:[data.riskOnY]}},[3]);
  await Plotly.restyle(usPlot,{{x:[data.riskOffX],y:[data.riskOffY]}},[4]);
  await Plotly.restyle(usPlot,{{customdata:[usBuildCustomData()],hovertemplate:usHoverTemplate()}},[5]);
  usUpdateWindowText();
  await usApplyRange(usActiveRange);
 }}catch(error){{console.error('切換美股判讀均線失敗：',error);await usApplyRange(usActiveRange)}}
 finally{{usWindowSelect.disabled=false;usWindowSelect.focus({{preventScroll:true}})}}
}}

const initial=usCurrentWindowData();
const traces=[
 {{type:'bar',x:usDates,y:usChartData.ratios,name:'美股電金比',marker:{{color:'#9b641f',line:{{color:'#d48a31',width:.35}}}},hoverinfo:'skip'}},
 {{type:'scatter',mode:'lines',x:usDates,y:initial.ma,name:'MA'+usSelectedWindow,line:{{color:'#f1f1f1',width:1.6}},hoverinfo:'skip'}},
 {{type:'scatter',mode:'lines',x:usDates,y:usChartData.spyNorm,name:'SPY 標準化',yaxis:'y2',visible:'legendonly',line:{{color:'#58a6ff',width:1.3}},hoverinfo:'skip'}},
 {{type:'scatter',mode:'markers',x:initial.riskOnX,y:initial.riskOnY,name:'Risk On',marker:{{color:'#ff2c55',size:10,line:{{color:'#ff8aa1',width:1}}}},hoverinfo:'skip'}},
 {{type:'scatter',mode:'markers',x:initial.riskOffX,y:initial.riskOffY,name:'Risk Off',marker:{{color:'#00df45',size:10,line:{{color:'#9affb5',width:1}}}},hoverinfo:'skip'}},
 {{type:'scatter',mode:'markers',x:usDates,y:usChartData.ratios,name:'查價',marker:{{size:18,color:'rgba(0,0,0,0)'}},showlegend:false,customdata:usBuildCustomData(),hovertemplate:usHoverTemplate()}}
];
const layout={{
 paper_bgcolor:'#050505',plot_bgcolor:'#050505',margin:{{l:52,r:65,t:20,b:55}},
 hovermode:'x unified',dragmode:'zoom',bargap:.12,
 legend:{{orientation:'h',x:0,y:1.08,font:{{color:'#ddd'}}}},font:{{color:'#ddd'}},
 xaxis:{{type:'category',categoryorder:'array',categoryarray:usDates,autorange:false,gridcolor:'#252525',zeroline:false,fixedrange:false}},
 yaxis:{{title:'美股電金比',autorange:false,side:'right',gridcolor:'#303030',zeroline:false,fixedrange:false}},
 yaxis2:{{title:'SPY 標準化',autorange:false,overlaying:'y',side:'left',showgrid:false,visible:false,fixedrange:false}},
 uirevision:'us-electric-finance-ratio-v3'
}};
Plotly.newPlot(usPlot,traces,layout,{{responsive:true,scrollZoom:true,displaylogo:false,modeBarButtonsToRemove:['lasso2d','select2d','toggleSpikelines'],doubleClick:'reset'}})
.then(async()=>{{await usApplyRange(usActiveRange);usUpdateWindowText();document.documentElement.dataset.usChartReady='true'}});
usButtons.forEach(b=>b.addEventListener('click',()=>usApplyRange(b.dataset.usRange)));
usWindowSelect.addEventListener('change',e=>usSwitchWindow(e.target.value));
usPlot.on('plotly_legendclick',event=>{{
 if(event.curveNumber!==2)return;
 const current=usPlot.data[2].visible,hidden=current==='legendonly'||current===false;
 Plotly.restyle(usPlot,{{visible:hidden?true:'legendonly'}},[2]);Plotly.relayout(usPlot,{{'yaxis2.visible':hidden}});return false;
}});
usPlot.on('plotly_hover',event=>{{const point=event.points.find(item=>item.curveNumber===5);if(point)usUpdateQuote(point.pointIndex)}});
}})();
</script>
{SCRIPT_END}
'''


def patch_html(
    frame: pd.DataFrame,
    windows: tuple[int, ...],
    default_window: int,
    buffer_pct: float,
    default_range_years: int,
) -> None:
    if not HTML_PATH.exists():
        raise FileNotFoundError(
            f"找不到 {HTML_PATH}。請先執行 update_chart.py。"
        )

    html_text = HTML_PATH.read_text(encoding="utf-8")
    html_text = remove_marker_block(
        html_text, SECTION_START, SECTION_END
    )
    html_text = remove_marker_block(
        html_text, SCRIPT_START, SCRIPT_END
    )
    if "</main>" not in html_text or "</body>" not in html_text:
        raise RuntimeError(
            "docs/index.html 缺少 </main> 或 </body>。"
        )

    html_text = html_text.replace(
        "</main>",
        build_section(frame, default_window, buffer_pct)
        + "\n</main>",
        1,
    )
    html_text = html_text.replace(
        "</body>",
        build_script(
            frame, windows, default_window,
            buffer_pct, default_range_years,
        )
        + "\n</body>",
        1,
    )
    HTML_PATH.write_text(html_text, encoding="utf-8")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = parse_args()
    (
        windows,
        default_window,
        buffer_pct,
        default_range_years,
    ) = load_display_config()

    if args.no_fetch:
        frame = read_existing(windows, buffer_pct)
    else:
        try:
            prices = download_adjusted_prices()
            frame = compute_ratio(prices, windows, buffer_pct)
        except Exception as exc:
            if US_DATA_CSV.exists():
                LOGGER.exception(
                    "美股下載失敗，沿用既有 CSV：%s", exc
                )
                frame = read_existing(windows, buffer_pct)
            else:
                raise

    frame = add_signal_columns(frame, windows, buffer_pct)
    save_data(frame, windows)
    patch_html(
        frame, windows, default_window,
        buffer_pct, default_range_years,
    )

    latest = frame.iloc[-1]
    LOGGER.info(
        "美股電金比完成：%s 至 %s，共 %d 筆；最新 %.4f。",
        pd.Timestamp(frame.iloc[0]["date"]).date(),
        pd.Timestamp(latest["date"]).date(),
        len(frame),
        float(latest["us_ratio"]),
    )


if __name__ == "__main__":
    main()
