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
DEFAULT_WINDOWS = (20, 120)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="抓取 XLK、XLF、SPY 還原後價格，建立美股電金比並插入既有 GitHub Pages。"
    )
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="不連線抓資料，改用既有 data/us_technology_finance_ratio.csv 重建網頁。",
    )
    return parser.parse_args()


def load_display_config() -> tuple[tuple[int, ...], int, float, int]:
    """沿用台股網頁的均線、緩衝區與預設期間設定。"""
    raw: dict[str, Any] = {}
    if CONFIG_PATH.exists():
        try:
            raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("config.json 讀取失敗，改用預設值：%s", exc)

    windows = tuple(int(v) for v in raw.get("moving_averages", list(DEFAULT_WINDOWS)))
    windows = tuple(v for v in windows if v > 1) or DEFAULT_WINDOWS
    display_window = 20 if 20 in windows else windows[0]
    buffer_pct = max(0.0, float(raw.get("buffer_pct", 0.0)))
    default_range_years = max(1, int(raw.get("default_chart_range_years", 1)))
    return windows, display_window, buffer_pct, default_range_years


def extract_adj_close(downloaded: pd.DataFrame, ticker: str) -> pd.Series:
    """兼容 yfinance 的兩種 MultiIndex 排列，取得指定代號的 Adj Close。"""
    if downloaded.empty:
        raise RuntimeError("yfinance 回傳空資料。")

    if isinstance(downloaded.columns, pd.MultiIndex):
        candidates = [
            ("Adj Close", ticker),
            (ticker, "Adj Close"),
        ]
        for candidate in candidates:
            if candidate in downloaded.columns:
                return pd.to_numeric(downloaded[candidate], errors="coerce").rename(ticker)

        close_candidates = [
            ("Close", ticker),
            (ticker, "Close"),
        ]
        for candidate in close_candidates:
            if candidate in downloaded.columns:
                LOGGER.warning("找不到 Adj Close，改用 yfinance 已還原的 Close：%s", ticker)
                return pd.to_numeric(downloaded[candidate], errors="coerce").rename(ticker)
    else:
        if "Adj Close" in downloaded.columns:
            return pd.to_numeric(downloaded["Adj Close"], errors="coerce").rename(ticker)
        if "Close" in downloaded.columns:
            LOGGER.warning("找不到 Adj Close，改用 yfinance 已還原的 Close：%s", ticker)
            return pd.to_numeric(downloaded["Close"], errors="coerce").rename(ticker)

    raise RuntimeError(
        f"找不到 {ticker} 的 Adj Close。實際欄位：{list(downloaded.columns)[:12]}"
    )


def download_adjusted_prices() -> pd.DataFrame:
    """完整重抓最長歷史；失敗時由呼叫端決定是否沿用舊 CSV。"""
    end_date = (datetime.now(TAIPEI).date() + timedelta(days=2)).isoformat()
    last_error: Exception | None = None

    for attempt in range(1, 4):
        try:
            LOGGER.info(
                "下載美股還原資料，第 %d/3 次：%s 至 %s",
                attempt,
                START_DATE,
                end_date,
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
            parts = [extract_adj_close(downloaded, ticker) for ticker in TICKERS]
            prices = pd.concat(parts, axis=1)
            prices.columns = list(TICKERS)
            prices.index = pd.to_datetime(prices.index, errors="coerce")

            if getattr(prices.index, "tz", None) is not None:
                prices.index = prices.index.tz_localize(None)

            prices = prices[~prices.index.isna()]
            prices = prices.sort_index()
            prices = prices[~prices.index.duplicated(keep="last")]
            prices = prices.dropna(subset=["XLK", "XLF"])
            prices = prices[(prices["XLK"] > 0) & (prices["XLF"] > 0)]

            # SPY 只作大盤對照；單日缺值以前值補齊，不改變 XLK/XLF 共同起日。
            prices["SPY"] = pd.to_numeric(prices["SPY"], errors="coerce").ffill()
            prices = prices.dropna(subset=["SPY"])
            prices = prices[prices["SPY"] > 0]

            if len(prices) < 5000:
                raise RuntimeError(f"有效共同交易日只有 {len(prices)} 筆，低於合理門檻。")

            LOGGER.info(
                "美股資料完成：%s 至 %s，共 %d 個共同交易日。",
                prices.index.min().date(),
                prices.index.max().date(),
                len(prices),
            )
            return prices
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            LOGGER.warning("第 %d 次下載失敗：%s", attempt, exc)
            if attempt < 3:
                time.sleep(8 * attempt)

    raise RuntimeError(f"美股資料下載三次皆失敗：{last_error}")


def add_signal_columns(
    frame: pd.DataFrame,
    windows: tuple[int, ...],
    buffer_pct: float,
) -> pd.DataFrame:
    """使用與台股版相同的均線狀態延續與 Risk On／Risk Off 切換規則。"""
    result = frame.copy().sort_values("date").reset_index(drop=True)

    for window in windows:
        ma_col = f"ma{window}"
        slope_col = f"slope{window}"
        state_col = f"state{window}"
        risk_on_col = f"bull_turn{window}"
        risk_off_col = f"bear_turn{window}"
        cross_on_col = f"cross_on{window}"
        cross_off_col = f"cross_off{window}"

        result[ma_col] = result["us_ratio"].rolling(window, min_periods=window).mean()
        result[slope_col] = result[ma_col].diff()

        # 保留舊版欄位名稱，避免既有 CSV 或其他程式讀取失敗。
        result[f"ma{window}_slope"] = result[slope_col]

        states: list[str] = []
        risk_on_signals: list[bool] = []
        risk_off_signals: list[bool] = []
        current_state = "Neutral"

        for ratio_value, ma_value in zip(result["us_ratio"], result[ma_col]):
            risk_on = False
            risk_off = False
            if pd.notna(ma_value):
                upper = float(ma_value) * (1 + buffer_pct)
                lower = float(ma_value) * (1 - buffer_pct)
                if float(ratio_value) > upper and current_state != "Risk On":
                    current_state = "Risk On"
                    risk_on = True
                elif float(ratio_value) < lower and current_state != "Risk Off":
                    current_state = "Risk Off"
                    risk_off = True

            states.append(current_state)
            risk_on_signals.append(risk_on)
            risk_off_signals.append(risk_off)

        result[state_col] = states
        result[risk_on_col] = risk_on_signals
        result[risk_off_col] = risk_off_signals
        result[cross_on_col] = risk_on_signals
        result[cross_off_col] = risk_off_signals

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
        result["xlk_adj_close"] / float(first["xlk_adj_close"]) * 100.0
    )
    result["xlf_normalized"] = (
        result["xlf_adj_close"] / float(first["xlf_adj_close"]) * 100.0
    )
    result["spy_normalized"] = (
        result["spy_adj_close"] / float(first["spy_adj_close"]) * 100.0
    )
    result["us_ratio"] = result["xlk_normalized"] / result["xlf_normalized"]

    if not math.isclose(float(result["us_ratio"].iloc[0]), 1.0, rel_tol=0, abs_tol=1e-12):
        raise RuntimeError("標準化後起始日美股電金比不等於 1。")

    return add_signal_columns(result.reset_index(), windows, buffer_pct)


def read_existing(
    windows: tuple[int, ...],
    buffer_pct: float,
) -> pd.DataFrame:
    if not US_DATA_CSV.exists():
        raise FileNotFoundError(f"找不到既有美股資料：{US_DATA_CSV}")

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
        raise RuntimeError(f"既有美股 CSV 缺少欄位：{sorted(missing)}")

    numeric_columns = [
        "xlk_adj_close",
        "xlf_adj_close",
        "spy_adj_close",
        "xlk_normalized",
        "xlf_normalized",
        "spy_normalized",
        "us_ratio",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna(subset=list(required))
    frame = frame.sort_values("date").drop_duplicates("date", keep="last")
    return add_signal_columns(frame.reset_index(drop=True), windows, buffer_pct)


def last_signal_date(frame: pd.DataFrame, column: str) -> str:
    matched = frame.loc[frame[column].fillna(False), "date"]
    if matched.empty:
        return "—"
    return pd.Timestamp(matched.iloc[-1]).strftime("%Y-%m-%d")


def format_value(value: Any, digits: int = 4, signed: bool = False) -> str:
    if value is None or pd.isna(value):
        return "—"
    number = float(value)
    return f"{number:+.{digits}f}" if signed else f"{number:.{digits}f}"


def save_data(frame: pd.DataFrame, display_window: int) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    frame.to_csv(US_DATA_CSV, index=False, date_format="%Y-%m-%d")
    frame.to_csv(
        US_WEB_CSV,
        index=False,
        date_format="%Y-%m-%d",
        encoding="utf-8-sig",
    )

    ma_col = f"ma{display_window}"
    slope_col = f"slope{display_window}"
    state_col = f"state{display_window}"
    risk_on_col = f"bull_turn{display_window}"
    risk_off_col = f"bear_turn{display_window}"

    signals = frame.loc[
        frame[risk_on_col].fillna(False) | frame[risk_off_col].fillna(False),
        ["date", "us_ratio", ma_col, slope_col, state_col, risk_on_col, risk_off_col],
    ].copy()
    signals["signal"] = signals.apply(
        lambda row: "Risk On" if bool(row[risk_on_col]) else "Risk Off",
        axis=1,
    )
    signals = signals.rename(
        columns={
            "us_ratio": "ratio",
            ma_col: f"ma{display_window}",
            slope_col: "slope",
            state_col: "state",
        }
    )
    signals[
        ["date", "signal", "ratio", f"ma{display_window}", "slope", "state"]
    ].to_csv(
        US_SIGNALS_CSV,
        index=False,
        date_format="%Y-%m-%d",
        encoding="utf-8-sig",
    )

    LOGGER.info("已儲存：%s", US_DATA_CSV)
    LOGGER.info("已儲存：%s", US_WEB_CSV)
    LOGGER.info("已儲存：%s", US_SIGNALS_CSV)


def as_json_number(value: Any, digits: int | None = None) -> float | None:
    if value is None or pd.isna(value):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return round(number, digits) if digits is not None else number


def remove_marker_block(text: str, start: str, end: str) -> str:
    pattern = re.compile(
        re.escape(start) + r".*?" + re.escape(end) + r"\s*",
        flags=re.DOTALL,
    )
    return pattern.sub("", text)


def build_section(
    frame: pd.DataFrame,
    window: int,
    buffer_pct: float,
) -> str:
    latest = frame.iloc[-1]
    first_date = pd.Timestamp(frame.iloc[0]["date"]).strftime("%Y-%m-%d")
    latest_date = pd.Timestamp(latest["date"]).strftime("%Y-%m-%d")
    generated_at = datetime.now(TAIPEI).strftime("%Y-%m-%d %H:%M:%S Asia/Taipei")

    ma_col = f"ma{window}"
    slope_col = f"slope{window}"
    state_col = f"state{window}"
    risk_on_col = f"bull_turn{window}"
    risk_off_col = f"bear_turn{window}"

    state = str(latest.get(state_col, "—"))
    state_class = "on" if state == "Risk On" else "off" if state == "Risk Off" else "neutral"
    latest_ma = latest.get(ma_col)
    latest_slope = latest.get(slope_col)
    last_on = last_signal_date(frame, risk_on_col)
    last_off = last_signal_date(frame, risk_off_col)
    buffer_text = f"{buffer_pct * 100:.2f}%"

    signal_rows: list[str] = []
    subset = frame.loc[
        frame[risk_on_col].fillna(False) | frame[risk_off_col].fillna(False),
        ["date", "us_ratio", ma_col, risk_on_col, risk_off_col],
    ].tail(10)
    for _, row in subset.iloc[::-1].iterrows():
        kind = "Risk On" if bool(row[risk_on_col]) else "Risk Off"
        css_class = "bull" if kind == "Risk On" else "bear"
        signal_rows.append(
            "<tr>"
            f"<td>{pd.Timestamp(row['date']).strftime('%Y-%m-%d')}</td>"
            f"<td class='{css_class}'>{kind}</td>"
            f"<td>{float(row['us_ratio']):.4f}</td>"
            f"<td>{float(row[ma_col]):.4f}</td>"
            "</tr>"
        )

    return f"""
{SECTION_START}
  <style>
    .us-range-button {{ appearance:none; border:1px solid #444; background:#191919; color:#ddd; border-radius:8px; padding:6px 11px; cursor:pointer; font:inherit; }}
    .us-range-button:hover {{ border-color:#777; }}
    .us-range-button.active {{ background:#75501d; border-color:#c58a35; color:#fff; font-weight:700; }}
    #us-interactive-chart {{ width:100%; height:clamp(580px,65vh,760px); min-height:580px; }}
    @media (max-width:900px) {{ #us-interactive-chart {{ height:600px; }} }}
  </style>

  <section class="section note" style="margin-top:48px;">
    <h2 style="margin-top:0; color:#f5f5f5;">美股電金比風險偏好指標</h2>
    <div class="sub">資料範圍：{first_date}～{latest_date}｜共 {len(frame):,} 個共同交易日｜美股資料更新：{generated_at}</div>
    <p>
      使用含股息與拆分調整的還原後收盤價；共同起始日將 XLK 與 XLF 都標準化為 100，
      美股電金比＝XLK 標準化指數 ÷ XLF 標準化指數，因此共同起始日固定為 1。
    </p>
  </section>

  <div class="summary" style="margin-top:14px;">
    <section class="ratio-card">
      <div>XLK 科技標準化指數 ÷ XLF 金融標準化指數</div>
      <div class="value">{float(latest["us_ratio"]):.4f}</div>
      <dl>
        <div><dt>XLK 標準化指數</dt><dd>{float(latest["xlk_normalized"]):,.2f}</dd></div>
        <div><dt>XLF 標準化指數</dt><dd>{float(latest["xlf_normalized"]):,.2f}</dd></div>
      </dl>
    </section>
    <section class="metric-card">
      <div class="metric-title">{window} 日趨勢</div>
      <div class="state {state_class}">{state}</div>
      <dl>
        <div><dt>MA{window}</dt><dd>{format_value(latest_ma)}</dd></div>
        <div><dt>斜率</dt><dd>{format_value(latest_slope, 5, True)}</dd></div>
        <div><dt>最近 Risk On</dt><dd>{last_on}</dd></div>
        <div><dt>最近 Risk Off</dt><dd>{last_off}</dd></div>
      </dl>
    </section>
  </div>

  <section class="chart-shell" id="us-ratio-section">
    <div class="quote-panel">
      <div class="quote-item"><div class="quote-label">查價日期</div><div class="quote-value" id="us-q-date">{latest_date}</div></div>
      <div class="quote-item"><div class="quote-label">美股電金比</div><div class="quote-value" id="us-q-ratio">{float(latest["us_ratio"]):.4f}</div></div>
      <div class="quote-item"><div class="quote-label">MA{window}</div><div class="quote-value" id="us-q-ma">{format_value(latest_ma)}</div></div>
      <div class="quote-item"><div class="quote-label">XLK 標準化</div><div class="quote-value" id="us-q-xlk">{float(latest["xlk_normalized"]):,.2f}</div></div>
      <div class="quote-item"><div class="quote-label">XLF 標準化</div><div class="quote-value" id="us-q-xlf">{float(latest["xlf_normalized"]):,.2f}</div></div>
      <div class="quote-item"><div class="quote-label">SPY 標準化</div><div class="quote-value" id="us-q-spy">{float(latest["spy_normalized"]):,.2f}</div></div>
      <div class="quote-item"><div class="quote-label">狀態</div><div class="quote-value" id="us-q-state">{state}</div></div>
      <div class="quote-item"><div class="quote-label">訊號</div><div class="quote-value" id="us-q-signal">—</div></div>
    </div>

    <div class="plot-heading">
      <div>
        <div class="plot-title">美國科技類股 ÷ 金融類股</div>
        <div class="plot-subtitle">XLK／XLF 還原後總報酬資料｜MA{window}｜{buffer_text} 緩衝訊號</div>
      </div>
    </div>

    <div class="range-controls" aria-label="美股圖表顯示期間">
      <button class="us-range-button" type="button" data-us-range="1">1年</button>
      <button class="us-range-button" type="button" data-us-range="3">3年</button>
      <button class="us-range-button" type="button" data-us-range="5">5年</button>
      <button class="us-range-button" type="button" data-us-range="10">10年</button>
      <button class="us-range-button" type="button" data-us-range="20">20年</button>
      <button class="us-range-button" type="button" data-us-range="all">全部</button>
    </div>

    <div id="us-interactive-chart" aria-label="美股電金比互動查價圖"></div>
    <div class="chart-help">
      移動滑鼠可查價；拖曳框選可放大；滑鼠滾輪可縮放。X 軸只排列實際交易日，週六、週日與休市日不留空白。
      可切換 1／3／5／10／20 年或全部資料；「SPY 標準化」預設隱藏，點圖例可開關。
    </div>
  </section>

  <div class="links">
    <a href="us_data.csv">下載美股完整每日資料 CSV</a>
    <a href="us_signals.csv">下載美股反轉訊號 CSV</a>
  </div>

  <section class="section note">
    <h2>美股版判讀規則</h2>
    <p><strong>美股電金比＝XLK 科技總報酬標準化指數 ÷ XLF 金融總報酬標準化指數。</strong>比值上升代表科技相對金融強，通常視為風險偏好提高；比值下降代表金融相對科技強，通常視為風險偏好降低或產業輪動轉向金融。</p>
    <p>緩衝區設定為 <code>{buffer_text}</code>。粉紅點代表突破 MA{window} 上方緩衝區並切換為 <strong>Risk On</strong>；綠點代表跌破下方緩衝區並切換為 <strong>Risk Off</strong>；緩衝區內延續前一狀態。</p>
    <p>比值大於或小於 1 只表示相對共同起始日的累積表現差異；目前狀態與切換訊號以 MA{window} 為準，不能只用是否大於 1 判斷。</p>
  </section>

  <section class="section">
    <h2>近期 MA{window} Risk On／Risk Off 切換</h2>
    <table>
      <thead><tr><th>日期</th><th>訊號</th><th>美股電金比</th><th>MA{window}</th></tr></thead>
      <tbody>{''.join(signal_rows) or '<tr><td colspan="4">目前尚無足夠資料形成切換訊號。</td></tr>'}</tbody>
    </table>
  </section>
{SECTION_END}
"""


def build_script(
    frame: pd.DataFrame,
    window: int,
    default_range_years: int,
) -> str:
    shown = frame.copy().reset_index(drop=True)
    shown["date_label"] = shown["date"].map(
        lambda value: pd.Timestamp(value).strftime("%Y-%m-%d")
    )

    ma_col = f"ma{window}"
    slope_col = f"slope{window}"
    state_col = f"state{window}"
    risk_on_col = f"bull_turn{window}"
    risk_off_col = f"bear_turn{window}"

    shown["signal"] = "—"
    shown.loc[shown[risk_on_col].fillna(False), "signal"] = "Risk On"
    shown.loc[shown[risk_off_col].fillna(False), "signal"] = "Risk Off"

    payload = {
        "dates": shown["date_label"].tolist(),
        "ratios": [as_json_number(v, 8) for v in shown["us_ratio"]],
        "ma": [as_json_number(v, 8) for v in shown[ma_col]],
        "spyNorm": [as_json_number(v, 6) for v in shown["spy_normalized"]],
        "riskOnX": shown.loc[shown[risk_on_col].fillna(False), "date_label"].tolist(),
        "riskOnY": [
            as_json_number(v, 8)
            for v in shown.loc[shown[risk_on_col].fillna(False), "us_ratio"]
        ],
        "riskOffX": shown.loc[shown[risk_off_col].fillna(False), "date_label"].tolist(),
        "riskOffY": [
            as_json_number(v, 8)
            for v in shown.loc[shown[risk_off_col].fillna(False), "us_ratio"]
        ],
        "customData": [
            [
                pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
                as_json_number(row["xlk_normalized"], 6),
                as_json_number(row["xlf_normalized"], 6),
                as_json_number(row[ma_col], 8),
                str(row.get(state_col, "—")),
                str(row.get("signal", "—")),
                as_json_number(row[slope_col], 8),
                as_json_number(row["spy_normalized"], 6),
                as_json_number(row["xlk_adj_close"], 6),
                as_json_number(row["xlf_adj_close"], 6),
            ]
            for _, row in shown.iterrows()
        ],
        "window": window,
        "defaultRangeYears": default_range_years,
    }
    payload_json = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )

    return f"""
{SCRIPT_START}
<script>
(() => {{
  const usChartData = {payload_json};
  const usDates = usChartData.dates;
  const usPlot = document.getElementById('us-interactive-chart');
  const usButtons = Array.from(document.querySelectorAll('.us-range-button'));
  if (!usPlot || !window.Plotly) return;

  const usRatioTrace = {{
    type:'bar', x:usDates, y:usChartData.ratios, name:'美股電金比',
    marker:{{color:'#9b641f',line:{{color:'#d48a31',width:0.7}}}}, hoverinfo:'skip'
  }};
  const usMaTrace = {{
    type:'scatter', mode:'lines', x:usDates, y:usChartData.ma,
    name:'MA'+usChartData.window, line:{{color:'#f3f3f3',width:2}}, hoverinfo:'skip'
  }};
  const usSpyTrace = {{
    type:'scatter', mode:'lines', x:usDates, y:usChartData.spyNorm,
    name:'SPY 標準化', yaxis:'y2', visible:'legendonly',
    line:{{color:'#36b0ff',width:2.1}}, hoverinfo:'skip'
  }};
  const usRiskOnTrace = {{
    type:'scatter', mode:'markers', x:usChartData.riskOnX, y:usChartData.riskOnY,
    name:'Risk On', marker:{{size:15,color:'#ff2cba',line:{{color:'#ff8ee3',width:1}}}}, hoverinfo:'skip'
  }};
  const usRiskOffTrace = {{
    type:'scatter', mode:'markers', x:usChartData.riskOffX, y:usChartData.riskOffY,
    name:'Risk Off', marker:{{size:15,color:'#00df45',line:{{color:'#9affb5',width:1}}}}, hoverinfo:'skip'
  }};
  const usHoverTrace = {{
    type:'scatter', mode:'lines+markers', x:usDates, y:usChartData.ratios,
    customdata:usChartData.customData, line:{{width:0}}, marker:{{size:18,opacity:0.002}},
    showlegend:false,
    hovertemplate:'<b>%{{customdata[0]}}</b><br>'+ 
      '美股電金比：%{{y:.4f}}<br>'+ 
      'MA'+usChartData.window+'：%{{customdata[3]:.4f}}<br>'+ 
      'XLK 標準化：%{{customdata[1]:,.2f}}<br>'+ 
      'XLF 標準化：%{{customdata[2]:,.2f}}<br>'+ 
      'SPY 標準化：%{{customdata[7]:,.2f}}<br>'+ 
      '狀態：%{{customdata[4]}}<br>'+ 
      '訊號：%{{customdata[5]}}<extra></extra>'
  }};

  const usLayout = {{
    paper_bgcolor:'#050505', plot_bgcolor:'#050505',
    margin:{{l:48,r:72,t:74,b:58}}, barmode:'overlay', bargap:0.18,
    hovermode:'closest', dragmode:'zoom', showlegend:true,
    legend:{{orientation:'h',x:0.01,xanchor:'left',y:1.10,yanchor:'top',font:{{color:'#ddd',size:14}},bgcolor:'rgba(0,0,0,0)',traceorder:'normal'}},
    xaxis:{{
      type:'category', categoryorder:'array', categoryarray:usDates,
      tickfont:{{color:'#ccc',size:11}}, showgrid:true, gridcolor:'#292929', gridwidth:1,
      showline:true, linecolor:'#555', fixedrange:false,
      showspikes:true, spikemode:'across', spikesnap:'cursor', spikecolor:'#f4f4f4', spikethickness:1
    }},
    yaxis:{{
      side:'right', tickformat:'.2f', tickfont:{{color:'#ccc',size:11}},
      showgrid:true, gridcolor:'#292929', zeroline:false, showline:true, linecolor:'#555', fixedrange:false,
      showspikes:true, spikemode:'across', spikesnap:'cursor', spikecolor:'#777', spikethickness:1, spikedash:'dot'
    }},
    yaxis2:{{
      visible:false, overlaying:'y', side:'left', tickfont:{{color:'#59beff',size:11}},
      title:{{text:'SPY 標準化總報酬指數',font:{{color:'#59beff',size:12}}}},
      showgrid:false, zeroline:false, showline:true, linecolor:'#2b6f99', fixedrange:false
    }},
    shapes:[{{
      type:'line',xref:'paper',x0:0,x1:1,yref:'y',y0:1,y1:1,
      line:{{color:'#777',width:1,dash:'dash'}}
    }}]
  }};

  function usStartIndexForYears(rangeValue) {{
    if (rangeValue === 'all') return 0;
    const years = Number(rangeValue);
    const latest = new Date(usDates[usDates.length-1]+'T00:00:00');
    const cutoff = new Date(latest);
    cutoff.setFullYear(cutoff.getFullYear()-years);
    const cutoffText = cutoff.toISOString().slice(0,10);
    const index = usDates.findIndex(value => value >= cutoffText);
    return index < 0 ? 0 : index;
  }}

  function usBuildTicks(startIndex, rangeValue) {{
    const years = rangeValue === 'all' ? 99 : Number(rangeValue);
    const monthStep = years >= 10 ? 12 : years >= 5 ? 6 : years >= 3 ? 3 : 1;
    const vals = [];
    const texts = [];
    let previousKey = '';
    for (let i=startIndex; i<usDates.length; i++) {{
      const value = usDates[i];
      const year = Number(value.slice(0,4));
      const month = Number(value.slice(5,7));
      const bucketMonth = Math.floor((month-1)/monthStep)*monthStep+1;
      const key = year+'-'+String(bucketMonth).padStart(2,'0');
      if (key !== previousKey && (month-1)%monthStep === 0) {{
        previousKey = key;
        vals.push(value);
        texts.push(monthStep === 12 ? String(year) : value.slice(0,7));
      }}
    }}
    return {{vals,texts}};
  }}

  function usFiniteRange(values, startIndex, minimumPadding) {{
    const numeric = values.slice(startIndex)
      .filter(value => value !== null && Number.isFinite(Number(value)))
      .map(Number);
    if (!numeric.length) return null;
    const minValue = Math.min(...numeric);
    const maxValue = Math.max(...numeric);
    const padding = Math.max((maxValue-minValue)*0.075,minimumPadding);
    return [minValue-padding,maxValue+padding];
  }}

  function usApplyRange(rangeValue) {{
    const startIndex = usStartIndexForYears(rangeValue);
    const ticks = usBuildTicks(startIndex,rangeValue);
    const ratioRange = usFiniteRange(
      usChartData.ratios.slice(startIndex).concat(usChartData.ma.slice(startIndex)),
      0,
      0.015
    );
    const spyRange = usFiniteRange(usChartData.spyNorm,startIndex,5);
    const changes = {{
      'xaxis.range':[startIndex-0.5,usDates.length-0.5],
      'xaxis.tickmode':'array',
      'xaxis.tickvals':ticks.vals,
      'xaxis.ticktext':ticks.texts
    }};
    if (ratioRange) changes['yaxis.range'] = ratioRange;
    if (spyRange) changes['yaxis2.range'] = spyRange;
    Plotly.relayout(usPlot,changes);
    usButtons.forEach(button => button.classList.toggle('active',button.dataset.usRange === String(rangeValue)));
  }}

  Plotly.newPlot(
    usPlot,
    [usRatioTrace,usMaTrace,usSpyTrace,usRiskOnTrace,usRiskOffTrace,usHoverTrace],
    usLayout,
    {{
      responsive:true,scrollZoom:true,displaylogo:false,
      modeBarButtonsToRemove:['lasso2d','select2d','toggleSpikelines'],doubleClick:'reset'
    }}
  ).then(() => {{
    usApplyRange(String(usChartData.defaultRangeYears));
    document.documentElement.dataset.usChartReady = 'true';
  }});

  usButtons.forEach(button => button.addEventListener('click',() => usApplyRange(button.dataset.usRange)));

  const usSpyTraceIndex = 2;
  usPlot.on('plotly_legendclick', event => {{
    if (event.curveNumber !== usSpyTraceIndex) return;
    const current = usPlot.data[usSpyTraceIndex].visible;
    const isHidden = current === 'legendonly' || current === false;
    Plotly.restyle(usPlot,{{visible:isHidden ? true : 'legendonly'}},[usSpyTraceIndex]);
    Plotly.relayout(usPlot,{{'yaxis2.visible':isHidden}});
    return false;
  }});

  const usFmt = (value,digits=2) =>
    (value === null || value === undefined || Number.isNaN(Number(value)))
      ? '—'
      : Number(value).toLocaleString('zh-TW',{{minimumFractionDigits:digits,maximumFractionDigits:digits}});

  usPlot.on('plotly_hover', event => {{
    const point = event.points.find(item => item.data === usHoverTrace)
      || event.points[event.points.length-1];
    if (!point || !point.customdata) return;
    document.getElementById('us-q-date').textContent = point.customdata[0];
    document.getElementById('us-q-ratio').textContent = usFmt(point.y,4);
    document.getElementById('us-q-ma').textContent = usFmt(point.customdata[3],4);
    document.getElementById('us-q-xlk').textContent = usFmt(point.customdata[1],2);
    document.getElementById('us-q-xlf').textContent = usFmt(point.customdata[2],2);
    document.getElementById('us-q-spy').textContent = usFmt(point.customdata[7],2);
    document.getElementById('us-q-state').textContent = point.customdata[4] || '—';
    document.getElementById('us-q-signal').textContent = point.customdata[5] || '—';
  }});
}})();
</script>
{SCRIPT_END}
"""


def patch_html(
    frame: pd.DataFrame,
    window: int,
    buffer_pct: float,
    default_range_years: int,
) -> None:
    if not HTML_PATH.exists():
        raise FileNotFoundError(
            f"找不到 {HTML_PATH}。請先執行 python src/update_chart.py。"
        )

    html_text = HTML_PATH.read_text(encoding="utf-8")
    html_text = remove_marker_block(html_text, SECTION_START, SECTION_END)
    html_text = remove_marker_block(html_text, SCRIPT_START, SCRIPT_END)

    if "</main>" not in html_text or "</body>" not in html_text:
        raise RuntimeError("docs/index.html 缺少 </main> 或 </body>，無法安全插入美股圖。")

    html_text = html_text.replace(
        "</main>",
        build_section(frame, window, buffer_pct) + "\n</main>",
        1,
    )
    html_text = html_text.replace(
        "</body>",
        build_script(frame, window, default_range_years) + "\n</body>",
        1,
    )
    HTML_PATH.write_text(html_text, encoding="utf-8")
    LOGGER.info("已把與台股版同格式的美股電金比插入網頁下方：%s", HTML_PATH)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = parse_args()
    windows, display_window, buffer_pct, default_range_years = load_display_config()

    if args.no_fetch:
        frame = read_existing(windows, buffer_pct)
    else:
        try:
            prices = download_adjusted_prices()
            frame = compute_ratio(prices, windows, buffer_pct)
        except Exception as exc:  # noqa: BLE001
            if US_DATA_CSV.exists():
                LOGGER.exception("美股下載失敗，沿用既有 CSV：%s", exc)
                frame = read_existing(windows, buffer_pct)
            else:
                raise

    # 每次都重算狀態、同步 CSV 與網頁，避免台股程式重建首頁後美股區塊消失。
    frame = add_signal_columns(frame, windows, buffer_pct)
    save_data(frame, display_window)
    patch_html(frame, display_window, buffer_pct, default_range_years)

    state_col = f"state{display_window}"
    LOGGER.info(
        "美股電金比完成：%s 至 %s，共 %d 筆；最新 %.4f；MA%d 狀態 %s。",
        pd.Timestamp(frame.iloc[0]["date"]).date(),
        pd.Timestamp(frame.iloc[-1]["date"]).date(),
        len(frame),
        float(frame.iloc[-1]["us_ratio"]),
        display_window,
        frame.iloc[-1].get(state_col, "—"),
    )


if __name__ == "__main__":
    main()
