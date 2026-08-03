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
DATA_DIR = ROOT / "data"
DOCS_DIR = ROOT / "docs"
HTML_PATH = DOCS_DIR / "index.html"
US_DATA_CSV = DATA_DIR / "us_technology_finance_ratio.csv"
US_WEB_CSV = DOCS_DIR / "us_data.csv"

TAIPEI = ZoneInfo("Asia/Taipei")
LOGGER = logging.getLogger("us_technology_finance_ratio")

SECTION_START = "<!-- US_RATIO_SECTION_START -->"
SECTION_END = "<!-- US_RATIO_SECTION_END -->"
SCRIPT_START = "<!-- US_RATIO_SCRIPT_START -->"
SCRIPT_END = "<!-- US_RATIO_SCRIPT_END -->"

TICKERS = ("XLK", "XLF", "SPY")
START_DATE = "1998-01-01"


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
                series = downloaded[candidate]
                return pd.to_numeric(series, errors="coerce").rename(ticker)

        # 某些版本／參數組合可能只有已自動還原的 Close。
        close_candidates = [
            ("Close", ticker),
            (ticker, "Close"),
        ]
        for candidate in close_candidates:
            if candidate in downloaded.columns:
                LOGGER.warning("找不到 Adj Close，改用 yfinance 已還原的 Close：%s", ticker)
                series = downloaded[candidate]
                return pd.to_numeric(series, errors="coerce").rename(ticker)
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

            # SPY 僅作大盤對照；若單日缺值，以前值補齊，不影響 XLK/XLF 共同起日。
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


def compute_ratio(prices: pd.DataFrame) -> pd.DataFrame:
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
    result["ma20"] = result["us_ratio"].rolling(20, min_periods=20).mean()
    result["ma120"] = result["us_ratio"].rolling(120, min_periods=120).mean()
    result["ma20_slope"] = result["ma20"].diff()
    result["ma120_slope"] = result["ma120"].diff()

    if not math.isclose(float(result["us_ratio"].iloc[0]), 1.0, rel_tol=0, abs_tol=1e-12):
        raise RuntimeError("標準化後起始日美股電金比不等於 1。")

    return result.reset_index()


def read_existing() -> pd.DataFrame:
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
        "ma20",
        "ma120",
        "ma20_slope",
        "ma120_slope",
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"既有美股 CSV 缺少欄位：{sorted(missing)}")
    frame = frame.sort_values("date").drop_duplicates("date", keep="last")
    return frame.reset_index(drop=True)


def save_data(frame: pd.DataFrame) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(US_DATA_CSV, index=False, date_format="%Y-%m-%d")
    frame.to_csv(
        US_WEB_CSV,
        index=False,
        date_format="%Y-%m-%d",
        encoding="utf-8-sig",
    )
    LOGGER.info("已儲存：%s", US_DATA_CSV)
    LOGGER.info("已儲存：%s", US_WEB_CSV)


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


def build_section(frame: pd.DataFrame) -> str:
    latest = frame.iloc[-1]
    first_date = pd.Timestamp(frame.iloc[0]["date"]).strftime("%Y-%m-%d")
    latest_date = pd.Timestamp(latest["date"]).strftime("%Y-%m-%d")
    generated_at = datetime.now(TAIPEI).strftime("%Y-%m-%d %H:%M:%S Asia/Taipei")

    ma120 = latest["ma120"]
    if pd.isna(ma120):
        trend_text = "資料累積中"
        trend_class = "neutral"
    elif float(latest["us_ratio"]) >= float(ma120):
        trend_text = "科技相對強"
        trend_class = "on"
    else:
        trend_text = "金融相對強"
        trend_class = "off"

    return f"""
{SECTION_START}
  <section class="section note" style="margin-top:48px;">
    <h2 style="margin-top:0;">美股電金比｜XLK ÷ XLF</h2>
    <p>
      使用含股息與拆分調整的還原後收盤價。共同起始日將 XLK 與 XLF 都標準化為 100，
      再計算「XLK 標準化指數 ÷ XLF 標準化指數」，因此起始日比值固定為 1。
    </p>
    <p>資料範圍：{first_date}～{latest_date}｜共 {len(frame):,} 個共同交易日｜美股資料更新：{generated_at}</p>
  </section>

  <div class="summary" style="margin-top:14px;">
    <section class="ratio-card">
      <div>科技標準化指數 ÷ 金融標準化指數</div>
      <div class="value">{float(latest["us_ratio"]):.4f}</div>
      <dl>
        <div><dt>XLK 標準化指數</dt><dd>{float(latest["xlk_normalized"]):,.2f}</dd></div>
        <div><dt>XLF 標準化指數</dt><dd>{float(latest["xlf_normalized"]):,.2f}</dd></div>
      </dl>
    </section>
    <section class="metric-card">
      <div class="metric-title">120 日相對趨勢</div>
      <div class="state {trend_class}">{trend_text}</div>
      <dl>
        <div><dt>MA20</dt><dd>{float(latest["ma20"]):.4f}</dd></div>
        <div><dt>MA120</dt><dd>{float(latest["ma120"]):.4f}</dd></div>
        <div><dt>MA20 斜率</dt><dd>{float(latest["ma20_slope"]):+.5f}</dd></div>
        <div><dt>MA120 斜率</dt><dd>{float(latest["ma120_slope"]):+.5f}</dd></div>
      </dl>
    </section>
  </div>

  <section class="chart-shell" id="us-ratio-section">
    <div class="quote-panel">
      <div class="quote-item"><div class="quote-label">查價日期</div><div class="quote-value" id="us-q-date">{latest_date}</div></div>
      <div class="quote-item"><div class="quote-label">美股電金比</div><div class="quote-value" id="us-q-ratio">{float(latest["us_ratio"]):.4f}</div></div>
      <div class="quote-item"><div class="quote-label">MA20</div><div class="quote-value" id="us-q-ma20">{float(latest["ma20"]):.4f}</div></div>
      <div class="quote-item"><div class="quote-label">MA120</div><div class="quote-value" id="us-q-ma120">{float(latest["ma120"]):.4f}</div></div>
      <div class="quote-item"><div class="quote-label">XLK 標準化</div><div class="quote-value" id="us-q-xlk-norm">{float(latest["xlk_normalized"]):,.2f}</div></div>
      <div class="quote-item"><div class="quote-label">XLF 標準化</div><div class="quote-value" id="us-q-xlf-norm">{float(latest["xlf_normalized"]):,.2f}</div></div>
      <div class="quote-item"><div class="quote-label">XLK 還原價</div><div class="quote-value" id="us-q-xlk">{float(latest["xlk_adj_close"]):,.2f}</div></div>
      <div class="quote-item"><div class="quote-label">XLF 還原價</div><div class="quote-value" id="us-q-xlf">{float(latest["xlf_adj_close"]):,.2f}</div></div>
    </div>

    <div class="plot-heading">
      <div>
        <div class="plot-title">美國科技類股相對金融類股</div>
        <div class="plot-subtitle">XLK／XLF 還原後總報酬資料｜共同起始日＝1</div>
      </div>
    </div>

    <div class="range-controls" aria-label="美股圖表顯示期間">
      <button class="us-range-button" type="button" data-us-range="1" style="appearance:none;border:1px solid #444;background:#191919;color:#ddd;border-radius:8px;padding:6px 11px;cursor:pointer;font:inherit;">1年</button>
      <button class="us-range-button" type="button" data-us-range="3" style="appearance:none;border:1px solid #444;background:#191919;color:#ddd;border-radius:8px;padding:6px 11px;cursor:pointer;font:inherit;">3年</button>
      <button class="us-range-button" type="button" data-us-range="5" style="appearance:none;border:1px solid #444;background:#191919;color:#ddd;border-radius:8px;padding:6px 11px;cursor:pointer;font:inherit;">5年</button>
      <button class="us-range-button" type="button" data-us-range="10" style="appearance:none;border:1px solid #444;background:#191919;color:#ddd;border-radius:8px;padding:6px 11px;cursor:pointer;font:inherit;">10年</button>
      <button class="us-range-button" type="button" data-us-range="20" style="appearance:none;border:1px solid #444;background:#191919;color:#ddd;border-radius:8px;padding:6px 11px;cursor:pointer;font:inherit;">20年</button>
      <button class="us-range-button" type="button" data-us-range="all" style="appearance:none;border:1px solid #444;background:#191919;color:#ddd;border-radius:8px;padding:6px 11px;cursor:pointer;font:inherit;">全部</button>
    </div>

    <div id="us-interactive-chart" aria-label="美股電金比互動查價圖" style="width:100%;height:clamp(580px,65vh,760px);min-height:580px;"></div>
    <div class="chart-help">
      黃線為美股電金比；白線為 MA20；藍綠線為 MA120；虛線 1 代表共同起始日基準。
      圖例中的 XLK、XLF 與 SPY 標準化指數預設隱藏，點擊後可分辨比值上升是科技上漲或金融下跌所造成。
    </div>
  </section>

  <div class="links">
    <a href="us_data.csv">下載美股電金比完整每日資料 CSV</a>
  </div>
{SECTION_END}
"""


def build_script(frame: pd.DataFrame) -> str:
    dates = [
        pd.Timestamp(value).strftime("%Y-%m-%d")
        for value in frame["date"]
    ]
    payload = {
        "dates": dates,
        "ratio": [as_json_number(v, 8) for v in frame["us_ratio"]],
        "ma20": [as_json_number(v, 8) for v in frame["ma20"]],
        "ma120": [as_json_number(v, 8) for v in frame["ma120"]],
        "xlkNorm": [as_json_number(v, 6) for v in frame["xlk_normalized"]],
        "xlfNorm": [as_json_number(v, 6) for v in frame["xlf_normalized"]],
        "spyNorm": [as_json_number(v, 6) for v in frame["spy_normalized"]],
        "xlkAdj": [as_json_number(v, 6) for v in frame["xlk_adj_close"]],
        "xlfAdj": [as_json_number(v, 6) for v in frame["xlf_adj_close"]],
        "customData": [
            [
                as_json_number(row["ma20"], 8),
                as_json_number(row["ma120"], 8),
                as_json_number(row["xlk_normalized"], 6),
                as_json_number(row["xlf_normalized"], 6),
                as_json_number(row["spy_normalized"], 6),
                as_json_number(row["xlk_adj_close"], 6),
                as_json_number(row["xlf_adj_close"], 6),
            ]
            for _, row in frame.iterrows()
        ],
        "defaultRangeYears": 1,
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
    type:'scatter', mode:'lines', x:usDates, y:usChartData.ratio,
    name:'美股電金比', line:{{color:'#d99a36',width:2.0}}, hoverinfo:'skip'
  }};
  const usMa20Trace = {{
    type:'scatter', mode:'lines', x:usDates, y:usChartData.ma20,
    name:'MA20', line:{{color:'#f2f2f2',width:1.7}}, hoverinfo:'skip'
  }};
  const usMa120Trace = {{
    type:'scatter', mode:'lines', x:usDates, y:usChartData.ma120,
    name:'MA120', line:{{color:'#37d3cf',width:1.8}}, hoverinfo:'skip'
  }};
  const usXlkTrace = {{
    type:'scatter', mode:'lines', x:usDates, y:usChartData.xlkNorm,
    name:'XLK 標準化', yaxis:'y2', visible:'legendonly',
    line:{{color:'#ff5b73',width:1.7}}, hoverinfo:'skip'
  }};
  const usXlfTrace = {{
    type:'scatter', mode:'lines', x:usDates, y:usChartData.xlfNorm,
    name:'XLF 標準化', yaxis:'y2', visible:'legendonly',
    line:{{color:'#55e07b',width:1.7}}, hoverinfo:'skip'
  }};
  const usSpyTrace = {{
    type:'scatter', mode:'lines', x:usDates, y:usChartData.spyNorm,
    name:'SPY 標準化', yaxis:'y2', visible:'legendonly',
    line:{{color:'#55aaff',width:1.7}}, hoverinfo:'skip'
  }};
  const usHoverTrace = {{
    type:'scatter', mode:'lines+markers', x:usDates, y:usChartData.ratio,
    customdata:usChartData.customData,
    line:{{width:0}}, marker:{{size:18,opacity:0.002}},
    showlegend:false,
    hovertemplate:'<b>%{{x}}</b><br>'+
      '美股電金比：%{{y:.4f}}<br>'+
      'MA20：%{{customdata[0]:.4f}}<br>'+
      'MA120：%{{customdata[1]:.4f}}<br>'+
      'XLK 標準化：%{{customdata[2]:,.2f}}<br>'+
      'XLF 標準化：%{{customdata[3]:,.2f}}<br>'+
      'SPY 標準化：%{{customdata[4]:,.2f}}<br>'+
      'XLK 還原價：%{{customdata[5]:,.2f}}<br>'+
      'XLF 還原價：%{{customdata[6]:,.2f}}<extra></extra>'
  }};

  const usLayout = {{
    paper_bgcolor:'#050505',
    plot_bgcolor:'#050505',
    margin:{{l:76,r:72,t:82,b:58}},
    hovermode:'closest',
    dragmode:'zoom',
    showlegend:true,
    legend:{{
      orientation:'h',x:0.01,xanchor:'left',y:1.12,yanchor:'top',
      font:{{color:'#ddd',size:13}},bgcolor:'rgba(0,0,0,0)'
    }},
    xaxis:{{
      type:'category',categoryorder:'array',categoryarray:usDates,
      tickfont:{{color:'#ccc',size:11}},
      showgrid:true,gridcolor:'#292929',gridwidth:1,
      showline:true,linecolor:'#555',fixedrange:false,
      showspikes:true,spikemode:'across',spikesnap:'cursor',
      spikecolor:'#f4f4f4',spikethickness:1
    }},
    yaxis:{{
      side:'right',tickformat:'.2f',tickfont:{{color:'#ccc',size:11}},
      title:{{text:'美股電金比',font:{{color:'#d99a36',size:12}}}},
      showgrid:true,gridcolor:'#292929',zeroline:false,
      showline:true,linecolor:'#555',fixedrange:false,
      showspikes:true,spikemode:'across',spikesnap:'cursor',
      spikecolor:'#777',spikethickness:1,spikedash:'dot'
    }},
    yaxis2:{{
      overlaying:'y',side:'left',tickfont:{{color:'#9ccfff',size:11}},
      title:{{text:'標準化總報酬指數',font:{{color:'#9ccfff',size:12}}}},
      showgrid:false,zeroline:false,showline:true,linecolor:'#2b6f99',
      fixedrange:false
    }},
    shapes:[{{
      type:'line',xref:'paper',x0:0,x1:1,yref:'y',y0:1,y1:1,
      line:{{color:'#888',width:1.2,dash:'dash'}}
    }}],
    annotations:[{{
      xref:'paper',x:1,yref:'y',y:1,
      text:'起始基準 1',showarrow:false,xanchor:'right',yanchor:'bottom',
      font:{{color:'#aaa',size:11}}
    }}]
  }};

  function usStartIndexForYears(rangeValue) {{
    if (rangeValue === 'all') return 0;
    const years = Number(rangeValue);
    const latest = new Date(usDates[usDates.length - 1] + 'T00:00:00');
    const cutoff = new Date(latest);
    cutoff.setFullYear(cutoff.getFullYear() - years);
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
    for (let i = startIndex; i < usDates.length; i++) {{
      const value = usDates[i];
      const year = Number(value.slice(0,4));
      const month = Number(value.slice(5,7));
      const bucketMonth = Math.floor((month - 1) / monthStep) * monthStep + 1;
      const key = year + '-' + String(bucketMonth).padStart(2,'0');
      if (key !== previousKey && (month - 1) % monthStep === 0) {{
        previousKey = key;
        vals.push(value);
        texts.push(monthStep === 12 ? String(year) : value.slice(0,7));
      }}
    }}
    return {{vals,texts}};
  }}

  function usFiniteRange(seriesList, startIndex, minimumPadding, includeValue=null) {{
    let minValue = Infinity;
    let maxValue = -Infinity;
    for (const values of seriesList) {{
      for (let i = startIndex; i < values.length; i++) {{
        const number = Number(values[i]);
        if (!Number.isFinite(number)) continue;
        if (number < minValue) minValue = number;
        if (number > maxValue) maxValue = number;
      }}
    }}
    if (Number.isFinite(Number(includeValue))) {{
      minValue = Math.min(minValue, Number(includeValue));
      maxValue = Math.max(maxValue, Number(includeValue));
    }}
    if (!Number.isFinite(minValue) || !Number.isFinite(maxValue)) return null;
    const padding = Math.max((maxValue - minValue) * 0.075, minimumPadding);
    return [minValue - padding, maxValue + padding];
  }}

  function usApplyRange(rangeValue) {{
    const startIndex = usStartIndexForYears(rangeValue);
    const ticks = usBuildTicks(startIndex, rangeValue);
    const ratioRange = usFiniteRange(
      [usChartData.ratio, usChartData.ma20, usChartData.ma120],
      startIndex,
      0.03,
      1
    );
    const normalizedRange = usFiniteRange(
      [usChartData.xlkNorm, usChartData.xlfNorm, usChartData.spyNorm],
      startIndex,
      5
    );
    const changes = {{
      'xaxis.range':[startIndex - 0.5, usDates.length - 0.5],
      'xaxis.tickmode':'array',
      'xaxis.tickvals':ticks.vals,
      'xaxis.ticktext':ticks.texts
    }};
    if (ratioRange) changes['yaxis.range'] = ratioRange;
    if (normalizedRange) changes['yaxis2.range'] = normalizedRange;
    Plotly.relayout(usPlot, changes);

    usButtons.forEach(button => {{
      const active = button.dataset.usRange === String(rangeValue);
      button.style.background = active ? '#75501d' : '#191919';
      button.style.borderColor = active ? '#c58a35' : '#444';
      button.style.color = '#fff';
      button.style.fontWeight = active ? '700' : '400';
    }});
  }}

  const usConfig = {{
    responsive:true,
    scrollZoom:true,
    displaylogo:false,
    modeBarButtonsToRemove:['lasso2d','select2d','toggleSpikelines'],
    doubleClick:'reset'
  }};

  Plotly.newPlot(
    usPlot,
    [usRatioTrace,usMa20Trace,usMa120Trace,usXlkTrace,usXlfTrace,usSpyTrace,usHoverTrace],
    usLayout,
    usConfig
  ).then(() => {{
    usApplyRange(String(usChartData.defaultRangeYears));
    document.documentElement.dataset.usChartReady = 'true';
  }});

  usButtons.forEach(button => {{
    button.addEventListener('click', () => usApplyRange(button.dataset.usRange));
  }});

  const usFmt = (value, digits=2) => {{
    if (value === null || value === undefined || Number.isNaN(Number(value))) return '—';
    return Number(value).toLocaleString('zh-TW', {{
      minimumFractionDigits:digits,
      maximumFractionDigits:digits
    }});
  }};

  usPlot.on('plotly_hover', event => {{
    const point = event.points.find(item => item.data === usHoverTrace)
      || event.points[event.points.length - 1];
    if (!point || !point.customdata) return;
    document.getElementById('us-q-date').textContent = point.x;
    document.getElementById('us-q-ratio').textContent = usFmt(point.y,4);
    document.getElementById('us-q-ma20').textContent = usFmt(point.customdata[0],4);
    document.getElementById('us-q-ma120').textContent = usFmt(point.customdata[1],4);
    document.getElementById('us-q-xlk-norm').textContent = usFmt(point.customdata[2],2);
    document.getElementById('us-q-xlf-norm').textContent = usFmt(point.customdata[3],2);
    document.getElementById('us-q-xlk').textContent = usFmt(point.customdata[5],2);
    document.getElementById('us-q-xlf').textContent = usFmt(point.customdata[6],2);
  }});
}})();
</script>
{SCRIPT_END}
"""


def patch_html(frame: pd.DataFrame) -> None:
    if not HTML_PATH.exists():
        raise FileNotFoundError(
            f"找不到 {HTML_PATH}。請先執行 python src/update_chart.py。"
        )

    html_text = HTML_PATH.read_text(encoding="utf-8")
    html_text = remove_marker_block(html_text, SECTION_START, SECTION_END)
    html_text = remove_marker_block(html_text, SCRIPT_START, SCRIPT_END)

    if "</main>" not in html_text or "</body>" not in html_text:
        raise RuntimeError("docs/index.html 缺少 </main> 或 </body>，無法安全插入美股圖。")

    html_text = html_text.replace("</main>", build_section(frame) + "\n</main>", 1)
    html_text = html_text.replace("</body>", build_script(frame) + "\n</body>", 1)
    HTML_PATH.write_text(html_text, encoding="utf-8")
    LOGGER.info("已把美股電金比插入網頁下方：%s", HTML_PATH)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = parse_args()

    if args.no_fetch:
        frame = read_existing()
    else:
        try:
            prices = download_adjusted_prices()
            frame = compute_ratio(prices)
            save_data(frame)
        except Exception as exc:  # noqa: BLE001
            if US_DATA_CSV.exists():
                LOGGER.exception("美股下載失敗，沿用既有 CSV：%s", exc)
                frame = read_existing()
            else:
                raise

    # --no-fetch 時也同步 docs/us_data.csv，確保網頁下載連結存在。
    if args.no_fetch:
        save_data(frame)

    patch_html(frame)
    LOGGER.info(
        "美股電金比完成：%s 至 %s，共 %d 筆；最新 %.4f。",
        pd.Timestamp(frame.iloc[0]["date"]).date(),
        pd.Timestamp(frame.iloc[-1]["date"]).date(),
        len(frame),
        float(frame.iloc[-1]["us_ratio"]),
    )


if __name__ == "__main__":
    main()
