from __future__ import annotations

import argparse
import logging
import math
import random
import re
import time
from datetime import date, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = ROOT / "data" / "electronics_finance_ratio.csv"

EF_ENDPOINTS = (
    "https://www.twse.com.tw/rwd/zh/TAIEX/EFTRI_HIST",
    "https://www.twse.com.tw/indicesReport/EFTRI_HIST",
)
TAIEX_ENDPOINTS = (
    "https://www.twse.com.tw/rwd/zh/TAIEX/MI_5MINS_HIST",
    "https://www.twse.com.tw/indicesReport/MI_5MINS_HIST",
)

LOGGER = logging.getLogger("history_backfill")


class SimpleTableParser(HTMLParser):
    """只用標準函式庫抽取 HTML 表格，避免額外安裝 lxml。"""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[list[list[str]]] = []
        self._table: list[list[str]] | None = None
        self._row: list[str] | None = None
        self._cell_parts: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "table":
            self._table = []
        elif tag == "tr" and self._table is not None:
            self._row = []
        elif tag in {"th", "td"} and self._row is not None:
            self._cell_parts = []

    def handle_data(self, data: str) -> None:
        if self._cell_parts is not None:
            self._cell_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"th", "td"} and self._row is not None and self._cell_parts is not None:
            cell = re.sub(r"\s+", " ", "".join(self._cell_parts)).strip()
            self._row.append(cell)
            self._cell_parts = None
        elif tag == "tr" and self._table is not None and self._row is not None:
            if any(cell.strip() for cell in self._row):
                self._table.append(self._row)
            self._row = None
        elif tag == "table" and self._table is not None:
            if self._table:
                self.tables.append(self._table)
            self._table = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 TWSE 月資料回補電子、金融與加權指數歷史資料。"
    )
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument("--sleep", type=float, default=1.6)
    parser.add_argument("--max-attempts", type=int, default=6)
    return parser.parse_args()


def build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        status=3,
        backoff_factor=1.2,
        status_forcelist=(408, 425, 429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4)
    session.mount("https://", adapter)
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/130.0 Safari/537.36"
            ),
            "Accept": "application/json,text/html,application/xhtml+xml,text/plain,*/*",
            "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.7",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "Referer": "https://www.twse.com.tw/zh/indices/taiex/eftri-hist.html",
        }
    )
    return session


def normalize(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "").replace("\u3000", " "))


def parse_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    text = (
        str(value)
        .strip()
        .replace(",", "")
        .replace("−", "-")
        .replace("－", "-")
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


def parse_twse_date(value: Any) -> pd.Timestamp | None:
    text = normalize(value)
    match = re.fullmatch(r"(\d{2,4})/(\d{1,2})/(\d{1,2})", text)
    if not match:
        return None
    raw_year, month, day = map(int, match.groups())
    year = raw_year + 1911 if raw_year < 1911 else raw_year
    try:
        return pd.Timestamp(date(year, month, day))
    except ValueError:
        return None


def candidate_tables_from_json(payload: dict[str, Any]) -> list[tuple[list[Any], list[Any]]]:
    output: list[tuple[list[Any], list[Any]]] = []
    tables = payload.get("tables")
    if isinstance(tables, list):
        for table in tables:
            if not isinstance(table, dict):
                continue
            fields = table.get("fields") or table.get("field") or []
            rows = table.get("data") or []
            if isinstance(fields, list) and isinstance(rows, list):
                output.append((fields, rows))

    fields = payload.get("fields")
    rows = payload.get("data")
    if isinstance(fields, list) and isinstance(rows, list):
        output.append((fields, rows))

    for key, fields in payload.items():
        if not isinstance(key, str) or not key.startswith("fields"):
            continue
        suffix = key[6:]
        rows = payload.get(f"data{suffix}")
        if isinstance(fields, list) and isinstance(rows, list):
            output.append((fields, rows))
    return output


def candidate_tables_from_html(text: str) -> list[tuple[list[Any], list[Any]]]:
    parser = SimpleTableParser()
    parser.feed(text)
    output: list[tuple[list[Any], list[Any]]] = []
    for table in parser.tables:
        header_index: int | None = None
        for index, row in enumerate(table):
            compact = [normalize(cell) for cell in row]
            if any(cell == "日期" or cell.endswith("日期") for cell in compact):
                header_index = index
                break
        if header_index is None:
            continue
        fields = table[header_index]
        rows = table[header_index + 1 :]
        if fields and rows:
            output.append((fields, rows))
    return output


def response_tables(response: requests.Response) -> list[tuple[list[Any], list[Any]]]:
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if isinstance(payload, dict):
        tables = candidate_tables_from_json(payload)
        if tables:
            return tables

    text = response.text
    if "<table" in text.lower() or "日期" in text:
        return candidate_tables_from_html(text)
    return []


def fetch_tables(
    session: requests.Session,
    endpoints: tuple[str, ...],
    month_key: str,
    max_attempts: int,
) -> list[tuple[list[Any], list[Any]]]:
    errors: list[str] = []

    for attempt in range(1, max_attempts + 1):
        # 每輪都同時試 JSON 與 HTML。舊端點有時即使要求 JSON 仍回傳有效 HTML。
        for endpoint in endpoints:
            for response_type in ("json", "html"):
                try:
                    response = session.get(
                        endpoint,
                        params={
                            "date": month_key,
                            "response": response_type,
                            "_": str(int(time.time() * 1000)),
                        },
                        timeout=(15, 60),
                        allow_redirects=True,
                    )
                except requests.RequestException as exc:
                    errors.append(
                        f"第{attempt}輪 {endpoint} {response_type}: {type(exc).__name__}: {exc}"
                    )
                    continue

                redirect_chain = "→".join(str(item.status_code) for item in response.history)
                status_note = (
                    f"HTTP {response.status_code}"
                    + (f"（redirect {redirect_chain}）" if redirect_chain else "")
                )
                if response.status_code != 200:
                    errors.append(f"第{attempt}輪 {endpoint} {response_type}: {status_note}")
                    continue

                tables = response_tables(response)
                if tables:
                    return tables

                content_type = response.headers.get("content-type", "unknown")
                errors.append(
                    f"第{attempt}輪 {endpoint} {response_type}: {status_note}，"
                    f"但無可解析資料表（{content_type}）"
                )

        if attempt < max_attempts:
            delay = min(75.0, 5.0 * (2 ** (attempt - 1))) + random.uniform(0.5, 2.5)
            LOGGER.warning(
                "%s 第 %d/%d 輪失敗，等待 %.1f 秒後重試。",
                month_key[:6],
                attempt,
                max_attempts,
                delay,
            )
            time.sleep(delay)
            # WAF 或限流時重建 session，避免沿用有問題的連線／cookie。
            session.close()
            session = build_session()

    tail = errors[-12:]
    raise RuntimeError(" | ".join(tail))


def find_field(fields: list[Any], exact_names: tuple[str, ...], fallback: int) -> int:
    compact_fields = [normalize(field) for field in fields]
    compact_targets = {normalize(name) for name in exact_names}
    for index, field in enumerate(compact_fields):
        if field in compact_targets:
            return index
    return fallback


def parse_electronics_finance(
    tables: Iterable[tuple[list[Any], list[Any]]]
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for fields, rows in tables:
        if not rows:
            continue
        date_idx = find_field(fields, ("日期", "日 期", "日　期"), 0)
        elec_idx = find_field(fields, ("電子類指數", "電子工業類指數"), 1)
        finance_idx = find_field(fields, ("金融保險類指數",), len(fields) - 1)

        for row in rows:
            if not isinstance(row, (list, tuple)):
                continue
            if len(row) <= max(date_idx, elec_idx, finance_idx):
                continue
            day = parse_twse_date(row[date_idx])
            electronics = parse_number(row[elec_idx])
            finance = parse_number(row[finance_idx])
            if day is None or electronics is None or finance is None:
                continue
            if electronics <= 0 or finance <= 0:
                continue
            records.append(
                {
                    "date": day,
                    "electronics_index": electronics,
                    "finance_index": finance,
                }
            )
        if records:
            break
    return pd.DataFrame(records).drop_duplicates("date", keep="last")


def parse_taiex(tables: Iterable[tuple[list[Any], list[Any]]]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for fields, rows in tables:
        if not rows:
            continue
        date_idx = find_field(fields, ("日期", "日 期", "日　期", "Date"), 0)
        close_idx = find_field(
            fields,
            ("收盤指數", "收盤", "Closing Index", "ClosingIndex"),
            len(fields) - 1,
        )
        for row in rows:
            if not isinstance(row, (list, tuple)):
                continue
            if len(row) <= max(date_idx, close_idx):
                continue
            day = parse_twse_date(row[date_idx])
            close = parse_number(row[close_idx])
            if day is None or close is None or close <= 0:
                continue
            records.append({"date": day, "weighted_index": close})
        if records:
            break
    return pd.DataFrame(records).drop_duplicates("date", keep="last")


def read_existing() -> pd.DataFrame:
    columns = [
        "date",
        "electronics_index",
        "finance_index",
        "weighted_index",
        "ratio",
    ]
    if not DATA_CSV.exists():
        return pd.DataFrame(columns=columns)
    frame = pd.read_csv(DATA_CSV, parse_dates=["date"])
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    frame = frame[columns].copy()
    for column in columns[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["date"]).drop_duplicates("date", keep="last")


def save_month(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([existing, incoming], ignore_index=True)
    combined = combined.drop_duplicates(subset=["date"], keep="last")
    combined = combined.dropna(
        subset=[
            "date",
            "electronics_index",
            "finance_index",
            "weighted_index",
            "ratio",
        ]
    )
    combined = combined.sort_values("date", kind="stable").reset_index(drop=True)
    DATA_CSV.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(DATA_CSV, index=False, date_format="%Y-%m-%d")
    return combined


def month_keys(start_year: int, end_year: int) -> list[str]:
    """建立要回補的完整月份清單。

    歷史回補只處理「已結束月份」。本月的逐日資料交給 daily workflow 更新，
    避免在月初尚無交易日（例如本月前兩天是週末）時，把正常空資料誤判成失敗。
    """
    today = datetime.now().date()
    keys: list[str] = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            # 本月及未來月份不屬於完整歷史月份，交由每日更新流程處理。
            if (year, month) >= (today.year, today.month):
                break
            keys.append(f"{year}{month:02d}01")
    return keys


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    if args.start_year < 2000 or args.end_year < args.start_year:
        raise SystemExit("年份範圍不正確；電子／金融官方歷史資料自 2000 年開始。")
    if args.max_attempts < 1:
        raise SystemExit("max-attempts 必須至少為 1。")

    session = build_session()
    combined = read_existing()
    successful_months = 0
    failed_months: list[str] = []
    keys = month_keys(args.start_year, args.end_year)

    for index, month_key in enumerate(keys, start=1):
        label = f"{month_key[:4]}-{month_key[4:6]}"
        LOGGER.info("[%d/%d] 抓取 %s 月資料", index, len(keys), label)
        try:
            ef_tables = fetch_tables(session, EF_ENDPOINTS, month_key, args.max_attempts)
            time.sleep(max(args.sleep, 0.0))
            taiex_tables = fetch_tables(
                session, TAIEX_ENDPOINTS, month_key, args.max_attempts
            )

            ef = parse_electronics_finance(ef_tables)
            taiex = parse_taiex(taiex_tables)
            if ef.empty:
                raise RuntimeError("電子／金融月資料解析為空")
            if taiex.empty:
                raise RuntimeError("加權指數月資料解析為空")

            merged = ef.merge(taiex, on="date", how="inner", validate="one_to_one")
            if merged.empty:
                raise RuntimeError("三項指數沒有共同交易日")
            merged["ratio"] = merged["electronics_index"] / merged["finance_index"]

            # 每個月成功後立刻寫入本機 CSV；即使後續月份失敗，也保留已完成進度。
            combined = save_month(combined, merged)
            successful_months += 1
            LOGGER.info("%s 完成，共 %d 個交易日，已寫入主 CSV。", label, len(merged))
        except Exception as exc:  # noqa: BLE001 - 需要將單月錯誤彙總後回報
            LOGGER.error("%s 最終失敗：%s", label, exc)
            failed_months.append(label)

        time.sleep(max(args.sleep, 0.0) + random.uniform(0.2, 0.8))

        # 每半年主動冷卻，降低 TWSE WAF／限流機率。
        if index % 6 == 0 and index < len(keys):
            LOGGER.info("已完成 %d 個月份，暫停 12 秒降低 TWSE 限流風險。", index)
            time.sleep(12)

    if combined.empty:
        LOGGER.error("沒有任何資料可寫入。")
        return 2

    LOGGER.info(
        "本次完成 %d 個月份；主資料共 %d 筆，範圍 %s 至 %s。",
        successful_months,
        len(combined),
        combined["date"].min().strftime("%Y-%m-%d"),
        combined["date"].max().strftime("%Y-%m-%d"),
    )

    if failed_months:
        LOGGER.error(
            "仍有 %d 個月份未完成：%s。已成功月份已寫入 CSV，可重新執行相同年份續補。",
            len(failed_months),
            ", ".join(failed_months),
        )
        return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
