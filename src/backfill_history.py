from __future__ import annotations

import argparse
import logging
import math
import re
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 TWSE 月資料回補電子、金融與加權指數歷史資料。"
    )
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument("--sleep", type=float, default=0.45)
    return parser.parse_args()


def build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        status=4,
        backoff_factor=1.0,
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
                "AppleWebKit/537.36 Chrome/130 Safari/537.36"
            ),
            "Accept": "application/json,text/plain,*/*",
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
    text = str(value).strip().replace(",", "").replace("−", "-").replace("－", "-")
    if not text or text in {"--", "---", "N/A", "nan", "None"}:
        return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def parse_roc_date(value: Any) -> pd.Timestamp | None:
    text = normalize(value)
    match = re.fullmatch(r"(\d{2,3})/(\d{1,2})/(\d{1,2})", text)
    if not match:
        return None
    roc_year, month, day = map(int, match.groups())
    try:
        return pd.Timestamp(date(roc_year + 1911, month, day))
    except ValueError:
        return None


def candidate_tables(payload: dict[str, Any]) -> list[tuple[list[Any], list[Any]]]:
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


def fetch_json(session: requests.Session, endpoints: tuple[str, ...], month_key: str) -> dict[str, Any]:
    errors: list[str] = []
    for endpoint in endpoints:
        try:
            response = session.get(
                endpoint,
                params={"date": month_key, "response": "json"},
                timeout=(12, 45),
            )
        except requests.RequestException as exc:
            errors.append(f"{endpoint}: {exc}")
            continue
        if response.status_code != 200:
            errors.append(f"{endpoint}: HTTP {response.status_code}")
            continue
        try:
            payload = response.json()
        except ValueError:
            errors.append(f"{endpoint}: 非 JSON 回應")
            continue
        if candidate_tables(payload):
            return payload
        errors.append(f"{endpoint}: stat={payload.get('stat', 'unknown')} 無資料表")
    raise RuntimeError(" | ".join(errors))


def find_field(fields: list[Any], exact_names: tuple[str, ...], fallback: int) -> int:
    compact_fields = [normalize(field) for field in fields]
    compact_targets = {normalize(name) for name in exact_names}
    for index, field in enumerate(compact_fields):
        if field in compact_targets:
            return index
    return fallback


def parse_electronics_finance(payload: dict[str, Any]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for fields, rows in candidate_tables(payload):
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
            day = parse_roc_date(row[date_idx])
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
    return pd.DataFrame(records)


def parse_taiex(payload: dict[str, Any]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for fields, rows in candidate_tables(payload):
        if not rows:
            continue
        date_idx = find_field(fields, ("日期", "日 期", "日　期"), 0)
        close_idx = find_field(fields, ("收盤指數", "收盤"), len(fields) - 1)
        for row in rows:
            if not isinstance(row, (list, tuple)):
                continue
            if len(row) <= max(date_idx, close_idx):
                continue
            day = parse_roc_date(row[date_idx])
            close = parse_number(row[close_idx])
            if day is None or close is None or close <= 0:
                continue
            records.append({"date": day, "weighted_index": close})
        if records:
            break
    return pd.DataFrame(records)


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


def month_keys(start_year: int, end_year: int) -> list[str]:
    today = datetime.now().date()
    keys: list[str] = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            if (year, month) > (today.year, today.month):
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

    session = build_session()
    monthly_frames: list[pd.DataFrame] = []
    keys = month_keys(args.start_year, args.end_year)

    for index, month_key in enumerate(keys, start=1):
        label = f"{month_key[:4]}-{month_key[4:6]}"
        LOGGER.info("[%d/%d] 抓取 %s 月資料", index, len(keys), label)
        try:
            ef_payload = fetch_json(session, EF_ENDPOINTS, month_key)
            time.sleep(max(args.sleep, 0.0))
            taiex_payload = fetch_json(session, TAIEX_ENDPOINTS, month_key)
        except RuntimeError as exc:
            LOGGER.error("%s 抓取失敗：%s", label, exc)
            return 2

        ef = parse_electronics_finance(ef_payload)
        taiex = parse_taiex(taiex_payload)
        if ef.empty:
            LOGGER.error("%s 電子／金融月資料解析為空。", label)
            return 3
        if taiex.empty:
            LOGGER.error("%s 加權指數月資料解析為空。", label)
            return 4

        merged = ef.merge(taiex, on="date", how="inner", validate="one_to_one")
        if merged.empty:
            LOGGER.error("%s 三項指數沒有共同交易日。", label)
            return 5
        merged["ratio"] = merged["electronics_index"] / merged["finance_index"]
        monthly_frames.append(merged)
        LOGGER.info("%s 完成，共 %d 個交易日。", label, len(merged))
        time.sleep(max(args.sleep, 0.0))

    incoming = pd.concat(monthly_frames, ignore_index=True)
    existing = read_existing()
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

    expected_new = set(pd.to_datetime(incoming["date"]))
    actual = set(pd.to_datetime(combined["date"]))
    missing = sorted(expected_new - actual)
    if missing:
        LOGGER.error("寫入後遺失 %d 個已抓到的交易日，例如 %s", len(missing), missing[:5])
        return 6

    DATA_CSV.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(DATA_CSV, index=False, date_format="%Y-%m-%d")
    LOGGER.info(
        "完成 %d–%d 回補：新增／覆蓋 %d 筆；主資料共 %d 筆，範圍 %s 至 %s。",
        args.start_year,
        args.end_year,
        len(incoming),
        len(combined),
        combined["date"].min().strftime("%Y-%m-%d"),
        combined["date"].max().strftime("%Y-%m-%d"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
