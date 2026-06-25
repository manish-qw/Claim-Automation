"""Date calculation helpers."""
from datetime import datetime, date
from typing import Optional


def parse_date(date_str: Optional[str]) -> Optional[date]:
    if not date_str:
        return None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    return None


def days_between(date1_str: Optional[str], date2_str: Optional[str]) -> Optional[int]:
    d1 = parse_date(date1_str)
    d2 = parse_date(date2_str)
    if d1 and d2:
        return abs((d2 - d1).days)
    return None


def days_from_today(date_str: Optional[str]) -> Optional[int]:
    d = parse_date(date_str)
    if d:
        return (date.today() - d).days
    return None
