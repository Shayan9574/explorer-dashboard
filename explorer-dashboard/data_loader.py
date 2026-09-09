"""Load the interview workbook from Google Drive (public link) with a repo fallback."""
import io
import re
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

VEHICLES = ["Explorer", "Traverse", "Highlander", "Pilot", "Palisade", "Telluride"]
COMPETITORS = [v for v in VEHICLES if v != "Explorer"]

# Google Drive file id of the uploaded xlsx (a full share link also works here or in
# .streamlit/secrets.toml as GDRIVE_FILE_ID). Overridable via secrets.
GDRIVE_FILE_ID = "1gHvrtZ6vm-C44K1_ti6XGMVICHQYsuKR"

_FALLBACK = Path(__file__).parent / "data" / "fallback.xlsx"


def _extract_id(value: str) -> str:
    """Accept a bare id or any Drive/Sheets share link."""
    m = re.search(r"/d/([A-Za-z0-9_-]{20,})", value) or re.search(r"[?&]id=([A-Za-z0-9_-]{20,})", value)
    return m.group(1) if m else value.strip()


def _drive_bytes(file_id: str) -> bytes:
    """Try both public endpoints; validate we got a real xlsx (zip magic PK)."""
    urls = [
        f"https://drive.google.com/uc?export=download&id={file_id}",
        f"https://docs.google.com/spreadsheets/d/{file_id}/export?format=xlsx",
    ]
    last_err = None
    for url in urls:
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            if r.content[:2] == b"PK":  # xlsx is a zip container
                return r.content
            last_err = ValueError("Response was not an xlsx (is the file shared publicly?)")
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise last_err


@st.cache_data(show_spinner="Loading data ...")
def load_data():
    """Return (monthly, quarterly, source_label)."""
    try:
        file_id = st.secrets.get("GDRIVE_FILE_ID", GDRIVE_FILE_ID)
    except Exception:
        file_id = GDRIVE_FILE_ID

    content, source = None, "Google Drive"
    if file_id:
        try:
            content = _drive_bytes(_extract_id(file_id))
        except Exception:
            content = None
    if content is None:
        if _FALLBACK.exists():
            content = _FALLBACK.read_bytes()
            source = "Repository fallback copy"
        else:
            st.error(
                "Could not load the data. The Google Drive download failed and no fallback "
                "copy exists in the repository. Fix either one: (1) in Drive, share the xlsx "
                "as 'Anyone with the link: Viewer' and set GDRIVE_FILE_ID in Streamlit "
                "secrets, or (2) add the workbook to the repo at data/fallback.xlsx."
            )
            st.stop()

    raw = pd.read_excel(io.BytesIO(content), sheet_name="Data", header=None)

    # Monthly block: rows whose first cell parses as a date and second as a number
    monthly_rows = []
    for _, r in raw.iterrows():
        d, p = r.iloc[0], r.iloc[1]
        if pd.notna(d) and isinstance(p, (int, float)) and not isinstance(d, str):
            monthly_rows.append(r)
    m = pd.DataFrame(monthly_rows).reset_index(drop=True).iloc[:21]

    monthly = pd.DataFrame({"date": pd.to_datetime(m.iloc[:, 0])})
    for i, v in enumerate(VEHICLES):
        monthly[f"atp_{v}"] = m.iloc[:, 1 + i].astype(float)
    monthly["atp_Segment"] = m.iloc[:, 7].astype(float)
    for i, v in enumerate(VEHICLES):
        monthly[f"vol_{v}"] = m.iloc[:, 10 + i].astype(float)

    monthly["quarter"] = monthly["date"].dt.to_period("Q").astype(str)
    monthly["refresh"] = (monthly["date"] >= "2024-08-01").astype(int)
    monthly["progA"] = monthly["date"].between("2025-04-01", "2025-06-30").astype(int)
    monthly["progB"] = monthly["date"].between("2025-07-01", "2025-09-30").astype(int)

    # Quarterly aggregation (volume weighted ATP, matches the workbook's quarterly tab)
    rows = []
    for q, g in monthly.groupby("quarter"):
        row = {"quarter": q}
        for v in VEHICLES:
            vol = g[f"vol_{v}"].sum()
            row[f"vol_{v}"] = vol
            row[f"atp_{v}"] = (g[f"atp_{v}"] * g[f"vol_{v}"]).sum() / vol
        rows.append(row)
    quarterly = pd.DataFrame(rows).sort_values("quarter").reset_index(drop=True)

    return monthly, quarterly, source
