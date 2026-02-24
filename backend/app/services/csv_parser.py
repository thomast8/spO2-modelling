"""
CSV Parser for Apnea Session Data
==================================

Parses CSV files exported from apnea training apps. The expected format has:
1. A header section with session metadata and round definitions
2. A Biometrics section with per-second HR and SpO2 readings

The parser auto-detects apnea holds by identifying intervals where
the Interval Type is "Apnea".
"""

import csv
import io
from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np
from loguru import logger


@dataclass
class RoundInfo:
    """Metadata for a training round."""

    round_number: int
    intervals: list[dict[str, str]] = field(default_factory=list)


@dataclass
class HoldData:
    """Data for a single detected hold (apnea interval)."""

    hold_number: int              # 1-indexed within session
    start_time_abs_s: int         # Absolute start time in session (seconds)
    end_time_abs_s: int           # Absolute end time in session (seconds)
    duration_s: int               # Duration in seconds
    elapsed_s: np.ndarray         # Time array relative to hold start
    spo2: np.ndarray              # SpO2 readings
    hr: np.ndarray                # Heart rate readings
    min_spo2: float               # Minimum SpO2 during hold
    min_hr: float                 # Minimum HR during hold


@dataclass
class SessionData:
    """Parsed session data from a CSV file."""

    name: str                     # Session type name (e.g., "Piramid")
    session_date: date            # Session date
    rounds: list[RoundInfo]       # Round definitions from header
    holds: list[HoldData]         # Detected apnea holds with data
    total_intervals: int          # Total number of detected intervals


def _parse_time_to_seconds(time_str: str) -> int:
    """Convert MM:SS time string to seconds."""
    parts = time_str.strip().split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    raise ValueError(f"Cannot parse time: {time_str!r}")


def _parse_date(date_str: str) -> date:
    """Parse date string in DD/MM/YYYY format."""
    # Try DD/MM/YYYY HH:MM:SS first, then DD/MM/YYYY
    for fmt in ("%d/%m/%Y %H:%M:%S", "%d/%m/%Y"):
        try:
            return datetime.strptime(date_str.strip(), fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date: {date_str!r}")


def _remove_ischaemic_dip(
    spo2: np.ndarray,
    elapsed_s: np.ndarray,
    hr: np.ndarray,
    dip_threshold: float = 2.0,
    search_window_s: int = 120,
    recovery_margin: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replace initial ischaemic dip artifact with baseline SpO2.

    At the start of some holds, SpO2 briefly drops due to ischaemic blood
    pooling before the real desaturation begins. This function detects and
    flattens that dip so it doesn't distort the model fit. It also fills
    any time gaps left by missing readings during the dip.

    Args:
        spo2: SpO2 readings array
        elapsed_s: Corresponding time array (seconds from hold start)
        hr: Heart rate readings array
        dip_threshold: Minimum drop from baseline to count as a dip
        search_window_s: How far into the hold to scan for dips
        recovery_margin: How close to baseline counts as "recovered"

    Returns:
        Tuple of (cleaned_spo2, filled_elapsed_s, filled_hr)
    """
    if len(spo2) < 3:
        return spo2, elapsed_s, hr

    baseline = spo2[0]

    # Find first sample that dips below threshold within the search window
    dip_start = None
    for i in range(1, len(spo2)):
        if elapsed_s[i] > search_window_s:
            break
        if spo2[i] < baseline - dip_threshold:
            dip_start = i
            break

    if dip_start is None:
        return spo2, elapsed_s, hr

    # Find recovery point: first sample at or near baseline after the dip
    recovery_end = None
    for j in range(dip_start + 1, len(spo2)):
        if elapsed_s[j] > search_window_s:
            break
        if spo2[j] >= baseline - recovery_margin:
            recovery_end = j
            break

    if recovery_end is None:
        return spo2, elapsed_s, hr

    cleaned_spo2 = spo2.copy()
    cleaned_spo2[dip_start:recovery_end] = baseline

    # Fill any time gaps in the dip region with 1-second intervals
    # This prevents holes in the chart where the sensor skipped readings
    t_before = int(elapsed_s[dip_start - 1]) if dip_start > 0 else 0
    t_after = int(elapsed_s[recovery_end])
    gap_times = np.arange(t_before + 1, t_after)
    existing_times = set(int(t) for t in elapsed_s[dip_start:recovery_end])
    missing_times = np.array([t for t in gap_times if t not in existing_times])

    if len(missing_times) > 0:
        # Interpolate HR for the filled gap points
        hr_baseline = hr[dip_start - 1] if dip_start > 0 else hr[0]
        hr_recovery = hr[recovery_end]
        fill_hr = np.linspace(hr_baseline, hr_recovery, len(missing_times))

        elapsed_s = np.concatenate([elapsed_s, missing_times])
        cleaned_spo2 = np.concatenate([cleaned_spo2, np.full(len(missing_times), baseline)])
        hr = np.concatenate([hr, fill_hr])

        # Sort all arrays by time
        order = np.argsort(elapsed_s)
        elapsed_s = elapsed_s[order]
        cleaned_spo2 = cleaned_spo2[order]
        hr = hr[order]

    logger.info(
        f"Removed ischaemic dip: t={t_before + 1}-{t_after - 1}s, "
        f"filled {len(missing_times)} missing points, "
        f"baseline={baseline}, min_dip={float(np.min(spo2[dip_start:recovery_end]))}"
    )
    return cleaned_spo2, elapsed_s, hr


def parse_csv(csv_content: str | bytes) -> SessionData:
    """Parse a complete apnea session CSV file.

    Args:
        csv_content: Raw CSV content as string or bytes

    Returns:
        SessionData with session metadata and detected holds

    Raises:
        ValueError: If the CSV format is not recognized
    """
    if isinstance(csv_content, bytes):
        csv_content = csv_content.decode("utf-8-sig")

    # Remove BOM if present
    if csv_content.startswith("\ufeff"):
        csv_content = csv_content[1:]

    reader = csv.reader(io.StringIO(csv_content))
    rows = list(reader)

    if not rows:
        raise ValueError("Empty CSV file")

    # --- Parse header section ---
    session_name = ""
    session_date = date.today()
    rounds: list[RoundInfo] = []
    biometrics_start = None

    i = 0
    # First row: sep=, or session info
    if rows[0][0].startswith("sep="):
        i = 1

    # Session name and date
    if i < len(rows) and len(rows[i]) >= 2:
        session_name = rows[i][0].strip()
        try:
            session_date = _parse_date(rows[i][1])
        except ValueError:
            logger.warning(f"Could not parse date from row {i}: {rows[i]}")
        i += 1

    # Parse rounds until we hit "Biometrics"
    while i < len(rows):
        row = rows[i]
        if row and row[0].strip() == "Biometrics":
            biometrics_start = i + 1  # Skip the header row too
            break

        if row and row[0].strip().startswith("Round"):
            round_num = int(row[0].strip().split()[-1])
            current_round = RoundInfo(round_number=round_num)
            i += 1  # Skip "Number,Type,Time" header

            # Read intervals for this round
            i += 1
            while i < len(rows) and rows[i] and rows[i][0].strip().startswith("Interval"):
                interval_row = rows[i]
                current_round.intervals.append({
                    "number": interval_row[0].strip(),
                    "type": interval_row[1].strip(),
                    "time": interval_row[2].strip() if len(interval_row) > 2 else "",
                })
                i += 1
            rounds.append(current_round)
        else:
            i += 1

    if biometrics_start is None:
        raise ValueError("No 'Biometrics' section found in CSV")

    # --- Parse biometrics section ---
    # Skip header row: "Time,Interval Time,Interval Type,HR,SpO2"
    bio_header = rows[biometrics_start]
    logger.debug(f"Biometrics header: {bio_header}")
    bio_start = biometrics_start + 1

    # Parse all biometric rows
    intervals: list[tuple[str, list[dict]]] = []
    current_type: str | None = None
    current_block: list[dict] = []

    for row in rows[bio_start:]:
        if not row or len(row) < 5:
            continue

        abs_time = row[0].strip()
        interval_time = row[1].strip()
        interval_type = row[2].strip()
        try:
            hr = int(row[3].strip())
            spo2 = int(row[4].strip())
        except (ValueError, IndexError):
            logger.warning(f"Skipping malformed biometric row: {row}")
            continue

        # Detect interval transitions
        if interval_type != current_type:
            if current_block:
                intervals.append((current_type or "Unknown", current_block))
            current_block = []
            current_type = interval_type

        current_block.append({
            "abs_time": abs_time,
            "interval_time": interval_time,
            "type": interval_type,
            "hr": hr,
            "spo2": spo2,
        })

    if current_block:
        intervals.append((current_type or "Unknown", current_block))

    logger.info(
        f"Parsed {len(intervals)} intervals: "
        f"{', '.join(f'{t}({len(b)})' for t, b in intervals)}"
    )

    # --- Extract apnea holds ---
    holds: list[HoldData] = []
    hold_num = 0

    min_hold_duration_s = 30  # Filter out spurious short fragments

    for interval_type, block in intervals:
        if interval_type != "Apnea":
            continue

        abs_start_s = _parse_time_to_seconds(block[0]["abs_time"])
        abs_end_s = _parse_time_to_seconds(block[-1]["abs_time"])
        duration = abs_end_s - abs_start_s

        if duration < min_hold_duration_s:
            logger.debug(
                f"Skipping short apnea fragment ({duration}s < {min_hold_duration_s}s) "
                f"at {block[0]['abs_time']}"
            )
            continue

        hold_num += 1

        elapsed = np.array([_parse_time_to_seconds(r["interval_time"]) for r in block])
        spo2 = np.array([r["spo2"] for r in block], dtype=float)
        hr = np.array([r["hr"] for r in block], dtype=float)

        # Clean ischaemic dip artifact at hold start
        spo2, elapsed, hr = _remove_ischaemic_dip(spo2, elapsed, hr)

        hold = HoldData(
            hold_number=hold_num,
            start_time_abs_s=abs_start_s,
            end_time_abs_s=abs_end_s,
            duration_s=abs_end_s - abs_start_s,
            elapsed_s=elapsed,
            spo2=spo2,
            hr=hr,
            min_spo2=float(np.min(spo2)),
            min_hr=float(np.min(hr)),
        )
        holds.append(hold)
        logger.info(
            f"Hold {hold_num}: {hold.duration_s}s, "
            f"SpO2 {float(np.max(spo2)):.0f}→{hold.min_spo2:.0f}%, "
            f"HR {float(np.min(hr)):.0f}→{float(np.max(hr)):.0f}"
        )

    return SessionData(
        name=session_name,
        session_date=session_date,
        rounds=rounds,
        holds=holds,
        total_intervals=len(intervals),
    )
