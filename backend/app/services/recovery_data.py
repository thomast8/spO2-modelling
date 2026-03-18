"""
Recovery Data Extraction
========================

Extracts post-apnea recovery data from stored CSV blobs. Stage A needs recovery
data to fit the sensor model through the nadir and into recovery, because the
sensor lag means the SpO2 nadir occurs after the apnea ends.
"""

import csv
import io

import numpy as np
from loguru import logger


def _parse_time_to_seconds(time_str: str) -> int:
    """Parse a time string (M:SS or H:MM:SS) to total seconds."""
    parts = time_str.strip().split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    raise ValueError(f"Cannot parse time: {time_str!r}")


def extract_holds_with_recovery(
    csv_blob: bytes,
    holds: list[dict],
    recovery_max_s: int = 90,
    recovery_spo2_ceiling: int = 97,
) -> list[dict]:
    """Extract hold data with post-apnea recovery appended from raw CSV.

    Args:
        csv_blob: Raw CSV bytes from the session record
        holds: List of hold dicts, each with keys:
            - id: hold database ID
            - type: hold type (FRC/RV/FL)
            - t: numpy array of elapsed seconds (apnea portion)
            - spo2: numpy array of SpO2 values (apnea portion)
            - hr: numpy array of HR values (apnea portion)
        recovery_max_s: Maximum recovery duration to append (seconds)
        recovery_spo2_ceiling: Stop appending when SpO2 reaches this level

    Returns:
        Enriched hold dicts with additional keys:
            - t_end: last time point of apnea
            - t_apnea: apnea-only time array
            - spo2_apnea: apnea-only SpO2 array
            - t_recovery: recovery time array
            - spo2_recovery: recovery SpO2 array
    """
    csv_text = csv_blob.decode("utf-8-sig")
    if csv_text.startswith("\ufeff"):
        csv_text = csv_text[1:]
    reader = csv.reader(io.StringIO(csv_text))
    rows = list(reader)

    # Find Biometrics section
    bio_start = None
    for i, row in enumerate(rows):
        if row and row[0].strip() == "Biometrics":
            bio_start = i + 2
            break
    if bio_start is None:
        logger.warning("No Biometrics section in CSV, returning holds without recovery")
        return _wrap_holds_no_recovery(holds)

    # Parse intervals from CSV
    intervals = []
    current_type = None
    current_block: list[dict] = []

    for row in rows[bio_start:]:
        if not row or len(row) < 5:
            continue
        itype = row[2].strip()
        try:
            hr = int(row[3].strip())
            spo2 = int(row[4].strip())
        except (ValueError, IndexError):
            continue
        if itype != current_type:
            if current_block:
                intervals.append((current_type, current_block))
            current_block = []
            current_type = itype
        current_block.append({
            "abs_time": row[0].strip(),
            "int_time": row[1].strip(),
            "type": itype,
            "hr": hr,
            "spo2": spo2,
        })
    if current_block:
        intervals.append((current_type, current_block))

    # Build a map from hold_id -> hold dict for quick lookup
    hold_map = {h["id"]: h for h in holds}
    result = []

    apnea_idx = 0
    for i, (itype, block) in enumerate(intervals):
        if itype != "Apnea":
            continue
        duration = (
            _parse_time_to_seconds(block[-1]["abs_time"])
            - _parse_time_to_seconds(block[0]["abs_time"])
        )
        if duration < 30:
            continue
        apnea_idx += 1
        hold_id = apnea_idx

        if hold_id not in hold_map:
            continue

        db_hold = hold_map[hold_id]
        t_end = float(db_hold["t"][-1])

        t_recovery = np.array([], dtype=float)
        spo2_recovery = np.array([], dtype=float)

        # Look for Rest/Cooldown interval following this apnea
        if i + 1 < len(intervals):
            next_type, next_block = intervals[i + 1]
            if next_type in ("Rest", "Cooldown"):
                for r in next_block:
                    rt = _parse_time_to_seconds(r["int_time"])
                    if rt > recovery_max_s:
                        break
                    if rt > 5 and r["spo2"] >= recovery_spo2_ceiling:
                        break
                    t_recovery = np.append(t_recovery, t_end + rt + 1)
                    spo2_recovery = np.append(spo2_recovery, float(r["spo2"]))

        t_full = np.concatenate([db_hold["t"], t_recovery])
        spo2_full = np.concatenate([db_hold["spo2"], spo2_recovery])

        result.append({
            "id": hold_id,
            "type": db_hold["type"],
            "t": t_full,
            "spo2": spo2_full,
            "t_end": t_end,
            "t_apnea": db_hold["t"],
            "spo2_apnea": db_hold["spo2"],
            "t_recovery": t_recovery,
            "spo2_recovery": spo2_recovery,
        })

    logger.info(
        f"Extracted recovery data for {len(result)} holds "
        f"(max {recovery_max_s}s, ceiling {recovery_spo2_ceiling}%)"
    )
    return result


def _wrap_holds_no_recovery(holds: list[dict]) -> list[dict]:
    """Wrap holds without recovery data (fallback when CSV parsing fails)."""
    result = []
    for h in holds:
        result.append({
            "id": h["id"],
            "type": h["type"],
            "t": h["t"],
            "spo2": h["spo2"],
            "t_end": float(h["t"][-1]),
            "t_apnea": h["t"],
            "spo2_apnea": h["spo2"],
            "t_recovery": np.array([], dtype=float),
            "spo2_recovery": np.array([], dtype=float),
        })
    return result
