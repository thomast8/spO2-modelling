"""Tests for the CSV parser."""

from datetime import date
from pathlib import Path

import numpy as np
import pytest

from app.services.csv_parser import HoldData, SessionData, parse_csv


class TestParseCSV:
    def test_parses_sample_csv(self, sample_csv_content: str):
        """Should successfully parse the sample session CSV."""
        result = parse_csv(sample_csv_content)
        assert isinstance(result, SessionData)

    def test_session_metadata(self, sample_csv_content: str):
        """Should extract correct session name and date."""
        result = parse_csv(sample_csv_content)
        assert result.name == "Piramid"
        assert result.session_date == date(2026, 2, 21)

    def test_detects_rounds(self, sample_csv_content: str):
        """Should detect all 7 rounds."""
        result = parse_csv(sample_csv_content)
        assert len(result.rounds) == 7

    def test_detects_6_apnea_holds(self, sample_csv_content: str):
        """Should detect exactly 6 apnea holds."""
        result = parse_csv(sample_csv_content)
        assert len(result.holds) == 6

    def test_hold_numbering(self, sample_csv_content: str):
        """Holds should be numbered 1-6."""
        result = parse_csv(sample_csv_content)
        numbers = [h.hold_number for h in result.holds]
        assert numbers == [1, 2, 3, 4, 5, 6]

    def test_hold_6_is_longest(self, sample_csv_content: str):
        """Hold 6 should be the longest (6:13 = 373s)."""
        result = parse_csv(sample_csv_content)
        hold_6 = result.holds[5]
        assert hold_6.duration_s >= 370  # ~6:13

    def test_hold_data_has_arrays(self, sample_csv_content: str):
        """Each hold should have numpy arrays for elapsed_s, spo2, hr."""
        result = parse_csv(sample_csv_content)
        for hold in result.holds:
            assert isinstance(hold.elapsed_s, np.ndarray)
            assert isinstance(hold.spo2, np.ndarray)
            assert isinstance(hold.hr, np.ndarray)
            assert len(hold.elapsed_s) == len(hold.spo2) == len(hold.hr)

    def test_spo2_in_valid_range(self, sample_csv_content: str):
        """SpO2 values should be between 0 and 100."""
        result = parse_csv(sample_csv_content)
        for hold in result.holds:
            assert np.all(hold.spo2 >= 0)
            assert np.all(hold.spo2 <= 100)

    def test_min_spo2_computed(self, sample_csv_content: str):
        """min_spo2 should match the minimum of the spo2 array."""
        result = parse_csv(sample_csv_content)
        for hold in result.holds:
            assert hold.min_spo2 == float(np.min(hold.spo2))

    def test_bytes_input(self, sample_csv_content: str):
        """Should accept bytes input."""
        result = parse_csv(sample_csv_content.encode("utf-8"))
        assert len(result.holds) == 6

    def test_empty_csv_raises(self):
        """Should raise ValueError for empty CSV."""
        with pytest.raises(ValueError, match="Empty CSV"):
            parse_csv("")

    def test_no_biometrics_raises(self):
        """Should raise ValueError if no Biometrics section."""
        csv = "Piramid,21/02/2026\nRound 1,,\nNumber,Type,Time\nInterval 1,Rest,03:00"
        with pytest.raises(ValueError, match="No 'Biometrics' section"):
            parse_csv(csv)
