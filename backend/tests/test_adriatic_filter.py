"""
Unit tests for the Adriatic/Ionian maritime corridor filter.

Run with:
    python3 -m unittest backend/tests/test_adriatic_filter.py

Coverage: corridor config loading (all four zones), coordinate -> zone
resolution, and the four AIS anomaly detectors exposed by
``backend.services.adriatic_radar``:

  ANOMALOUS_STOP      vessel stopped/drifting in open sea
  HIGH_SPEED_TRANSIT  speed above the zone speed limit
  DRAFT_HAZARD        reported draft exceeds the zone safe depth
  TRANSPONDER_GAP     AIS report older than the zone gap threshold

The filter is pure stdlib (no network / database), so these tests run offline.
"""

from __future__ import annotations

import os
import sys
import unittest
from datetime import datetime, timedelta, timezone

# Make the repository root importable so "backend.*" modules resolve when this
# file is executed directly via `python3 -m unittest backend/tests/...`.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from backend.services.adriatic_radar import (  # noqa: E402
    ANOMALOUS_STOP,
    DRAFT_HAZARD,
    HIGH_SPEED_TRANSIT,
    TRANSPONDER_GAP,
    load_filter,
)

# Fixed clock so staleness maths is deterministic.
NOW = datetime(2026, 9, 19, 12, 0, 0, tzinfo=timezone.utc)

EXPECTED_ZONE_IDS = {
    "durres_port_approach",
    "vlore_bay",
    "strait_of_otranto",
    "sarande_corfu_channel",
}

# Representative in-zone coordinates (WGS84 lon, lat).
DURRES_LON, DURRES_LAT = 19.45, 41.28        # durres_port_approach
VLORE_LON, VLORE_LAT = 19.48, 40.45          # vlore_bay
OTRANTO_LON, OTRANTO_LAT = 18.90, 40.20      # strait_of_otranto (open sea)
SARANDE_LON, SARANDE_LAT = 19.95, 39.80      # sarande_corfu_channel

# Open Adriatic, well outside every corridor bounding box.
OUTSIDE_LON, OUTSIDE_LAT = 12.5, 41.9


def vessel(mmsi: int = 200000001, **overrides) -> dict:
    """A well-behaved AIS report: slow, shallow draft, freshly received."""
    signal = {
        "mmsi": mmsi,
        "lon": VLORE_LON,
        "lat": VLORE_LAT,
        "speed_kn": 5.0,
        "draft_m": 4.0,
        "last_seen": (NOW - timedelta(minutes=1)).isoformat(),
        "at_berth": False,
    }
    signal.update(overrides)
    return signal


def anomaly_types(anomalies) -> list[str]:
    return [a.anomaly_type for a in anomalies]


class TestAdriaticFilter(unittest.TestCase):
    """Corridor config loading, zone matching and the four anomaly detectors."""

    def setUp(self) -> None:
        self.corridor = load_filter()

    # ---------------------------------------------------------------- config

    def test_config_loads_zones(self) -> None:
        """load_filter() must expose all four corridor zones."""
        zone_ids = {zone.zone_id for zone in self.corridor.zones}
        self.assertEqual(zone_ids, EXPECTED_ZONE_IDS)
        self.assertEqual(len(self.corridor.zones), 4)

        # Every zone must carry a usable bbox and positive limits.
        for zone in self.corridor.zones:
            with self.subTest(zone=zone.zone_id):
                self.assertEqual(len(zone.bbox), 4)
                self.assertLess(zone.bbox[0], zone.bbox[2])  # min_lon < max_lon
                self.assertLess(zone.bbox[1], zone.bbox[3])  # min_lat < max_lat
                self.assertGreater(zone.speed_limit_kn, 0.0)
                self.assertGreater(zone.max_draft_m, 0.0)

        # get_zone() resolves by id, and the defaults block is loaded.
        self.assertIsNotNone(self.corridor.get_zone("durres_port_approach"))
        self.assertIsNone(self.corridor.get_zone("no_such_zone"))
        self.assertIn("transponder_gap_minutes", self.corridor.defaults)

    # -------------------------------------------------------------- geometry

    def test_point_matching(self) -> None:
        """Durres matches its port approach; a point outside the corridor is None."""
        zone = self.corridor.match_zone(DURRES_LON, DURRES_LAT)
        self.assertIsNotNone(zone)
        self.assertEqual(zone.zone_id, "durres_port_approach")
        self.assertTrue(self.corridor.is_in_corridor(DURRES_LON, DURRES_LAT))

        # 12.5E 41.9N is open Adriatic, outside every bounding box.
        self.assertIsNone(self.corridor.match_zone(OUTSIDE_LON, OUTSIDE_LAT))
        self.assertFalse(self.corridor.is_in_corridor(OUTSIDE_LON, OUTSIDE_LAT))
        self.assertEqual(self.corridor.match_zones(OUTSIDE_LON, OUTSIDE_LAT), [])

        # An out-of-corridor signal yields no anomalies at all.
        signal = vessel(lon=OUTSIDE_LON, lat=OUTSIDE_LAT, speed_kn=0.1, draft_m=30.0)
        self.assertEqual(self.corridor.evaluate(signal, now=NOW), [])

    # -------------------------------------------------------------- detectors

    def test_anomalous_stop_detection(self) -> None:
        """A drifting vessel in the open-sea Strait of Otranto is anomalous."""
        signal = vessel(
            mmsi=247100001,
            lon=OTRANTO_LON,
            lat=OTRANTO_LAT,
            speed_kn=0.2,
            draft_m=8.0,
        )
        anomalies = self.corridor.evaluate(signal, now=NOW)
        self.assertIn(ANOMALOUS_STOP, anomaly_types(anomalies))

        stop = next(a for a in anomalies if a.anomaly_type == ANOMALOUS_STOP)
        self.assertEqual(stop.zone_id, "strait_of_otranto")
        self.assertEqual(stop.mmsi, 247100001)
        self.assertEqual(stop.severity, "high")
        self.assertAlmostEqual(stop.observed_value, 0.2)

    def test_high_speed_detection(self) -> None:
        """35 kn exceeds the Vlore Bay limit of 10 kn."""
        signal = vessel(mmsi=247100010, lon=VLORE_LON, lat=VLORE_LAT, speed_kn=35.0)
        anomalies = self.corridor.evaluate(signal, now=NOW)
        self.assertIn(HIGH_SPEED_TRANSIT, anomaly_types(anomalies))

        speeding = next(a for a in anomalies if a.anomaly_type == HIGH_SPEED_TRANSIT)
        self.assertEqual(speeding.zone_id, "vlore_bay")
        self.assertEqual(speeding.mmsi, 247100010)
        self.assertAlmostEqual(speeding.observed_value, 35.0)
        self.assertEqual(speeding.severity, "high")

        # Zone speed limit for vlore_bay is 10.0 kn.
        zone = self.corridor.get_zone("vlore_bay")
        self.assertAlmostEqual(zone.speed_limit_kn, 10.0)
        self.assertAlmostEqual(speeding.threshold_value, zone.speed_limit_kn)

    def test_draft_hazard(self) -> None:
        """13.5 m draft exceeds the 11.5 m safe depth of the Durres approach."""
        signal = vessel(
            mmsi=247100020,
            lon=DURRES_LON,
            lat=DURRES_LAT,
            speed_kn=4.0,
            draft_m=13.5,
        )
        anomalies = self.corridor.evaluate(signal, now=NOW)
        self.assertIn(DRAFT_HAZARD, anomaly_types(anomalies))

        hazard = next(a for a in anomalies if a.anomaly_type == DRAFT_HAZARD)
        self.assertEqual(hazard.zone_id, "durres_port_approach")
        self.assertEqual(hazard.mmsi, 247100020)
        self.assertAlmostEqual(hazard.observed_value, 13.5)

        zone = self.corridor.get_zone("durres_port_approach")
        self.assertAlmostEqual(zone.max_draft_m, 11.5)
        self.assertTrue(13.5 > zone.max_draft_m)

        # A compliant draft in the same zone is clean.
        safe = vessel(
            mmsi=247100021, lon=DURRES_LON, lat=DURRES_LAT, speed_kn=4.0, draft_m=9.0
        )
        self.assertIsNone(self.corridor.detect_draft_hazard(safe, zone))

    def test_transponder_gap(self) -> None:
        """A report older than the zone gap threshold raises TRANSPONDER_GAP."""
        zone = self.corridor.get_zone("sarande_corfu_channel")
        gap_limit = float(
            self.corridor._threshold(zone, "transponder_gap_minutes", 20.0)
        )
        stale_minutes = gap_limit + 25.0

        signal = vessel(
            mmsi=247100030,
            lon=SARANDE_LON,
            lat=SARANDE_LAT,
            speed_kn=9.0,
            draft_m=4.0,
            last_seen=(NOW - timedelta(minutes=stale_minutes)).isoformat(),
        )
        anomalies = self.corridor.evaluate(signal, now=NOW)
        self.assertIn(TRANSPONDER_GAP, anomaly_types(anomalies))

        gap = next(a for a in anomalies if a.anomaly_type == TRANSPONDER_GAP)
        self.assertEqual(gap.zone_id, "sarande_corfu_channel")
        self.assertAlmostEqual(gap.observed_value, stale_minutes, places=1)
        self.assertAlmostEqual(gap.threshold_value, gap_limit)
        self.assertGreater(gap.observed_value, gap.threshold_value)

        # A fresh report on the same track is not a gap.
        fresh = vessel(
            mmsi=247100031,
            lon=SARANDE_LON,
            lat=SARANDE_LAT,
            speed_kn=9.0,
            draft_m=4.0,
        )
        self.assertIsNone(
            self.corridor.detect_transponder_gap(fresh, zone, now=NOW)
        )

    # ------------------------------------------------------------- clean pass

    def test_normal_vessel_eval(self) -> None:
        """A regular vessel produces zero anomalies and a False alert flag."""
        signal = vessel(mmsi=247100040, speed_kn=6.0, draft_m=5.0, lon=VLORE_LON, lat=VLORE_LAT)
        anomalies = self.corridor.evaluate(signal, now=NOW)

        self.assertEqual(anomalies, [])
        self.assertEqual(anomaly_types(anomalies), [])

        alert = bool(anomalies)
        self.assertFalse(alert)

        # Same result through the batch/scan paths and other zones.
        self.assertEqual(self.corridor.scan([signal], now=NOW), [])
        for lon, lat, speed, draft in (
            (DURRES_LON, DURRES_LAT, 5.0, 6.0),
            (VLORE_LON, VLORE_LAT, 6.0, 5.0),
            (OTRANTO_LON, OTRANTO_LAT, 12.0, 8.0),
            (SARANDE_LON, SARANDE_LAT, 8.0, 6.0),
        ):
            with self.subTest(lon=lon, lat=lat):
                clean = vessel(
                    mmsi=247100041, lon=lon, lat=lat, speed_kn=speed, draft_m=draft
                )
                self.assertEqual(self.corridor.evaluate(clean, now=NOW), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
