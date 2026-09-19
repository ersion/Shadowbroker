"""
Adriatic/Ionian maritime corridor filter for AIS vessel signals.

Loads the corridor bounding-box config (backend/config/adriatic_corridor.json),
resolves vessel coordinates to corridor zones and evaluates AIS reports for
navigational anomalies:

  ANOMALOUS_STOP      vessel stopped/drifting in open sea
  HIGH_SPEED_TRANSIT  speed above the zone speed limit / tactical speed
  DRAFT_HAZARD        reported draft exceeds the safe depth of the zone
  TRANSPONDER_GAP     AIS report older than the zone gap threshold

Pure stdlib, no network or database access — safe to import from tests.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "adriatic_corridor.json"

# Anomaly types emitted by the filter.
ANOMALOUS_STOP = "ANOMALOUS_STOP"
HIGH_SPEED_TRANSIT = "HIGH_SPEED_TRANSIT"
DRAFT_HAZARD = "DRAFT_HAZARD"
TRANSPONDER_GAP = "TRANSPONDER_GAP"

# Zones whose interior is bounded or pilotage water rather than open sea: a
# stationary vessel inside these is considered berthed/anchored/awaiting berth,
# not drifting in open sea. A port approach is included because vessels legally
# anchor there waiting for a berth; open-sea zones (straits, transit channels)
# are where a dead-stop vessel is the anomaly.
ENCLOSED_ZONE_KINDS = {"port_approach", "bay", "harbor", "harbour"}

# Fields that are optional on a vessel signal; "unknown" never triggers an
# anomaly because a missing value is not evidence of a violation.
_UNKNOWN = (None, "", "unknown")


@dataclass(frozen=True)
class CorridorZone:
    """A single bounding-box zone of the maritime corridor."""

    zone_id: str
    name: str
    bbox: tuple[float, float, float, float]  # min_lon, min_lat, max_lon, max_lat
    speed_limit_kn: float
    max_draft_m: float
    min_safe_depth_m: float
    tactical_speed_kn: float
    anomaly_thresholds: Mapping[str, Any] = field(default_factory=dict)
    kind: str = "open_sea"

    @property
    def min_lon(self) -> float:
        return self.bbox[0]

    @property
    def min_lat(self) -> float:
        return self.bbox[1]

    @property
    def max_lon(self) -> float:
        return self.bbox[2]

    @property
    def max_lat(self) -> float:
        return self.bbox[3]

    def contains(self, lon: float, lat: float) -> bool:
        """True when (lon, lat) falls inside the zone bounding box (inclusive)."""
        return (
            self.min_lon <= lon <= self.max_lon
            and self.min_lat <= lat <= self.max_lat
        )

    @property
    def is_enclosed(self) -> bool:
        return self.kind in ENCLOSED_ZONE_KINDS

    def to_dict(self) -> dict[str, Any]:
        return {
            "zone_id": self.zone_id,
            "name": self.name,
            "bbox": list(self.bbox),
            "speed_limit_kn": self.speed_limit_kn,
            "max_draft_m": self.max_draft_m,
            "min_safe_depth_m": self.min_safe_depth_m,
            "tactical_speed_kn": self.tactical_speed_kn,
            "kind": self.kind,
        }


@dataclass(frozen=True)
class Anomaly:
    """A detected anomaly tied to a vessel and (optionally) a corridor zone."""

    anomaly_type: str
    mmsi: Any
    zone_id: Optional[str]
    severity: str
    detail: str
    observed_value: Optional[float] = None
    threshold_value: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "anomaly_type": self.anomaly_type,
            "mmsi": self.mmsi,
            "zone_id": self.zone_id,
            "severity": self.severity,
            "detail": self.detail,
            "observed_value": self.observed_value,
            "threshold_value": self.threshold_value,
        }


class ZoneConfigError(ValueError):
    """Raised when the corridor config file is missing or malformed."""


def _parse_timestamp(value: Any) -> Optional[datetime]:
    """Parse an AIS timestamp to an aware UTC datetime, or None if unusable."""
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, (int, float)):
        dt = datetime.fromtimestamp(float(value), tz=timezone.utc)
    elif isinstance(value, str) and value.strip():
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None
    else:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _as_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_unknown(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in ("", "unknown", "n/a", "nan")
    return False


class AdriaticRadarFilter:
    """Evaluate AIS vessel signals against the Adriatic corridor zones."""

    def __init__(
        self,
        config_path: Optional[str | Path] = None,
        config: Optional[Mapping[str, Any]] = None,
        clock: Optional[Any] = None,
    ) -> None:
        if config is not None:
            self._raw = dict(config)
        else:
            path = Path(config_path) if config_path is not None else CONFIG_PATH
            if not path.exists():
                raise ZoneConfigError(f"corridor config not found: {path}")
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    self._raw = json.load(handle)
            except json.JSONDecodeError as exc:
                raise ZoneConfigError(f"corridor config is not valid JSON: {exc}") from exc
        self.config_path = str(config_path or CONFIG_PATH)
        self.zones: list[CorridorZone] = self._parse_zones(self._raw)
        self.defaults: dict[str, Any] = dict(self._raw.get("defaults", {}))
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        logger.debug(
            "AdriaticRadarFilter loaded %d corridor zones from %s",
            len(self.zones),
            self.config_path,
        )

    # ------------------------------------------------------------------ config

    @staticmethod
    def _parse_zones(raw: Mapping[str, Any]) -> list[CorridorZone]:
        zones: list[CorridorZone] = []
        for entry in raw.get("zones", []):
            bbox = entry.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                raise ZoneConfigError(f"zone {entry.get('zone_id')!r} has an invalid bbox")
            zones.append(
                CorridorZone(
                    zone_id=str(entry["zone_id"]),
                    name=str(entry.get("name", entry["zone_id"])),
                    bbox=tuple(float(v) for v in bbox),  # type: ignore[arg-type]
                    speed_limit_kn=float(entry.get("speed_limit_kn", 0.0)),
                    max_draft_m=float(entry.get("max_draft_m", 0.0)),
                    min_safe_depth_m=float(entry.get("min_safe_depth_m", 0.0)),
                    tactical_speed_kn=float(entry.get("tactical_speed_kn", 0.0)),
                    anomaly_thresholds=dict(entry.get("anomaly_thresholds", {})),
                    kind=str(entry.get("kind", "open_sea")),
                )
            )
        if not zones:
            raise ZoneConfigError("corridor config contains no zones")
        return zones

    def get_zone(self, zone_id: str) -> Optional[CorridorZone]:
        for zone in self.zones:
            if zone.zone_id == zone_id:
                return zone
        return None

    def _threshold(self, zone: Optional[CorridorZone], key: str, default: Any) -> Any:
        values = [self.defaults.get(key)]
        if zone is not None:
            values.append(zone.anomaly_thresholds.get(key))
            values.append(getattr(zone, {"min_safe_depth_m": "min_safe_depth_m"}.get(key, ""), None))
        for value in values:
            if value is not None:
                return value
        return default

    # -------------------------------------------------------------- geometry

    def match_zone(self, lon: float, lat: float) -> Optional[CorridorZone]:
        """Return the zone containing (lon, lat), smallest area first."""
        matches = [z for z in self.zones if z.contains(lon, lat)]
        if not matches:
            return None
        return min(matches, key=lambda z: (z.max_lon - z.min_lon) * (z.max_lat - z.min_lat))

    def match_zones(self, lon: float, lat: float) -> list[CorridorZone]:
        """Return every zone containing (lon, lat), smallest area first."""
        matches = [z for z in self.zones if z.contains(lon, lat)]
        matches.sort(key=lambda z: (z.max_lon - z.min_lon) * (z.max_lat - z.min_lat))
        return matches

    def is_in_corridor(self, lon: float, lat: float) -> bool:
        return self.match_zone(lon, lat) is not None

    @staticmethod
    def _coords(signal: Mapping[str, Any]) -> tuple[Optional[float], Optional[float]]:
        lon = _as_float(signal.get("lon", signal.get("longitude")))
        lat = _as_float(signal.get("lat", signal.get("latitude")))
        return lon, lat

    @staticmethod
    def _mmsi(signal: Mapping[str, Any]) -> Any:
        for key in ("mmsi", "MMSI", "vessel_id"):
            if signal.get(key) not in (None, ""):
                return signal[key]
        return None

    # -------------------------------------------------------------- detection

    def detect_anomalous_stop(
        self,
        signal: Mapping[str, Any],
        zone: Optional[CorridorZone] = None,
        now: Optional[datetime] = None,
    ) -> Optional[Anomaly]:
        """Flag a vessel stopped in open sea (not a port approach, bay or berth)."""
        if zone is None or zone.is_enclosed:
            return None
        speed = _as_float(signal.get("speed_kn", signal.get("sog")))
        if speed is None or speed > float(self._threshold(zone, "stop_speed_kn", 0.5)):
            return None
        if bool(signal.get("at_berth")):
            return None
        return Anomaly(
            anomaly_type=ANOMALOUS_STOP,
            mmsi=self._mmsi(signal),
            zone_id=zone.zone_id,
            severity="high",
            detail=(
                f"vessel stopped in open-sea zone {zone.zone_id} "
                f"at {speed:.1f} kn (limit {self._threshold(zone, 'stop_speed_kn', 0.5)} kn)"
            ),
            observed_value=speed,
        )

    def detect_high_speed_transit(
        self,
        signal: Mapping[str, Any],
        zone: CorridorZone,
        tactical: bool = False,
    ) -> Optional[Anomaly]:
        """Flag speed above the zone speed limit (or its tactical limit)."""
        speed = _as_float(signal.get("speed_kn", signal.get("sog")))
        if speed is None:
            return None
        if tactical and zone.tactical_speed_kn:
            limit = zone.tactical_speed_kn
            label = "tactical"
        else:
            limit = zone.speed_limit_kn
            label = "zone"
        if speed <= limit:
            return None
        return Anomaly(
            anomaly_type=HIGH_SPEED_TRANSIT,
            mmsi=self._mmsi(signal),
            zone_id=zone.zone_id,
            severity="high",
            detail=(
                f"{speed:.1f} kn exceeds {label} speed limit {limit:.1f} kn "
                f"in {zone.zone_id}"
            ),
            observed_value=speed,
            threshold_value=limit,
        )

    def detect_draft_hazard(
        self,
        signal: Mapping[str, Any],
        zone: CorridorZone,
    ) -> Optional[Anomaly]:
        """Flag a vessel whose draft exceeds the safe depth of the zone."""
        draft = _as_float(signal.get("draft_m", signal.get("draught")))
        if draft is None:
            return None
        if not zone.max_draft_m or draft <= zone.max_draft_m:
            return None
        severity = "critical" if zone.min_safe_depth_m and draft > zone.min_safe_depth_m else "high"
        return Anomaly(
            anomaly_type=DRAFT_HAZARD,
            mmsi=self._mmsi(signal),
            zone_id=zone.zone_id,
            severity=severity,
            detail=(
                f"draft {draft:.1f} m exceeds safe depth {zone.max_draft_m:.1f} m "
                f"for {zone.zone_id} (min safe depth {zone.min_safe_depth_m:.1f} m)"
            ),
            observed_value=draft,
            threshold_value=zone.max_draft_m,
        )

    def detect_transponder_gap(
        self,
        signal: Mapping[str, Any],
        zone: CorridorZone,
        now: Optional[datetime] = None,
    ) -> Optional[Anomaly]:
        """Flag a stale/missing AIS report relative to the zone gap threshold."""
        gap_limit = float(self._threshold(zone, "transponder_gap_minutes", 20.0))
        current = now or self._clock()
        last = _parse_timestamp(
            signal.get("last_seen")
            or signal.get("timestamp")
            or signal.get("ts")
            or signal.get("time")
        )
        if last is None:
            if _is_unknown(signal.get("last_seen")):
                return Anomaly(
                    anomaly_type=TRANSPONDER_GAP,
                    mmsi=self._mmsi(signal),
                    zone_id=zone.zone_id,
                    severity="high",
                    detail=f"no AIS report timestamp for {zone.zone_id}; transponder gap",
                    threshold_value=gap_limit,
                )
            return None
        gap_minutes = (current - last).total_seconds() / 60.0
        if gap_minutes <= gap_limit:
            return None
        return Anomaly(
            anomaly_type=TRANSPONDER_GAP,
            mmsi=self._mmsi(signal),
            zone_id=zone.zone_id,
            severity="high",
            detail=(
                f"AIS report {gap_minutes:.1f} min old exceeds {gap_limit:.1f} min "
                f"gap threshold in {zone.zone_id}"
            ),
            observed_value=round(gap_minutes, 2),
            threshold_value=gap_limit,
        )

    # ------------------------------------------------------------------ public

    def evaluate(
        self,
        signal: Mapping[str, Any],
        now: Optional[datetime] = None,
    ) -> list[Anomaly]:
        """Evaluate one AIS signal and return every anomaly detected."""
        lon, lat = self._coords(signal)
        if lon is None or lat is None:
            logger.debug("AIS signal without usable coordinates: %r", signal)
            return []
        zone = self.match_zone(lon, lat)
        if zone is None:
            return []
        return self.evaluate_signals([signal], now=now)

    def evaluate_signals(
        self,
        signals: Iterable[Mapping[str, Any]],
        now: Optional[datetime] = None,
    ) -> list[Anomaly]:
        """Evaluate a batch of AIS signals against their matching zones."""
        current = now or self._clock()
        anomalies: list[Anomaly] = []
        for signal in signals:
            lon, lat = self._coords(signal)
            if lon is None or lat is None:
                continue
            for zone in self.match_zones(lon, lat):
                for check in (
                    lambda: self.detect_anomalous_stop(signal, zone, now=current),
                    lambda: self.detect_high_speed_transit(signal, zone),
                    lambda: self.detect_draft_hazard(signal, zone),
                    lambda: self.detect_transponder_gap(signal, zone, now=current),
                ):
                    try:
                        anomaly = check()
                    except Exception as exc:  # pragma: no cover - defensive
                        logger.warning("anomaly check failed: %s", exc)
                        continue
                    if anomaly is not None:
                        anomalies.append(anomaly)
        return anomalies

    def scan(
        self,
        signals: Iterable[Mapping[str, Any]],
        now: Optional[datetime] = None,
    ) -> list[dict[str, Any]]:
        """Evaluation helper returning JSON-serialisable anomaly dicts."""
        return [a.to_dict() for a in self.evaluate_signals(signals, now=now)]

    def summary(self) -> dict[str, Any]:
        """Static description of the loaded corridor configuration."""
        return {
            "config_path": self.config_path,
            "zone_count": len(self.zones),
            "zones": [zone.to_dict() for zone in self.zones],
        }


def load_filter(config_path: Optional[str | Path] = None) -> AdriaticRadarFilter:
    """Convenience loader used by routers/tests."""
    return AdriaticRadarFilter(config_path=config_path)


__all__ = [
    "AdriaticRadarFilter",
    "Anomaly",
    "CorridorZone",
    "ZoneConfigError",
    "load_filter",
    "ANOMALOUS_STOP",
    "HIGH_SPEED_TRANSIT",
    "DRAFT_HAZARD",
    "TRANSPONDER_GAP",
    "CONFIG_PATH",
]
