"""
src/analytics/access_types.py
-----------------------------
Data classes, models, and type definitions for Stage 7: Access Analytics & Report Generation.

Features:
- Rigorous statistical models for overall access, user activity, security events, and time-based trends.
- Explicit definition of percentage denominators.
- Strict privacy guarantees: PINs, salts, hashes, and session secrets are prohibited.
- UTC ISO 8601 normalization with timezone validation.
"""

from dataclasses import dataclass, field
import datetime
import json
from typing import Any, Dict, List, Optional, Tuple


def normalize_iso_timestamp(ts_str: Optional[str], is_end: bool = False) -> Optional[str]:
    """
    Normalizes a timestamp string to an ISO 8601 UTC string.
    Supports:
    - Pure date strings ('2026-11-04'):
      - start: 2026-11-04T00:00:00+00:00
      - end:   2026-11-04T23:59:59.999999+00:00
    - Full ISO strings with or without 'Z' or timezone offsets.
    """
    if not ts_str:
        return None

    cleaned = ts_str.strip()
    if not cleaned:
        return None

    # Handle pure date (YYYY-MM-DD)
    if len(cleaned) == 10 and cleaned.count("-") == 2:
        try:
            d = datetime.date.fromisoformat(cleaned)
            if is_end:
                t = datetime.time.max
            else:
                t = datetime.time.min
            dt = datetime.datetime.combine(d, t, tzinfo=datetime.timezone.utc)
            return dt.isoformat()
        except ValueError as e:
            raise ValueError(f"Invalid date format '{ts_str}'. Expected 'YYYY-MM-DD': {e}")

    # Handle full ISO datetime string
    cleaned_iso = cleaned.replace("Z", "+00:00")
    try:
        dt = datetime.datetime.fromisoformat(cleaned_iso)
        if dt.tzinfo is None:
            # Assume UTC if not specified
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        else:
            # Convert to UTC
            dt = dt.astimezone(datetime.timezone.utc)
        return dt.isoformat()
    except ValueError as e:
        raise ValueError(f"Invalid timestamp format '{ts_str}'. Expected ISO 8601: {e}")


@dataclass
class AccessFilter:
    """
    Multi-dimensional filter criteria for access analytics and reporting.
    Guarantees strict timezone and range validation.
    """
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    user_id: Optional[str] = None
    role: Optional[str] = None
    event_type: Optional[str] = None
    result: Optional[str] = None

    def __post_init__(self):
        # Normalize timestamps
        self.start_date = normalize_iso_timestamp(self.start_date, is_end=False)
        self.end_date = normalize_iso_timestamp(self.end_date, is_end=True)

        if self.start_date and self.end_date:
            if self.start_date > self.end_date:
                raise ValueError(
                    f"Invalid date range: start_date '{self.start_date}' is later than end_date '{self.end_date}'."
                )

        if self.role:
            self.role = self.role.strip().upper()
        if self.event_type:
            self.event_type = self.event_type.strip().upper()
        if self.result:
            self.result = self.result.strip().upper()
        if self.user_id:
            self.user_id = self.user_id.strip()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "user_id": self.user_id,
            "role": self.role,
            "event_type": self.event_type,
            "result": self.result,
        }


@dataclass
class OverallAccessMetrics:
    """
    Aggregate access metrics across all authentication attempts.
    Denominator is strictly defined as N_total = N_success + N_failure.
    """
    total_attempts: int = 0
    successful_attempts: int = 0
    failed_attempts: int = 0
    success_rate_pct: float = 0.0
    denial_rate_pct: float = 0.0
    denominator_definition: str = (
        "Total authentication attempts (Successful + Failed authentication decisions)"
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_attempts": self.total_attempts,
            "successful_attempts": self.successful_attempts,
            "failed_attempts": self.failed_attempts,
            "success_rate_pct": round(self.success_rate_pct, 2),
            "denial_rate_pct": round(self.denial_rate_pct, 2),
            "denominator_definition": self.denominator_definition,
        }


@dataclass
class UserActivityMetrics:
    """Per-user access activity metrics."""
    user_id: str
    username: str
    role: Optional[str]
    total_attempts: int = 0
    successful_attempts: int = 0
    failed_attempts: int = 0
    success_rate_pct: float = 0.0
    last_access_time: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "username": self.username,
            "role": self.role,
            "total_attempts": self.total_attempts,
            "successful_attempts": self.successful_attempts,
            "failed_attempts": self.failed_attempts,
            "success_rate_pct": round(self.success_rate_pct, 2),
            "last_access_time": self.last_access_time,
        }


@dataclass
class SecurityActivityMetrics:
    """Critical security events and anomaly counters."""
    lockout_count: int = 0
    repeated_failed_auth_count: int = 0
    emergency_stop_count: int = 0
    unauthorized_admin_operations: int = 0
    guest_account_expiration_events: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lockout_count": self.lockout_count,
            "repeated_failed_auth_count": self.repeated_failed_auth_count,
            "emergency_stop_count": self.emergency_stop_count,
            "unauthorized_admin_operations": self.unauthorized_admin_operations,
            "guest_account_expiration_events": self.guest_account_expiration_events,
        }


@dataclass
class DailyTrend:
    """Daily aggregated authentication trend record."""
    date: str  # YYYY-MM-DD
    total_attempts: int = 0
    successful_attempts: int = 0
    failed_attempts: int = 0
    success_rate_pct: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "date": self.date,
            "total_attempts": self.total_attempts,
            "successful_attempts": self.successful_attempts,
            "failed_attempts": self.failed_attempts,
            "success_rate_pct": round(self.success_rate_pct, 2),
        }


@dataclass
class HourlyTrend:
    """Hourly aggregated access distribution (0-23 UTC)."""
    hour: int
    hour_label: str
    total_attempts: int = 0
    successful_attempts: int = 0
    failed_attempts: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hour": self.hour,
            "hour_label": self.hour_label,
            "total_attempts": self.total_attempts,
            "successful_attempts": self.successful_attempts,
            "failed_attempts": self.failed_attempts,
        }


@dataclass
class WeeklyTrend:
    """Weekly aggregated access trend (YYYY-Www)."""
    week: str
    total_attempts: int = 0
    successful_attempts: int = 0
    failed_attempts: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "week": self.week,
            "total_attempts": self.total_attempts,
            "successful_attempts": self.successful_attempts,
            "failed_attempts": self.failed_attempts,
        }


@dataclass
class PeakPeriodMetrics:
    """Identified peak access periods."""
    peak_hour: Dict[str, Any] = field(default_factory=lambda: {"hour": None, "label": "N/A", "attempts": 0})
    peak_day: Dict[str, Any] = field(default_factory=lambda: {"date": "N/A", "attempts": 0})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "peak_hour": self.peak_hour,
            "peak_day": self.peak_day,
        }


@dataclass
class TimeActivityMetrics:
    """Comprehensive time-based trend breakdowns."""
    daily_trends: List[DailyTrend] = field(default_factory=list)
    hourly_trends: List[HourlyTrend] = field(default_factory=list)
    weekly_trends: List[WeeklyTrend] = field(default_factory=list)
    peak_periods: PeakPeriodMetrics = field(default_factory=PeakPeriodMetrics)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "daily_trends": [d.to_dict() for d in self.daily_trends],
            "hourly_trends": [h.to_dict() for h in self.hourly_trends],
            "weekly_trends": [w.to_dict() for w in self.weekly_trends],
            "peak_periods": self.peak_periods.to_dict(),
        }


@dataclass
class AccessRecord:
    """
    Sanitized individual access audit record.
    Strictly excludes sensitive authentication secrets (PINs, salts, hashes, tokens).
    """
    event_id: int
    timestamp: str
    event_type: str
    user_id: Optional[str]
    username: Optional[str]
    role: Optional[str]
    result: str
    reason: str
    lock_state: str
    source: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "user_id": self.user_id,
            "username": self.username,
            "role": self.role,
            "result": self.result,
            "reason": self.reason,
            "lock_state": self.lock_state,
            "source": self.source,
        }

    def to_csv_row(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "user_id": self.user_id or "",
            "username": self.username or "",
            "role": self.role or "",
            "result": self.result,
            "reason": self.reason,
            "lock_state": self.lock_state,
            "source": self.source,
        }


@dataclass
class AccessAnalyticsSummary:
    """Unified summary of access analytics and security telemetry."""
    overall: OverallAccessMetrics = field(default_factory=OverallAccessMetrics)
    security: SecurityActivityMetrics = field(default_factory=SecurityActivityMetrics)
    user_activity: List[UserActivityMetrics] = field(default_factory=list)
    most_active_users: List[UserActivityMetrics] = field(default_factory=list)
    time_activity: TimeActivityMetrics = field(default_factory=TimeActivityMetrics)
    applied_filters: AccessFilter = field(default_factory=AccessFilter)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall": self.overall.to_dict(),
            "security": self.security.to_dict(),
            "user_activity": [u.to_dict() for u in self.user_activity],
            "most_active_users": [u.to_dict() for u in self.most_active_users],
            "time_activity": self.time_activity.to_dict(),
            "applied_filters": self.applied_filters.to_dict(),
        }


@dataclass
class AccessReport:
    """
    Complete downloadable access analytics report.
    Includes metadata, summary metrics, applied filters, and sanitized detailed records.
    """
    report_id: str
    generated_at: str
    selected_date_range: Dict[str, Optional[str]]
    applied_filters: Dict[str, Any]
    summary_metrics: AccessAnalyticsSummary
    detailed_records: List[AccessRecord]

    def to_dict(self) -> Dict[str, Any]:
        records_list = [r.to_dict() for r in self.detailed_records]
        return {
            "report_metadata": {
                "report_id": self.report_id,
                "generated_at": self.generated_at,
                "system": "RTOS Voice-Controlled Smart Lock Access Analytics",
                "selected_date_range": self.selected_date_range,
                "applied_filters": self.applied_filters,
                "record_count": len(self.detailed_records),
            },
            "summary_metrics": self.summary_metrics.to_dict(),
            "overall_metrics": self.summary_metrics.overall.to_dict(),
            "detailed_access_records": records_list,
            "records": records_list,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serializes report to structured JSON format."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_csv(self) -> str:
        """
        Serializes report to clean RFC 4180 CSV with descriptive header metadata.
        Comment lines (#) allow direct parsing by CSV libraries (e.g. pandas comment='#').
        """
        import io
        import csv

        out = io.StringIO()
        out.write("# ==============================================================================\n")
        out.write("# RTOS Smart Lock Access Analytics Report\n")
        out.write("# RTOS Access Control Security Audit Report\n")
        out.write(f"# Report ID: {self.report_id}\n")
        out.write(f"# Generated At (UTC): {self.generated_at}\n")
        out.write(f"# Date Range: {self.selected_date_range.get('start') or 'Earliest'} to {self.selected_date_range.get('end') or 'Latest'}\n")
        out.write(f"# Applied Filters: user_id={self.applied_filters.get('user_id')}, role={self.applied_filters.get('role')}, event_type={self.applied_filters.get('event_type')}, result={self.applied_filters.get('result')}\n")
        out.write(f"# Summary: Total Attempts={self.summary_metrics.overall.total_attempts}, Success={self.summary_metrics.overall.successful_attempts}, Failures={self.summary_metrics.overall.failed_attempts}, Success Rate={self.summary_metrics.overall.success_rate_pct:.2f}%, Denial Rate={self.summary_metrics.overall.denial_rate_pct:.2f}%\n")
        out.write(f"# Denominator: {self.summary_metrics.overall.denominator_definition}\n")
        out.write(f"# Security Events: Lockouts={self.summary_metrics.security.lockout_count}, Emergency Stops={self.summary_metrics.security.emergency_stop_count}, Repeated Failures={self.summary_metrics.security.repeated_failed_auth_count}, Unauthorized Ops={self.summary_metrics.security.unauthorized_admin_operations}, Guest Expirations={self.summary_metrics.security.guest_account_expiration_events}\n")
        out.write("# ==============================================================================\n")

        fieldnames = [
            "event_id",
            "timestamp",
            "event_type",
            "user_id",
            "username",
            "role",
            "result",
            "reason",
            "lock_state",
            "source",
        ]
        writer = csv.DictWriter(out, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for r in self.detailed_records:
            writer.writerow(r.to_csv_row())

        return out.getvalue()

