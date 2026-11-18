"""
src/analytics/access_analytics.py
---------------------------------
Access Analytics and Report Generation Engine.

Features:
- Aggregates security audit database into structured access statistics.
- Strict denominator definitions: N_total = N_success + N_failure for authentication attempts.
- Separates authentication events from system/lifecycle/lock events.
- Multi-dimensional filtering by Start Date, End Date, User, Role, Event Type, and Result.
- User activity breakdowns and ranking of most active users.
- Security event analysis (Lockouts, Emergency Stops, Repeated Failures, Unauthorized Operations).
- Hourly (00-23 UTC), daily, and weekly trend analysis with peak period detection.
- Clean CSV and JSON report generation with guaranteed sanitization of credentials and secrets.
"""

from collections import defaultdict
import datetime
import json
import logging
import os
import pathlib
import sqlite3
import threading
from typing import Any, Dict, List, Optional, Tuple

from src.analytics.access_types import (
    AccessFilter,
    AccessRecord,
    AccessReport,
    AccessAnalyticsSummary,
    DailyTrend,
    HourlyTrend,
    OverallAccessMetrics,
    PeakPeriodMetrics,
    SecurityActivityMetrics,
    TimeActivityMetrics,
    UserActivityMetrics,
    WeeklyTrend,
)
from src.security.audit_logger import AuditLogger, AuditEventType, AuditResult

logger = logging.getLogger("rtos.access_analytics")

# Event types that represent authentication decisions
AUTH_ATTEMPT_EVENT_TYPES = (
    AuditEventType.AUTH_SUCCESS.value,
    AuditEventType.USER_ACCESS_GRANTED.value,
    AuditEventType.AUTH_FAILURE.value,
    AuditEventType.USER_ACCESS_DENIED.value,
    AuditEventType.AUTH_TIMEOUT.value,
)

AUTH_SUCCESS_EVENT_TYPES = (
    AuditEventType.AUTH_SUCCESS.value,
    AuditEventType.USER_ACCESS_GRANTED.value,
)

AUTH_FAILURE_EVENT_TYPES = (
    AuditEventType.AUTH_FAILURE.value,
    AuditEventType.USER_ACCESS_DENIED.value,
    AuditEventType.AUTH_TIMEOUT.value,
)


class AccessAnalyticsManager:
    """
    Subsystem manager for access analytics aggregation and report generation.
    Connects to the persistent security audit database and user manager.
    """

    def __init__(
        self,
        db_path: str | pathlib.Path = "data/security_audit.db",
        user_manager: Optional[Any] = None,
        audit_logger: Optional[AuditLogger] = None,
        output_dir: Optional[str | pathlib.Path] = "results/reports",
    ):
        self.db_path = pathlib.Path(db_path)
        self.user_manager = user_manager
        self.audit_logger = audit_logger
        self.output_dir = pathlib.Path(output_dir or "results/reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _get_read_connection(self) -> Optional[sqlite3.Connection]:
        """Opens an isolated read-only connection to the audit database."""
        if not self.db_path.exists():
            return None
        try:
            conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True, timeout=3.0)
            conn.row_factory = sqlite3.Row
            return conn
        except sqlite3.OperationalError:
            # Fallback if URI read-only mode is unsupported
            try:
                conn = sqlite3.connect(str(self.db_path), timeout=3.0)
                conn.row_factory = sqlite3.Row
                return conn
            except Exception as e:
                logger.error(f"[ACCESS ANALYTICS] Failed to connect to database '{self.db_path}': {e}")
                return None

    def _get_user_role_map(self) -> Dict[str, str]:
        """Caches user ID to Role string mapping from UserManager if available."""
        role_map: Dict[str, str] = {}
        if self.user_manager is not None and hasattr(self.user_manager, "get_all_users"):
            try:
                users = self.user_manager.get_all_users()
                for u in users:
                    role_val = u.role.value if hasattr(u.role, "value") else str(u.role)
                    role_map[u.user_id] = role_val
            except Exception as e:
                logger.debug(f"[ACCESS ANALYTICS] Error querying UserManager for roles: {e}")
        return role_map

    def _flush_audit_logger(self):
        """Flushes pending in-flight audit records before generating analytics."""
        if self.audit_logger is not None and hasattr(self.audit_logger, "flush"):
            try:
                self.audit_logger.flush(timeout=0.1)
            except Exception as e:
                logger.debug(f"[ACCESS ANALYTICS] Flush error: {e}")

    # ==========================================================================
    # Aggregation & Analytics Core
    # ==========================================================================

    def get_summary(self, filters: Optional[AccessFilter] = None) -> AccessAnalyticsSummary:
        """Alias for compute_summary() returning current access metrics and trends."""
        return self.compute_summary(filters=filters)

    def compute_summary(self, filters: Optional[AccessFilter] = None) -> AccessAnalyticsSummary:
        """
        Computes overall access metrics, user activity, security activity,
        and time-based trends across the filtered dataset.
        """
        self._flush_audit_logger()
        f = filters or AccessFilter()
        conn = self._get_read_connection()

        if conn is None:
            # Return empty models if database does not exist yet
            return AccessAnalyticsSummary(applied_filters=f)

        role_map = self._get_user_role_map()

        try:
            with conn:
                # 1. Fetch all raw events matching basic filters
                query_sql = """
                SELECT event_id, timestamp, event_type, user_id, username, result, reason, lock_state, source, metadata
                FROM audit_events
                WHERE 1=1
                """
                params: List[Any] = []

                if f.start_date:
                    query_sql += " AND timestamp >= ?"
                    params.append(f.start_date)
                if f.end_date:
                    query_sql += " AND timestamp <= ?"
                    params.append(f.end_date)
                if f.user_id:
                    query_sql += " AND user_id = ?"
                    params.append(f.user_id)
                if f.event_type:
                    query_sql += " AND event_type = ?"
                    params.append(f.event_type)
                if f.result:
                    query_sql += " AND result = ?"
                    params.append(f.result)

                query_sql += " ORDER BY event_id ASC;"
                cursor = conn.cursor()
                cursor.execute(query_sql, params)
                all_rows = cursor.fetchall()

            # 2. Filter by Role in Python (supporting metadata JSON and UserManager)
            filtered_rows = []
            for row in all_rows:
                u_id = row["user_id"]
                row_meta = {}
                if row["metadata"]:
                    try:
                        row_meta = json.loads(row["metadata"])
                    except Exception:
                        pass

                row_role = row_meta.get("role") or role_map.get(u_id)
                if not row_role and u_id:
                    if "admin" in u_id.lower() or "administrator" in str(row["username"]).lower():
                        row_role = "ADMIN"
                    elif "guest" in u_id.lower():
                        row_role = "GUEST"
                    else:
                        row_role = "REGULAR_USER"

                if f.role and (not row_role or row_role.upper() != f.role):
                    continue

                filtered_rows.append((row, row_role))

            # 3. Overall Access Aggregation
            total_attempts = 0
            successful_attempts = 0
            failed_attempts = 0

            # 4. User Activity Aggregation
            user_stats = defaultdict(lambda: {
                "username": "Unknown",
                "role": None,
                "total": 0,
                "success": 0,
                "fail": 0,
                "last_access": None,
            })

            # 5. Security Activity Aggregation
            lockout_count = 0
            emergency_stop_count = 0
            unauthorized_admin_ops = 0
            guest_expirations = 0

            # Repeated failure tracker (consecutive failures >= 3)
            repeated_failed_auth_count = 0
            consecutive_failures = 0

            # 6. Time Activity Aggregation
            daily_stats = defaultdict(lambda: {"total": 0, "success": 0, "fail": 0})
            hourly_stats = {h: {"total": 0, "success": 0, "fail": 0} for h in range(24)}
            weekly_stats = defaultdict(lambda: {"total": 0, "success": 0, "fail": 0})

            for row, role_val in filtered_rows:
                e_type = row["event_type"]
                res = row["result"]
                u_id = row["user_id"] or "system"
                u_name = row["username"] or "Unknown"
                ts = row["timestamp"]

                # Track security-specific events
                if e_type == AuditEventType.LOCKOUT_ACTIVATED.value:
                    lockout_count += 1
                elif e_type in (AuditEventType.EMERGENCY_STOP.value, AuditEventType.EMERGENCY_DASHBOARD_TRIGGER.value):
                    emergency_stop_count += 1
                elif e_type == AuditEventType.UNAUTHORIZED_OP.value:
                    unauthorized_admin_ops += 1
                elif e_type == AuditEventType.GUEST_ACCOUNT_EXPIRED.value:
                    guest_expirations += 1

                # Check if this row represents an authentication attempt
                is_auth_event = (
                    e_type in AUTH_ATTEMPT_EVENT_TYPES
                    or (f.event_type and f.event_type in (e_type, res))
                )

                if is_auth_event:
                    total_attempts += 1
                    is_success = (res == AuditResult.SUCCESS.value or e_type in AUTH_SUCCESS_EVENT_TYPES)
                    is_failure = (res == AuditResult.FAILURE.value or e_type in AUTH_FAILURE_EVENT_TYPES)

                    if is_success:
                        successful_attempts += 1
                        consecutive_failures = 0
                    elif is_failure:
                        failed_attempts += 1
                        consecutive_failures += 1
                        if consecutive_failures >= 3:
                            repeated_failed_auth_count += 1
                            consecutive_failures = 0  # reset for next sequence

                    # User Activity
                    u_entry = user_stats[u_id]
                    u_entry["username"] = u_name
                    u_entry["role"] = role_val or u_entry["role"]
                    u_entry["total"] += 1
                    if is_success:
                        u_entry["success"] += 1
                    elif is_failure:
                        u_entry["fail"] += 1
                    u_entry["last_access"] = ts  # chronological

                    # Time breakdown
                    try:
                        # Extract date string YYYY-MM-DD
                        day_str = ts[:10]
                        daily_stats[day_str]["total"] += 1
                        if is_success:
                            daily_stats[day_str]["success"] += 1
                        elif is_failure:
                            daily_stats[day_str]["fail"] += 1

                        # Extract hour int 0-23
                        # Supports '2026-11-04T20:37:50...'
                        if "T" in ts:
                            h_int = int(ts.split("T")[1][:2])
                        elif " " in ts:
                            h_int = int(ts.split(" ")[1][:2])
                        else:
                            h_int = 0

                        if 0 <= h_int <= 23:
                            hourly_stats[h_int]["total"] += 1
                            if is_success:
                                hourly_stats[h_int]["success"] += 1
                            elif is_failure:
                                hourly_stats[h_int]["fail"] += 1

                        # Weekly ISO format
                        dt = datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        iso_yr, iso_wk, _ = dt.isocalendar()
                        wk_str = f"{iso_yr}-W{iso_wk:02d}"
                        weekly_stats[wk_str]["total"] += 1
                        if is_success:
                            weekly_stats[wk_str]["success"] += 1
                        elif is_failure:
                            weekly_stats[wk_str]["fail"] += 1

                    except Exception as ex:
                        logger.debug(f"[ACCESS ANALYTICS] Timestamp parse error for '{ts}': {ex}")

            # Compute percentage rates
            succ_rate = (successful_attempts / total_attempts * 100.0) if total_attempts > 0 else 0.0
            denial_rate = (failed_attempts / total_attempts * 100.0) if total_attempts > 0 else 0.0

            overall = OverallAccessMetrics(
                total_attempts=total_attempts,
                successful_attempts=successful_attempts,
                failed_attempts=failed_attempts,
                success_rate_pct=succ_rate,
                denial_rate_pct=denial_rate,
            )

            # Build User Activity list
            user_list: List[UserActivityMetrics] = []
            for uid, udata in user_stats.items():
                u_tot = udata["total"]
                u_succ = udata["success"]
                u_rate = (u_succ / u_tot * 100.0) if u_tot > 0 else 0.0
                user_list.append(UserActivityMetrics(
                    user_id=uid,
                    username=udata["username"],
                    role=udata["role"],
                    total_attempts=u_tot,
                    successful_attempts=u_succ,
                    failed_attempts=udata["fail"],
                    success_rate_pct=u_rate,
                    last_access_time=udata["last_access"],
                ))

            # Sort most active users descending by total attempts
            most_active = sorted(user_list, key=lambda x: x.total_attempts, reverse=True)

            # Build Security Activity
            security = SecurityActivityMetrics(
                lockout_count=lockout_count,
                repeated_failed_auth_count=repeated_failed_auth_count,
                emergency_stop_count=emergency_stop_count,
                unauthorized_admin_operations=unauthorized_admin_ops,
                guest_account_expiration_events=guest_expirations,
            )

            # Build Daily Trends
            daily_trends: List[DailyTrend] = []
            for d_str in sorted(daily_stats.keys()):
                d_data = daily_stats[d_str]
                d_tot = d_data["total"]
                d_succ = d_data["success"]
                d_rate = (d_succ / d_tot * 100.0) if d_tot > 0 else 0.0
                daily_trends.append(DailyTrend(
                    date=d_str,
                    total_attempts=d_tot,
                    successful_attempts=d_succ,
                    failed_attempts=d_data["fail"],
                    success_rate_pct=d_rate,
                ))

            # Build Hourly Trends (0 to 23)
            hourly_trends: List[HourlyTrend] = []
            for h in range(24):
                h_data = hourly_stats[h]
                h_label = f"{h:02d}:00 - {(h+1)%24:02d}:00 UTC"
                hourly_trends.append(HourlyTrend(
                    hour=h,
                    hour_label=h_label,
                    total_attempts=h_data["total"],
                    successful_attempts=h_data["success"],
                    failed_attempts=h_data["fail"],
                ))

            # Build Weekly Trends
            weekly_trends: List[WeeklyTrend] = []
            for w_str in sorted(weekly_stats.keys()):
                w_data = weekly_stats[w_str]
                weekly_trends.append(WeeklyTrend(
                    week=w_str,
                    total_attempts=w_data["total"],
                    successful_attempts=w_data["success"],
                    failed_attempts=w_data["fail"],
                ))

            # Peak Period Detection
            peak_hour_entry = max(hourly_trends, key=lambda h: h.total_attempts, default=None)
            peak_day_entry = max(daily_trends, key=lambda d: d.total_attempts, default=None)

            peak_periods = PeakPeriodMetrics(
                peak_hour={
                    "hour": peak_hour_entry.hour if peak_hour_entry and peak_hour_entry.total_attempts > 0 else None,
                    "label": peak_hour_entry.hour_label if peak_hour_entry and peak_hour_entry.total_attempts > 0 else "N/A",
                    "attempts": peak_hour_entry.total_attempts if peak_hour_entry else 0,
                },
                peak_day={
                    "date": peak_day_entry.date if peak_day_entry and peak_day_entry.total_attempts > 0 else "N/A",
                    "attempts": peak_day_entry.total_attempts if peak_day_entry else 0,
                },
            )

            time_activity = TimeActivityMetrics(
                daily_trends=daily_trends,
                hourly_trends=hourly_trends,
                weekly_trends=weekly_trends,
                peak_periods=peak_periods,
            )

            return AccessAnalyticsSummary(
                overall=overall,
                security=security,
                user_activity=user_list,
                most_active_users=most_active,
                time_activity=time_activity,
                applied_filters=f,
            )

        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    # ==========================================================================
    # Detailed Records Query & Sanitization
    # ==========================================================================

    def query_detailed_records(
        self,
        filters: Optional[AccessFilter] = None,
        limit: int = 5000,
        offset: int = 0,
    ) -> List[AccessRecord]:
        """
        Retrieves sanitized individual access records matching the filter criteria.
        Guarantees that credentials, hashes, and session secrets are omitted.
        """
        self._flush_audit_logger()
        f = filters or AccessFilter()
        conn = self._get_read_connection()
        if conn is None:
            return []

        role_map = self._get_user_role_map()

        try:
            with conn:
                query_sql = """
                SELECT event_id, timestamp, event_type, user_id, username, result, reason, lock_state, source, metadata
                FROM audit_events
                WHERE 1=1
                """
                params: List[Any] = []

                if f.start_date:
                    query_sql += " AND timestamp >= ?"
                    params.append(f.start_date)
                if f.end_date:
                    query_sql += " AND timestamp <= ?"
                    params.append(f.end_date)
                if f.user_id:
                    query_sql += " AND user_id = ?"
                    params.append(f.user_id)
                if f.event_type:
                    query_sql += " AND event_type = ?"
                    params.append(f.event_type)
                if f.result:
                    query_sql += " AND result = ?"
                    params.append(f.result)

                query_sql += " ORDER BY event_id DESC LIMIT ? OFFSET ?;"
                params.extend([limit, offset])

                cursor = conn.cursor()
                cursor.execute(query_sql, params)
                rows = cursor.fetchall()

            records: List[AccessRecord] = []
            for r in rows:
                u_id = r["user_id"]
                row_meta = {}
                if r["metadata"]:
                    try:
                        row_meta = json.loads(r["metadata"])
                    except Exception:
                        pass

                row_role = row_meta.get("role") or role_map.get(u_id)
                if not row_role and u_id:
                    if "admin" in u_id.lower() or "administrator" in str(r["username"]).lower():
                        row_role = "ADMIN"
                    elif "guest" in u_id.lower():
                        row_role = "GUEST"
                    else:
                        row_role = "REGULAR_USER"

                if f.role and (not row_role or row_role.upper() != f.role):
                    continue

                records.append(AccessRecord(
                    event_id=r["event_id"],
                    timestamp=r["timestamp"],
                    event_type=r["event_type"],
                    user_id=r["user_id"],
                    username=r["username"],
                    role=row_role,
                    result=r["result"],
                    reason=r["reason"],
                    lock_state=r["lock_state"],
                    source=r["source"],
                ))

            return records

        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    # ==========================================================================
    # Report Generation (JSON & CSV)
    # ==========================================================================

    def generate_report(
        self,
        filters: Optional[AccessFilter] = None,
        limit: int = 5000,
        fmt: Optional[str] = None,
    ) -> AccessReport:
        """
        Builds a comprehensive AccessReport containing summary metrics,
        filter metadata, and sanitized detailed records.
        """
        f = filters or AccessFilter()
        summary = self.compute_summary(filters=f)
        detailed_records = self.query_detailed_records(filters=f, limit=limit)

        now_utc = datetime.datetime.now(datetime.timezone.utc)
        rep_id = f"rep_{now_utc.strftime('%Y%m%d_%H%M%S')}_{os.urandom(2).hex()}"

        return AccessReport(
            report_id=rep_id,
            generated_at=now_utc.isoformat(),
            selected_date_range={
                "start": f.start_date,
                "end": f.end_date,
            },
            applied_filters=f.to_dict(),
            summary_metrics=summary,
            detailed_records=detailed_records,
        )

    def export_report_files(
        self,
        report: AccessReport,
        base_filename: Optional[str] = None,
    ) -> Tuple[pathlib.Path, pathlib.Path]:
        """
        Exports the AccessReport to both JSON and CSV files on disk.
        Returns paths to (json_filepath, csv_filepath).
        """
        fname = base_filename or report.report_id
        json_path = self.output_dir / f"{fname}.json"
        csv_path = self.output_dir / f"{fname}.csv"

        with open(json_path, "w", encoding="utf-8") as f:
            f.write(report.to_json(indent=2))

        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            f.write(report.to_csv())

        logger.info(f"[ACCESS ANALYTICS] Exported reports to '{json_path}' and '{csv_path}'.")
        return json_path, csv_path

