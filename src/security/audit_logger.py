"""
src/security/audit_logger.py
----------------------------
Asynchronous, Thread-Safe, Persistent Security Audit Logger backed by SQLite.

Responsibilities:
1. Implements a persistent, queryable security audit trail answering:
   - Who attempted access?
   - When did the attempt happen (UTC ISO8601)?
   - Was access granted or denied?
   - What was the specific reason?
   - What was the lock state (LOCKED, PIN_ENTRY, UNLOCKED, LOCKOUT)?
   - Did an administrator or user perform an operation?
   - Did a security or system fault occur?
2. Non-Blocking Guarantees:
   - Audio callbacks and real-time scheduling threads NEVER block on database I/O.
   - Uses a bounded asynchronous queue with a dedicated background writer thread.
   - Bounded queue prevents memory exhaustion with explicit queue-full accounting.
3. Reliability & Fault Tolerance:
   - Uses explicit SQLite transactions (WAL mode enabled for concurrent read performance).
   - Handles database errors without crashing callers.
   - Safe serialization of metadata with sanitization.
   - Clean flush and shutdown mechanics.
4. Privacy:
   - Strictly prohibits storing raw PINs, PIN hashes, passwords, or speech recordings.
"""

from dataclasses import asdict, dataclass, field
import datetime
from enum import Enum
import json
import logging
import os
import pathlib
import queue
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("rtos.security.audit")


class AuditEventType(str, Enum):
    """Categorized security and lifecycle events."""
    # Authentication Events
    AUTH_SUBMISSION = "AUTH_SUBMISSION"
    AUTH_SUCCESS = "AUTH_SUCCESS"
    AUTH_FAILURE = "AUTH_FAILURE"
    AUTH_INCOMPLETE = "AUTH_INCOMPLETE"
    AUTH_TIMEOUT = "AUTH_TIMEOUT"
    LOCKOUT_ACTIVATED = "LOCKOUT_ACTIVATED"
    LOCKOUT_EXPIRED = "LOCKOUT_EXPIRED"
    USER_ACCESS_GRANTED = "USER_ACCESS_GRANTED"
    USER_ACCESS_DENIED = "USER_ACCESS_DENIED"
    GUEST_ACCOUNT_EXPIRED = "GUEST_ACCOUNT_EXPIRED"

    # Lock Events
    LOCK_ACTIVATED = "LOCK_ACTIVATED"
    LOCK_RELEASED = "LOCK_RELEASED"
    LOCK_AUTO_RELOCK = "LOCK_AUTO_RELOCK"
    LOCK_MANUAL_RELOCK = "LOCK_MANUAL_RELOCK"
    LOCK_STARTUP_INIT = "LOCK_STARTUP_INIT"

    # Security & Administration Events
    EMERGENCY_STOP = "EMERGENCY_STOP"
    EMERGENCY_STOP_REJECTED = "EMERGENCY_STOP_REJECTED"
    EMERGENCY_CANCEL_PENDING = "EMERGENCY_CANCEL_PENDING"
    EMERGENCY_DASHBOARD_TRIGGER = "EMERGENCY_DASHBOARD_TRIGGER"
    USER_ADDED = "USER_ADDED"
    USER_REMOVED = "USER_REMOVED"
    USER_DISABLED = "USER_DISABLED"
    PIN_CHANGED = "PIN_CHANGED"
    ADMIN_ACTION = "ADMIN_ACTION"
    UNAUTHORIZED_OP = "UNAUTHORIZED_OP"
    ADMIN_CREATED_USER = "ADMIN_CREATED_USER"
    ADMIN_DISABLED_USER = "ADMIN_DISABLED_USER"
    ADMIN_ENABLED_USER = "ADMIN_ENABLED_USER"
    ADMIN_REMOVED_USER = "ADMIN_REMOVED_USER"
    ADMIN_CHANGED_ROLE = "ADMIN_CHANGED_ROLE"
    ADMIN_RESET_PIN = "ADMIN_RESET_PIN"

    # System Events
    SYSTEM_STARTUP = "SYSTEM_STARTUP"
    SYSTEM_SHUTDOWN = "SYSTEM_SHUTDOWN"
    DATABASE_ERROR = "DATABASE_ERROR"
    AUDIO_FAILURE = "AUDIO_FAILURE"
    QUEUE_OVERFLOW = "QUEUE_OVERFLOW"
    DEADLINE_MISS = "DEADLINE_MISS"
    FAULT_DETECTED = "FAULT_DETECTED"
    FAULT_RECOVERY = "FAULT_RECOVERY"


class AuditResult(str, Enum):
    """Event outcomes."""
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    WARNING = "WARNING"
    INFO = "INFO"


class AuditSource(str, Enum):
    """Event originator."""
    VOICE = "voice"
    ADMINISTRATOR = "administrator"
    SYSTEM = "system"
    API = "api"
    DASHBOARD = "dashboard"


@dataclass
class AuditRecord:
    """Represents an individual security audit event."""
    event_type: str
    result: str
    reason: str
    lock_state: str
    event_id: Optional[int] = None
    timestamp: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    user_id: Optional[str] = "voice_user"
    username: Optional[str] = "Voice User"
    command_type: Optional[str] = None
    source: str = "voice"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "event_type": str(self.event_type),
            "user_id": self.user_id,
            "username": self.username,
            "result": str(self.result),
            "reason": self.reason,
            "lock_state": str(self.lock_state),
            "command_type": self.command_type,
            "source": str(self.source),
            "metadata": self.metadata,
        }


class AuditLogger:
    """
    Thread-safe, non-blocking asynchronous audit logging manager.
    Batches database writes in a background worker thread.
    """

    def __init__(
        self,
        db_path: str | pathlib.Path = "data/security_audit.db",
        queue_capacity: int = 1000,
        enable_background_worker: bool = True,
    ):
        self.db_path = pathlib.Path(db_path)
        self.queue_capacity = queue_capacity
        self.enable_background_worker = enable_background_worker

        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Thread-safe bounded queue
        self._queue: queue.Queue = queue.Queue(maxsize=self.queue_capacity)
        self._lock = threading.RLock()

        # Telemetry & Reliability counters
        self.total_logged_count: int = 0
        self.total_written_count: int = 0
        self.total_dropped_count: int = 0
        self.total_db_errors: int = 0

        # Initialize SQLite database schema
        self._init_db()

        # Worker thread lifecycle
        self._is_running = False
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        if self.enable_background_worker:
            self.start()

    def _init_db(self):
        """Initializes SQLite schema and indexes."""
        conn = None
        try:
            conn = sqlite3.connect(str(self.db_path), timeout=3.0)
            with conn:
                # Enable WAL mode for concurrent non-blocking reads
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA synchronous=NORMAL;")

                conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    user_id TEXT,
                    username TEXT,
                    result TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    lock_state TEXT NOT NULL,
                    command_type TEXT,
                    source TEXT NOT NULL,
                    metadata TEXT
                );
                """)

                # Create indexes for efficient dashboard querying
                conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_events (timestamp);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_event_type ON audit_events (event_type);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_user_id ON audit_events (user_id);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_result ON audit_events (result);")
            logger.info(f"[AUDIT LOGGER] Initialized SQLite schema at '{self.db_path}'.")
        except sqlite3.Error as e:
            self.total_db_errors += 1
            logger.error(f"[AUDIT LOGGER] Failed to initialize SQLite database: {e}")
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    def start(self):
        """Starts the background worker thread."""
        with self._lock:
            if self._is_running:
                return
            self._is_running = True
            self._stop_event.clear()
            self._worker_thread = threading.Thread(
                target=self._worker_loop,
                name="AuditLogWorkerThread",
                daemon=True,
            )
            self._worker_thread.start()
            logger.debug("[AUDIT LOGGER] Background worker thread started.")

    def log(
        self,
        event_type: str | AuditEventType,
        result: str | AuditResult,
        reason: str,
        lock_state: str,
        user_id: Optional[str] = "voice_user",
        username: Optional[str] = "Voice User",
        command_type: Optional[str] = None,
        source: str | AuditSource = "voice",
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[str] = None,
    ) -> bool:
        """
        Asynchronously enqueues an audit event.
        Guaranteed to be non-blocking with defined queue-full policy.
        """
        # Ensure enums convert to strings
        evt_type = event_type.value if isinstance(event_type, Enum) else str(event_type)
        res = result.value if isinstance(result, Enum) else str(result)
        src = source.value if isinstance(source, Enum) else str(source)
        l_state = lock_state.value if isinstance(lock_state, Enum) else str(lock_state)

        # Sanitize metadata (prohibit raw secrets)
        safe_meta = self._sanitize_metadata(metadata or {})

        rec = AuditRecord(
            event_type=evt_type,
            result=res,
            reason=reason,
            lock_state=l_state,
            user_id=user_id,
            username=username,
            command_type=command_type,
            source=src,
            metadata=safe_meta,
            timestamp=timestamp or datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )

        with self._lock:
            self.total_logged_count += 1

            # Check if queue is full
            if self._queue.full():
                # Critical events (FAILURE, WARNING, or security critical) take precedence
                is_critical = res in ("FAILURE", "WARNING") or "LOCKOUT" in evt_type or "STOP" in evt_type
                if is_critical:
                    try:
                        # Drop oldest item if queue is full
                        _ = self._queue.get_nowait()
                        self._queue.task_done()
                        self.total_dropped_count += 1
                        logger.warning("[AUDIT LOGGER] Queue capacity saturated! Evicted oldest audit record to preserve critical event.")
                    except queue.Empty:
                        pass

                try:
                    self._queue.put_nowait(rec)
                    return True
                except queue.Full:
                    self.total_dropped_count += 1
                    logger.warning("[AUDIT LOGGER] Queue capacity saturated! Dropped audit record.")
                    return False
            else:
                try:
                    self._queue.put_nowait(rec)
                    return True
                except queue.Full:
                    self.total_dropped_count += 1
                    return False

    def log_sync(self, record: AuditRecord) -> bool:
        """Synchronously persists a record directly to SQLite (used in tests or critical paths)."""
        return self._write_records_to_db([record])

    def flush(self, timeout: float = 2.0) -> bool:
        """Flushes all queued events to disk within timeout."""
        if not self._is_running:
            # Drain and write synchronously if worker is not running
            batch = []
            while not self._queue.empty():
                try:
                    batch.append(self._queue.get_nowait())
                    self._queue.task_done()
                except queue.Empty:
                    break
            if batch:
                self._write_records_to_db(batch)
            return True

        t_start = time.perf_counter()
        while self._queue.unfinished_tasks > 0:
            time.sleep(0.005)
            if time.perf_counter() - t_start > timeout:
                logger.warning("[AUDIT LOGGER] Flush timed out with records remaining in queue.")
                return False
        return True

    def stop(self, timeout: float = 2.0):
        """Flushes remaining records and terminates the worker thread cleanly."""
        self.flush(timeout=timeout)

        with self._lock:
            if not self._is_running:
                return
            self._is_running = False
            self._stop_event.set()

        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout)
        logger.info("[AUDIT LOGGER] Audit worker stopped cleanly.")

    # ==========================================================================
    # Worker Loop & Database Batch Writing
    # ==========================================================================

    def _worker_loop(self):
        """Dedicated background thread draining the queue and writing to SQLite."""
        while not self._stop_event.is_set() or not self._queue.empty():
            batch: List[AuditRecord] = []
            try:
                # Fetch record with 50ms timeout
                rec = self._queue.get(timeout=0.05)
                batch.append(rec)
                # Drain any immediately available records up to batch limit 50
                while len(batch) < 50:
                    try:
                        rec = self._queue.get_nowait()
                        batch.append(rec)
                    except queue.Empty:
                        break
            except queue.Empty:
                pass

            if batch:
                self._write_records_to_db(batch)
                for _ in batch:
                    self._queue.task_done()
                batch.clear()

    def _write_records_to_db(self, records: List[AuditRecord]) -> bool:
        """Writes a batch of records inside a single SQLite transaction."""
        if not records:
            return True

        rows = []
        for r in records:
            meta_json = "{}"
            try:
                meta_json = json.dumps(r.metadata, default=str)
            except Exception:
                meta_json = json.dumps({"raw": str(r.metadata)})

            rows.append((
                r.timestamp,
                r.event_type,
                r.user_id,
                r.username,
                r.result,
                r.reason,
                r.lock_state,
                r.command_type,
                r.source,
                meta_json,
            ))

        conn = None
        try:
            conn = sqlite3.connect(str(self.db_path), timeout=3.0)
            with conn:
                conn.executemany("""
                INSERT INTO audit_events (
                    timestamp, event_type, user_id, username, result,
                    reason, lock_state, command_type, source, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """, rows)

            with self._lock:
                self.total_written_count += len(rows)
            return True
        except sqlite3.Error as e:
            with self._lock:
                self.total_db_errors += 1
            logger.error(f"[AUDIT LOGGER] SQLite transaction error writing {len(rows)} records: {e}")
            return False
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    # ==========================================================================
    # Query API & Telemetry
    # ==========================================================================

    def query(
        self,
        event_type: Optional[str] = None,
        result: Optional[str] = None,
        user_id: Optional[str] = None,
        category: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Queries audit events with optional filters, sorted newest to oldest.
        Safe for concurrent invocation alongside worker thread.
        """
        # Ensure any in-memory items are flushed before querying
        self.flush(timeout=0.2)

        query_sql = "SELECT event_id, timestamp, event_type, user_id, username, result, reason, lock_state, command_type, source, metadata FROM audit_events WHERE 1=1"
        params: List[Any] = []

        if event_type:
            query_sql += " AND event_type = ?"
            params.append(event_type)

        if result:
            query_sql += " AND result = ?"
            params.append(result)

        if user_id:
            query_sql += " AND user_id = ?"
            params.append(user_id)

        if category:
            cat_upper = category.upper()
            if cat_upper == "AUTH":
                query_sql += " AND (event_type LIKE 'AUTH%' OR event_type LIKE 'LOCKOUT%' OR event_type LIKE 'USER_ACCESS%' OR event_type = 'GUEST_ACCOUNT_EXPIRED')"
            elif cat_upper == "LOCK":
                query_sql += " AND event_type LIKE 'LOCK%'"
            elif cat_upper == "SECURITY":
                query_sql += " AND (event_type IN ('EMERGENCY_STOP', 'UNAUTHORIZED_OP', 'PIN_CHANGED', 'ADMIN_ACTION') OR event_type LIKE 'USER_%' OR event_type LIKE 'ADMIN_%')"
            elif cat_upper == "SYSTEM":
                query_sql += " AND (event_type LIKE 'SYSTEM_%' OR event_type IN ('DATABASE_ERROR', 'AUDIO_FAILURE', 'QUEUE_OVERFLOW', 'DEADLINE_MISS', 'FAULT_DETECTED', 'FAULT_RECOVERY'))"

        if start_time:
            query_sql += " AND timestamp >= ?"
            params.append(start_time)

        if end_time:
            query_sql += " AND timestamp <= ?"
            params.append(end_time)

        query_sql += " ORDER BY event_id DESC LIMIT ? OFFSET ?;"
        params.extend([limit, offset])

        results = []
        conn = None
        try:
            conn = sqlite3.connect(str(self.db_path), timeout=3.0)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query_sql, params)
            for row in cursor.fetchall():
                meta = {}
                if row["metadata"]:
                    try:
                        meta = json.loads(row["metadata"])
                    except Exception:
                        meta = {"raw": row["metadata"]}

                results.append({
                    "event_id": row["event_id"],
                    "timestamp": row["timestamp"],
                    "event_type": row["event_type"],
                    "user_id": row["user_id"],
                    "username": row["username"],
                    "result": row["result"],
                    "reason": row["reason"],
                    "lock_state": row["lock_state"],
                    "command_type": row["command_type"],
                    "source": row["source"],
                    "metadata": meta,
                })
        except sqlite3.Error as e:
            with self._lock:
                self.total_db_errors += 1
            logger.error(f"[AUDIT LOGGER] Failed to query audit events: {e}")
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Returns diagnostic and reliability metrics for dashboard/monitoring."""
        with self._lock:
            q_size = self._queue.qsize()

        db_size_kb = 0.0
        total_in_db = 0
        if self.db_path.exists():
            conn = None
            try:
                db_size_kb = round(self.db_path.stat().st_size / 1024.0, 2)
                conn = sqlite3.connect(str(self.db_path), timeout=3.0)
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) FROM audit_events;")
                total_in_db = cur.fetchone()[0]
            except Exception:
                pass
            finally:
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass

        return {
            "db_path": str(self.db_path),
            "db_size_kb": db_size_kb,
            "total_in_db": total_in_db,
            "total_logged": self.total_logged_count,
            "total_written": self.total_written_count,
            "total_dropped": self.total_dropped_count,
            "total_db_errors": self.total_db_errors,
            "queue_occupancy": q_size,
            "queue_capacity": self.queue_capacity,
        }

    # ==========================================================================
    # Sanitization
    # ==========================================================================

    def _sanitize_metadata(self, meta: Any) -> Any:
        """Removes any sensitive credential fields from metadata before storage (recursively handles nested structures)."""
        prohibited_keys = {"pin", "pin_hash", "salt", "salt_hex", "hash", "hash_hex", "password", "secret"}
        if isinstance(meta, dict):
            sanitized = {}
            for k, v in meta.items():
                if str(k).lower() in prohibited_keys:
                    continue
                sanitized[k] = self._sanitize_metadata(v)
            return sanitized
        elif isinstance(meta, (list, tuple)):
            return [self._sanitize_metadata(item) for item in meta]
        return meta
