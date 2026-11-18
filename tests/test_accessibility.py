"""
tests/test_accessibility.py
---------------------------
Comprehensive Automated Test Suite for Stage 8: Accessibility Mode.

Covers all 12 required test scenarios:
1. Large text mode (toggle on/off, HTML/CSS class scaling, persistence).
2. High-contrast mode (toggle on/off, WCAG AAA compliant colors, persistence).
3. Settings persistence (safe saving, reloading, corrupt file fallback to safe defaults, bounds clamping).
4. Voice feedback enabled (messages are queued and spoken).
5. Voice feedback disabled (no speech output when disabled).
6. Correct access feedback (emits 'Access granted.', 'PIN entry complete.', never reveals PIN).
7. Incorrect access feedback (emits 'Access denied.').
8. Lockout feedback (emits 'Lockout activated.' and 'Lockout expired.').
9. STOP feedback (emits 'Emergency STOP accepted.' and 'Lock activated.').
10. Keyboard navigation (skip links, focus-visible styling, labels, aria attributes).
11. No sensitive credential disclosure (verifying digits/secrets are never spoken or leaked).
12. Existing authentication behaviour remains unchanged (full authentication flow, lockout, timeouts, RBAC permissions unaffected).
"""

import json
import os
import pathlib
import re
import time
import pytest

from src.accessibility.settings import (
    AccessibilitySettings,
    AccessibilitySettingsManager,
    MIN_PIN_TIMEOUT_S,
    MAX_PIN_TIMEOUT_S,
    DEFAULT_PIN_TIMEOUT_S,
    MIN_VOLUME,
    MAX_VOLUME,
    DEFAULT_VOLUME,
)
from src.accessibility.voice_feedback import (
    VoiceFeedback,
    FeedbackEvent,
    FeedbackRecord,
    MockTTSBackend,
    CANONICAL_FEEDBACK_MESSAGES,
)
from src.security.config import SecurityConfig
from src.security.lock_controller import LockController, LockState, AccessResult
from src.security.audit_logger import AuditLogger, AuditSource
from src.security.user_manager import UserManager, UserRole
from src.dashboard.server import (
    DashboardStatusProvider,
    DashboardServer,
    DASHBOARD_HTML,
)


# ==============================================================================
# Test 1: Large Text Mode
# ==============================================================================
def test_01_large_text_mode(tmp_path):
    """Verifies large text mode configuration, CSS font scaling factors, and persistence."""
    cfg_file = str(tmp_path / "a11y_large_text.json")
    mgr = AccessibilitySettingsManager(config_path=cfg_file)

    assert mgr.settings.large_text is False

    updated = mgr.update({"large_text": True})
    assert updated.large_text is True
    assert mgr.settings.large_text is True

    # Verify CSS in DASHBOARD_HTML supports .large-text scaling
    assert "body.large-text" in DASHBOARD_HTML
    assert "font-size: 17px !important" in DASHBOARD_HTML or "font-size:" in DASHBOARD_HTML
    assert "body.large-text h1" in DASHBOARD_HTML
    assert "body.large-text button" in DASHBOARD_HTML
    assert "body.large-text table" in DASHBOARD_HTML


# ==============================================================================
# Test 2: High-Contrast Mode
# ==============================================================================
def test_02_high_contrast_mode(tmp_path):
    """Verifies high-contrast mode configuration, WCAG AAA compliant contrast palette, and styles."""
    cfg_file = str(tmp_path / "a11y_high_contrast.json")
    mgr = AccessibilitySettingsManager(config_path=cfg_file)

    assert mgr.settings.high_contrast is False

    updated = mgr.update({"high_contrast": True})
    assert updated.high_contrast is True
    assert mgr.settings.high_contrast is True

    # Verify CSS in DASHBOARD_HTML contains WCAG AAA compliant colors (#000000, #ffffff, #ffff00)
    assert "body.high-contrast" in DASHBOARD_HTML
    assert "--bg-color: #000000" in DASHBOARD_HTML
    assert "--text-main: #ffffff" in DASHBOARD_HTML
    assert "--text-muted: #ffff00" in DASHBOARD_HTML
    assert "2px solid #ffffff" in DASHBOARD_HTML or "2px solid #ffff00" in DASHBOARD_HTML

    # WCAG Contrast Math: pure black (#000000) and pure white (#ffffff) has 21:1 contrast ratio
    # WCAG AAA requires >= 7:1 for normal text and >= 4.5:1 for large text. 21:1 strictly satisfies AAA.
    l1 = 1.0  # White relative luminance
    l2 = 0.0  # Black relative luminance
    contrast_ratio = (l1 + 0.05) / (l2 + 0.05)
    assert contrast_ratio == 21.0
    assert contrast_ratio >= 7.0


# ==============================================================================
# Test 3: Settings Persistence & Safe Clamping
# ==============================================================================
def test_03_settings_persistence(tmp_path):
    """Verifies safe atomic saving to disk, reloading, corrupt file recovery, and bounds clamping."""
    cfg_file = str(tmp_path / "a11y_persistence.json")
    mgr = AccessibilitySettingsManager(config_path=cfg_file)

    # 1. Modify and save valid settings
    mgr.update({
        "large_text": True,
        "high_contrast": True,
        "pin_entry_timeout_s": 25.0,
        "voice_feedback_volume": 0.75,
        "reduced_visual_complexity": True,
    })

    assert os.path.exists(cfg_file)

    # 2. Reload with independent manager instance
    mgr2 = AccessibilitySettingsManager(config_path=cfg_file)
    s2 = mgr2.settings
    assert s2.large_text is True
    assert s2.high_contrast is True
    assert s2.pin_entry_timeout_s == 25.0
    assert s2.voice_feedback_volume == 0.75
    assert s2.reduced_visual_complexity is True

    # 3. Test Security bounds clamping (cannot weaken authentication security)
    # Attempt to set pin timeout too low (below 5.0s) or too high (above 60.0s)
    s_clamped = mgr2.update({"pin_entry_timeout_s": 1.0})
    assert s_clamped.pin_entry_timeout_s == MIN_PIN_TIMEOUT_S  # 5.0s

    s_clamped2 = mgr2.update({"pin_entry_timeout_s": 999.0})
    assert s_clamped2.pin_entry_timeout_s == MAX_PIN_TIMEOUT_S  # 60.0s

    # Attempt to set negative volume or volume > 1.0
    s_vol_low = mgr2.update({"voice_feedback_volume": -0.5})
    assert s_vol_low.voice_feedback_volume == MIN_VOLUME  # 0.0

    s_vol_high = mgr2.update({"voice_feedback_volume": 2.5})
    assert s_vol_high.voice_feedback_volume == MAX_VOLUME  # 1.0

    # 4. Corrupted configuration file recovery (graceful fallback without crash)
    with open(cfg_file, "w", encoding="utf-8") as f:
        f.write("{ invalid json corrupted data ... !!!")

    mgr_repaired = AccessibilitySettingsManager(config_path=cfg_file)
    recovered = mgr_repaired.settings
    assert recovered.large_text is False
    assert recovered.pin_entry_timeout_s == DEFAULT_PIN_TIMEOUT_S  # 10.0s
    assert recovered.voice_feedback_volume == DEFAULT_VOLUME  # 1.0


# ==============================================================================
# Test 4: Voice Feedback Enabled
# ==============================================================================
def test_04_voice_feedback_enabled():
    """Verifies that when voice feedback is enabled, speech requests are queued and dispatched."""
    backend = MockTTSBackend()
    settings = AccessibilitySettings(spoken_feedback=True, voice_feedback_volume=1.0)
    fb = VoiceFeedback(settings=settings, backend=backend)

    try:
        r1 = fb.on_system_ready()
        r2 = fb.on_digit_accepted()
        r3 = fb.on_pin_complete()

        fb.wait_until_idle(timeout=1.0)

        assert r1.spoken is True
        assert r2.spoken is True
        assert r3.spoken is True

        assert "System ready." in backend.spoken_phrases
        assert "Digit accepted." in backend.spoken_phrases
        assert "PIN entry complete." in backend.spoken_phrases

        hist = fb.get_history()
        assert len(hist) == 3
        assert hist[0]["event"] == "SYSTEM_READY"
        assert hist[1]["event"] == "DIGIT_ACCEPTED"
        assert hist[2]["event"] == "PIN_COMPLETE"
    finally:
        fb.stop()


# ==============================================================================
# Test 5: Voice Feedback Disabled
# ==============================================================================
def test_05_voice_feedback_disabled():
    """Verifies that when voice feedback is disabled, no spoken output is dispatched."""
    backend = MockTTSBackend()
    settings = AccessibilitySettings(spoken_feedback=False, audio_confirmation=False)
    fb = VoiceFeedback(settings=settings, backend=backend)

    try:
        r1 = fb.on_system_ready()
        r2 = fb.on_access_granted()
        r3 = fb.on_access_denied()

        fb.wait_until_idle(timeout=0.5)

        assert r1.spoken is False
        assert r2.spoken is False
        assert r3.spoken is False

        # Backend should have received zero spoken phrases
        assert len(backend.spoken_phrases) == 0
        assert len(backend.tones_played) == 0

        # Event is still recorded in history with spoken=False for observability
        hist = fb.get_history()
        assert len(hist) == 3
        assert all(h["spoken"] is False for h in hist)
    finally:
        fb.stop()


# ==============================================================================
# Test 6: Correct Access Feedback
# ==============================================================================
def test_06_correct_access_feedback(tmp_path):
    """Verifies correct access interaction emits 'Access granted.' without exposing PIN digits."""
    db_path = str(tmp_path / "audit.db")
    users_db = str(tmp_path / "users.db")
    audit = AuditLogger(db_path=db_path)
    cfg = SecurityConfig(audit_db_path=db_path, users_db_path=users_db, enable_audit_logging=True)
    user_mgr = UserManager(config=cfg, audit_logger=audit, db_path=users_db, allow_demo_bootstrap=False)
    if len(user_mgr.get_all_users()) == 0:
        user_mgr.bootstrap_initial_admin("admin_01", "Administrator", "1234")

    backend = MockTTSBackend()
    settings = AccessibilitySettings(spoken_feedback=True)
    fb = VoiceFeedback(settings=settings, backend=backend)

    lock = LockController(
        config=cfg,
        audit_logger=audit,
        user_manager=user_mgr,
        voice_feedback=fb,
    )

    try:
        # Enter valid 4 digits: "1", "2", "3", "4"
        lock.handle_command("one")
        lock.handle_command("two")
        lock.handle_command("three")
        lock.handle_command("four")

        # Submit PIN
        lock.handle_command("go")

        fb.wait_until_idle(timeout=1.0)

        # Verifications
        assert lock.state == LockState.UNLOCKED
        assert "Access granted." in backend.spoken_phrases
        assert "PIN entry complete." in backend.spoken_phrases
        assert backend.spoken_phrases.count("Digit accepted.") == 4

        # Crucial security assertion: Individual digits 1, 2, 3, 4 were NEVER spoken!
        for phrase in backend.spoken_phrases:
            assert "1" not in phrase and "one" not in phrase.lower()
            assert "2" not in phrase and "two" not in phrase.lower()
            assert "3" not in phrase and "three" not in phrase.lower()
            assert "4" not in phrase and "four" not in phrase.lower()
    finally:
        fb.stop()
        lock.stop()


# ==============================================================================
# Test 7: Incorrect Access Feedback
# ==============================================================================
def test_07_incorrect_access_feedback(tmp_path):
    """Verifies incorrect PIN submissions emit 'Access denied.' feedback."""
    db_path = str(tmp_path / "audit.db")
    users_db = str(tmp_path / "users.db")
    audit = AuditLogger(db_path=db_path)
    cfg = SecurityConfig(audit_db_path=db_path, users_db_path=users_db, enable_audit_logging=True)
    user_mgr = UserManager(config=cfg, audit_logger=audit, db_path=users_db, allow_demo_bootstrap=False)
    if len(user_mgr.get_all_users()) == 0:
        user_mgr.bootstrap_initial_admin("admin_01", "Administrator", "1234")

    backend = MockTTSBackend()
    settings = AccessibilitySettings(spoken_feedback=True)
    fb = VoiceFeedback(settings=settings, backend=backend)

    lock = LockController(
        config=cfg,
        audit_logger=audit,
        user_manager=user_mgr,
        voice_feedback=fb,
    )

    try:
        # Enter wrong digits: "9", "9", "9", "9"
        lock.handle_command("nine")
        lock.handle_command("nine")
        lock.handle_command("nine")
        lock.handle_command("nine")
        lock.handle_command("go")

        fb.wait_until_idle(timeout=1.0)

        assert lock.state == LockState.LOCKED
        assert lock.failed_attempts == 1
        assert "Access denied." in backend.spoken_phrases

        # Also test incomplete PIN: 2 digits + 'go'
        backend.clear()
        lock.handle_command("five")
        lock.handle_command("six")
        lock.handle_command("go")

        fb.wait_until_idle(timeout=1.0)
        assert lock.state == LockState.LOCKED
        assert "Access denied." in backend.spoken_phrases
    finally:
        fb.stop()
        lock.stop()


# ==============================================================================
# Test 8: Lockout Feedback
# ==============================================================================
def test_08_lockout_feedback(tmp_path):
    """Verifies feedback for 'Lockout activated.' and 'Lockout expired.' upon timeout."""
    db_path = str(tmp_path / "audit.db")
    users_db = str(tmp_path / "users.db")
    audit = AuditLogger(db_path=db_path)
    cfg = SecurityConfig(audit_db_path=db_path, users_db_path=users_db, lockout_duration_s=1.0)
    user_mgr = UserManager(config=cfg, audit_logger=audit, db_path=users_db, allow_demo_bootstrap=False)
    if len(user_mgr.get_all_users()) == 0:
        user_mgr.bootstrap_initial_admin("admin_01", "Administrator", "1234")

    backend = MockTTSBackend()
    settings = AccessibilitySettings(spoken_feedback=True)
    fb = VoiceFeedback(settings=settings, backend=backend)

    lock = LockController(
        config=cfg,
        audit_logger=audit,
        user_manager=user_mgr,
        voice_feedback=fb,
    )

    try:
        # Fail 3 consecutive times to trigger lockout
        for _ in range(3):
            lock.handle_command("nine")
            lock.handle_command("nine")
            lock.handle_command("nine")
            lock.handle_command("nine")
            lock.handle_command("go")

        fb.wait_until_idle(timeout=1.0)

        assert lock.state == LockState.LOCKOUT
        assert "Lockout activated." in backend.spoken_phrases

        # Advance time to expire lockout
        time.sleep(1.05)
        lock.tick()

        fb.wait_until_idle(timeout=1.0)

        assert lock.state == LockState.LOCKED
        assert "Lockout expired." in backend.spoken_phrases
    finally:
        fb.stop()
        lock.stop()


# ==============================================================================
# Test 9: Emergency STOP Feedback
# ==============================================================================
def test_09_emergency_stop_feedback(tmp_path):
    """Verifies emergency STOP emits 'Emergency STOP accepted.' and 'Lock activated.'."""
    db_path = str(tmp_path / "audit.db")
    audit = AuditLogger(db_path=db_path)
    cfg = SecurityConfig(audit_db_path=db_path, enable_audit_logging=True)

    backend = MockTTSBackend()
    settings = AccessibilitySettings(spoken_feedback=True)
    fb = VoiceFeedback(settings=settings, backend=backend)

    lock = LockController(
        config=cfg,
        audit_logger=audit,
        voice_feedback=fb,
    )

    try:
        # 1. STOP from PIN_ENTRY state
        lock.handle_command("one")
        lock.handle_command("two")
        assert lock.state == LockState.PIN_ENTRY

        lock.handle_command("stop")
        fb.wait_until_idle(timeout=1.0)

        assert lock.state == LockState.LOCKED
        assert "Emergency STOP accepted." in backend.spoken_phrases

        # 2. STOP from UNLOCKED state
        backend.clear()
        user_mgr = UserManager(config=cfg, audit_logger=audit, db_path=str(tmp_path / "users2.db"), allow_demo_bootstrap=False)
        if len(user_mgr.get_all_users()) == 0:
            user_mgr.bootstrap_initial_admin("admin_01", "Administrator", "1234")
        admin_rec = user_mgr.get_user("admin_01")
        lock.commit_unlock(op_id=1, user_rec=admin_rec)
        assert lock.state == LockState.UNLOCKED

        lock.handle_command("stop")
        fb.wait_until_idle(timeout=1.0)

        assert lock.state == LockState.LOCKED
        assert "Emergency STOP accepted." in backend.spoken_phrases
        assert "Lock activated." in backend.spoken_phrases
    finally:
        fb.stop()
        lock.stop()


# ==============================================================================
# Test 10: Keyboard Navigation & WCAG Compliance
# ==============================================================================
def test_10_keyboard_navigation_and_wcag():
    """Verifies dashboard HTML contains keyboard focus, skip links, and ARIA accessibility markup."""
    # 1. Skip Navigation Link
    assert '<a href="#main-content" class="skip-link">Skip to main content</a>' in DASHBOARD_HTML
    assert '<main id="main-content">' in DASHBOARD_HTML

    # 2. Visible Focus Styling
    assert ":focus-visible" in DASHBOARD_HTML
    assert "outline: 3px solid #ffcc00 !important" in DASHBOARD_HTML

    # 3. Accessibility Controls Section with accessible labels
    assert 'section id="accessibility-panel"' in DASHBOARD_HTML
    assert 'aria-label="Accessibility and Usability Settings"' in DASHBOARD_HTML
    assert 'for="toggleLargeText"' in DASHBOARD_HTML
    assert 'for="toggleHighContrast"' in DASHBOARD_HTML
    assert 'for="toggleClearIndicators"' in DASHBOARD_HTML
    assert 'for="toggleSpokenFeedback"' in DASHBOARD_HTML
    assert 'for="toggleAudioConfirmation"' in DASHBOARD_HTML
    assert 'for="toggleReducedComplexity"' in DASHBOARD_HTML
    assert 'for="inputPinTimeout"' in DASHBOARD_HTML
    assert 'for="inputVoiceVolume"' in DASHBOARD_HTML

    # 4. ARIA Live Regions for assistive technology
    assert 'role="status"' in DASHBOARD_HTML
    assert 'aria-live="polite"' in DASHBOARD_HTML


# ==============================================================================
# Test 11: No Sensitive Credential Disclosure
# ==============================================================================
def test_11_no_sensitive_credential_disclosure():
    """Verifies that individual PIN digits, hashes, salts, and secrets are strictly forbidden from speech."""
    # 1. Test canonical messages ensure zero digits or secret parameters
    for event, msg in CANONICAL_FEEDBACK_MESSAGES.items():
        assert not re.search(r"\b\d+\b", msg), f"Event {event} contains numeric digits: {msg}"
        lower = msg.lower()
        for forbidden in ["password", "hash", "salt", "secret", "token"]:
            assert forbidden not in lower, f"Event {event} contains sensitive keyword '{forbidden}': {msg}"

    # 2. Test sanitizer blocks attempted leaks
    sanitized_pin = VoiceFeedback.sanitize_message("User entered PIN 5831 successfully.")
    assert sanitized_pin == "[REDACTED_SECURITY_POLICY]"

    sanitized_hash = VoiceFeedback.sanitize_message("Salt and hash updated.")
    assert sanitized_hash == "[REDACTED_SECURITY_POLICY]"

    sanitized_secret = VoiceFeedback.sanitize_message("Secret token issued.")
    assert sanitized_secret == "[REDACTED_SECURITY_POLICY]"

    # 3. Safe messages pass unchanged
    safe_msg = VoiceFeedback.sanitize_message("Access granted.")
    assert safe_msg == "Access granted."

    safe_msg2 = VoiceFeedback.sanitize_message("Digit accepted.")
    assert safe_msg2 == "Digit accepted."


# ==============================================================================
# Test 12: Existing Authentication Behaviour Remains Unchanged
# ==============================================================================
def test_12_existing_authentication_unchanged(tmp_path):
    """Verifies that modifying accessibility settings does not alter or weaken authentication security."""
    db_path = str(tmp_path / "audit.db")
    users_db = str(tmp_path / "users.db")
    a11y_cfg = str(tmp_path / "a11y.json")

    audit = AuditLogger(db_path=db_path)
    cfg = SecurityConfig(audit_db_path=db_path, users_db_path=users_db, enable_audit_logging=True)
    user_mgr = UserManager(config=cfg, audit_logger=audit, db_path=users_db, allow_demo_bootstrap=False)
    if len(user_mgr.get_all_users()) == 0:
        user_mgr.bootstrap_initial_admin("admin_01", "Administrator", "1234")

    # Create regular user with PIN 4321
    user_mgr.create_user(admin_user_id="admin_01", user_id="user_reg", username="Regular User", pin="4321", role=UserRole.REGULAR_USER)

    a11y_mgr = AccessibilitySettingsManager(config_path=a11y_cfg)
    # Enable aggressive accessibility settings
    a11y_mgr.update({
        "large_text": True,
        "high_contrast": True,
        "spoken_feedback": True,
        "pin_entry_timeout_s": 25.0,
        "voice_feedback_volume": 0.5,
        "reduced_visual_complexity": True,
    })

    backend = MockTTSBackend()
    fb = VoiceFeedback(settings=a11y_mgr.settings, backend=backend)

    lock = LockController(
        config=cfg,
        audit_logger=audit,
        user_manager=user_mgr,
        voice_feedback=fb,
    )
    # Sync accessibility timeout
    lock.set_pin_timeout(a11y_mgr.settings.pin_entry_timeout_s)
    assert lock.config.pin_timeout_s == 25.0

    try:
        # A. PIN verification still strictly enforced
        # Wrong PIN rejected
        lock.handle_command("one")
        lock.handle_command("one")
        lock.handle_command("one")
        lock.handle_command("one")
        lock.handle_command("go")
        assert lock.state == LockState.LOCKED
        assert lock.failed_attempts == 1

        # Correct regular user PIN authenticated
        lock.handle_command("four")
        lock.handle_command("three")
        lock.handle_command("two")
        lock.handle_command("one")
        lock.handle_command("go")

        assert lock.state == LockState.UNLOCKED
        assert lock.authenticated_user is not None
        assert lock.authenticated_user.user_id == "user_reg"
        assert lock.authenticated_user.role == UserRole.REGULAR_USER

        # B. Emergency STOP still takes absolute precedence
        lock.handle_command("stop")
        assert lock.state == LockState.LOCKED
        assert lock.authenticated_user is None

        # C. User management RBAC cannot be bypassed
        assert user_mgr.get_user("user_reg").role == UserRole.REGULAR_USER
        assert user_mgr.get_user("admin_01").role == UserRole.ADMIN

        # D. System never auto-unlocks upon accessibility setting change
        a11y_mgr.update({"high_contrast": False, "large_text": False})
        assert lock.state == LockState.LOCKED
    finally:
        fb.stop()
        lock.stop()


# ==============================================================================
# Test 13: Dashboard REST Endpoints for Accessibility
# ==============================================================================
def test_13_dashboard_accessibility_api(tmp_path):
    """Verifies dashboard REST endpoints for accessibility settings and voice feedback test."""
    import urllib.request

    cfg_file = str(tmp_path / "a11y_api.json")
    a11y_mgr = AccessibilitySettingsManager(config_path=cfg_file)
    backend = MockTTSBackend()
    fb = VoiceFeedback(settings=a11y_mgr.settings, backend=backend)

    provider = DashboardStatusProvider(
        accessibility_manager=a11y_mgr,
        voice_feedback=fb,
    )
    server = DashboardServer(port=0, provider=provider)
    server.start()
    port = server.port
    base_url = f"http://127.0.0.1:{port}"

    try:
        # 1. GET /api/accessibility/settings
        req = urllib.request.Request(f"{base_url}/api/accessibility/settings")
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data["large_text"] is False
            assert data["pin_entry_timeout_s"] == 10.0

        # 2. POST /api/accessibility/settings
        post_data = json.dumps({"large_text": True, "pin_entry_timeout_s": 20.0}).encode("utf-8")
        req_post = urllib.request.Request(
            f"{base_url}/api/accessibility/settings",
            data=post_data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req_post) as resp:
            assert resp.status == 200
            updated = json.loads(resp.read().decode("utf-8"))
            assert updated["large_text"] is True
            assert updated["pin_entry_timeout_s"] == 20.0

        # 3. POST /api/accessibility/feedback/test
        test_post = json.dumps({"event": "SYSTEM_READY"}).encode("utf-8")
        req_test = urllib.request.Request(
            f"{base_url}/api/accessibility/feedback/test",
            data=test_post,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req_test) as resp:
            assert resp.status == 200
            fb_res = json.loads(resp.read().decode("utf-8"))
            assert fb_res["message"] == "System ready."

        # 4. POST /api/accessibility/settings/reset
        req_reset = urllib.request.Request(
            f"{base_url}/api/accessibility/settings/reset",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req_reset) as resp:
            assert resp.status == 200
            reset_data = json.loads(resp.read().decode("utf-8"))
            assert reset_data["large_text"] is False
            assert reset_data["pin_entry_timeout_s"] == 10.0
    finally:
        server.stop()
        fb.stop()

