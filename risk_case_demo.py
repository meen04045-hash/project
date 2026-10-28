"""Opt-in, synthetic-only risk case demonstration.

No chatbot prediction, assessment result, webhook event, or notification is
consumed here. Routes are registered only when RISK_DEMO_ENABLED=1.
"""

from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import os
import re
import secrets
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from flask import Blueprint, abort, jsonify, redirect, render_template, request, url_for
from firebase_admin import firestore
from google.api_core.exceptions import Aborted
from werkzeug.security import check_password_hash


CASE_STATUSES = ("NEW", "UNDER_REVIEW", "CLOSED")
CASE_ID_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
PHONE_PATTERN = re.compile(r"\+?[0-9][0-9 -]{6,19}\Z")
EMAIL_PATTERN = re.compile(r"[^\s@]+@[^\s@]+\.[^\s@]+\Z")
CONTACT_RETENTION = timedelta(days=7)
CASE_RETENTION = timedelta(days=7)
LOGIN_WINDOW = timedelta(minutes=15)
SESSION_LIFETIME = timedelta(minutes=30)
MAX_LOGIN_FAILURES = 5
REVIEWER_COOKIE = "risk_demo_reviewer_session"
SYNTHETIC_EVENTS = {
    "text_general": ("synthetic_text_event", None),
    "text_review": ("synthetic_text_event", "SYNTHETIC_REVIEW"),
    "text_urgent": ("synthetic_text_event", "SYNTHETIC_URGENT"),
    "assessment_no_case": ("synthetic_assessment_event", None),
    "assessment_review": ("synthetic_assessment_event", "SYNTHETIC_REVIEW"),
    "assessment_urgent": ("synthetic_assessment_event", "SYNTHETIC_URGENT"),
}


def verify_line_identity(id_token: str, client_id: str) -> str | None:
    """Return LINE sub only after LINE's ID-token verification accepts it."""
    from json import loads

    verification = Request(
        "https://api.line.me/oauth2/v2.1/verify",
        data=urlencode({"id_token": id_token, "client_id": client_id}).encode("utf-8"),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with urlopen(verification, timeout=5) as response:
            claims = loads(response.read())
    except HTTPError as error:
        if error.code in (400, 401):
            return None
        raise RuntimeError("LINE identity verification unavailable") from error
    except (URLError, TimeoutError, ValueError) as error:
        raise RuntimeError("LINE identity verification unavailable") from error
    if (
        not isinstance(claims, dict)
        or not isinstance(claims.get("sub"), str)
        or not claims["sub"]
        or claims.get("aud") != client_id
        or claims.get("iss") != "https://access.line.me"
    ):
        return None
    return claims["sub"]


def register_risk_demo(app, db, line_client_id: str, identity_verifier=None,
                       transaction_runner=None, enable_event_simulator=False,
                       contact_only=False, contact_settings=None,
                       allowed_subject_keys=None, contact_auth_mode="legacy_allowlist"):
    """Attach demo routes; refuse to start the demo without secure setup."""
    required = ["RISK_DEMO_IDENTITY_HMAC_KEY", "RISK_DEMO_LIFF_ID"]
    if not contact_only:
        required.extend(("RISK_DEMO_REVIEWER_USERNAME", "RISK_DEMO_REVIEWER_PASSWORD_HASH"))
    if contact_settings is not None and not contact_only:
        raise RuntimeError("Explicit contact settings require contact-only mode")
    if allowed_subject_keys is not None and not contact_only:
        raise RuntimeError("Explicit identity allowlist requires contact-only mode")
    config = ({name: contact_settings.get(name, "") for name in required}
              if contact_settings is not None else
              {name: os.environ.get(name, "") for name in required})
    missing = [name for name, value in config.items() if not value]
    if missing:
        raise RuntimeError("Risk demo configuration missing: " + ", ".join(missing))
    if len(config["RISK_DEMO_IDENTITY_HMAC_KEY"]) < 32:
        raise RuntimeError("Risk demo HMAC key must contain at least 32 characters")
    if not line_client_id:
        raise RuntimeError("LINE Login channel ID is required for the risk demo")

    identity_verifier = identity_verifier or verify_line_identity
    transaction_runner = transaction_runner or firestore.transactional
    hmac_key = config["RISK_DEMO_IDENTITY_HMAC_KEY"].encode("utf-8")
    demo = Blueprint("risk_demo", __name__)

    def now():
        return datetime.now(timezone.utc)

    def subject_key(line_sub):
        return hmac.new(hmac_key, ("subject:" + line_sub).encode("utf-8"), hashlib.sha256).hexdigest()

    def reviewer_session_ref():
        token = request.cookies.get(REVIEWER_COOKIE, "")
        if not re.fullmatch(r"[A-Za-z0-9_-]{40,60}", token):
            return None
        digest = hashlib.sha256(token.encode("ascii")).hexdigest()
        return db.collection("risk_demo_reviewer_sessions").document(digest)

    def reviewer_state():
        ref = reviewer_session_ref()
        state = ref.get().to_dict() if ref else None
        if (
            not state
            or state.get("reviewer") != config["RISK_DEMO_REVIEWER_USERNAME"]
            or not isinstance(state.get("expires_at"), datetime)
            or state["expires_at"] <= now()
        ):
            return None
        return state

    def login_attempt_ref():
        # Never trust X-Forwarded-For. Store only an HMAC of the direct peer.
        peer = request.remote_addr or "unknown"
        digest = hmac.new(hmac_key, ("login:" + peer).encode("utf-8"), hashlib.sha256).hexdigest()
        return db.collection("risk_demo_login_attempts").document(digest)

    def locked(record, timestamp):
        return bool(record and isinstance(record.get("locked_until"), datetime)
                    and record["locked_until"] > timestamp)

    @transaction_runner
    def record_failed_login(transaction, ref, timestamp):
        record = ref.get(transaction=transaction).to_dict() or {}
        if locked(record, timestamp):
            return True
        started = record.get("window_started")
        count = (record.get("failures", 0) + 1
                 if isinstance(started, datetime) and started + LOGIN_WINDOW > timestamp else 1)
        locked_until = timestamp + LOGIN_WINDOW if count >= MAX_LOGIN_FAILURES else None
        transaction.set(ref, {
            "failures": count, "window_started": started if count > 1 else timestamp,
            "locked_until": locked_until, "expires_at": timestamp + LOGIN_WINDOW * 2,
        })
        return locked_until is not None

    @transaction_runner
    def transition_case_status(transaction, ref, event_ref, new_status, timestamp):
        record = ref.get(transaction=transaction).to_dict()
        if not active_case(record):
            return "missing", None
        old_status = record["status"]
        if old_status == new_status:
            return "unchanged", old_status
        if (old_status, new_status) not in (("NEW", "UNDER_REVIEW"), ("UNDER_REVIEW", "CLOSED")):
            return "conflict", old_status
        transaction.update(ref, {"status": new_status, "updated_at": timestamp})
        transaction.set(event_ref, {
            "action": "status_changed", "from_status": old_status,
            "to_status": new_status,
            "reviewer": config["RISK_DEMO_REVIEWER_USERNAME"], "at": timestamp,
            "expires_at": record["expires_at"],
        })
        return "changed", new_status

    def bearer_subject():
        scheme, separator, token = request.headers.get("Authorization", "").partition(" ")
        if scheme.lower() != "bearer" or not separator or not token.strip():
            abort(401)
        try:
            line_sub = identity_verifier(token.strip(), line_client_id)
        except RuntimeError:
            abort(503)
        if not isinstance(line_sub, str) or not line_sub:
            abort(401)
        key = subject_key(line_sub)
        if contact_only and contact_auth_mode != "verified_line_self_service" and key not in (allowed_subject_keys or ()):
            abort(403)
        return key

    def require_reviewer():
        state = reviewer_state()
        if state is None:
            abort(401)
        return state

    def require_csrf():
        expected = require_reviewer()["csrf"]
        supplied = request.headers.get("X-Demo-CSRF", "")
        if not expected or not hmac.compare_digest(expected, supplied):
            abort(403)

    def contact_public(record):
        if (
            not record or record.get("consent") is not True
            or not isinstance(record.get("expires_at"), datetime)
            or record["expires_at"] <= now()
        ):
            return {"consent": False}
        return {
            "consent": True,
            "preferred_name": record.get("preferred_name", ""),
            "contact_method": record.get("contact_method", "none"),
            "contact_value": record.get("contact_value", ""),
        }

    def case_public(case_id, record):
        item = {
            "case_id": case_id,
            "status": record["status"],
            "signal_type": record["signal_type"],
            "source_kind": record["source_kind"],
            "scenario": record.get("scenario"),
            "simulation_actor": record.get("simulation_actor"),
            "assessment_source": record.get("assessment_source"),
            "ml_source": record.get("ml_source"),
            "synthetic": True,
            "created_at": record.get("created_at").isoformat() if isinstance(record.get("created_at"), datetime) else None,
            "updated_at": record.get("updated_at").isoformat() if isinstance(record.get("updated_at"), datetime) else None,
        }
        key = record.get("subject_key")
        contact = db.collection("risk_demo_contacts").document(key).get().to_dict() if key else None
        item["contact"] = contact_public(contact)
        return item

    def active_case(record):
        return bool(record and record.get("synthetic") is True
                    and isinstance(record.get("expires_at"), datetime)
                    and record["expires_at"] > now())

    @demo.after_request
    def no_store(response):
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @demo.get("/contact-form")
    def contact_form():
        return render_template("risk_demo_contact.html", liff_id=config["RISK_DEMO_LIFF_ID"],
                               contact_only=contact_only)

    @demo.route("/api/contact", methods=["GET", "PUT", "DELETE"])
    def contact():
        key = bearer_subject()
        ref = db.collection("risk_demo_contacts").document(key)
        if request.method == "GET":
            return jsonify(contact_public(ref.get().to_dict()))
        if request.method == "DELETE":
            # Overwrite rather than merge: all personal contact fields are removed.
            ref.set({"consent": False, "updated_at": now(), "expires_at": now() + CONTACT_RETENTION})
            return jsonify(consent=False)

        data = request.get_json(silent=True)
        if not isinstance(data, dict) or set(data) != {"preferred_name", "contact_method", "contact_value", "consent"}:
            abort(400)
        name, method, value = data["preferred_name"], data["contact_method"], data["contact_value"]
        if data["consent"] is not True or not isinstance(name, str) or not isinstance(value, str):
            abort(400)
        if not isinstance(method, str):
            abort(400)
        name, value = name.strip(), value.strip()
        if not name or len(name) > 80 or any(ord(char) < 32 for char in name):
            abort(400)
        if method == "none" and value:
            abort(400)
        if method == "phone" and not PHONE_PATTERN.fullmatch(value):
            abort(400)
        if method == "email" and (len(value) > 254 or not EMAIL_PATTERN.fullmatch(value)):
            abort(400)
        if method not in ("none", "phone", "email") or len(value) > 254:
            abort(400)
        ref.set({
            "consent": True, "preferred_name": name,
            "contact_method": method, "contact_value": value,
            "updated_at": now(), "expires_at": now() + CONTACT_RETENTION,
        })
        return jsonify(consent=True)

    if contact_only:
        app.register_blueprint(demo, url_prefix="/demo")
        return demo

    @demo.post("/api/cases/sample")
    def user_sample_case():
        key = bearer_subject()
        contact = db.collection("risk_demo_contacts").document(key).get().to_dict()
        if contact_public(contact).get("consent") is not True:
            abort(403)
        data = request.get_json(silent=True)
        if (not isinstance(data, dict)
                or set(data) not in ({"idempotency_key"}, {"idempotency_key", "scenario"})):
            abort(400)
        idem = data["idempotency_key"]
        if not isinstance(idem, str) or not re.fullmatch(r"[A-Za-z0-9_-]{16,80}", idem):
            abort(400)
        scenario = data.get("scenario", "review")
        if scenario not in ("review", "urgent"):
            abort(400)
        # Preserve the existing review case ID for old clients. An urgent
        # simulation with the same key is a separate synthetic case.
        namespace = "sample:" if scenario == "review" else "sample:urgent:"
        case_id = hmac.new(hmac_key, (namespace + key + ":" + idem).encode("utf-8"), hashlib.sha256).hexdigest()
        ref = db.collection("risk_demo_cases").document(case_id)
        timestamp = now()
        record = {
            "subject_key": key, "status": "NEW",
            "signal_type": "SYNTHETIC_URGENT" if scenario == "urgent" else "SYNTHETIC_REVIEW",
            "source_kind": "synthetic_demo", "scenario": scenario,
            "assessment_source": None,
            "ml_source": None, "synthetic": True,
            "created_at": timestamp, "updated_at": timestamp,
            "expires_at": timestamp + CASE_RETENTION,
        }
        try:
            ref.create(record)
            return jsonify(case_id=case_id, synthetic=True, duplicate=False), 201
        except Exception as error:
            # Firestore create is atomic; only an existing record is a retry.
            from google.api_core.exceptions import AlreadyExists
            if not isinstance(error, AlreadyExists):
                raise
            return jsonify(case_id=case_id, synthetic=True, duplicate=True), 200

    if enable_event_simulator:
        @demo.post("/api/events/simulate")
        def simulate_synthetic_event():
            key = bearer_subject()
            actor = {subject_key("synthetic-sub-a"): "A",
                     subject_key("synthetic-sub-b"): "B"}.get(key)
            if actor is None:
                abort(403)
            data = request.get_json(silent=True)
            if not isinstance(data, dict) or set(data) != {"scenario", "idempotency_key"}:
                abort(400)
            scenario, idem = data["scenario"], data["idempotency_key"]
            if not isinstance(scenario, str) or scenario not in SYNTHETIC_EVENTS:
                abort(400)
            if not isinstance(idem, str) or not re.fullmatch(r"[A-Za-z0-9_-]{16,80}", idem):
                abort(400)
            source_kind, signal_type = SYNTHETIC_EVENTS[scenario]
            if signal_type is None:
                return jsonify(created=False, case_id=None, scenario=scenario,
                               source_kind=source_kind, synthetic=True), 200
            contact = db.collection("risk_demo_contacts").document(key).get().to_dict()
            if contact_public(contact).get("consent") is not True:
                abort(403)
            case_id = hmac.new(hmac_key, ("event:" + scenario + ":" + key + ":" + idem).encode("utf-8"),
                               hashlib.sha256).hexdigest()
            timestamp = now()
            ref = db.collection("risk_demo_cases").document(case_id)
            record = {
                "subject_key": key, "simulation_actor": actor, "status": "NEW",
                "source_kind": source_kind, "scenario": scenario,
                "signal_type": signal_type, "synthetic": True,
                "assessment_source": None, "ml_source": None,
                "created_at": timestamp, "updated_at": timestamp,
                "expires_at": timestamp + CASE_RETENTION,
            }
            try:
                ref.create(record)
                return jsonify(created=True, case_id=case_id, duplicate=False,
                               scenario=scenario, source_kind=source_kind, synthetic=True), 201
            except Exception as error:
                from google.api_core.exceptions import AlreadyExists
                if not isinstance(error, AlreadyExists):
                    raise
                return jsonify(created=True, case_id=case_id, duplicate=True,
                               scenario=scenario, source_kind=source_kind, synthetic=True), 200

    @demo.get("/reviewer/login")
    def reviewer_login_page():
        return render_template("risk_demo_login.html")

    @demo.post("/api/reviewer/login")
    def reviewer_login():
        data = request.get_json(silent=True)
        if not isinstance(data, dict) or set(data) != {"username", "password"}:
            abort(400)
        username, password = data["username"], data["password"]
        if (not isinstance(username, str) or not isinstance(password, str)
                or len(username) > 80 or len(password) > 1024):
            abort(400)
        attempt = login_attempt_ref()
        if locked(attempt.get().to_dict(), now()):
            abort(429)
        username_ok = hmac.compare_digest(
            username.encode("utf-8"), config["RISK_DEMO_REVIEWER_USERNAME"].encode("utf-8"))
        password_ok = check_password_hash(config["RISK_DEMO_REVIEWER_PASSWORD_HASH"], password)
        if not username_ok or not password_ok:
            if record_failed_login(db.transaction(), attempt, now()):
                abort(429)
            abort(401)
        if locked(attempt.get().to_dict(), now()):
            abort(429)
        attempt.delete()
        token = secrets.token_urlsafe(32)
        digest = hashlib.sha256(token.encode("ascii")).hexdigest()
        timestamp = now()
        db.collection("risk_demo_reviewer_sessions").document(digest).create({
            "reviewer": username, "csrf": secrets.token_urlsafe(32),
            "created_at": timestamp, "expires_at": timestamp + SESSION_LIFETIME,
        })
        response = jsonify(ok=True)
        response.set_cookie(
            REVIEWER_COOKIE, token, max_age=int(SESSION_LIFETIME.total_seconds()),
            httponly=True, secure=os.environ.get("RISK_DEMO_COOKIE_SECURE", "1") == "1",
            samesite="Strict", path="/demo",
        )
        return response

    @demo.post("/api/reviewer/logout")
    def reviewer_logout():
        require_reviewer()
        require_csrf()
        reviewer_session_ref().delete()
        response = jsonify(ok=True)
        response.delete_cookie(REVIEWER_COOKIE, path="/demo")
        return response

    @demo.get("/reviewer")
    def reviewer_page():
        state = reviewer_state()
        if state is None:
            return redirect(url_for("risk_demo.reviewer_login_page"))
        return render_template("risk_demo_reviewer.html", csrf_token=state["csrf"])

    @demo.get("/api/reviewer/cases")
    def reviewer_cases():
        require_reviewer()
        cases = db.collection("risk_demo_cases").limit(50).stream()
        return jsonify(cases=[case_public(snap.id, record) for snap in cases
                             if active_case(record := snap.to_dict())])

    @demo.get("/api/reviewer/cases/<case_id>")
    def reviewer_case_detail(case_id):
        require_reviewer()
        if not CASE_ID_PATTERN.fullmatch(case_id):
            abort(404)
        ref = db.collection("risk_demo_cases").document(case_id)
        record = ref.get().to_dict()
        if not active_case(record):
            abort(404)
        item = case_public(case_id, record)
        item["audit_events"] = [snap.to_dict() for snap in ref.collection("risk_demo_audit_events").limit(50).stream()]
        for event in item["audit_events"]:
            if isinstance(event.get("at"), datetime):
                event["at"] = event["at"].isoformat()
        return jsonify(item)

    @demo.patch("/api/reviewer/cases/<case_id>/status")
    def reviewer_update_status(case_id):
        require_reviewer()
        require_csrf()
        if not CASE_ID_PATTERN.fullmatch(case_id):
            abort(404)
        data = request.get_json(silent=True)
        if not isinstance(data, dict) or set(data) != {"status"} or data["status"] not in CASE_STATUSES:
            abort(400)
        ref = db.collection("risk_demo_cases").document(case_id)
        event_ref = ref.collection("risk_demo_audit_events").document(secrets.token_hex(16))
        try:
            outcome, status = transition_case_status(db.transaction(), ref, event_ref, data["status"], now())
        except Aborted:
            return jsonify(error="transaction_conflict", retryable=True), 409
        except ValueError as error:
            # firestore.transactional raises ValueError after bounded Aborted retries.
            # Do not mask unrelated programming errors as a retryable conflict.
            if not isinstance(error.__cause__, Aborted):
                raise
            return jsonify(error="transaction_conflict", retryable=True), 409
        if outcome == "missing":
            abort(404)
        if outcome == "conflict":
            abort(409)
        return jsonify(status=status)

    @demo.post("/api/reviewer/sample")
    def reviewer_sample_case():
        require_reviewer()
        require_csrf()
        data = request.get_json(silent=True)
        if data is None:
            if request.content_length not in (None, 0):
                abort(400)
            data = {"scenario": "review"}
        if (not isinstance(data, dict) or set(data) != {"scenario"}
                or data["scenario"] not in ("general", "review", "urgent")):
            abort(400)
        scenario = data["scenario"]
        if scenario == "general":
            return jsonify(created=False, scenario=scenario, synthetic=True), 200
        case_id = secrets.token_hex(32)
        timestamp = now()
        db.collection("risk_demo_cases").document(case_id).create({
            "subject_key": None, "status": "NEW",
            "signal_type": "SYNTHETIC_URGENT" if scenario == "urgent" else "SYNTHETIC_REVIEW",
            "source_kind": "synthetic_demo", "scenario": scenario,
            "assessment_source": None,
            "ml_source": None, "synthetic": True,
            "created_at": timestamp, "updated_at": timestamp,
            "expires_at": timestamp + CASE_RETENTION,
        })
        return jsonify(case_id=case_id, created=True, scenario=scenario, synthetic=True), 201

    app.register_blueprint(demo, url_prefix="/demo")
    return demo
