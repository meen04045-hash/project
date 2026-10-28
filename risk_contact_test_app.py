"""Isolated TEST-only contact form service; never imports the chatbot app."""

import os
import re

from flask import Flask, jsonify

from risk_case_demo import register_risk_demo
from risk_contact_test_client import TEST_PROJECT_ID, create_contact_test_client


def create_app(identity_verifier=None):
    bootstrap = os.environ.get("RISK_CONTACT_TEST_BOOTSTRAP", "")
    if bootstrap not in ("", "0", "1"):
        raise RuntimeError("Invalid TEST bootstrap setting")
    if bootstrap == "1":
        # Deliberately create no Firebase client or contact blueprint. The
        # initial Render deploy can expose only a process health endpoint.
        project = os.environ.get("RISK_CONTACT_TEST_PROJECT_ID", "")
        if (project not in ("", TEST_PROJECT_ID)
                or os.environ.get("RISK_CONTACT_TEST_CREDENTIALS_JSON")
                or os.environ.get("FIREBASE_CREDENTIALS_JSON")
                or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
                or os.environ.get("FIRESTORE_EMULATOR_HOST")
                or os.environ.get("RISK_DEMO_ENABLED") == "1"):
            raise RuntimeError("Conflicting TEST bootstrap configuration")
        app = Flask(__name__)
        @app.get("/health")
        def bootstrap_health():
            response = jsonify(status="ok")
            response.headers["Cache-Control"] = "no-store"
            return response
        return app

    if (os.environ.get("RISK_CONTACT_TEST_PROJECT_ID") != TEST_PROJECT_ID
            or not os.environ.get("RISK_CONTACT_TEST_CREDENTIALS_JSON")):
        raise RuntimeError("Contact TEST Firebase configuration is incomplete")
    if (os.environ.get("RISK_DEMO_ENABLED") == "1"
            or os.environ.get("FIREBASE_CREDENTIALS_JSON")
            or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
            or os.environ.get("FIRESTORE_EMULATOR_HOST")):
        raise RuntimeError("Contact TEST service has conflicting Firebase configuration")
    settings = {
        "RISK_DEMO_IDENTITY_HMAC_KEY": os.environ.get("RISK_CONTACT_TEST_IDENTITY_HMAC_KEY", ""),
        "RISK_DEMO_LIFF_ID": os.environ.get("RISK_CONTACT_TEST_LIFF_ID", ""),
    }
    channel_id = os.environ.get("RISK_CONTACT_TEST_LINE_LOGIN_CHANNEL_ID", "")
    if not all(settings.values()) or not channel_id:
        raise RuntimeError("Contact TEST service configuration is incomplete")
    raw_allowlist = os.environ.get("RISK_CONTACT_TEST_ALLOWED_SUBJECT_KEYS", "")
    entries = raw_allowlist.split(",") if raw_allowlist else []
    if any(not re.fullmatch(r"[0-9a-f]{64}", entry) for entry in entries) or len(entries) != len(set(entries)):
        raise RuntimeError("Invalid TEST identity allowlist")
    db = create_contact_test_client()
    app = Flask(__name__)
    register_risk_demo(app, db, channel_id, identity_verifier=identity_verifier,
                       contact_only=True, contact_settings=settings,
                       allowed_subject_keys=frozenset(entries))

    @app.get("/health")
    def health():
        response = jsonify(status="ok")
        response.headers["Cache-Control"] = "no-store"
        return response

    return app


app = create_app()
