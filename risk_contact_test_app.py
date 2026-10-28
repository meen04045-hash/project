"""Isolated TEST-only contact form service; never imports the chatbot app."""

import os

from flask import Flask, jsonify

from risk_case_demo import register_risk_demo
from risk_contact_test_client import TEST_PROJECT_ID, create_contact_test_client


def create_app(identity_verifier=None):
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
    db = create_contact_test_client()
    app = Flask(__name__)
    register_risk_demo(app, db, channel_id, identity_verifier=identity_verifier,
                       contact_only=True, contact_settings=settings)

    @app.get("/health")
    def health():
        response = jsonify(status="ok")
        response.headers["Cache-Control"] = "no-store"
        return response

    return app


app = create_app()
