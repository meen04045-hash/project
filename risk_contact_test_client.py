"""Create a separate, explicitly checked TEST Firestore client for contact-only mode."""

import json
import os

import firebase_admin
from firebase_admin import credentials, firestore

TEST_PROJECT_ID = "heal-heart-b51de"
PRODUCTION_PROJECT_ID = "heal-heart-e869a"

def create_contact_test_client(primary_project=None):
    expected = os.environ.get("RISK_CONTACT_TEST_PROJECT_ID", "")
    raw = os.environ.get("RISK_CONTACT_TEST_CREDENTIALS_JSON", "")
    if (expected != TEST_PROJECT_ID or expected == PRODUCTION_PROJECT_ID
            or not raw or (primary_project is not None and expected == primary_project)):
        raise RuntimeError("Contact TEST Firebase configuration is missing or not isolated")
    try:
        data = json.loads(raw)
        if not isinstance(data, dict) or data.get("project_id") != expected:
            raise ValueError("project mismatch")
        credential = credentials.Certificate(data)
        if credential.project_id != expected:
            raise ValueError("credential project mismatch")
        contact_app = firebase_admin.initialize_app(
            credential, {"projectId": expected}, name="risk-contact-test")
        if contact_app.project_id != expected:
            raise ValueError("Admin app project mismatch")
        client = firestore.client(app=contact_app)
        if client.project != expected:
            raise ValueError("client project mismatch")
    except Exception:
        raise RuntimeError("Contact TEST Firebase project verification failed") from None
    return client
