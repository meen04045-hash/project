"""Run the contact-only API against a fresh, owned Firestore Emulator.

This test deliberately patches the TEST client factory and LINE verifier. It
cannot authenticate a real account or connect to Firebase Cloud.
"""

import hashlib
import hmac
import importlib
import json
import os
from pathlib import Path
import signal
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from google.auth.credentials import AnonymousCredentials
from google.cloud import firestore


HOST = "127.0.0.1"
PORT = 8091
PROJECT = "demo-risk-contact-flow"
TEST_PROJECT = "heal-heart-b51de"
EMULATOR_HOST = f"{HOST}:{PORT}"
FAKE_KEY = "synthetic-test-only-hmac-key-40-characters"
FAKE_SUB_A = "synthetic-contact-user-a"
FAKE_SUB_B = "synthetic-contact-user-b"
CONTACTS = "risk_demo_contacts"


def port_open():
    try:
        with socket.create_connection((HOST, PORT), timeout=0.3):
            return True
    except OSError:
        return False


def allowed_key(sub):
    return hmac.new(FAKE_KEY.encode(), ("subject:" + sub).encode(),
                    hashlib.sha256).hexdigest()


def require(condition, label):
    if not condition:
        raise AssertionError(label)
    print(f"PASS {label}")


def make_app(module, db, allowlist, auth_mode=None):
    test_env = {
        "RISK_CONTACT_TEST_PROJECT_ID": TEST_PROJECT,
        "RISK_CONTACT_TEST_CREDENTIALS_JSON": json.dumps({"project_id": TEST_PROJECT}),
        "RISK_CONTACT_TEST_IDENTITY_HMAC_KEY": FAKE_KEY,
        "RISK_CONTACT_TEST_LIFF_ID": "synthetic-liff",
        "RISK_CONTACT_TEST_LINE_LOGIN_CHANNEL_ID": "synthetic-channel",
        "RISK_CONTACT_TEST_ALLOWED_SUBJECT_KEYS": allowlist,
        "RISK_CONTACT_TEST_AUTH_MODE": auth_mode or ("legacy_allowlist" if allowlist else "deny_all"),
    }
    verifier = lambda token, channel: {
        "synthetic-token-a": FAKE_SUB_A,
        "synthetic-token-b": FAKE_SUB_B,
    }.get(token) if channel == "synthetic-channel" else None
    # Remove the emulator host only during app construction because production
    # startup must reject that variable. The already-built client stays bound
    # to the owned local emulator; no runtime authorization code is changed.
    with patch.dict(os.environ, test_env, clear=True), \
            patch.object(module, "create_contact_test_client", return_value=db):
        return module.create_app(identity_verifier=verifier)


def run_flow(db):
    require(db.project == PROJECT and
            db._firestore_api.transport._host == EMULATOR_HOST,
            "anonymous client bound to isolated emulator")
    require(not list(db.collections()), "fresh emulator database is empty")

    import risk_contact_test_client as client_module
    with patch.dict(os.environ, {
            "RISK_CONTACT_TEST_PROJECT_ID": TEST_PROJECT,
            "RISK_CONTACT_TEST_CREDENTIALS_JSON": json.dumps({"project_id": TEST_PROJECT}),
            "RISK_CONTACT_TEST_IDENTITY_HMAC_KEY": FAKE_KEY,
            "RISK_CONTACT_TEST_LIFF_ID": "synthetic-liff",
            "RISK_CONTACT_TEST_LINE_LOGIN_CHANNEL_ID": "synthetic-channel",
        }, clear=True), patch.object(client_module, "create_contact_test_client", return_value=db):
        module = importlib.import_module("risk_contact_test_app")

    with patch("risk_case_demo.urlopen", side_effect=AssertionError("Unexpected LINE call")):
        denied = make_app(module, db, "").test_client()
        a = {"Authorization": "Bearer synthetic-token-a"}
        b = {"Authorization": "Bearer synthetic-token-b"}
        require(denied.get("/demo/api/contact", headers=a).status_code == 403 and
                not list(db.collections()), "deny-all blocks contact before Firestore access")

        app = make_app(module, db, allowed_key(FAKE_SUB_A))
        routes = {rule.rule for rule in app.url_map.iter_rules() if rule.endpoint != "static"}
        require(routes == {"/health", "/demo/contact-form", "/demo/api/contact"} and
                "app" not in sys.modules, "TEST routes only; production app not imported")
        client = app.test_client()
        ref_a = db.collection(CONTACTS).document(allowed_key(FAKE_SUB_A))
        ref_b = db.collection(CONTACTS).document(allowed_key(FAKE_SUB_B))
        first = {"preferred_name": "Synthetic A", "contact_method": "email",
                 "contact_value": "contact-a@example.invalid", "consent": True}
        changed = {**first, "preferred_name": "Synthetic A Updated",
                   "contact_value": "contact-a-updated@example.invalid"}

        require(client.put("/demo/api/contact", json={**first, "consent": False},
                           headers=a).status_code == 400 and not ref_a.get().exists,
                "consent required before save")
        require(client.put("/demo/api/contact", json=first, headers=a).status_code == 200
                and ref_a.get().to_dict()["contact_value"] == first["contact_value"],
                "save fictional contact in emulator")
        require(client.get("/demo/api/contact", headers=a).json["contact_value"]
                == first["contact_value"], "read same subject contact")
        require(client.get("/demo/api/contact", headers=b).status_code == 403 and
                client.put("/demo/api/contact", json=first, headers=b).status_code == 403
                and not ref_b.get().exists, "other identity denied and isolated")
        require(client.put("/demo/api/contact", json=changed, headers=a).status_code == 200
                and client.get("/demo/api/contact", headers=a).json["contact_value"]
                == changed["contact_value"], "update fictional contact")
        saved = ref_a.get().to_dict()
        saved["expires_at"] = datetime.now(timezone.utc) - timedelta(seconds=1)
        ref_a.set(saved)
        require(client.get("/demo/api/contact", headers=a).json == {"consent": False},
                "expired contact is hidden")
        saved["expires_at"] = datetime.now(timezone.utc) + timedelta(days=1)
        ref_a.set(saved)
        require(client.delete("/demo/api/contact", headers=a).status_code == 200 and
                client.get("/demo/api/contact", headers=a).json == {"consent": False}
                and not any(key in ref_a.get().to_dict() for key in
                            ("preferred_name", "contact_method", "contact_value")),
                "withdraw consent and reject stale contact disclosure")

        # The manual cleanup tool is deliberately locked to the TEST project.
        # Patch only that guard inside this synthetic test so its transaction
        # can be exercised against the owned demo-* emulator project.
        from risk_contact_test_cleanup import CONFIRMATION, cleanup_contacts
        expired = ref_a.get().to_dict()
        expired["expires_at"] = datetime.now(timezone.utc) - timedelta(days=1)
        ref_a.set(expired)
        with patch("risk_contact_test_cleanup.TEST_PROJECT_ID", PROJECT):
            dry = cleanup_contacts(db)
            require(dry == {"eligible": 1, "deleted": 0, "skipped": 0}
                    and ref_a.get().exists, "cleanup dry-run does not delete")
            result = cleanup_contacts(db, execute=True, confirm_count=1,
                                      confirmation=CONFIRMATION)
        require(result == {"eligible": 1, "deleted": 1, "skipped": 0}
                and not ref_a.get().exists and not ref_b.get().exists,
                "cleanup deletes only expired synthetic contact")

        self_service = make_app(module, db, "", "verified_line_self_service").test_client()
        require(self_service.get("/demo/api/contact", headers=a).json == {"consent": False}
                and self_service.get("/demo/api/contact", headers=b).json == {"consent": False},
                "self-service starts with separate empty contacts")
        saved_a = self_service.put("/demo/api/contact", json=first, headers=a)
        saved_b = self_service.put("/demo/api/contact", json={**first,
                    "preferred_name": "Synthetic B"}, headers=b)
        require(saved_a.status_code == 200 and saved_b.status_code == 200,
                "both verified synthetic users may consent independently")
        receipt_a = saved_a.json["demo_receipt"]
        require(ref_a.get().to_dict()["demo_receipt"] == receipt_a
                and self_service.get("/demo/api/contact", headers=a).json["demo_receipt"]
                    == receipt_a and receipt_a != saved_b.json["demo_receipt"],
                "random receipt matches only its saved TEST contact record")
        require(self_service.get("/demo/api/contact", headers=a).json["preferred_name"]
                == "Synthetic A" and
                self_service.get("/demo/api/contact?user_id=synthetic-contact-user-a",
                                 headers=b).json["preferred_name"] == "Synthetic B",
                "self-service never resolves client-supplied account identity")
        require(self_service.delete("/demo/api/contact", headers=a).status_code == 200
                and self_service.get("/demo/api/contact", headers=a).json == {"consent": False}
                and "demo_receipt" not in ref_a.get().to_dict()
                and self_service.get("/demo/api/contact", headers=b).json["consent"] is True,
                "self-service withdrawal does not affect another user")


def main():
    if port_open():
        raise RuntimeError("8091 is already in use; refusing to attach to an existing emulator")
    firebase = shutil.which("firebase")
    if not firebase or not shutil.which("java"):
        raise RuntimeError("Firebase CLI or Java unavailable")
    if any(os.environ.get(name) for name in (
            "GOOGLE_APPLICATION_CREDENTIALS", "FIREBASE_CREDENTIALS_JSON",
            "RISK_CONTACT_TEST_CREDENTIALS_JSON")):
        raise RuntimeError("Cloud credential environment detected; refusing to start")

    with tempfile.TemporaryDirectory(prefix="risk-contact-8091-") as folder:
        root = Path(folder)
        config = root / "firebase.json"
        config.write_text(json.dumps({"emulators": {"firestore": {
            "host": HOST, "port": PORT}, "ui": {"enabled": False}}}), encoding="utf-8")
        # Do not pass any inherited project, identity, or cloud configuration
        # to the child emulator process, even when the caller has it locally.
        env = {key: value for key, value in os.environ.items()
               if not key.startswith(("RISK_", "FIREBASE_", "FIRESTORE_",
                                      "GOOGLE_", "GCLOUD_", "LINE_"))}
        env.update({"FIRESTORE_EMULATOR_HOST": EMULATOR_HOST,
                    "FIREBASE_CLI_DISABLE_TELEMETRY": "1", "CI": "1"})
        command = [firebase, "emulators:start", "--only", "firestore",
                   "--project", PROJECT, "--config", str(config)]
        with (root / "emulator.log").open("w", encoding="utf-8") as log:
            flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
            process = subprocess.Popen(command, cwd=root, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, creationflags=flags,
                                       start_new_session=os.name != "nt")
            try:
                deadline = time.monotonic() + 90
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError("isolated emulator exited before ready")
                    log.flush()
                    startup = (root / "emulator.log").read_text(encoding="utf-8", errors="replace")
                    if "All emulators ready" in startup and port_open():
                        break
                    time.sleep(0.5)
                else:
                    raise RuntimeError("isolated emulator not ready within 90 seconds")
                with patch.dict(os.environ, {"FIRESTORE_EMULATOR_HOST": EMULATOR_HOST}):
                    db = firestore.Client(project=PROJECT, credentials=AnonymousCredentials())
                    run_flow(db)
            finally:
                if process.poll() is None:
                    if os.name == "nt":
                        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                       capture_output=True, check=False)
                    else:
                        os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        if os.name == "nt":
                            process.kill()
                        else:
                            os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                deadline = time.monotonic() + 5
                while port_open() and time.monotonic() < deadline:
                    time.sleep(0.1)
                if port_open():
                    raise RuntimeError("owned emulator port remains open after teardown")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, RuntimeError) as error:
        print(f"FAIL {error}", file=sys.stderr)
        raise SystemExit(1)
