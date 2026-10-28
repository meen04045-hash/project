"""Isolated TEST entrypoint checks. All Firestore and LINE calls are mocked."""

import copy
import importlib
import os
import unittest
from unittest.mock import patch


class FakeSnapshot:
    def __init__(self, data):
        self.data = data

    def to_dict(self):
        return copy.deepcopy(self.data)


class FakeDoc:
    def __init__(self, store, path):
        self.store = store
        self.path = path

    def get(self):
        return FakeSnapshot(self.store.get(self.path))

    def set(self, data):
        self.store[self.path] = copy.deepcopy(data)


class FakeCollection:
    def __init__(self, store, name):
        self.store = store
        self.name = name

    def document(self, key):
        return FakeDoc(self.store, self.name + "/" + key)


class FakeDb:
    project = "heal-heart-b51de"

    def __init__(self):
        self.store = {}

    def collection(self, name):
        return FakeCollection(self.store, name)


TEST_ENV = {
    "RISK_CONTACT_TEST_PROJECT_ID": "heal-heart-b51de",
    "RISK_CONTACT_TEST_CREDENTIALS_JSON": '{"project_id":"heal-heart-b51de"}',
    "RISK_CONTACT_TEST_IDENTITY_HMAC_KEY": "h" * 40,
    "RISK_CONTACT_TEST_LIFF_ID": "synthetic-liff-id",
    "RISK_CONTACT_TEST_LINE_LOGIN_CHANNEL_ID": "synthetic-channel-id",
}


class IsolatedContactAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = FakeDb()
        # Import must create its application from the TEST-only client; no
        # production app or real Admin SDK client is initialized in this test.
        with patch.dict(os.environ, TEST_ENV, clear=True), \
                patch("risk_contact_test_client.create_contact_test_client", return_value=cls.db):
            cls.module = importlib.import_module("risk_contact_test_app")

    def setUp(self):
        self.db.store.clear()

    def make_app(self, verifier=lambda token, channel: "synthetic-user-a"):
        with patch.dict(os.environ, TEST_ENV, clear=True), \
                patch.object(self.module, "create_contact_test_client", return_value=self.db):
            return self.module.create_app(identity_verifier=verifier)

    def test_route_isolation_and_health(self):
        app = self.make_app()
        routes = {rule.rule: set(rule.methods) - {"HEAD", "OPTIONS"}
                  for rule in app.url_map.iter_rules() if rule.endpoint != "static"}
        self.assertEqual(routes, {
            "/health": {"GET"},
            "/demo/contact-form": {"GET"},
            "/demo/api/contact": {"GET", "PUT", "DELETE"},
        })
        client = app.test_client()
        health = client.get("/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json, {"status": "ok"})
        self.assertEqual(health.headers["Cache-Control"], "no-store")
        for path in ("/callback", "/send", "/demo/reviewer",
                     "/demo/api/cases/sample", "/demo/api/events/simulate"):
            with self.subTest(path=path):
                self.assertEqual(client.get(path).status_code, 404)
                self.assertEqual(client.post(path).status_code, 404)

    def test_bootstrap_has_health_only_and_no_external_side_effects(self):
        # No normal-mode config is needed. The WSGI module is also reloaded
        # under the flag to check its import-time app creation path.
        with patch.dict(os.environ, {"RISK_CONTACT_TEST_BOOTSTRAP": "1"}, clear=True), \
                patch.object(self.module, "create_contact_test_client") as create_client, \
                patch("firebase_admin.initialize_app") as initialize_admin, \
                patch("risk_case_demo.urlopen") as call_line, \
                patch("risk_contact_test_cleanup.cleanup_contacts") as cleanup:
            app = importlib.reload(self.module).app
            routes = {rule.rule: set(rule.methods) - {"HEAD", "OPTIONS"}
                      for rule in app.url_map.iter_rules() if rule.endpoint != "static"}
            self.assertEqual(routes, {"/health": {"GET"}})
            client = app.test_client()
            response = client.get("/health")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json, {"status": "ok"})
            self.assertEqual(response.headers["Cache-Control"], "no-store")
            for path in ("/demo/contact-form", "/demo/api/contact", "/callback",
                         "/send", "/demo/reviewer", "/demo/api/cases/sample",
                         "/demo/api/events/simulate", "/local-demo/users"):
                for method in ("GET", "PUT", "DELETE", "POST"):
                    with self.subTest(path=path, method=method):
                        self.assertEqual(client.open(path, method=method).status_code, 404)
            create_client.assert_not_called()
            initialize_admin.assert_not_called()
            call_line.assert_not_called()
            cleanup.assert_not_called()

    def test_bootstrap_exact_flag_and_conflicting_settings_fail_closed(self):
        for value in ("true", "TRUE", "yes", "01", " 1 ", "2"):
            with self.subTest(value=value), patch.dict(os.environ, {
                    "RISK_CONTACT_TEST_BOOTSTRAP": value}, clear=True), \
                    patch.object(self.module, "create_contact_test_client") as factory:
                with self.assertRaisesRegex(RuntimeError, "Invalid TEST bootstrap"):
                    self.module.create_app()
                factory.assert_not_called()
        for name, value in (("RISK_CONTACT_TEST_PROJECT_ID", "heal-heart-e869a"),
                            ("RISK_CONTACT_TEST_CREDENTIALS_JSON", "synthetic"),
                            ("FIREBASE_CREDENTIALS_JSON", "synthetic"),
                            ("GOOGLE_APPLICATION_CREDENTIALS", "synthetic"),
                            ("FIRESTORE_EMULATOR_HOST", "127.0.0.1:8089"),
                            ("RISK_DEMO_ENABLED", "1")):
            with self.subTest(name=name), patch.dict(os.environ, {
                    "RISK_CONTACT_TEST_BOOTSTRAP": "1", name: value}, clear=True), \
                    patch.object(self.module, "create_contact_test_client") as factory:
                with self.assertRaisesRegex(RuntimeError, "Conflicting TEST bootstrap"):
                    self.module.create_app()
                factory.assert_not_called()

    def test_zero_flag_keeps_normal_mode_and_missing_config_fails(self):
        with patch.dict(os.environ, {**TEST_ENV, "RISK_CONTACT_TEST_BOOTSTRAP": "0"}, clear=True), \
                patch.object(self.module, "create_contact_test_client", return_value=self.db):
            app = self.module.create_app(identity_verifier=lambda token, channel: None)
            self.assertIn("/demo/api/contact", {rule.rule for rule in app.url_map.iter_rules()})
        with patch.dict(os.environ, {"RISK_CONTACT_TEST_BOOTSTRAP": "0"}, clear=True), \
                patch.object(self.module, "create_contact_test_client") as factory:
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                self.module.create_app()
            factory.assert_not_called()

    def test_missing_and_conflicting_configuration_fails_before_client_creation(self):
        for missing in TEST_ENV:
            with self.subTest(missing=missing), patch.dict(os.environ, {
                    **TEST_ENV, missing: ""}, clear=True), \
                    patch.object(self.module, "create_contact_test_client") as create_client:
                with self.assertRaises(RuntimeError):
                    self.module.create_app()
                create_client.assert_not_called()
        for wrong_project in ("heal-heart-e869a", "other-project"):
            with self.subTest(wrong_project=wrong_project), patch.dict(os.environ, {
                    **TEST_ENV, "RISK_CONTACT_TEST_PROJECT_ID": wrong_project}, clear=True), \
                    patch.object(self.module, "create_contact_test_client") as create_client:
                with self.assertRaises(RuntimeError):
                    self.module.create_app()
                create_client.assert_not_called()
        for conflicting in ("RISK_DEMO_ENABLED", "FIREBASE_CREDENTIALS_JSON",
                            "GOOGLE_APPLICATION_CREDENTIALS", "FIRESTORE_EMULATOR_HOST"):
            value = "1" if conflicting == "RISK_DEMO_ENABLED" else "synthetic-conflict"
            with self.subTest(conflicting=conflicting), patch.dict(os.environ, {
                    **TEST_ENV, conflicting: value}, clear=True), \
                    patch.object(self.module, "create_contact_test_client") as create_client:
                with self.assertRaises(RuntimeError):
                    self.module.create_app()
                create_client.assert_not_called()

    def test_invalid_token_and_verify_outage_fail_closed(self):
        for verifier, expected in ((lambda token, channel: None, 401),
                                   (lambda token, channel: (_ for _ in ()).throw(
                                       RuntimeError("verify unavailable")), 503)):
            with self.subTest(expected=expected):
                client = self.make_app(verifier).test_client()
                response = client.get("/demo/api/contact", headers={
                    "Authorization": "Bearer synthetic-invalid-token"})
                self.assertEqual(response.status_code, expected)
                self.assertEqual(response.headers["Cache-Control"], "no-store")
                self.assertNotIn("synthetic-invalid-token", response.get_data(as_text=True))

    def test_contact_isolation_consent_and_withdrawal(self):
        def verifier(token, channel):
            self.assertEqual(channel, TEST_ENV["RISK_CONTACT_TEST_LINE_LOGIN_CHANNEL_ID"])
            return {"synthetic-token-a": "synthetic-user-a",
                    "synthetic-token-b": "synthetic-user-b"}.get(token)

        client = self.make_app(verifier).test_client()
        a = {"Authorization": "Bearer synthetic-token-a"}
        b = {"Authorization": "Bearer synthetic-token-b"}
        payload = {"preferred_name": "Test A", "contact_method": "none",
                   "contact_value": "", "consent": True}
        self.assertEqual(client.put("/demo/api/contact", json=payload, headers=a).status_code, 200)
        self.assertEqual(client.get("/demo/api/contact", headers=a).json["preferred_name"], "Test A")
        self.assertEqual(client.get("/demo/api/contact", headers=b).json, {"consent": False})
        self.assertEqual(client.delete("/demo/api/contact", headers=a).status_code, 200)
        self.assertEqual(client.get("/demo/api/contact", headers=a).json, {"consent": False})
        self.assertEqual(client.get("/demo/api/contact", headers=b).json, {"consent": False})
        self.assertTrue(all(path.startswith("risk_demo_contacts/") for path in self.db.store))
        self.assertFalse(any("synthetic-user-a" in path for path in self.db.store))

    def test_client_identity_fields_are_rejected(self):
        client = self.make_app().test_client()
        payload = {"preferred_name": "Test A", "contact_method": "none",
                   "contact_value": "", "consent": True, "user_id": "synthetic-user-b"}
        response = client.put("/demo/api/contact", json=payload,
                              headers={"Authorization": "Bearer synthetic-token-a"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.db.store, {})


if __name__ == "__main__":
    unittest.main()
