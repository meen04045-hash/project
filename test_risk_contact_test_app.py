"""Isolated TEST entrypoint checks. All Firestore and LINE calls are mocked."""

import copy
import hashlib
import hmac
import importlib
import json
import os
import unittest
from unittest.mock import Mock, patch


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
    "RISK_CONTACT_TEST_AUTH_MODE": "legacy_allowlist",
}


def allowed_key(sub):
    return hmac.new(TEST_ENV["RISK_CONTACT_TEST_IDENTITY_HMAC_KEY"].encode(),
                    ("subject:" + sub).encode(), hashlib.sha256).hexdigest()

TEST_ENV["RISK_CONTACT_TEST_ALLOWED_SUBJECT_KEYS"] = allowed_key("synthetic-user-a")


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

    def make_app(self, verifier=lambda token, channel: "synthetic-user-a", env=None):
        with patch.dict(os.environ, env if env is not None else TEST_ENV, clear=True), \
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
        self.assertIn("ข้อมูลสมมติ", client.get("/demo/contact-form").get_data(as_text=True))
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
        for missing in TEST_ENV.keys() - {"RISK_CONTACT_TEST_ALLOWED_SUBJECT_KEYS",
                                          "RISK_CONTACT_TEST_AUTH_MODE"}:
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
        saved_a = client.put("/demo/api/contact", json=payload, headers=a)
        self.assertEqual(saved_a.status_code, 200)
        receipt_a = saved_a.json["demo_receipt"]
        self.assertRegex(receipt_a, r"^[0-9A-F]{16}$")
        self.assertEqual(self.db.store["risk_demo_contacts/" + allowed_key("synthetic-user-a")]
                         ["demo_receipt"], receipt_a)
        self.assertEqual(client.get("/demo/api/contact", headers=a).json["demo_receipt"], receipt_a)
        self.assertIn("risk_demo_contacts/" + allowed_key("synthetic-user-a"), self.db.store)
        self.assertEqual(client.get("/demo/api/contact", headers=a).json["preferred_name"], "Test A")
        self.assertEqual(client.get("/demo/api/contact", headers=b).status_code, 403)
        self.assertEqual(client.delete("/demo/api/contact", headers=a).status_code, 200)
        self.assertEqual(client.get("/demo/api/contact", headers=a).json, {"consent": False})
        self.assertEqual(client.get("/demo/api/contact", headers=b).status_code, 403)
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

    def test_allowlist_missing_empty_and_malformed_fail_closed(self):
        for raw in (None, ""):
            with self.subTest(raw=raw):
                env = {**TEST_ENV, "RISK_CONTACT_TEST_ALLOWED_SUBJECT_KEYS": raw or ""}
                client = self.make_app(env=env).test_client()
                self.assertEqual(client.get("/demo/api/contact", headers={
                    "Authorization": "Bearer synthetic-token"}).status_code, 403)
                self.assertEqual(self.db.store, {})
        good = allowed_key("synthetic-user-a")
        for raw in ("wrong", good + ",", good + "," + good, good.upper(), " " + good):
            with self.subTest(raw=raw), patch.dict(os.environ, {
                    **TEST_ENV, "RISK_CONTACT_TEST_ALLOWED_SUBJECT_KEYS": raw}, clear=True), \
                    patch.object(self.module, "create_contact_test_client") as factory:
                with self.assertRaisesRegex(RuntimeError, "allowlist"):
                    self.module.create_app()
                factory.assert_not_called()

    def test_default_deny_all_and_invalid_modes_fail_closed(self):
        for mode in (None, "deny_all"):
            with self.subTest(mode=mode):
                env = {**TEST_ENV, "RISK_CONTACT_TEST_ALLOWED_SUBJECT_KEYS": ""}
                if mode is None:
                    env.pop("RISK_CONTACT_TEST_AUTH_MODE")
                else:
                    env["RISK_CONTACT_TEST_AUTH_MODE"] = mode
                client = self.make_app(env=env).test_client()
                self.assertEqual(client.get("/demo/api/contact", headers={
                    "Authorization": "Bearer synthetic-token"}).status_code, 403)
        for mode in ("", "open", "VERIFIED_LINE_SELF_SERVICE"):
            with self.subTest(mode=mode), patch.dict(os.environ, {
                    **TEST_ENV, "RISK_CONTACT_TEST_AUTH_MODE": mode}, clear=True), \
                    patch.object(self.module, "create_contact_test_client") as factory:
                with self.assertRaisesRegex(RuntimeError, "authorization mode"):
                    self.module.create_app()
                factory.assert_not_called()

    def test_self_service_requires_verified_identity_and_isolates_subjects(self):
        env = {**TEST_ENV, "RISK_CONTACT_TEST_AUTH_MODE": "verified_line_self_service",
               "RISK_CONTACT_TEST_ALLOWED_SUBJECT_KEYS": ""}
        verifier = lambda token, channel: {"a": "synthetic-user-a",
                                           "b": "synthetic-user-b"}.get(token)
        client = self.make_app(verifier, env).test_client()
        a = {"Authorization": "Bearer a"}
        b = {"Authorization": "Bearer b"}
        payload = {"preferred_name": "Synthetic A", "contact_method": "email",
                   "contact_value": "a@example.invalid", "consent": True}
        self.assertEqual(client.get("/demo/api/contact").status_code, 401)
        self.assertEqual(client.get("/demo/api/contact", headers={
            "Authorization": "Bearer invalid"}).status_code, 401)
        self.assertEqual(client.put("/demo/api/contact", json={**payload, "consent": False},
                                    headers=a).status_code, 400)
        self.assertEqual(client.put("/demo/api/contact", json={**payload,
            "user_id": "synthetic-user-b"}, headers=a).status_code, 400)
        saved_a = client.put("/demo/api/contact", json=payload, headers=a)
        self.assertEqual(saved_a.status_code, 200)
        receipt_a = saved_a.json["demo_receipt"]
        self.assertRegex(receipt_a, r"^[0-9A-F]{16}$")
        self.assertEqual(self.db.store["risk_demo_contacts/" + allowed_key("synthetic-user-a")]
                         ["demo_receipt"], receipt_a)
        self.assertEqual(client.get("/demo/api/contact", headers=b).json, {"consent": False})
        self.assertEqual(client.get("/demo/api/contact?user_id=synthetic-user-a",
                                    headers=b).json, {"consent": False})
        self.assertEqual(client.put("/demo/api/contact", json={**payload,
            "preferred_name": "Synthetic B"}, headers=b).status_code, 200)
        self.assertEqual(client.get("/demo/api/contact", headers=a).json["preferred_name"],
                         "Synthetic A")
        self.assertEqual(client.get("/demo/api/contact", headers=b).json["preferred_name"],
                         "Synthetic B")
        self.assertEqual(client.get("/demo/api/contact?demo_receipt=" + receipt_a,
                                    headers=b).json["preferred_name"], "Synthetic B")
        self.assertEqual(client.delete("/demo/api/contact", headers=a).status_code, 200)
        self.assertEqual(client.get("/demo/api/contact", headers=a).json, {"consent": False})
        self.assertNotIn("demo_receipt", self.db.store[
            "risk_demo_contacts/" + allowed_key("synthetic-user-a")])
        self.assertEqual(client.get("/demo/api/contact", headers=b).json["preferred_name"],
                         "Synthetic B")
        self.assertEqual({rule.rule for rule in self.make_app(verifier, env).url_map.iter_rules()
                          if rule.endpoint != "static"},
                         {"/health", "/demo/contact-form", "/demo/api/contact"})

    def test_self_service_rejects_nonempty_legacy_allowlist_before_client(self):
        with patch.dict(os.environ, {**TEST_ENV,
                "RISK_CONTACT_TEST_AUTH_MODE": "verified_line_self_service"}, clear=True), \
                patch.object(self.module, "create_contact_test_client") as factory:
            with self.assertRaisesRegex(RuntimeError, "conflicts with legacy allowlist"):
                self.module.create_app()
            factory.assert_not_called()

    def test_self_service_burst_limits_and_window_expiry(self):
        env = {**TEST_ENV, "RISK_CONTACT_TEST_AUTH_MODE": "verified_line_self_service",
               "RISK_CONTACT_TEST_ALLOWED_SUBJECT_KEYS": ""}
        client = self.make_app(lambda token, channel: {
            "a": "synthetic-user-a", "b": "synthetic-user-b"}.get(token), env).test_client()
        payload = {"preferred_name": "Synthetic", "contact_method": "none",
                   "contact_value": "", "consent": True}
        with patch("risk_case_demo.monotonic", return_value=100):
            for _ in range(30):
                self.assertEqual(client.get("/demo/api/contact").status_code, 401)
            limited = client.get("/demo/api/contact")
            self.assertEqual(limited.status_code, 429)
            self.assertEqual(limited.headers["Retry-After"], "60")
            self.assertEqual(limited.headers["Cache-Control"], "no-store")
        with patch("risk_case_demo.monotonic", return_value=161):
            self.assertEqual(client.get("/demo/api/contact").status_code, 401)
            for _ in range(6):
                self.assertEqual(client.put("/demo/api/contact", json=payload,
                                            headers={"Authorization": "Bearer a"}).status_code, 200)
            self.assertEqual(client.put("/demo/api/contact", json=payload,
                                        headers={"Authorization": "Bearer a"}).status_code, 429)
            self.assertEqual(client.put("/demo/api/contact", json=payload,
                                        headers={"Authorization": "Bearer b"}).status_code, 200)
        with patch("risk_case_demo.monotonic", return_value=222):
            self.assertEqual(client.put("/demo/api/contact", json=payload,
                                        headers={"Authorization": "Bearer a"}).status_code, 200)

    def test_self_service_rejects_oversize_and_malformed_before_line_verify(self):
        env = {**TEST_ENV, "RISK_CONTACT_TEST_AUTH_MODE": "verified_line_self_service",
               "RISK_CONTACT_TEST_ALLOWED_SUBJECT_KEYS": ""}
        verifier = Mock(return_value="synthetic-user-a")
        client = self.make_app(verifier, env).test_client()
        path = "/demo/api/contact"
        self.assertEqual(client.put(path, data="plain", content_type="text/plain").status_code, 415)
        self.assertEqual(client.put(path, data="{", content_type="application/json").status_code, 400)
        self.assertEqual(client.put(path, json={"preferred_name": "x" * 2000,
            "contact_method": "none", "contact_value": "", "consent": True}).status_code, 413)
        self.assertEqual(client.post(path).status_code, 405)
        verifier.assert_not_called()
        self.assertEqual(self.db.store, {})

    def test_contact_limit_cache_is_bounded(self):
        from risk_case_demo import _BoundedWindowLimiter
        limiter = _BoundedWindowLimiter(1, max_keys=2)
        with patch("risk_case_demo.monotonic", return_value=100):
            for key in ("synthetic-a", "synthetic-b", "synthetic-c"):
                self.assertTrue(limiter.allow(key))
            self.assertLessEqual(len(limiter.events), 2)

    def test_synthetic_thai_utf8_form_fits_request_limit(self):
        env = {**TEST_ENV, "RISK_CONTACT_TEST_AUTH_MODE": "verified_line_self_service",
               "RISK_CONTACT_TEST_ALLOWED_SUBJECT_KEYS": ""}
        client = self.make_app(env=env).test_client()
        payload = {"preferred_name": "ผู้ใช้ทดลอง", "contact_method": "email",
                   "contact_value": "demo@example.invalid", "consent": True}
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.assertLessEqual(len(body), 1024)
        response = client.put("/demo/api/contact", data=body,
                              content_type="application/json",
                              headers={"Authorization": "Bearer synthetic-token"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(client.get("/demo/api/contact", headers={
            "Authorization": "Bearer synthetic-token"}).json["preferred_name"],
                         "ผู้ใช้ทดลอง")

    def test_unapproved_verified_sub_has_no_contact_reads_or_writes(self):
        client = self.make_app(lambda token, channel: "synthetic-user-b").test_client()
        headers = {"Authorization": "Bearer synthetic-token"}
        payload = {"preferred_name": "Test B", "contact_method": "none",
                   "contact_value": "", "consent": True}
        for method in ("GET", "PUT", "DELETE"):
            with self.subTest(method=method), patch.object(self.db, "collection", wraps=self.db.collection) as collection:
                response = client.open("/demo/api/contact", method=method,
                                       json=payload if method == "PUT" else None, headers=headers)
                self.assertEqual(response.status_code, 403)
                collection.assert_not_called()
        self.assertEqual(self.db.store, {})

    def test_two_approved_subjects_are_isolated_and_spoofing_fails(self):
        env = {**TEST_ENV, "RISK_CONTACT_TEST_ALLOWED_SUBJECT_KEYS": ",".join(
            (allowed_key("synthetic-user-a"), allowed_key("synthetic-user-b")))}
        verifier = lambda token, channel: {"a": "synthetic-user-a", "b": "synthetic-user-b"}.get(token)
        client = self.make_app(verifier, env).test_client()
        payload = {"preferred_name": "Test A", "contact_method": "none",
                   "contact_value": "", "consent": True}
        self.assertEqual(client.put("/demo/api/contact", json=payload,
                                    headers={"Authorization": "Bearer a"}).status_code, 200)
        self.assertEqual(client.get("/demo/api/contact?user_id=synthetic-user-a",
                                    headers={"Authorization": "Bearer b"}).json, {"consent": False})
        self.assertEqual(client.delete("/demo/api/contact?user_id=synthetic-user-a",
                                       headers={"Authorization": "Bearer b"}).status_code, 200)
        self.assertEqual(client.get("/demo/api/contact",
                                    headers={"Authorization": "Bearer a"}).json["preferred_name"], "Test A")

    def test_credential_email_is_pinned_before_admin_initialization(self):
        import json
        from risk_contact_test_client import create_contact_test_client, TEST_CLIENT_EMAIL
        base = {"project_id": "heal-heart-b51de", "client_email": TEST_CLIENT_EMAIL}
        for email in ("other@heal-heart-b51de.iam.gserviceaccount.com",
                      "risk-contact-test@heal-heart-e869a.iam.gserviceaccount.com", ""):
            with self.subTest(email=email), patch.dict(os.environ, {
                    **TEST_ENV, "RISK_CONTACT_TEST_CREDENTIALS_JSON": json.dumps({**base, "client_email": email})}, clear=True), \
                    patch("firebase_admin.initialize_app") as initialize, \
                    patch("risk_contact_test_client.credentials.Certificate") as certificate:
                with self.assertRaisesRegex(RuntimeError, "verification failed"):
                    create_contact_test_client()
                certificate.assert_not_called()
                initialize.assert_not_called()
        with patch.dict(os.environ, {**TEST_ENV,
                "RISK_CONTACT_TEST_CREDENTIALS_JSON": json.dumps(base)}, clear=True), \
                patch("risk_contact_test_client.credentials.Certificate") as certificate, \
                patch("firebase_admin.initialize_app") as initialize, \
                patch("risk_contact_test_client.firestore.client") as firestore_client:
            certificate.return_value.project_id = base["project_id"]
            initialize.return_value.project_id = base["project_id"]
            firestore_client.return_value.project = base["project_id"]
            self.assertIs(create_contact_test_client(), firestore_client.return_value)

    def test_wrong_line_audience_and_issuer_are_rejected(self):
        from risk_case_demo import verify_line_identity
        import json
        for claims in ({"sub": "synthetic-user-a", "aud": "wrong", "iss": "https://access.line.me"},
                       {"sub": "synthetic-user-a", "aud": "synthetic-channel-id", "iss": "wrong"}):
            with self.subTest(claims=claims), patch("risk_case_demo.urlopen") as urlopen:
                urlopen.return_value.__enter__.return_value.read.return_value = json.dumps(claims).encode()
                self.assertIsNone(verify_line_identity("synthetic-token", "synthetic-channel-id"))

    def test_line_verify_expired_token_response_is_rejected(self):
        from urllib.error import HTTPError
        from risk_case_demo import verify_line_identity

        with patch("risk_case_demo.urlopen", side_effect=HTTPError(
                "https://api.line.me/oauth2/v2.1/verify", 400,
                "synthetic expired token", {}, None)):
            self.assertIsNone(verify_line_identity("synthetic-expired-token",
                                                    "synthetic-channel-id"))


if __name__ == "__main__":
    unittest.main()
