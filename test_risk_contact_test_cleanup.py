"""No cloud client is created; all cleanup tests use an in-memory database."""

from datetime import datetime, timedelta, timezone
import os
import unittest
from unittest.mock import patch

from risk_contact_test_cleanup import CONFIRMATION, cleanup_contacts, main

NOW = datetime(2026, 10, 28, tzinfo=timezone.utc)


class Snapshot:
    def __init__(self, reference, data):
        self.reference = reference
        self.data = data
        self.exists = data is not None

    def to_dict(self):
        return self.data


class Document:
    def __init__(self, db, path):
        self.db, self.path = db, path

    def get(self, transaction=None):
        return Snapshot(self, self.db.store.get(self.path))


class Collection:
    def __init__(self, db, name):
        self.db, self.name = db, name

    def stream(self):
        return [Document(self.db, key).get() for key in list(self.db.store)
                if key.startswith(self.name + "/") and key.count("/") == 1]


class Transaction:
    def __init__(self, db):
        self.db = db
        self.pending = []

    def delete(self, ref):
        self.pending.append(ref.path)

    def commit(self):
        for path in self.pending:
            self.db.store.pop(path)


class FakeDb:
    def __init__(self, project="heal-heart-b51de"):
        self.project = project
        self.store = {}

    def collection(self, name):
        return Collection(self, name)

    def transaction(self):
        return Transaction(self)


def run_transaction(fn):
    def wrapper(transaction, ref):
        result = fn(transaction, ref)
        transaction.commit()
        return result
    return wrapper


class CleanupTests(unittest.TestCase):
    def setUp(self):
        self.db = FakeDb()
        self.db.store.update({
            "risk_demo_contacts/expired-a": {"consent": True, "expires_at": NOW - timedelta(seconds=1)},
            "risk_demo_contacts/withdrawn-a": {"consent": False, "expires_at": NOW - timedelta(days=1)},
            "risk_demo_contacts/active-b": {"consent": True, "expires_at": NOW + timedelta(days=1)},
            "risk_demo_contacts/withdrawn-b": {"consent": False, "expires_at": NOW + timedelta(days=1)},
            "risk_demo_contacts/unknown": {"consent": True},
            "risk_demo_cases/other": {"expires_at": NOW - timedelta(days=1)},
        })

    def test_project_mismatch_fails_before_scan(self):
        self.db.project = "heal-heart-e869a"
        with self.assertRaises(RuntimeError):
            cleanup_contacts(self.db, cutoff=NOW)

    def test_cli_rejects_conflicting_environment_before_client_creation(self):
        with patch.dict(os.environ, {"FIRESTORE_EMULATOR_HOST": "127.0.0.1:8089"}), \
                patch("risk_contact_test_cleanup.create_contact_test_client") as factory:
            self.assertEqual(main([]), 2)
            factory.assert_not_called()

    def test_dry_run_is_default_and_never_deletes(self):
        before = self.db.store.copy()
        self.assertEqual(cleanup_contacts(self.db, cutoff=NOW),
                         {"eligible": 2, "deleted": 0, "skipped": 0})
        self.assertEqual(self.db.store, before)

    def test_confirmation_is_exact_and_scope_is_fixed(self):
        for count, phrase in ((None, None), (1, CONFIRMATION), (2, "wrong")):
            with self.assertRaises(RuntimeError):
                cleanup_contacts(self.db, cutoff=NOW, execute=True,
                                 confirm_count=count, confirmation=phrase)
        self.assertEqual(len(self.db.store), 6)

    def test_expired_and_withdrawn_only_after_retention(self):
        result = cleanup_contacts(self.db, cutoff=NOW, execute=True,
                                  confirm_count=2, confirmation=CONFIRMATION,
                                  transaction_runner=run_transaction)
        self.assertEqual(result, {"eligible": 2, "deleted": 2, "skipped": 0})
        self.assertEqual(set(self.db.store), {
            "risk_demo_contacts/active-b", "risk_demo_contacts/withdrawn-b",
            "risk_demo_contacts/unknown", "risk_demo_cases/other"})

    def test_renewed_contact_is_not_deleted(self):
        def renew_then_run(fn):
            def wrapper(transaction, ref):
                if ref.path.endswith("expired-a"):
                    self.db.store[ref.path]["expires_at"] = NOW + timedelta(days=7)
                result = fn(transaction, ref)
                transaction.commit()
                return result
            return wrapper
        result = cleanup_contacts(self.db, cutoff=NOW, execute=True,
                                  confirm_count=2, confirmation=CONFIRMATION,
                                  transaction_runner=renew_then_run)
        self.assertEqual(result, {"eligible": 2, "deleted": 1, "skipped": 1})
        self.assertIn("risk_demo_contacts/expired-a", self.db.store)


if __name__ == "__main__":
    unittest.main()
