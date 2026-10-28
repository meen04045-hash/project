"""Manual, TEST-only contact retention cleanup. Never runs during app startup."""

import argparse
from datetime import datetime, timezone
import os
import sys

from firebase_admin import firestore

from risk_contact_test_client import TEST_PROJECT_ID, create_contact_test_client

COLLECTION = "risk_demo_contacts"
CONFIRMATION = "DELETE EXPIRED TEST CONTACTS"


def _expired(record, cutoff):
    expiry = record.get("expires_at") if isinstance(record, dict) else None
    if not isinstance(expiry, datetime) or expiry.tzinfo is None:
        return False
    return expiry <= cutoff


def cleanup_contacts(db, *, cutoff=None, execute=False, confirm_count=None,
                     confirmation=None, transaction_runner=None):
    """Count eligible records; delete only after explicit, exact confirmation.

    Each deletion rechecks expiry within a Firestore transaction so a renewed
    contact cannot be removed using an older dry-run snapshot.
    """
    if db.project != TEST_PROJECT_ID:
        raise RuntimeError("Contact cleanup requires the approved TEST project")
    cutoff = cutoff or datetime.now(timezone.utc)
    if cutoff.tzinfo is None:
        raise ValueError("A timezone-aware cutoff is required")

    candidates = [snap.reference for snap in db.collection(COLLECTION).stream()
                  if _expired(snap.to_dict(), cutoff)]
    count = len(candidates)
    if not execute:
        return {"eligible": count, "deleted": 0, "skipped": 0}
    if (type(confirm_count) is not int or confirm_count != count
            or confirmation != CONFIRMATION):
        raise RuntimeError("Deletion requires the exact eligible count and confirmation phrase")

    runner = transaction_runner or firestore.transactional

    @runner
    def delete_if_expired(transaction, ref):
        snapshot = ref.get(transaction=transaction)
        if not snapshot.exists or not _expired(snapshot.to_dict(), cutoff):
            return False
        transaction.delete(ref)
        return True

    deleted = 0
    for ref in candidates:
        if delete_if_expired(db.transaction(), ref):
            deleted += 1
    return {"eligible": count, "deleted": deleted, "skipped": count - deleted}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Manual cleanup of expired TEST contacts only")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-count", type=int)
    parser.add_argument("--confirm-phrase")
    args = parser.parse_args(argv)
    if any(os.environ.get(name) for name in ("FIRESTORE_EMULATOR_HOST",
                                          "FIREBASE_CREDENTIALS_JSON",
                                          "GOOGLE_APPLICATION_CREDENTIALS")):
        print("Conflicting Firebase configuration; cleanup stopped", file=sys.stderr)
        return 2
    try:
        db = create_contact_test_client()
        result = cleanup_contacts(db, execute=args.execute,
                                  confirm_count=args.confirm_count,
                                  confirmation=args.confirm_phrase)
    except Exception:
        print("TEST contact cleanup failed; no further documents attempted", file=sys.stderr)
        return 2
    print("TEST contact cleanup:", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
