# Manual contact retention for Firebase TEST (Spark)

The Contact-Only TEST API stores `risk_demo_contacts/{opaque_subject_key}`.
Consent is optional. A successful PUT sets `expires_at` seven days ahead.
GET hides expired records. DELETE withdrawal overwrites the record with a
consent-false tombstone, removing the name and contact value immediately; the
tombstone expires seven days after withdrawal. No Managed TTL is enabled.

`risk_contact_test_cleanup.py` selects only documents in `risk_demo_contacts`
whose timezone-aware `expires_at` is at or before the scan time. It skips
active, future-dated, malformed and missing-expiry records. A second check
inside each Firestore transaction protects contacts renewed after the scan.
It never accepts a project or collection argument. The client factory checks
the configured project, credential, Admin App and Firestore client against the
approved TEST project. There is no production fallback.

Only an authorized TEST operator should run this from the TEST checkout after
privately verifying the TEST service account and project. Do not paste the
credential, HMAC key, token, document ID, name or contact value into a terminal
log, ticket or report. Keep cloud credentials out of the Local Emulator shell.

Dry-run (default; reads TEST Firestore but deletes nothing):

```powershell
python -B risk_contact_test_cleanup.py
```

The command prints counts only. Review the `eligible` count and the TEST
project in the Firebase Console before any deletion. A separate execution
requires both the exact current count and an explicit phrase:

```powershell
python -B risk_contact_test_cleanup.py --execute --confirm-count <COUNT> --confirm-phrase "DELETE EXPIRED TEST CONTACTS"
```

The execution performs a fresh scan. It stops if the count changed. Its output
contains only eligible, deleted and skipped counts. A changed document is
skipped. If a transaction or connection fails, the command exits with an error;
some earlier documents might already have been deleted. Re-run dry-run and
inspect counts before deciding whether a further approved execution is needed.

This is a manual process: expiry hides data from the API but does not guarantee
physical deletion by a deadline. Withdrawal removes the contact fields at once
but leaves a consent-false tombstone until manual cleanup. Do not use real
contact details until a retention owner, execution schedule, evidence record
and failure response have been agreed and tested. For rollback, stop running
the cleanup tool; it is not part of app startup or a scheduled service.
Deleted documents cannot be restored by this tool.

For a future, separately approved Render TEST service, use the branch's
`.python-version` (`3.11.9`), build command
`pip install -r requirements-contact-test.txt`, start command
`gunicorn risk_contact_test_app:app --bind 0.0.0.0:$PORT`, and health path
`/health`. The TEST requirements contain only Flask, Firebase Admin and
Gunicorn. `Werkzeug` and `google-api-core` are required imports supplied by
their respective pinned parent packages; installation on Python 3.11.9 and
actual Gunicorn startup remain unverified. The cleanup tool is a separate
operator command and must never be added to build, start or health commands.
