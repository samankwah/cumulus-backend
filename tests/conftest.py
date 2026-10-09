"""Test-wide environment, set before any test module imports the app.

Production hides the legacy random-forest endpoints and requires an admin key for mutating ones;
the existing suites exercise both freely, so they opt in here. ``test_security.py`` switches these
back off to check the locked-down defaults.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUMULUS_ENABLE_LEGACY_ENDPOINTS", "true")
os.environ.setdefault("CUMULUS_ALLOW_UNAUTHENTICATED_ADMIN", "true")
