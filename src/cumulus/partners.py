"""Issue partner keys for the open API: ``python -m cumulus.partners new nadmo "NADMO"``.

The API needs no key; a key only names a partner's calls in the logs. With ``--webhook-url`` the
partner is also sent a signed event for each new run, and a signing secret is printed for it.
Prints a new random key (give it to the partner once; it is not stored) and the JSON entry to add
to ``CUMULUS_PARTNERS`` in the deployment's environment. Revoke a partner by setting
``"revoked": true`` on its entry, or rotate by issuing a new key under the same id.
"""

from __future__ import annotations

import argparse
import json
import secrets

from cumulus.api.security import partner_key_hash


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Partner API keys.")
    commands = parser.add_subparsers(dest="command", required=True)
    new = commands.add_parser("new", help="Issue a key for a partner.")
    new.add_argument("partner_id", help="Short id, e.g. nadmo, vra, mofa, fao, insurer-acme.")
    new.add_argument("name", help="Display name, e.g. 'National Disaster Management Organisation'.")
    new.add_argument("--webhook-url", help="Where to POST run.published events.")
    args = parser.parse_args(argv)
    key = f"cmls_{secrets.token_urlsafe(32)}"
    entry: dict[str, object] = {"name": args.name, "key_sha256": partner_key_hash(key)}
    print(f"API key for {args.partner_id} (shown once; send it securely):\n  {key}\n")
    if args.webhook_url:
        # Unlike the key, the secret is stored: announcing a run has to sign with it.
        entry["webhook_url"] = args.webhook_url
        entry["webhook_secret"] = f"whsec_{secrets.token_urlsafe(32)}"
        print(f"Webhook signing secret (send it securely too):\n  {entry['webhook_secret']}\n")
    print("Add to CUMULUS_PARTNERS:")
    print(json.dumps({args.partner_id: entry}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
