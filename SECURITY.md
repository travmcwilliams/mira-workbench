# Security policy

## Reporting a vulnerability

Please **do not** open a public issue for security problems. Use GitHub's
[private vulnerability reporting](../../security/advisories/new) for this repository
instead. You should get an acknowledgement within a few days.

## Scope notes

- MIRA is a local, single-user tool. The Streamlit app is meant to run on
  `localhost`. Do not expose it to a network without adding authentication.
- API keys are read from environment variables only. Never commit a `.env` file.
- If you find credentials or personal data committed to this repository, report it
  privately as above.
