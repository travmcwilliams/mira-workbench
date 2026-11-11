# Contributing to MIRA

Thanks for your interest! Issues and pull requests are welcome.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .
python -m pytest -q
```

## Guidelines

- **Keep tests offline.** Network code sits behind a transport seam; test it with
  fixtures, never live requests.
- **Domain logic is data.** Source definitions, thresholds and domain vocab belong in
  `acquire/data/`, `acquire/registry/config/` or `packs/`, not hardcoded in Python.
- **No private data in the repo.** Never commit scraped corpora, databases, credentials,
  cookies, or anything about real people. `data/` is ignored by default; only small,
  synthetic or clearly public samples are allow-listed in `.gitignore`.
- **No machine-specific paths.** Read locations from environment variables with a
  repo-relative default.
- **Scrape politely.** New sources must respect `robots.txt` and the site's terms and
  go through the existing rate-limiting/backoff utilities.
- Run `ruff check .` and `python -m pytest -q` before opening a PR.
