# pattern-scanner

Python project for detecting technical and statistical patterns across S&P 500 stocks using data pulled via `yfinance`.

## Scope

- **Technical patterns**: moving average crossovers, RSI, support/resistance levels.
- **Statistical patterns**: volatility regimes, correlations, mean reversion.

## Project structure

```
pattern-scanner/
├── venv/                  # virtual environment (not committed)
├── src/
│   ├── ingestion/          # yfinance data fetching / caching
│   ├── technical/          # moving averages, RSI, support/resistance
│   ├── statistical/        # volatility regimes, correlations, mean reversion
│   └── utils/               # shared helpers (config, I/O, plotting)
├── data/
│   ├── raw/                 # unmodified data pulled from yfinance
│   └── processed/           # cleaned/derived datasets
├── outputs/
│   ├── charts/               # matplotlib figures
│   └── reports/              # generated summaries/CSVs
├── requirements.txt
└── CLAUDE.md
```

## Environment

- Virtual environment lives in `venv/`. Activate with:
  - PowerShell: `venv\Scripts\Activate.ps1`
  - Bash/Git Bash: `source venv/Scripts/activate`
- Dependencies are pinned in `requirements.txt` (installed: `yfinance`, `pandas`, `numpy`, `matplotlib`, `scipy`). Install with `pip install -r requirements.txt`.
- After adding a new dependency, refresh the lockfile with `pip freeze > requirements.txt`.

## Conventions

- Data flows one direction: `ingestion/` fetches raw data into `data/raw/` → processing steps write derived data into `data/processed/` → pattern detectors in `technical/`/`statistical/` consume `data/processed/` → results land in `outputs/`.
- Keep pattern-detection logic in `src/technical/` or `src/statistical/` depending on category; avoid mixing the two in one module.
- No secrets or API keys are required for `yfinance`; if any are introduced later, keep them in a `.env` file (already gitignored) rather than hardcoding.
- `data/` and `outputs/` contents are gitignored except for `.gitkeep` placeholders — treat them as local, regenerable artifacts, not source of truth.
