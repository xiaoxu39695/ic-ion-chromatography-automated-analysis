# IC Ion Chromatography Automated Analysis

An English-language Streamlit application for chromatogram integration and calibration-based concentration analysis, currently specialized for NaI / iodide.

**No calibration data, chromatograms, measured peak areas, fitted experimental coefficients or laboratory reports are included.** Bring your own standards and validate your analytical method.

## Run locally

Python 3.11 or newer is recommended.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

Open http://127.0.0.1:8501. The default configuration binds to localhost and limits each upload to 20 MB. Stop with Ctrl+C.

## Workflow

1. Upload your own calibration chromatograms named `NaI-<concentration>M.txt` or `.csv`. Concentrations are mol/L; an optional suffix supports replicates, such as `NaI-0.001M-replicate2.txt`.
2. Standards must contain tab-separated `Instrument Method` and `Volume (µl)` metadata before a `Time (min)` column header. The parser reads Chromeleon exports (time, step, signal) or two-column time/signal files. Use UTF-8, decimal points, time in minutes and signal in µS. At least 20 numeric rows are required. Samples without method or volume metadata require explicit confirmation of matching conditions.
3. Choose linear, quadratic, or piecewise fitting and included standards. The largest group sharing instrument method and injection volume is used. Linear fitting requires two distinct concentration levels; quadratic fitting requires four standards and a full-rank design. Piecewise fitting requires a 0.001 M boundary standard, at least two lower and two higher standards.
4. Review candidate peaks and ion identity. Select another peak or manually edit integration bounds for standards or samples. Recalculation follows each change. Adjust the automatic search window when needed.
5. Set the sample dilution factor. Original concentration equals measured concentration times this factor; instrument dilution metadata are not automatically applied.
6. Download analysis CSV and model JSON before ending the session. Removing a file from the uploader does not delete its session record; use **Clear all session data** to reset.

## Numerical method and limits

Positive peaks are located using a moving average, upper-median noise estimates and local baseline thresholds. Net peak area uses unsmoothed data, a straight baseline between integration endpoints and signed trapezoidal integration at actual time intervals. Incomplete peaks are excluded until reviewed.

Piecewise fitting uses an origin-constrained quadratic below 0.001 M and a linear curve above it. Both segments pass through the measured boundary area, making the curve continuous. The origin is a mathematical constraint, not an experimental blank; it is excluded from R² and RMSE. Coefficients are computed only from the standards supplied in the current session.

Method or injection-volume mismatches block quantification. The app flags concentrations outside the measured standard range. A 15% standard back-calculation bias flag prompts review and is not a validated acceptance criterion. High R² does not establish accuracy. No LOD, LOQ, uncertainty or leaching rate is calculated.

The chemistry assumes NaI and iodide have a 1:1 molar relationship. Mass conversions use 149.894 g/mol (NaI) and 126.90447 g/mol (iodide). Adapting the software to other ions requires changing these assumptions and validating the method. Automatic selection of the largest positive peak does not establish ion identity.

## Privacy and publishing

Processing takes place on the Python server. Local execution processes uploads locally; remote hosting transfers uploads to that server. The app retains uploaded chromatograms and manual edits in Streamlit session memory and does not intentionally write them to disk or send them to external analysis services. Hosting infrastructure may retain logs.

`.gitignore` excludes common raw data, calibration, report and credential paths. Review every file before publishing; ignore rules do not remove files already tracked by Git. This release contains only source, configuration, documentation, license and tests using generated mathematical signals. No original repository history is copied.

## Test

```sh
python -m unittest -v test_analysis.py
```

Tests cover an analytical Gaussian integral, monotonic time validation, continuous calibration and inversion, method mismatch protection, and empty-session startup. Tests generate numerical inputs in memory and use no experimental files.

## License

MIT. See [LICENSE](LICENSE).
