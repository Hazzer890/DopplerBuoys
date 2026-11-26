# DopplerBuoys Simulator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local web page, backed by Python, that shows how well a field of hydrophone buoys can track small boats in a 5 km square, and improves the buoy layout.

**Architecture:** One Python file holds the acoustic model, the accuracy bound, the tracker, the optimiser, the self-check and a small HTTP server. One HTML file draws the map and controls and calls three JSON endpoints. The model works at measurement level: range gives signal-to-noise, signal-to-noise gives an error on each measurement, and the errors give a position bound. One full-audio check confirms the error formulas.

**Tech Stack:** Python 3.14, numpy, scipy, stdlib `http.server`. Vanilla JS and canvas. Started with `uv run sim.py`.

**Spec:** `docs/design.md`

## Global Constraints

- Two source files: `sim.py` and `index.html`. Dependencies are numpy and scipy only, declared inline in `sim.py` (PEP 723).
- Area is a 5000 m square. Coordinates are metres, origin bottom-left.
- Score is mean position error over the grid, capped at 500 m per cell. "Tracked" threshold defaults to 50 m.
- Each assumption that drives the result is a named entry in `DEFAULTS` and a control on the page.
- Server binds to 127.0.0.1 only.
- Not a git repository. No commits.
- `uv run sim.py --check` runs the self-check and exits non-zero on failure.

## Review Focus

Each line names an input the page can produce. The owning task's check covers it.

1. No buoys, or one buoy: the map returns the capped error everywhere and the run returns no fixes. Nothing crashes. (Task 1, Task 3)
2. A boat or grid cell at the same point as a buoy: range is floored at 1 m, no division by zero. (Task 1)
3. All three measurement types switched off: same behaviour as no buoys. (Task 1)
4. Slider extremes (window 0.5 s, 3 hydrophones, 0 m array, sea noise 90 dB): finite numbers out. (Task 1)
5. Malformed or oversized request body: HTTP 400, server keeps running. Buoy count is clamped to 60. (Task 5)

---

### Task 1: Acoustic model and accuracy bound

**Files:** Create `sim.py`

**Interfaces (produces):**
- `DEFAULTS: dict[str, float]` with keys `c, sl, f0, harmonics, bb_offset, spread, alpha, nl, hydrophones, array_d, window, detect_db, compass_deg, clock_ms, wobble_pct, speed, band_lo, band_hi, use_bearing, use_doppler, use_tdoa, threshold`
- `params(overrides: dict) -> dict`: known keys only, cast to float, clamped to `LIMITS`
- `sigmas(r: ndarray, P) -> (sig_bearing, sig_g, heard, snr_bb)`: per-buoy bearing error (rad), fractional pitch error, tone-detected mask, broadband spectral SNR
- `detection_range(P) -> float` metres
- `crlb(pos: (n,2), vel: (n,2), buoys: (m,2), P) -> (n,)` position error in metres
- `accuracy_map(buoys, P, n=40) -> dict` with `grid` (n×n list), `score`, `within`, `detect_range`

**Model:**
- Loss: `spread*log10(r) + alpha*r/1000`, `r` floored at 1 m.
- Tone bin SNR (dB): `sl - loss - nl + 10*log10(window*hydrophones)`. Heard when `>= detect_db`.
- Pitch error (Hz, fundamental): `sqrt(6)/(2*pi*window*sqrt(snr)) / G`, `G = sqrt(sum(h^2 for h in 1..harmonics))`. Fractional error adds `wobble_pct/100` in quadrature.
- Bearing error (rad): `c/(pi*f0*array_d*sqrt(snr)*G)`, plus `compass_deg` in quadrature.
- Between-buoy arrival time, against the loudest buoy: Knapp-Carter bound over `band_lo..band_hi` with broadband spectral SNR `sl - bb_offset - loss - nl + 10*log10(hydrophones)`, plus `sqrt(2)*clock_ms` in quadrature. Usable when `sqrt(s1*s2*bandwidth*window)` reaches `detect_db`.
- State is `[x, y, vx, vy, q]`, `q` the pitch ratio. Fisher information sums analytic Jacobians over buoys, plus a weak prior (5000 m, 15 m/s, 0.2). Map averages over four headings at `speed`.

- [ ] Step 1: Write `check_model()` asserting: closing boat reads higher pitch than opening; `detection_range` falls when `nl` rises; four buoys give lower mean `crlb` than two; the five Review Focus cases 1 to 4 return finite, capped values.
- [ ] Step 2: Run `uv run sim.py --check`, expect failure.
- [ ] Step 3: Implement the model.
- [ ] Step 4: Run `uv run sim.py --check`, expect pass.

### Task 2: Full-audio check of the formulas

**Files:** Modify `sim.py`

**Interfaces:** Consumes `sigmas`, `DEFAULTS`. Produces `audio_check(trials=200, seed=0) -> dict` with `pitch_ratio` and `bearing_ratio` (measured RMS error over formula).

- [ ] Step 1: Add to `check()`: both ratios lie in 0.7 to 1.4, and mean measured pitch matches `f0*(1 - radial_speed/c)` within 3 formula sigmas.
- [ ] Step 2: Implement. Synthesise one tone at each hydrophone of a circular array at 4 kHz for one window, boat moving at constant velocity 2 km away, using emission time solved from `t_emit + r(t_emit)/c = t`. Add white noise at the model's level. Estimate pitch by periodogram peak with a bounded refine, bearing by steering the hydrophone phases.
- [ ] Step 3: Run `uv run sim.py --check`, expect pass.

### Task 3: Tracker and live run

**Files:** Modify `sim.py`

**Interfaces:** Consumes Task 1. Produces:
- `locate(meas: dict, buoys, P, guess=None) -> ndarray | None` (state of length 5)
- `run_scenario(buoys, P, seed=0) -> dict` with `dt`, and per boat `name`, `f0`, `true`, `est`, `err`, `bound`, `heard` (lists per step, `None` where no fix)

Three boats: a 20 kn transit, a 12 kn boat that stops for 5 minutes with its engine off, a 4 kn troller. A fix needs measurements from at least two buoys.

- [ ] Step 1: Add to `check()`: at a boat inside a ring of six buoys, RMS error of 200 `locate` fixes is 0.7 to 1.5 times `crlb`; `run_scenario` with no buoys returns all `None`; the stopped boat has `None` fixes while stopped.
- [ ] Step 2: Implement with `scipy.optimize.least_squares`, started from the previous fix or a coarse grid search.
- [ ] Step 3: Run `uv run sim.py --check`, expect pass.

### Task 4: Layout optimiser

**Files:** Modify `sim.py`

**Interfaces:** Consumes `accuracy_map`. Produces `optimise(buoys, P, iters=1500, seed=0) -> list[[x, y]]`.

- [ ] Step 1: Add to `check()`: starting from five buoys bunched in one corner, the optimised score is lower and each buoy stays inside the square.
- [ ] Step 2: Implement as a hill climb on a 15×15 grid: move one buoy by a random step that shrinks over the run, keep the move when the score drops.
- [ ] Step 3: Run `uv run sim.py --check`, expect pass.

### Task 5: Server and page

**Files:** Modify `sim.py`, create `index.html`

**Interfaces:** `POST /api/map`, `/api/optimise`, `/api/run`, each taking `{buoys: [[x,y]], params: {}, seed?}` and returning the matching function's result as JSON. `GET /` serves `index.html`.

- [ ] Step 1: Add to `check()`: start the server on a free port, post a valid map request (200, 40×40 grid), post `not json` (400), post 500 buoys (200, clamped to 60).
- [ ] Step 2: Implement the server.
- [ ] Step 3: Build the page: error heatmap with draggable buoys (click to add, right-click to remove), stats, measurement toggles, grouped sliders, Optimise and Run buttons, boat tracks over the map, error-over-time chart.
- [ ] Step 4: Open the page in a browser and confirm: map redraws on drag, optimise moves buoys and lowers the score, run draws tracks and the stopped boat's track breaks.

### Task 6: Calibrate defaults

**Files:** Modify `sim.py`, `index.html`

- [ ] Step 1: Look up published small-boat source levels and coastal ambient noise levels. Set `sl`, `nl`, `spread`, `alpha` from them.
- [ ] Step 2: Cite the sources on the page next to the settings they support, and mark settings with no source as unconfirmed.
- [ ] Step 3: Run `uv run sim.py --check`, expect pass.
