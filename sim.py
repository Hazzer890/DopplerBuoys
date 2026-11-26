# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy"]
# ///
"""DopplerBuoys: can a field of hydrophone buoys track small boats in a 5 km square?

Measurement-level model. See docs/design.md.
    uv run sim.py           serve the page on http://127.0.0.1:8000
    uv run sim.py --check   run the self-check
"""
import json
import sys
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, minimize_scalar

SIZE = 5000.0  # m, side of the square
CAP = 500.0  # m, per-cell cap on error in the score
MAX_BUOYS = 60
PRIOR = np.array([5000.0, 5000.0, 15.0, 15.0, 0.2, 10.0])  # weak prior sigma on [x, y, vx, vy, q, t0]
HEADINGS = np.radians([0, 45, 90, 135])  # opposite headings give the same bound

# name: (default, min, max). Every assumption that drives the result lives here.
SETTINGS = {
    "c": (1500, 1400, 1600),  # sound speed, m/s
    "sl": (150, 110, 180),  # level of each engine tone, dB re 1 uPa at 1 m (estimate)
    "f0": (120, 20, 500),  # engine fundamental, Hz
    "harmonics": (4, 1, 8),  # equal-level tones at f0, 2*f0, ...
    "bb_offset": (20, 0, 40),  # broadband spectrum level sits this far below the tone level, dB
    "spread": (17, 10, 25),  # loss = spread*log10(r) + alpha*r/1000
    "alpha": (3.5, 0, 20),  # extra shallow-water loss, dB/km
    "nl": (70, 40, 100),  # sea noise spectrum level, dB re 1 uPa^2/Hz
    "hydrophones": (4, 3, 12),  # per buoy, on a circle
    "array_d": (1.0, 0, 5),  # circle diameter, m
    "array_gain_db": (0, 0, 12),  # gain of the ring over sea noise; about 0 when closer than half a wavelength
    "window": (4, 0.5, 30),  # listening window per fix, s
    "detect_db": (16, 0, 30),  # SNR needed to use a measurement
    "compass_deg": (3, 0, 30),  # buoy heading error
    "clock_ms": (5, 0.5, 100),  # timing error per buoy (clock, buoy position, echoes; 3 m = 2 ms)
    "wobble_pct": (0.05, 0.01, 5),  # engine pitch wobble not common to all buoys
    "speed": (5, 0, 30),  # boat speed used for the map, m/s
    "band_lo": (100, 10, 5000),  # band used to time arrivals between buoys, Hz
    "band_hi": (1000, 20, 10000),
    "use_bearing": (1, 0, 1),
    "use_doppler": (1, 0, 1),
    "use_tdoa": (1, 0, 1),
    "threshold": (50, 5, 500),  # "tracked" means error under this, m
}
DEFAULTS = {k: float(v[0]) for k, v in SETTINGS.items()}
LIMITS = {k: v[1:] for k, v in SETTINGS.items()}
KINDS = {"bearing": "use_bearing", "doppler": "use_doppler", "tdoa": "use_tdoa"}


def params(overrides=None):
    """DEFAULTS with known overrides applied and clamped. Raises ValueError on non-numbers."""
    P = dict(DEFAULTS)
    for k, v in (overrides or {}).items():
        if k in LIMITS:
            v = float(v)
            if not np.isfinite(v):
                raise ValueError(k)
            P[k] = min(max(v, LIMITS[k][0]), LIMITS[k][1])
    P["band_hi"] = max(P["band_hi"], P["band_lo"] + 10)
    return P


def as_buoys(buoys):
    b = np.asarray(buoys, float).reshape(-1, 2)[:MAX_BUOYS]
    if not np.isfinite(b).all():
        raise ValueError("buoys")
    return np.clip(b, 0, SIZE)


def loss(r, P):
    r = np.maximum(r, 1.0)
    return P["spread"] * np.log10(r) + P["alpha"] * r / 1000


def tone_snr_db(r, P):
    """SNR of one engine tone in a 1/window bin at range r."""
    return P["sl"] - loss(r, P) - P["nl"] + 10 * np.log10(P["window"]) + P["array_gain_db"]


def sigmas(r, P):
    """Errors for one buoy at range r (m): bearing sigma (rad), pitch-ratio sigma,
    tone-heard mask, broadband spectral SNR (linear)."""
    M, T, D = P["hydrophones"], P["window"], P["array_d"]
    snr_db = tone_snr_db(r, P)
    snr = 10 ** (snr_db / 10)
    h = np.arange(1, int(P["harmonics"]) + 1)
    # Cramer-Rao bound for the frequency of a sinusoid in white noise, tones combined
    sig_f = np.sqrt(6) / (2 * np.pi * T * np.sqrt(snr) * np.sqrt((h**2).sum()))
    sig_g = np.hypot(sig_f / P["f0"], P["wobble_pct"] / 100)
    # Bearing from tone phase across the ring: M phases, each with independent noise at the
    # per-hydrophone SNR. Tones whose half wavelength is shorter than the hydrophone gap alias
    # and are left out. ponytail: tones only, broadband would add a little.
    clear = h[h * P["f0"] * D * np.sin(np.pi / M) <= P["c"] / 2]
    snr_1 = snr / 10 ** (P["array_gain_db"] / 10)
    with np.errstate(divide="ignore"):  # no spread or no clear tone means no bearing
        sig_arr = P["c"] / (np.pi * P["f0"] * D * np.sqrt(snr_1 * M) * np.sqrt((clear**2).sum()))
    sig_b = np.hypot(sig_arr, np.radians(P["compass_deg"]))
    snr_bb = 10 ** ((P["sl"] - P["bb_offset"] - loss(r, P) - P["nl"] + P["array_gain_db"]) / 10)
    return sig_b, sig_g, snr_db >= P["detect_db"], snr_bb


def toa_terms(sbb, P):
    """Arrival-time error per buoy (s), which buoys can time the sound, and the reference buoy,
    for broadband SNRs of shape (n, m). Each buoy times the sound by correlating against the
    loudest buoy (Knapp-Carter bound) and carries its own independent timing error."""
    i, ref = np.arange(len(sbb)), np.argmax(sbb, axis=1)
    s_ref = sbb[i, ref][:, None]
    eff = sbb * s_ref / (1 + sbb + s_ref)
    bt = (P["band_hi"] - P["band_lo"]) * P["window"]
    ok = 10 * np.log10(2 * bt * eff) >= P["detect_db"]  # correlator output SNR
    var = 3 / (eff * 8 * np.pi**2 * P["window"] * (P["band_hi"] ** 3 - P["band_lo"] ** 3))
    ok[i, ref], var[i, ref] = False, 0
    ok[i, ref] = ok.any(1)  # the reference counts once another buoy can be timed against it
    return np.sqrt(var + (P["clock_ms"] / 1000) ** 2), ok, ref


def detection_range(P):
    r = np.geomspace(1, 1e5, 2000)
    heard = sigmas(r, P)[2]
    return float(r[heard][-1]) if heard.any() else 0.0


def predict(s, buoys, P):
    """Noise-free measurements for state s = [x, y, vx, vy, q, t0]:
    bearing from each buoy (rad), pitch ratio heard at each buoy, range (m)."""
    d = s[:2] - buoys
    r = np.maximum(np.linalg.norm(d, axis=1), 1.0)
    return np.arctan2(d[:, 1], d[:, 0]), s[4] / (1 + (d @ s[2:4]) / (r * P["c"])), r


def covariance(pos, vel, buoys, P):
    """Covariance (n,6,6) of the best possible estimate of [x, y, vx, vy, q, t0]
    for boats at pos (n,2) moving at vel (n,2): inverse Fisher information plus a weak prior."""
    pos, vel, buoys = np.atleast_2d(pos).astype(float), np.atleast_2d(vel).astype(float), as_buoys(buoys)
    n, m, c = len(pos), len(buoys), P["c"]
    rows, w = [], []  # Jacobian rows (n,m,6) and weights (n,m) per measurement type
    if m:
        d = pos[:, None, :] - buoys[None, :, :]
        r = np.maximum(np.linalg.norm(d, axis=2), 1.0)
        u = d / r[..., None]  # unit vector buoy -> boat
        sig_b, sig_g, heard, sbb = sigmas(r, P)
        zero = np.zeros((n, m))
        if P["use_bearing"]:
            rows.append(np.stack([-u[..., 1] / r, u[..., 0] / r, zero, zero, zero, zero], -1))
            w.append(heard / sig_b**2)
        if P["use_doppler"]:
            a = (vel[:, None, :] * u).sum(-1) / c  # radial speed over c, positive = opening
            vperp = vel[:, None, :] - (a * c)[..., None] * u
            k = (1 / (1 + a) ** 2)[..., None]
            rows.append(np.concatenate([-k * vperp / (c * r[..., None]), -k * u / c, (1 / (1 + a))[..., None], zero[..., None]], -1))
            w.append(heard / sig_g**2)
        if P["use_tdoa"]:
            sig_t, ok, _ = toa_terms(sbb, P)
            rows.append(np.concatenate([u / c, np.zeros((n, m, 3)), np.ones((n, m, 1))], -1))
            w.append(ok / sig_t**2)
    J = np.tile(np.eye(6), (n, 1, 1))  # the prior, in units of PRIOR so the matrix stays well conditioned
    if rows:
        A = np.concatenate(rows, 1) * PRIOR
        J += np.einsum("nki,nk,nkj->nij", A, np.concatenate(w, 1), A)
    return np.linalg.inv(J) * PRIOR[:, None] * PRIOR[None, :]


def crlb(pos, vel, buoys, P):
    """Lowest achievable RMS position error (m)."""
    cov = covariance(pos, vel, buoys, P)
    return np.sqrt(cov[:, 0, 0] + cov[:, 1, 1])


def accuracy_map(buoys, P, n=40):
    xs = (np.arange(n) + 0.5) * SIZE / n
    pos = np.stack(np.meshgrid(xs, xs), -1).reshape(-1, 2)  # row = y, column = x
    vel = P["speed"] * np.repeat(np.column_stack([np.cos(HEADINGS), np.sin(HEADINGS)]), len(pos), axis=0)
    err = crlb(np.tile(pos, (len(HEADINGS), 1)), vel, buoys, P).reshape(len(HEADINGS), -1)
    err = np.sqrt((err**2).mean(0))  # RMS over headings
    capped = np.minimum(err, CAP)
    return {
        "grid": capped.reshape(n, n).round(1).tolist(),
        "score": float(capped.mean()),
        "within": float((err <= P["threshold"]).mean()),
        "detect_range": detection_range(P),
    }


def optimise(buoys, P, iters=1000, seed=0):
    """Improve a layout: nudge one buoy at a time and keep the move when the score drops."""
    # ponytail: local search from the current layout, so it can settle in a local best.
    # Swap in scipy.optimize.differential_evolution if layouts stop improving.
    b, rng = as_buoys(buoys).copy(), np.random.default_rng(seed)
    if not len(b):
        return []
    best = accuracy_map(b, P, 15)["score"]
    for i in range(iters):
        trial, k = b.copy(), rng.integers(len(b))
        step = 50 + 0.3 * SIZE * (1 - i / iters)  # big moves first, fine moves last
        trial[k] = np.clip(trial[k] + rng.normal(0, step, 2), 0, SIZE)
        score = accuracy_map(trial, P, 15)["score"]
        if score < best:
            b, best = trial, score
    return b.round().tolist()


def measure(s, buoys, P, rng):
    """One window of noisy measurements of a boat in state s, as the buoys would report them."""
    bear, g, r = predict(s, buoys, P)
    sig_b, sig_g, heard, sbb = sigmas(r, P)
    sig_t, ok, ref = (v[0] for v in toa_terms(sbb[None], P))
    has_b = heard & bool(P["use_bearing"]) & np.isfinite(sig_b)
    noise = rng.normal(size=(3, len(r)))
    return {
        "ref": int(ref),
        "bearing": bear + noise[0] * np.where(has_b, sig_b, 0), "sig_b": sig_b, "has_b": has_b,
        "pitch": g + noise[1] * sig_g, "sig_g": sig_g, "has_g": heard & bool(P["use_doppler"]),
        "toa": s[5] + r / P["c"] + noise[2] * sig_t, "sig_t": sig_t, "has_t": ok & bool(P["use_tdoa"]),
    }  # fmt: skip


def locate(meas, buoys, P, guess=None):
    """Least-squares fix [x, y, vx, vy, q, t0] from one window, or None if under two buoys contribute."""
    hb, hg, ht = meas["has_b"], meas["has_g"], meas["has_t"]
    if (hb | hg | ht).sum() < 2:
        return None
    centre = np.array([SIZE / 2, SIZE / 2, 0, 0, 1, 0])

    def resid(s):
        bear, g, r = predict(s, buoys, P)
        return np.concatenate([
            ((meas["bearing"] - bear + np.pi) % (2 * np.pi) - np.pi)[hb] / meas["sig_b"][hb],
            (meas["pitch"] - g)[hg] / meas["sig_g"][hg],
            (meas["toa"] - r / P["c"] - s[5])[ht] / meas["sig_t"][ht],
            (s - centre) / PRIOR,  # same weak prior as covariance()
        ])  # fmt: skip

    def fit(start):
        return least_squares(resid, start, x_scale=[100, 100, 1, 1, 1e-3, 1e-2])

    def start(x, y):  # at rest at (x, y), with the emission time that best fits the arrivals
        t0 = (meas["toa"] - np.linalg.norm([x, y] - buoys, axis=1) / P["c"])[ht].mean() if ht.any() else 0.0
        return np.array([x, y, 0, 0, meas["pitch"][hg].mean() if hg.any() else 1.0, t0])

    best = fit(guess) if guess is not None else None
    if best is None or 2 * best.cost > 4 * len(best.fun):  # no guess, or the guess led somewhere poor
        xs = (np.arange(12) + 0.5) * SIZE / 12
        grid = fit(min((start(x, y) for x in xs for y in xs), key=lambda s: (resid(s) ** 2).sum()))
        best = grid if best is None or grid.cost < best.cost else best
    return best.x


KN = 0.5144  # m/s per knot
# pitch = multiple of the f0 setting, so boats differ; stop = (waypoint index, seconds with engine off)
BOATS = [
    {"name": "Transit, 20 kn", "pitch": 1.125, "speed": 20 * KN, "path": [(0, 1200), (5000, 3600)], "stop": None},
    {"name": "Fisher, 12 kn, stops 5 min", "pitch": 0.92, "speed": 12 * KN,
     "path": [(2000, 0), (2800, 2600), (5000, 4400)], "stop": (1, 300)},
    {"name": "Troller, 4 kn", "pitch": 0.75, "speed": 4 * KN, "path": [(600, 4700), (1800, 3400), (1500, 1500)], "stop": None},
]  # fmt: skip


def boat_state(boat, t):
    """(position, velocity, engine_on) at time t, or None once the boat has finished its path."""
    path = np.array(boat["path"], float)
    legs = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(legs)])
    at, pause = boat["stop"] or (None, 0)
    t_stop = cum[at] / boat["speed"] if at is not None else np.inf
    if t_stop <= t < t_stop + pause:
        return path[at], np.zeros(2), False
    dist = boat["speed"] * (t - pause if t >= t_stop + pause else t)
    if dist >= cum[-1]:
        return None
    i = np.searchsorted(cum, dist, side="right") - 1
    u = (path[i + 1] - path[i]) / legs[i]
    return path[i] + u * (dist - cum[i]), u * boat["speed"], True


def run_scenario(buoys, P, seed=0):
    """Drive the three boats through the square and fix each one every step."""
    buoys, rng = as_buoys(buoys), np.random.default_rng(seed)
    dt = max(P["window"], 6.0)  # ponytail: about 300 steps over the half-hour scenario keeps a run to seconds
    out = {"dt": dt, "boats": []}
    for boat in BOATS:
        Pb = {**P, "f0": P["f0"] * boat["pitch"]}
        rec = {k: [] for k in ("true", "est", "err", "bound", "heard", "silent")}
        guess = None
        for t in np.arange(0, 1800, dt):
            st = boat_state(boat, t)
            fix = err = bound = None
            heard = 0
            if st and st[2] and len(buoys):
                s = np.array([*st[0], *st[1], 1.0, 0.0])
                heard = int(sigmas(predict(s, buoys, Pb)[2], Pb)[2].sum())
                bound = round(float(min(crlb(st[0], st[1], buoys, Pb)[0], 9999)), 1)
                guess = locate(measure(s, buoys, Pb, rng), buoys, Pb, guess)
                if guess is not None:
                    fix = [round(float(v), 1) for v in guess[:2]]
                    err = round(float(np.linalg.norm(guess[:2] - st[0])), 1)
            else:
                guess = None
            rec["true"].append([round(float(v), 1) for v in st[0]] if st else None)
            rec["est"].append(fix)
            rec["err"].append(err)
            rec["bound"].append(bound)
            rec["heard"].append(heard)
            rec["silent"].append(bool(st and not st[2]))
        present = sum(1 for tr in rec["true"] if tr)
        underway = present - sum(rec["silent"])
        errs = [e for e in rec["err"] if e is not None]
        good = sum(e <= P["threshold"] for e in errs)
        out["boats"].append({
            "name": boat["name"], "f0": round(Pb["f0"], 1), **rec,
            "median_err": float(np.median(errs)) if errs else None,
            "tracked": good / underway if underway else 0.0,  # share of engine-on time
            "tracked_all": good / present if present else 0.0,  # share of all time in the square
        })  # fmt: skip
    return out


def num(v, digits=2):
    """A JSON-safe number: None for anything not finite."""
    return round(float(v), digits) if np.isfinite(v) else None


def explain(buoys, P, boat, seed=0):
    """The working behind one fix of one boat [x, y, vx, vy, pitch multiple]: what each buoy
    receives and measures, the fix those measurements give, and the error each method allows."""
    buoys = as_buoys(buoys)
    b = np.asarray(boat, float).ravel()
    if b.shape != (5,) or not np.isfinite(b).all():
        raise ValueError("boat must be [x, y, vx, vy, pitch]")
    pos, vel = np.clip(b[:2], 0, SIZE), np.clip(b[2:4], -30, 30)
    P = {**P, "f0": P["f0"] * min(max(b[4], 0.25), 4.0)}
    s = np.array([*pos, *vel, 1.0, 0.0])
    out = {"f0": num(P["f0"], 1), "boat": [*pos.tolist(), *vel.tolist()], "buoys": [], "fix": None, "error": None, "shapes": {}}
    for kind in ("all", *KINDS):
        if kind != "all" and not P[KINDS[kind]]:
            continue
        only = {} if kind == "all" else {flag: float(flag == KINDS[kind]) for flag in KINDS.values()}
        cov = covariance(pos, vel, buoys, {**P, **only})[0, :2, :2]
        vals, vecs = np.linalg.eigh(cov)  # 1-sigma ellipse: semi-major, semi-minor (m), angle (rad)
        out["shapes"][kind] = {
            "error": num(np.sqrt(cov[0, 0] + cov[1, 1]), 1),
            "ellipse": [num(np.sqrt(vals[1]), 1), num(np.sqrt(vals[0]), 1), num(np.arctan2(vecs[1, 1], vecs[0, 1]), 4)],
        }
    if not len(buoys):
        return out
    bear, g, r = predict(s, buoys, P)
    sig_b, sig_g, heard, _ = sigmas(r, P)
    m = measure(s, buoys, P, np.random.default_rng(seed))
    fix = locate(m, buoys, P)
    if fix is not None:
        out["fix"] = [num(fix[0], 1), num(fix[1], 1)]
        out["error"] = num(np.linalg.norm(fix[:2] - pos), 1)
    first = m["toa"][m["has_t"]].min() if m["has_t"].any() else 0.0
    for i in range(len(buoys)):
        out["buoys"].append({
            "range": num(r[i], 0), "level": num(P["sl"] - loss(r[i], P), 1), "snr": num(tone_snr_db(r[i], P), 1),
            "heard": bool(heard[i]), "ref": i == m["ref"],
            "closing": num(-(pos - buoys[i]) @ vel / r[i], 2),  # m/s towards the buoy
            "bearing": num(m["bearing"][i], 4) if m["has_b"][i] else None, "bearing_sigma": num(sig_b[i], 4),
            "pitch": num(m["pitch"][i] * P["f0"], 3) if m["has_g"][i] else None,
            "pitch_true": num(g[i] * P["f0"], 3), "pitch_sigma": num(sig_g[i] * P["f0"], 3),
            "delay": num((m["toa"][i] - first) * 1000, 1) if m["has_t"][i] else None,  # ms after the first arrival
            "delay_sigma": num(m["sig_t"][i] * 1000, 1),
        })  # fmt: skip
    return out


def audio_check(trials=200, seed=0, snr_db=None):
    """Synthesise the hydrophone waveforms for one boat passing one buoy at 1 km, measure
    pitch and bearing from them, and return measured RMS error over the formulas in sigmas().
    snr_db sets the tone SNR by raising the sea noise; the default leaves it high."""
    M = int(DEFAULTS["hydrophones"])
    # white noise independent between hydrophones, so the ring gets its full gain here
    P = {**params({"harmonics": 1, "compass_deg": 0}), "wobble_pct": 0.0, "array_gain_db": 10 * np.log10(M)}
    if snr_db is not None:
        P["nl"] += tone_snr_db(1000, P) - snr_db
    rng = np.random.default_rng(seed)
    fs, f0, c, a = 1000.0, P["f0"], P["c"], P["array_d"] / 2
    ang = 2 * np.pi * np.arange(M) / M
    hyd = a * np.column_stack([np.cos(ang), np.sin(ang)])
    t = np.arange(int(P["window"] * fs)) / fs
    x0 = 1000 * np.array([np.cos(np.radians(20)), np.sin(np.radians(20))])
    v = 8 * np.array([np.cos(np.radians(80)), np.sin(np.radians(80))])  # 4 m/s opening

    def boat(tt):
        return x0 + v * (np.asarray(tt)[..., None] - t[-1] / 2)

    te = np.tile(t, (M, 1))  # emission time of the sound arriving at each hydrophone at time t
    for _ in range(5):
        te = t - np.linalg.norm(boat(te) - hyd[:, None, :], axis=2) / c
    ps_n0 = 10 ** ((P["sl"] - loss(1000, P) - P["nl"]) / 10)  # tone power over noise density
    clean = np.sqrt(2 * ps_n0) * np.sin(2 * np.pi * f0 * te)
    f_true = f0 * ((te[:, -1] - te[:, 0]) / (t[-1] - t[0])).mean()
    at = boat(te[:, len(t) // 2].mean())
    b_true = np.arctan2(at[1], at[0])
    sig_b, sig_g, _, _ = sigmas(np.linalg.norm(at), P)

    freqs = np.fft.rfftfreq(len(t), 1 / fs)
    near = np.abs(freqs - f0) < 2
    df = freqs[1]
    coarse = np.radians(np.arange(360))
    ferr, berr = [], []
    for _ in range(trials):
        x = clean + rng.normal(0, np.sqrt(fs / 2), clean.shape)  # unit noise density
        spec = (np.abs(np.fft.rfft(x, axis=1)) ** 2).sum(0)
        fk = freqs[near][np.argmax(spec[near])]

        def amps(f):
            return x @ np.exp(-2j * np.pi * f * t)

        f = minimize_scalar(
            lambda f: -(np.abs(amps(f)) ** 2).sum(), bounds=(fk - df, fk + df), method="bounded", options={"xatol": 1e-7}
        ).x
        X, k = amps(f), 2 * np.pi * f / c

        def steer(th):
            return -np.abs((X * np.exp(-1j * k * a * np.cos(np.atleast_1d(th)[:, None] - ang))).sum(1)) ** 2

        th0 = coarse[np.argmin(steer(coarse))]
        th = minimize_scalar(lambda th: steer(th)[0], bounds=(th0 - 0.03, th0 + 0.03), method="bounded").x
        ferr.append(f - f_true)
        berr.append((th - b_true + np.pi) % (2 * np.pi) - np.pi)
    ferr, berr = np.array(ferr), np.array(berr)
    return {
        "pitch_ratio": float(np.sqrt((ferr**2).mean()) / (sig_g * f0)),
        "bearing_ratio": float(np.sqrt((berr**2).mean()) / sig_b),
        "pitch_bias_sigmas": float(abs(ferr.mean()) / (sig_g * f0)),
        "doppler_hz": float(f_true),
        "predicted_hz": float(f0 * predict(np.array([*at, *v, 1.0, 0.0]), np.zeros((1, 2)), P)[1][0]),
    }


PAGE = Path(__file__).with_name("index.html")


def optimise_route(b, P, seed, req):
    better = optimise(b, P, seed=seed)
    return {"buoys": better, "before": accuracy_map(b, P)["score"], "after": accuracy_map(better, P)["score"]}


ROUTES = {
    "/api/map": lambda b, P, seed, req: accuracy_map(b, P),
    "/api/optimise": optimise_route,
    "/api/run": lambda b, P, seed, req: run_scenario(b, P, seed),
    "/api/explain": lambda b, P, seed, req: explain(b, P, req.get("boat"), seed),
}


class Handler(BaseHTTPRequestHandler):
    timeout = 10  # a stalled client cannot hold a thread for ever

    def log_message(self, *args):
        pass

    def reply(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            self.reply(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/settings":
            self.reply(200, json.dumps(SETTINGS).encode())
        else:
            self.reply(404, b"{}")

    def do_POST(self):
        if self.path not in ROUTES:
            return self.reply(404, b"{}")
        try:
            size = int(self.headers.get("Content-Length") or 0)
            if not 0 < size <= 100_000:
                raise ValueError("request too large or empty")
            req = json.loads(self.rfile.read(size))
            b, P, seed = as_buoys(req.get("buoys", [])), params(req.get("params")), int(req.get("seed", 0)) % 2**32
            body = json.dumps(ROUTES[self.path](b, P, seed, req), allow_nan=False).encode()
        except (ValueError, TypeError, AttributeError, OverflowError, RecursionError) as e:
            return self.reply(400, json.dumps({"error": f"Bad request: {e}"}).encode())
        except Exception as e:  # a model failure must reach the page as a message, not a dropped connection
            return self.reply(500, json.dumps({"error": f"The model failed on these settings: {e}"}).encode())
        self.reply(200, body)


def serve(port=8000):
    print(f"DopplerBuoys on http://127.0.0.1:{port}  (Ctrl+C to stop)")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


def check_model():
    P = params()
    assert P["array_gain_db"] == 0  # no free gain from hydrophones closer than half a wavelength
    here = np.array([[0.0, 0.0]])
    closing = predict(np.array([1000.0, 0, -5, 0, 1, 0]), here, P)[1][0]
    opening = predict(np.array([1000.0, 0, 5, 0, 1, 0]), here, P)[1][0]
    assert closing > 1 > opening, "closing boat must read higher pitch"

    assert 3000 < detection_range(P) < 5000, detection_range(P)  # PNNL-19169: 3 to 5 km
    assert detection_range(params({"nl": 80})) < detection_range(P)

    pos = np.random.default_rng(0).uniform(1500, 3500, (50, 2))
    vel = np.tile([5.0, 0], (50, 1))
    two = [[1000, 1000], [4000, 4000]]
    four = two + [[1000, 4000], [4000, 1000]]
    assert crlb(pos, vel, four, P).mean() < crlb(pos, vel, two, P).mean()

    # tones that alias across the hydrophone ring give no bearing
    assert np.isinf(sigmas(np.array([500.0]), params({"array_d": 5, "f0": 500, "compass_deg": 0}))[0]).all()

    # Review Focus 1 to 4: degenerate inputs give finite, capped results
    assert accuracy_map([], P)["score"] == CAP and accuracy_map([], P)["within"] == 0
    assert np.isfinite(accuracy_map([[2500, 2500]], P)["score"])
    assert np.isfinite(crlb([[2500, 2500]], [[5, 0]], [[2500, 2500], [100, 100]], P)).all()
    off = params({"use_bearing": 0, "use_doppler": 0, "use_tdoa": 0})
    assert accuracy_map(four, off)["score"] == CAP
    extreme = params({"window": 0.5, "hydrophones": 3, "array_d": 0, "nl": 90})
    m = accuracy_map(four, extreme)
    assert np.isfinite(m["grid"]).all() and np.isfinite(m["detect_range"])
    hot = params({"sl": 180, "spread": 10, "wobble_pct": 0, "harmonics": 8, "f0": 500})  # once a singular matrix
    assert np.isfinite(accuracy_map(four, hot)["grid"]).all()
    assert params({"hydrophones": 1, "junk": 5})["hydrophones"] == 3  # clamped, unknown key dropped


def check_audio():
    a = audio_check()
    assert 0.8 < a["pitch_ratio"] < 1.25, a  # measured error over formula
    assert 0.8 < a["bearing_ratio"] < 1.25, a
    assert a["pitch_bias_sigmas"] < 3, a
    assert abs(a["doppler_hz"] - a["predicted_hz"]) < 5e-4, a  # waveform Doppler matches predict()
    low = audio_check(snr_db=DEFAULTS["detect_db"])  # the weakest signal the model will use
    assert low["pitch_ratio"] < 1.5 and low["bearing_ratio"] < 1.5, low


def check_tracker():
    ring = [[2500 + 1500 * np.cos(a), 2500 + 1500 * np.sin(a)] for a in np.arange(6) * np.pi / 3]
    truth = np.array([2800.0, 2700, 5, 2, 1, 0])
    cases = [("all", {}), ("no arrival time", {"use_tdoa": 0}), ("arrival time only", {"use_bearing": 0, "use_doppler": 0})]
    for name, over in cases:
        P, rng = params(over), np.random.default_rng(1)
        fixes = [locate(measure(truth, as_buoys(ring), P, rng), as_buoys(ring), P) for _ in range(150)]
        rms = np.sqrt(np.mean([((f[:2] - truth[:2]) ** 2).sum() for f in fixes]))
        ratio = rms / crlb(truth[:2], truth[2:4], ring, P)[0]
        assert 0.7 < ratio < 1.5, (name, ratio)  # tracker reaches the bound the map draws

    assert all(e is None for b in run_scenario([], params())["boats"] for e in b["est"])
    grid9 = [[x, y] for x in (800, 2500, 4200) for y in (800, 2500, 4200)]
    run = run_scenario(grid9, params())
    fisher = run["boats"][1]
    stopped = [i for i, s in enumerate(fisher["silent"]) if s]
    assert len(stopped) * run["dt"] > 250, "fisher should sit silent for about 5 minutes"
    assert all(fisher["est"][i] is None and fisher["true"][i] is not None for i in stopped)
    assert fisher["tracked_all"] < fisher["tracked"]  # silent time counts against the all-time figure
    assert run["boats"][0]["median_err"] < 50, run["boats"][0]["median_err"]
    json.dumps(run, allow_nan=False)


def check_optimiser():
    P = params()
    corner = [[100 + 80 * i, 100 + 60 * i] for i in range(5)]
    better = optimise(corner, P, iters=300)
    assert accuracy_map(better, P)["score"] < 0.5 * accuracy_map(corner, P)["score"]
    assert len(better) == 5 and all(0 <= v <= SIZE for b in better for v in b)
    assert optimise([], P) == []


def check_explain():
    P = params()
    grid9 = [[x, y] for x in (800, 2500, 4200) for y in (800, 2500, 4200)]
    east = [2000, 2500, 5, 0, 1]  # boat heading east
    e = explain(grid9, P, east)
    assert len(e["buoys"]) == 9 and e["fix"] is not None
    ahead, behind = e["buoys"][7], e["buoys"][1]  # buoys at (4200, 2500) and (800, 2500)
    assert ahead["closing"] > 0 > behind["closing"]
    assert ahead["pitch_true"] > e["f0"] > behind["pitch_true"]
    shapes = e["shapes"]
    assert shapes["all"]["error"] <= min(shapes[k]["error"] for k in ("bearing", "doppler", "tdoa")) + 1e-6
    assert e["error"] < 6 * shapes["all"]["error"]
    assert "doppler" not in explain(grid9, params({"use_doppler": 0}), east)["shapes"]
    none = explain([], P, east)
    assert none["buoys"] == [] and none["fix"] is None
    json.dumps(explain(grid9, params({"array_d": 0}), east), allow_nan=False)  # no bearing, still valid JSON
    for bad in ([1, 2], "x", [1, 2, 3, 4, float("nan")], None):
        try:
            explain(grid9, P, bad)
            raise AssertionError(f"accepted boat {bad!r}")
        except ValueError:
            pass


def check_server():
    ROUTES["/api/boom"] = lambda b, P, seed, req: 1 / 0
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_port}"

    def post(path, body):
        try:
            with urllib.request.urlopen(urllib.request.Request(url + path, data=body, method="POST")) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    three = [[1000, 1000], [4000, 4000], [1000, 4000]]
    code, m = post("/api/map", json.dumps({"buoys": three[:2], "params": {"nl": 75}}).encode())
    assert code == 200 and len(m["grid"]) == 40 and len(m["grid"][0]) == 40
    assert post("/api/map", b"not json")[0] == 400
    assert post("/api/map", b"[1, 2]")[0] == 400
    assert post("/api/map", b'{"buoys": [[1, "x"]]}')[0] == 400
    assert post("/api/map", b'{"buoys": [[1, 2]], "params": {"nl": "loud"}}')[0] == 400
    assert post("/api/map", b'{"buoys": [], "seed": 1e999}')[0] == 400
    assert post("/api/map", b'{"buoys": ' + b" " * 200_000 + b"[]}")[0] == 400  # oversized
    assert post("/api/nope", b"{}")[0] == 404
    code, err = post("/api/boom", b"{}")
    assert code == 500 and "division" in err["error"]  # model failures reach the page as a message
    many = [[i, i] for i in range(500)]
    assert post("/api/map", json.dumps({"buoys": many}).encode())[0] == 200 and len(as_buoys(many)) == MAX_BUOYS
    code, run = post("/api/run", json.dumps({"buoys": three}).encode())
    assert code == 200 and len(run["boats"]) == 3
    code, opt = post("/api/optimise", json.dumps({"buoys": [[100, 100], [200, 200]]}).encode())
    assert code == 200 and len(opt["buoys"]) == 2 and opt["after"] <= opt["before"]
    code, ex = post("/api/explain", json.dumps({"buoys": three, "boat": [2500, 2500, 3, 4, 1]}).encode())
    assert code == 200 and len(ex["buoys"]) == 3
    assert post("/api/explain", json.dumps({"buoys": three, "boat": [1, 2]}).encode())[0] == 400
    assert post("/api/explain", json.dumps({"buoys": three}).encode())[0] == 400
    with urllib.request.urlopen(url + "/") as r:
        assert b"<canvas" in r.read()
    with urllib.request.urlopen(url + "/api/settings") as r:
        assert json.loads(r.read())["nl"] == [70, 40, 100]
    srv.shutdown()
    del ROUTES["/api/boom"]


def check():
    check_model()
    check_audio()
    check_tracker()
    check_optimiser()
    check_explain()
    check_server()
    print("check passed")


if __name__ == "__main__":
    check() if "--check" in sys.argv else serve()
