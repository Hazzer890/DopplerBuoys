# DopplerBuoys simulator: design

## Purpose

We want to know whether acoustic tracking of small boats holds up before we propose it for monitoring marine parks. A proposal has to state detection ranges, capabilities and limitations. The simulator produces those numbers for the acoustic part of the system. The camera is a separate piece and is not simulated.

## Goals

- Show tracking accuracy in metres across a 5 km square for any buoy layout.
- Let us move buoys by hand, add and remove them, and have an optimiser improve the layout.
- Show what each kind of measurement contributes: bearing, Doppler shift, and arrival time between buoys.
- Show boats being tracked live against their true paths.

## Out of scope

Camera, data link, power, cost, boat type classification, real hydrophone recordings.

## What would make it a failure

Optimistic numbers. Each assumption that drives the result is a visible setting with a stated default, so a reader can see what the result depends on.

## Approaches considered

1. **Full audio.** Generate the sound at each hydrophone and run real signal processing on it. This is the most faithful option. It is too slow to score the thousands of layouts an optimiser needs to try.
2. **Measurement level.** Skip the audio. Work out how loud the boat is at each buoy, convert that to an error on each measurement with standard formulas, and track from the noisy measurements. This is fast enough to optimise. It trusts the formulas.
3. **Measurement level with an audio check.** Option 2 for the map, the optimiser and the live tracking, plus one full-audio run of a boat passing a buoy to confirm the formulas match real processing.

We chose option 3. It is fast where speed matters and it tests the formulas it relies on.

## How it works

**Boats.** Each boat has a path, a speed, a set of engine tones and a loudness. A boat that stops and cuts its engine goes silent.

**Sound.** Loudness at a buoy is the boat's source level, less the loss over distance, compared against background sea noise. More hydrophones and a longer listening window both improve it.

**What each buoy measures.**
- Bearing to the boat, from arrival time differences between its own hydrophones.
- Pitch of the engine tone, which the Doppler shift moves up or down.
- Arrival time, compared between buoys.

**Errors.** Each measurement gets an error from its signal-to-noise ratio, plus three practical errors that have their own settings: buoy compass error, clock error between buoys, and engine pitch wobble.

**Accuracy map.** For each cell in a grid over the square, the simulator works out the lowest position error any tracker could reach with those measurements (the Cramér-Rao bound), averaged over boat headings. Each measurement type can be switched off to see what it adds.

**Score.** Average position error over the square, capped at 500 m per cell so that dead zones do not swamp the result. The page also reports the share of the square tracked to within a threshold (default 50 m).

**Optimiser.** Moves the buoys to lower the score and shows the before and after.

**Live run.** Boats follow set paths, including one that stops to fish with its engine off. The tracker fits position and velocity to the last few seconds of noisy measurements. The page draws the true and estimated tracks and plots the error over time.

**Fix view.** A second view of the map shows the working behind one fix. A probe boat can be dragged anywhere, or the view can follow a boat from the live run. The map draws the sound path to each buoy that hears the boat, each buoy's bearing with its uncertainty, the curve of places that fit each arrival-time gap, and the closing speed and pitch shift each buoy sees. A close-up shows where those lines cross, the fix, and the uncertainty of each method alone and of all three together. A panel under the map walks through the same fix in five steps with the numbers for each buoy.

**Audio check.** A command-line check generates the real waveform for one boat passing one buoy, measures bearing and pitch from it, and fails if the errors disagree with the formulas. It runs once with a strong signal and once at the weakest signal the model will use. That second run set the default signal threshold of 16 dB, because the pitch measurement breaks down below about 15 dB.

## Settings

| Setting | Why it matters |
|---|---|
| Engine tone loudness and pitch | Sets how far away a boat can be heard |
| Boat speed | Sets the size of the Doppler shift |
| Loss over distance | Shallow water carries sound differently to deep water |
| Sea noise | Rough weather hides quiet boats |
| Hydrophones per buoy, and their spread | Sets bearing accuracy |
| Listening window | Longer gives better pitch accuracy and a slower update |
| Buoy compass error | Adds directly to bearing error |
| Clock error between buoys | Limits arrival-time measurements |
| Engine pitch wobble | Limits Doppler measurements |

Defaults for boat loudness and sea noise come from published measurements, checked during the build and cited on the page.

## Built with

Python with numpy and scipy for the model. Python's built-in web server. One HTML page that draws on a canvas. Two source files, started with one command.

## Known limits

- A boat with its engine off is silent. The track ends at the last place it was heard.
- Sound travels in straight lines through uniform water. Echoes off the seabed and surface are not modelled, and the loss-over-distance setting stands in for them.
- Arrival timing assumes the processing corrects for the boat moving during each listening window. Without that correction a 10 kn boat smears arrival times by about 25 ms.
- Hydrophones on one buoy sit closer than half a wavelength, so they get no gain over sea noise by default. Bearings assume each hydrophone's own noise is independent.
- Accuracy figures are the lowest error any tracker could reach from the measurements. The live run shows a tracker reaching them, with errors drawn from the same formulas. Only the pitch and bearing formulas are tested against generated audio.
- Boats are told apart by engine pitch. Two boats with the same pitch are not handled.
- Buoys stay where they are placed. Drift on the mooring is folded into the timing error setting.
- Sound takes up to 3 seconds to cross the square. The tracker ignores that delay.

## Left to the builder

Page layout, optimiser method, grid resolution.

DopplerBuoys · v0.2, 26 Nov 2026
