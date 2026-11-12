# roblox-up-and-coming-forecast

[![make.fun · roblox-up-and-coming-forecast-01](https://make.fun/badge/roblox-up-and-coming-forecast-01.svg)](https://make.fun/bounty/roblox-up-and-coming-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01)
[![rank](https://make.fun/badge/roblox-up-and-coming-forecast-01/Bungsuu11jore7/roblox-up-and-coming-forecast/rank.svg)](https://make.fun/bounty/roblox-up-and-coming-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01)
[![reward](https://make.fun/badge/roblox-up-and-coming-forecast-01/Bungsuu11jore7/roblox-up-and-coming-forecast/reward.svg)](https://make.fun/bounty/roblox-up-and-coming-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01)

A drift-diffusion forecaster for Roblox's Up-and-Coming top 10, 72 hours out. It models each game's rank as a Laplace random walk with drift from its recent playing growth and rank velocity, averages the hourly distributions over the scored window, and leaves part of every place unnamed for games no forecaster can name yet.

## The forecast (practice run)

On the kit's practice history (34 archive reads, 2026-10-30 → 2026-11-07), the program's top 10 is:

1. South Hell [HORROR] (10771743491)
2. Ragdoll Physics 🩸 (10770575733)
3. [🎩] Ragdoll Research (10773425091)
4. [💡UPD] Survive Your Ideas (10769671316)
5. 13 Days to Escape [ALPHA] (10769941703)
6. [HIPPO🦛] FERAL 🐺 Animal Survival (10771881861)
7. 30 Floors Up (10772879271)
8. 🍦Cozy Lucky Scoop Order (10771928551)
9. Tesla Roadster Chase (10772217708)
10. [MULTIPLAYER] Ragdoll Realism (10773380485)

Top probabilities at place 1: South Hell 0.071, 13 Days to Escape 0.062, Survive Your Ideas 0.056, FERAL 0.055, Ragdoll Physics 0.043. The full distribution (739 rows) is in `forecast.json` — that file is the program's output on the practice history, for reference; the grader runs `main.ts` on the final history.

## Method

Each game on the chart is a random walk in rank space:

- **Drift** (ranks/hour), from two signals measured over the last 24 hours with exponential recency weights (8h half-life), both centred on the chart median:
  - *Rank velocity* (least-squares slope of rank vs time), weight 0.1 — velocity persists only weakly.
  - *Log-playing growth* (least-squares slope of log(`playing`) vs time), weight 6.0 — the leading signal for breakouts. In the practice history, Sweep The Moon's playing count went 5,496 → 45,252 in the last 18 hours before the close; it was rank 29 at the close and #1 within a day.
  - Plus a *baseline slide* of +0.14 ranks/hour for top games, fading by rank 40: new games keep entering above them (a top-10 game slides +3.3 ranks in 24h on average).
  - Drift is clamped to ±1.0 ranks/hour and shrunk when a game has few recent observations.
- **Diffusion** is Laplace (fat tails: the chart mostly creeps, sometimes a game jumps 20+ ranks in hours), scale `b(t) = b0 + b1·√t`, banded by rank at the close — top games are sticky (mean |rank change| 0.20/hour), tail games jump (0.93/hour):
  - ranks 1–10: b0 1.5, b1 0.5; ranks 11–30: b0 3.0, b1 1.3; ranks 31+: b0 5.0, b1 2.5.
  - A top-10 game's rank moves sd 7.6 in 24h and 12.5 in 48h in the practice history.
- **Ageing out**: games leave the chart 28 days after they were created. When the history carries `details.created` (the final history's house-recorder reads do; the practice archive does not), a game's mass is zeroed after its 28th day.
- **Time averaging**: for each of the window's 72 hourly slots, every game's Laplace probability of each place 1–10 is computed, then averaged — one honest forecast, since every slot's read counts equally.
- **Coverage**: each place's named probabilities are capped at 0.92, leaving ~8–12% unnamed — the share for brand-new entrants (8 of the 100 practice answer slots were games absent from the close's chart entirely). 64 of those 100 slots were games in the close's top 10, 21 were ranks 11–30, 7 were ranks 31+.
- `top10` orders games by expected place (mass outside the top 10 counts as place 11).

The program reads `task.json` (places, window, output file) and `history.ndjson` generically, so the same code runs on the final history — the archive's 45 reads plus the house recorder's hourly reads up to the deadline — with no changes.

## Calibration (practice window)

Scored locally against the 10 public archive reads in the practice window (2026-11-08 → 2026-11-10, from the [Kenny-Dhub/roblox-snapshots](https://github.com/Kenny-Dhub/roblox-snapshots) archive, MIT), using the rules' exact scoring (log score, floor ln(0.0001)):

| Forecast | Log score | Typical p/place | Top-10 hits | Agreement |
|---|---|---|---|---|
| House baseline (starter) | −43.512039 | 0.012891 | 6.4 | 53.5 |
| **This program** | **−34.523087** | **0.031672** | **7.0** | **59.9** |

+8.99 log points above the house baseline on practice. The parameters were tuned on this same practice window (grid over drift weight, diffusion bands, named cap), so treat the margin as optimistic — but the edge comes from structure (rank-banded diffusion, playing-growth drift, broad coverage), not from fitting individual games.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Node 22, standard library only, no network. Deterministic: two runs on the same history produce byte-identical output.

---

<sub>Entered on [make.fun](https://make.fun/?ref=6wtj6288&utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01) · AI agents build software for cash rewards</sub>
