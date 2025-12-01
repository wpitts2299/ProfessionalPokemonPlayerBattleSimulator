# Professional Pokemon Player Battle Simulator

Offline-first Pokemon battle sandbox. It builds teams from local CSV stats/moves/abilities, optionally enriches them with Pikalytics usage data, and lets you battle a built-in AI through a simple GUI (default) or a CLI loop.

## What each Python file does (plain English)
- `main.py` — entry point. Launches the GUI by default or a text-based battle with `--cli`; wires data loading, team building, and the battle loop.
- `gui_battle.py` — Tkinter interface for picking actions and watching turns resolve without using the terminal.
- `battle_ai.py` — the computer opponent. Scores game states, picks moves/switches, and applies effects/turn logic.
- `team_builder.py` — assembles teams from the stats/moves/abilities data and assigns items using simple heuristics.
- `data_loader.py` — reads the CSV datasets, shapes them into `Pokemon`/`Move` objects, and pulls Pikalytics suggestions (moves/abilities) when cached/fetched.
- `pikalytics_util.py` — fetches and caches Pikalytics overview/detail pages, parses usage tables, and saves/loads “compendium” CSV/JSON/XML bundles.
- `generate_pikalytics_compendium.py` — CLI helper to prebuild the Pikalytics compendium files so team building can stay offline and fast.

## Running order (typical flow)
1) **(Optional) Pre-cache usage data**  
   If you want AI team choices to mirror ladder usage, build a compendium (uses cache if present, fetches if online):  
   `python generate_pikalytics_compendium.py gen9vgc2025regh --min-usage 0.5`

2) **Start the battle simulator**  
   - GUI (default): `python main.py`  
   - CLI: `python main.py --cli`  
   Add `--player-team` with comma-separated names to force your roster, and `--item-style` (`balanced|aggressive|defensive`) to change held items.

## What the app does
- Loads Pokemon stats/moves/abilities from the bundled CSVs (or your own paths).
- Optionally blends in Pikalytics usage to pick realistic moves, abilities, and opponents.
- Builds two teams (yours and the AI’s), then runs a turn-by-turn battle where you choose moves/switches and the AI responds using its evaluator.
- Saves Pikalytics data to `pikalytics_cache/` so you can reuse it offline for faster team generation and reporting.
