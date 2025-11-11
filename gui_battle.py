"""Simple Tkinter interface for running battles and building teams."""

import re
from html import unescape
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import ttk
except Exception as e:
    raise ImportError(f"tkinter not available: {e}")

from typing import Dict, List, Optional, Tuple

from data_loader import load_local_data, create_pokemon_from_name, _lookup_move_in_db, Pokemon
from team_builder import generate_balanced_team, generate_flowchart_ai_team, pick_item
from pikalytics_util import fetch_overview, fetch_details, CACHE_DIR, _slugify
from battle_ai import BattleAI, BattleState, TYPE_EFFECTIVENESS


DETAILS_FORMAT = "gen9vgc2025regh"
TEAM_SIZE = 4
AI_BANNED_MOVES = {"protect"}


def _load_pikalytics_html(pokemon_name: str, format_slug: str = DETAILS_FORMAT) -> str:
    slug = _slugify(pokemon_name)
    cache_path = Path(CACHE_DIR) / f"details_{format_slug}_{slug}.html"
    if not cache_path.exists():
        return ""
    try:
        return cache_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _extract_moves_from_cache(
    pokemon_name: str,
    format_slug: str = DETAILS_FORMAT,
    html: Optional[str] = None,
) -> List[Tuple[str, float]]:
    html = html or _load_pikalytics_html(pokemon_name, format_slug)
    if not html:
        return []

    start = html.find('id="moves_wrapper"')
    if start == -1:
        return []
    end_markers = [
        'id="teammate_wrapper"',
        'Teammates</span>',
        'Teammates</div>',
    ]
    end = -1
    for marker in end_markers:
        end = html.find(marker, start)
        if end != -1:
            break
    if end == -1:
        end = start + 4000
    snippet = html[start:end]

    pattern = re.compile(
        r'<div\s+class="pokedex-move-entry-new">.*?<div[^>]*>(?P<name>[^<]+)</div>.*?<div[^>]*float:right;">\s*(?P<pct>[\d\.]+)%</div>',
        re.S | re.IGNORECASE,
    )
    moves: List[Tuple[str, float]] = []
    seen: set[str] = set()
    for match in pattern.finditer(snippet):
        raw_name = unescape(match.group("name") or "").strip()
        if not raw_name:
            continue
        key = raw_name.lower()
        if key in seen or key == "other":
            continue
        seen.add(key)
        try:
            pct = float(match.group("pct") or "0")
        except Exception:
            pct = 0.0
        moves.append((raw_name, pct))
    return moves


def _extract_simple_usage_from_cache(
    pokemon_name: str,
    section_id: str,
    limit: int = 8,
    format_slug: str = DETAILS_FORMAT,
    html: Optional[str] = None,
) -> List[Tuple[str, float]]:
    html = html or _load_pikalytics_html(pokemon_name, format_slug)
    if not html:
        return []
    match = re.search(rf'<div[^>]+id="{re.escape(section_id)}"[^>]*>', html)
    if not match:
        return []
    start = match.start()
    snippet = html[start:start + 4000]
    pattern = re.compile(
        r'>\s*([^<>]+?)\s*</div>\s*<div[^>]*float:right;">\s*([\d\.]+)%</div>',
        re.S | re.IGNORECASE,
    )
    results: List[Tuple[str, float]] = []
    seen: set[str] = set()
    for name, pct in pattern.findall(snippet):
        clean = unescape(name).strip()
        key = clean.lower()
        if not clean or key == "other" or key in seen or not any(ch.isalpha() for ch in clean):
            continue
        seen.add(key)
        try:
            pct_val = float(pct)
        except Exception:
            pct_val = 0.0
        results.append((clean, pct_val))
        if len(results) >= limit:
            break
    return results


def _extract_items_from_cache(
    pokemon_name: str,
    format_slug: str = DETAILS_FORMAT,
    html: Optional[str] = None,
) -> List[Tuple[str, float]]:
    return _extract_simple_usage_from_cache(pokemon_name, "items_wrapper", 10, format_slug, html)


def _extract_abilities_from_cache(
    pokemon_name: str,
    format_slug: str = DETAILS_FORMAT,
    html: Optional[str] = None,
) -> List[Tuple[str, float]]:
    return _extract_simple_usage_from_cache(pokemon_name, "abilities_wrapper", 6, format_slug, html)
class BattleGUI(tk.Tk):
    """Minimal playable GUI for single-battle turns.

    - Shows active AI and Player Pokémon with HP bars
    - Lets the player click a move button to act
    - AI responds automatically each turn
    """

    def __init__(self):
        super().__init__()
        self.title("Pokemon Battle")
        self.geometry("720x420")

        # Data + AI init
        self.stats_df, self.moves_df, self.abilities_df = load_local_data()
        self.ai_engine = BattleAI(recursion_depth=2)
        self.state: Optional[BattleState] = None
        # Player custom team settings
        self.custom_player_team_names = []  # type: ignore[var-annotated]
        self.custom_item_style = "balanced"
        self.custom_player_team_moves: Dict[str, List[str]] = {}
        self.custom_player_team_abilities: Dict[str, str] = {}
        self.custom_player_team_items: Dict[str, str] = {}
        self._bench_lookup: Dict[str, Pokemon] = {}

        # UI
        self._build_layout()
        self._new_battle()

    # ---------------- UI Construction ----------------
    def _build_layout(self):
        root = self
        root.columnconfigure(0, weight=1)

        # AI panel
        self.ai_frame = ttk.LabelFrame(root, text="AI")
        self.ai_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=6)
        self.ai_name = ttk.Label(self.ai_frame, text="AI: ")
        self.ai_name.grid(row=0, column=0, sticky="w", padx=6, pady=4)
        self.ai_hp_label = ttk.Label(self.ai_frame, text="HP:")
        self.ai_hp_label.grid(row=1, column=0, sticky="w", padx=6)
        self.ai_hp = ttk.Progressbar(self.ai_frame, length=480)
        self.ai_hp.grid(row=1, column=1, sticky="ew", padx=6)
        self.ai_frame.columnconfigure(1, weight=1)

        # Message log
        self.msg = tk.StringVar(value="Welcome! Choose a move.")
        ttk.Label(root, textvariable=self.msg, anchor="center").grid(row=1, column=0, sticky="ew", padx=10)

        # Player panel
        self.pl_frame = ttk.LabelFrame(root, text="Player")
        self.pl_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=6)
        self.pl_name = ttk.Label(self.pl_frame, text="Player: ")
        self.pl_name.grid(row=0, column=0, sticky="w", padx=6, pady=4)
        self.pl_hp_label = ttk.Label(self.pl_frame, text="HP:")
        self.pl_hp_label.grid(row=1, column=0, sticky="w", padx=6)
        self.pl_hp = ttk.Progressbar(self.pl_frame, length=480)
        self.pl_hp.grid(row=1, column=1, sticky="ew", padx=6)
        self.pl_frame.columnconfigure(1, weight=1)

        # Moves
        self.moves_frame = ttk.LabelFrame(root, text="Moves")
        self.moves_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=6)
        self.move_buttons = []
        for i in range(4):
            btn = ttk.Button(self.moves_frame, text=f"Move {i+1}", command=lambda idx=i: self._on_move(idx))
            btn.grid(row=0, column=i, sticky="ew", padx=4, pady=4)
            self.moves_frame.columnconfigure(i, weight=1)
            self.move_buttons.append(btn)

        # Switch options
        self.switch_frame = ttk.LabelFrame(root, text="Switch")
        self.switch_frame.grid(row=4, column=0, sticky="ew", padx=10, pady=(0, 6))
        self.switch_frame.columnconfigure(0, weight=1)
        self.switch_var = tk.StringVar()
        self.switch_combo = ttk.Combobox(self.switch_frame, textvariable=self.switch_var, state="disabled", width=28)
        self.switch_combo.grid(row=0, column=0, sticky="ew", padx=(6, 4), pady=4)
        self.switch_button = ttk.Button(self.switch_frame, text="Switch", command=self._on_switch, state=tk.DISABLED)
        self.switch_button.grid(row=0, column=1, sticky="ew", padx=(4, 6), pady=4)

        # Controls
        ctrl = ttk.Frame(root)
        ctrl.grid(row=5, column=0, sticky="ew", padx=10, pady=6)
        # Custom team builder lets the player define exact roster and item style.
        ttk.Button(ctrl, text="Build Team", command=self._open_team_builder).pack(side=tk.LEFT)
        ttk.Button(ctrl, text="New Battle", command=self._new_battle).pack(side=tk.LEFT, padx=(8,0))
        ttk.Button(ctrl, text="Quit", command=self.destroy).pack(side=tk.RIGHT)

    # ---------------- Battle Lifecycle ----------------
    def _new_battle(self):
        # Build AI allowed set from Pikalytics (>= 2.5% usage)
        allowed_ai = set()
        fmt = "gen9vgc2025regh"
        try:
            from pikalytics_util import load_compendium_single_csv, load_compendium_csv, CACHE_DIR
            import os
            single_csv = os.path.join(CACHE_DIR, f"compendium_{fmt}.csv")
            if os.path.exists(single_csv):
                comp = load_compendium_single_csv(fmt)
                allowed_ai = {name for name, data in comp.get("pokemon", {}).items() if data.get("usage", 0) >= 2.5}
            elif os.path.exists(os.path.join(CACHE_DIR, f"compendium_{fmt}_overview.csv")):
                comp = load_compendium_csv(fmt)
                allowed_ai = {name for name, data in comp.get("pokemon", {}).items() if data.get("usage", 0) >= 2.5}
            else:
                from pikalytics_util import fetch_overview
                overview = fetch_overview(fmt)
                allowed_ai = {name for name, usage in overview if usage >= 2.5}
        except Exception:
            allowed_ai = set()
        # Map allowed names to names present in stats_df (case-insensitive)
        allowed_ai_lc = {n.lower() for n in allowed_ai}
        stats_names = [str(n) for n in self.stats_df["pokemon"].unique()] if "pokemon" in self.stats_df.columns else []
        allowed_mapped = {n for n in stats_names if n.lower() in allowed_ai_lc}

        ai_allow = allowed_mapped if allowed_mapped else None
        try:
            ai_team = generate_flowchart_ai_team(
                self.stats_df,
                self.moves_df,
                self.abilities_df,
                n=TEAM_SIZE,
                format_slug=fmt,
                item_style="aggressive",
                allowed_names=ai_allow,
                banned_moves=AI_BANNED_MOVES,
            )
        except Exception:
            ai_team = generate_balanced_team(
                self.stats_df,
                self.moves_df,
                self.abilities_df,
                n=TEAM_SIZE,
                item_style="aggressive",
                allowed_names=ai_allow,
                banned_moves=AI_BANNED_MOVES,
            )
        # Use custom player team if provided; otherwise generate balanced
        chosen_names = list(self.custom_player_team_names) if self.custom_player_team_names else []
        custom_moves_map: Dict[str, List[str]] = {name: list(moves)[:4] for name, moves in (self.custom_player_team_moves or {}).items()}
        custom_ability_map: Dict[str, str] = {name: ability for name, ability in (self.custom_player_team_abilities or {}).items() if ability}
        custom_item_map: Dict[str, str] = {name: item for name, item in (self.custom_player_team_items or {}).items() if item}
        if chosen_names:
            team = []
            for nm in chosen_names[:TEAM_SIZE]:
                try:
                    p = create_pokemon_from_name(nm, self.stats_df, self.moves_df, self.abilities_df, preferred_item=pick_item(self.custom_item_style))
                    override_names = [mv for mv in custom_moves_map.get(nm, []) if mv]
                    if override_names:
                        override_moves = []
                        seen = set()
                        for mv_name in override_names:
                            mv = _lookup_move_in_db(mv_name, self.moves_df)
                            if mv and mv.name not in seen:
                                override_moves.append(mv)
                                seen.add(mv.name)
                        if override_moves:
                            existing = [mv for mv in p.moves if mv.name not in seen]
                            p.moves = (override_moves + existing)[:4]
                    override_ability = custom_ability_map.get(nm)
                    if override_ability:
                        p.ability = override_ability
                    override_item = custom_item_map.get(nm)
                    if override_item:
                        p.item = override_item
                    team.append(p)
                except Exception:
                    continue
            # Fill to reach the required team size if needed
            if len(team) < TEAM_SIZE:
                fill = generate_balanced_team(self.stats_df, self.moves_df, self.abilities_df, n=TEAM_SIZE, item_style=self.custom_item_style)
                for p in fill:
                    if p.name not in [q.name for q in team] and len(team) < TEAM_SIZE:
                        team.append(p)
            pl_team = team
        else:
            pl_team = generate_balanced_team(self.stats_df, self.moves_df, self.abilities_df, n=TEAM_SIZE, item_style="balanced")
        self.state = BattleState(ai_team, pl_team)
        self.msg.set("A new battle begins! Your turn.")
        self._refresh_ui()

    def _on_move(self, idx: int):
        if not self.state:
            return
        player = self.state.active_player()
        ai_p = self.state.active_ai()
        if not player or not ai_p:
            return
        if idx >= len(player.moves):
            return

        mv = player.moves[idx]
        wmap = self.ai_engine.get_weather_moves()
        setup = self.ai_engine.get_setup_moves()
        if mv.name in wmap:
            action = {"type": "setup", "move": mv, "weather": wmap[mv.name]}
        elif mv.name in setup:
            action = {"type": "setup_stat", "move": mv}
        else:
            action = {"type": "attack", "move": mv}

        # Player action
        self.state = self.ai_engine.simulate_action(self.state, attacker=player, defender=ai_p, action=action, weather_moves=wmap)
        self.msg.set(self.state.last_event or f"You used {mv.name}!")
        self._refresh_ui()
        self.after(400, self._ai_turn)

    def _on_switch(self):
        if not self.state:
            return
        player = self.state.active_player()
        ai_p = self.state.active_ai()
        if not player or not ai_p:
            return
        choice = (self.switch_var.get() or "").strip()
        target = self._bench_lookup.get(choice)
        if not target or target.is_fainted() or target is player:
            return
        action = {"type": "switch", "pokemon": target}
        wmap = self.ai_engine.get_weather_moves()
        self.state = self.ai_engine.simulate_action(self.state, attacker=player, defender=ai_p, action=action, weather_moves=wmap)
        self.msg.set(self.state.last_event or f"You switched to {target.name}.")
        self._refresh_ui()
        self.after(400, self._ai_turn)

    def _ai_turn(self):
        if not self.state or self.state.is_terminal():
            self._show_outcome()
            return
        if getattr(self.state, "skip_move", {}).get("ai"):
            a = {"type": "skip"}
        else:
            a = self.ai_engine.choose_ai_action(self.state)
        self.state = self.ai_engine.simulate_action(
            self.state,
            attacker=self.state.active_ai(),
            defender=self.state.active_player(),
            action=a,
            weather_moves=self.ai_engine.get_weather_moves(),
        )
        self.msg.set(self.state.last_event or "AI acted.")
        self._refresh_ui()
        if self.state.is_terminal():
            self._show_outcome()

    # ---------------- UI Helpers ----------------
    def _refresh_ui(self):
        if not self.state:
            return
        ai_p = self.state.active_ai()
        pl_p = self.state.active_player()

        # Update names
        if ai_p:
            self.ai_name.config(text=f"AI: {ai_p.name} ({'/'.join(ai_p.type)})  HP {ai_p.hp}/{ai_p.max_hp}")
            self._set_hp(self.ai_hp, ai_p.hp, ai_p.max_hp)
        if pl_p:
            self.pl_name.config(text=f"Player: {pl_p.name} ({'/'.join(pl_p.type)})  HP {pl_p.hp}/{pl_p.max_hp}")
            self._set_hp(self.pl_hp, pl_p.hp, pl_p.max_hp)

        # Update moves
        for i, btn in enumerate(self.move_buttons):
            if not pl_p or i >= len(pl_p.moves):
                btn.config(text="--", state=tk.DISABLED)
            else:
                m = pl_p.moves[i]
                btn.config(text=f"{m.name}\n[{m.type}] pow {m.power}", state=tk.NORMAL)

        # Update switch options
        bench_entries: List[str] = []
        self._bench_lookup = {}
        if self.state and pl_p:
            for idx, candidate in enumerate(self.state.player_team):
                if candidate is pl_p or candidate.is_fainted():
                    continue
                label = f"{candidate.name} #{idx + 1} (HP {candidate.hp}/{candidate.max_hp})"
                bench_entries.append(label)
                self._bench_lookup[label] = candidate

        if bench_entries:
            self.switch_combo.config(state="readonly", values=bench_entries)
            if self.switch_var.get() not in bench_entries:
                choice = bench_entries[0]
                self.switch_var.set(choice)
                self.switch_combo.set(choice)
            self.switch_button.config(state=tk.NORMAL)
        else:
            self.switch_var.set("")
            self.switch_combo.set("")
            self.switch_combo.config(state="disabled", values=[])
            self.switch_button.config(state=tk.DISABLED)
    def _set_hp(self, bar: ttk.Progressbar, hp: int, max_hp: int):
        bar.config(maximum=max(1, int(max_hp)))
        bar['value'] = max(0, int(hp))

    def _show_outcome(self):
        if not self.state:
            return
        ai_alive = any(not p.is_fainted() for p in self.state.ai_team)
        pl_alive = any(not p.is_fainted() for p in self.state.player_team)
        if pl_alive and not ai_alive:
            self.msg.set("You win!")
        elif ai_alive and not pl_alive:
            self.msg.set("AI wins!")
        else:
            self.msg.set("Battle ended.")

    # ---------------- Team Builder ----------------
    def _open_team_builder(self):
        dlg = TeamBuilderDialog(
            self,
            self.stats_df,
            self.moves_df,
            self.abilities_df,
            initial_names=list(self.custom_player_team_names),
            initial_style=self.custom_item_style,
            initial_moves=dict(self.custom_player_team_moves),
            initial_abilities=dict(self.custom_player_team_abilities),
            initial_items=dict(self.custom_player_team_items),
        )
        self.wait_window(dlg)
        if getattr(dlg, "result", None):
            self.custom_player_team_names = dlg.result.get("names", [])
            self.custom_item_style = dlg.result.get("item_style", "balanced")
            move_map = dlg.result.get("moves_by_pokemon", {})
            ability_map = dlg.result.get("ability_by_pokemon", {})
            item_map = dlg.result.get("item_by_pokemon", {})
            if self.custom_player_team_names:
                self.custom_player_team_moves = {name: move_map.get(name, []) for name in self.custom_player_team_names}
                self.custom_player_team_abilities = {name: ability_map.get(name, "") for name in self.custom_player_team_names}
                self.custom_player_team_items = {name: item_map.get(name, "") for name in self.custom_player_team_names}
            else:
                self.custom_player_team_moves = {}
                self.custom_player_team_abilities = {}
                self.custom_player_team_items = {}

            self._new_battle()

            if self.custom_player_team_names:
                self.msg.set(f"Custom team loaded ({len(self.custom_player_team_names)}) - your turn.")
            else:
                self.msg.set("Custom team cleared. Balanced roster ready.")







class TeamBuilderDialog(tk.Toplevel):
    def __init__(
        self,
        master,
        stats_df,
        moves_df,
        abilities_df,
        initial_names=None,
        initial_style: str = "balanced",
        initial_moves=None,
        initial_abilities=None,
        initial_items=None,
    ):
        super().__init__(master)
        self.title("Build Your Team")
        self.resizable(False, False)
        self.result = None
        self._stats_df = stats_df
        self._moves_df = moves_df
        self._abilities_df = abilities_df

        all_stats_names = [str(n) for n in stats_df['pokemon'].unique()] if 'pokemon' in stats_df.columns else []
        allowed = set()
        try:
            from pikalytics_util import load_compendium_single_csv, load_compendium_csv, CACHE_DIR
            import os
            fmt = "gen9vgc2025regh"
            single_csv = os.path.join(CACHE_DIR, f"compendium_{fmt}.csv")
            if os.path.exists(single_csv):
                comp = load_compendium_single_csv(fmt)
                allowed = {name for name, data in comp.get('pokemon', {}).items() if data.get('usage', 0) >= 0.05}
            elif os.path.exists(os.path.join(CACHE_DIR, f"compendium_{fmt}_overview.csv")):
                comp = load_compendium_csv(fmt)
                allowed = {name for name, data in comp.get('pokemon', {}).items() if data.get('usage', 0) >= 0.05}
            else:
                overview = fetch_overview(fmt)
                allowed = {n for (n, u) in overview if isinstance(u, (int, float)) and float(u) >= 0.05}
        except Exception:
            allowed = set()
        if allowed:
            allowed_lc = {n.lower() for n in allowed}
            self._all_names = sorted([n for n in all_stats_names if n.lower() in allowed_lc])
        else:
            self._all_names = sorted(all_stats_names)
        self._type_by_name = self._build_type_lookup(stats_df)
        type_pool = sorted({t for name in self._all_names for t in self._type_by_name.get(name, [])})
        self._type_options = ["All types"] + type_pool if type_pool else ["All types"]

        self._filtered = list(self._all_names)
        self._team: List[str] = list(initial_names or [])[:TEAM_SIZE]
        self._style = tk.StringVar(value=initial_style or "balanced")

        self._moves_by_pokemon: Dict[str, List[str]] = {
            name: list((initial_moves or {}).get(name, []))[:4] for name in self._team
        }
        self._ability_by_pokemon: Dict[str, str] = {
            name: str((initial_abilities or {}).get(name, "")) for name in self._team
        }
        self._item_by_pokemon: Dict[str, str] = {
            name: str((initial_items or {}).get(name, "")) for name in self._team
        }
        self._current_pokemon: Optional[str] = self._team[0] if self._team else None
        self._available_move_entries: List[Tuple[str, str]] = []

        frm = ttk.Frame(self, padding=10)
        frm.grid(row=0, column=0, sticky="nsew")
        frm.columnconfigure(0, weight=1)
        frm.columnconfigure(1, weight=1)
        frm.columnconfigure(2, weight=1)
        frm.rowconfigure(2, weight=1)

        filter_row = ttk.Frame(frm)
        filter_row.grid(row=0, column=0, columnspan=3, sticky="ew")
        filter_row.columnconfigure(1, weight=1)
        ttk.Label(filter_row, text="Search:").grid(row=0, column=0, sticky="w")
        self.search_var = tk.StringVar()
        self.search_var.trace_add("write", lambda *_: self._apply_filter())
        ttk.Entry(filter_row, textvariable=self.search_var, width=24).grid(row=0, column=1, sticky="ew", padx=(4, 10))
        ttk.Label(filter_row, text="Type:").grid(row=0, column=2, sticky="w")
        self.type_filter_var = tk.StringVar(value=self._type_options[0])
        self.type_filter_var.trace_add("write", lambda *_: self._apply_filter())
        self.type_combo = ttk.Combobox(filter_row, textvariable=self.type_filter_var, values=self._type_options, state="readonly", width=16)
        self.type_combo.grid(row=0, column=3, sticky="w", padx=(4, 0))

        ttk.Label(frm, text="Available").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Label(frm, text=f"Your Team (max {TEAM_SIZE})").grid(row=1, column=2, sticky="w", pady=(6, 0))

        self.list_available = tk.Listbox(frm, height=12, exportselection=False)
        self.list_available.grid(row=2, column=0, sticky="nsew")

        self.list_team = tk.Listbox(frm, height=12, exportselection=False)
        self.list_team.grid(row=2, column=2, sticky="nsew", padx=(12, 0))
        self.list_team.bind("<<ListboxSelect>>", self._on_team_selection)

        btns = ttk.Frame(frm)
        btns.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        ttk.Button(btns, text="Add", command=self._add_selected).pack(side=tk.LEFT)
        ttk.Button(btns, text="Remove", command=self._remove_selected).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(btns, text="Clear", command=self._clear_team).pack(side=tk.LEFT, padx=(6, 0))

        move_frame = ttk.LabelFrame(frm, text="Moves")
        move_frame.grid(row=4, column=0, columnspan=3, sticky="nsew", pady=(10, 0))
        move_frame.columnconfigure(0, weight=1)
        move_frame.columnconfigure(2, weight=1)

        ttk.Label(move_frame, text="Available moves").grid(row=0, column=0, sticky="w")
        ttk.Label(move_frame, text="Selected moves (max 4)").grid(row=0, column=2, sticky="w")

        self.list_moves_available = tk.Listbox(move_frame, height=8, exportselection=False, selectmode=tk.MULTIPLE)
        self.list_moves_available.grid(row=1, column=0, sticky="nsew", padx=(0, 6), pady=(4, 0))

        move_buttons = ttk.Frame(move_frame)
        move_buttons.grid(row=1, column=1, sticky="ns")
        ttk.Button(move_buttons, text="Add ?", command=self._add_moves).pack(pady=2)
        ttk.Button(move_buttons, text="? Remove", command=self._remove_moves).pack(pady=2)
        ttk.Button(move_buttons, text="Clear", command=self._clear_moves).pack(pady=2)

        self.list_moves_selected = tk.Listbox(move_frame, height=8, exportselection=False)
        self.list_moves_selected.grid(row=1, column=2, sticky="nsew", padx=(6, 0), pady=(4, 0))

        self.move_hint = tk.StringVar(value="Select a Pok?mon to configure moves.")
        ttk.Label(move_frame, textvariable=self.move_hint, anchor="w").grid(row=2, column=0, columnspan=3, sticky="ew", pady=(6, 0))

        gear = ttk.Frame(move_frame)
        gear.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        gear.columnconfigure(1, weight=1)
        gear.columnconfigure(3, weight=1)

        ttk.Label(gear, text="Ability:").grid(row=0, column=0, sticky="w")
        self._ability_var = tk.StringVar()
        self.ability_combo = ttk.Combobox(gear, textvariable=self._ability_var, state="disabled", width=24)
        self.ability_combo.grid(row=0, column=1, sticky="ew", padx=(4, 12))
        self.ability_combo.bind("<<ComboboxSelected>>", self._on_ability_selected)

        ttk.Label(gear, text="Item:").grid(row=0, column=2, sticky="w")
        self._item_var = tk.StringVar()
        self.item_combo = ttk.Combobox(gear, textvariable=self._item_var, state="disabled", width=24)
        self.item_combo.grid(row=0, column=3, sticky="ew")
        self.item_combo.bind("<<ComboboxSelected>>", self._on_item_selected)

        ttk.Label(frm, text="Item style:").grid(row=5, column=0, sticky="w", pady=(10, 0))
        style_box = ttk.Combobox(frm, values=["balanced", "aggressive", "defensive"], textvariable=self._style, state="readonly", width=18)
        style_box.grid(row=5, column=1, sticky="w", padx=(4, 0), pady=(10, 0))

        actions = ttk.Frame(frm)
        actions.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        ttk.Button(actions, text="Done", command=self._done).pack(side=tk.RIGHT)
        ttk.Button(actions, text="Cancel", command=self.destroy).pack(side=tk.RIGHT, padx=(0, 6))

        self._refresh_lists()

    def _apply_filter(self):
        query = (self.search_var.get() or "").strip().lower()
        names = list(self._all_names)
        if query:
            names = [n for n in names if query in n.lower()]

        type_var = getattr(self, "type_filter_var", None)
        if type_var is not None:
            selected = (type_var.get() or "").strip()
            if selected and selected.lower() != "all types":
                target = selected.lower()
                names = [
                    n
                    for n in names
                    if any(t.lower() == target for t in self._type_by_name.get(n, []) or [])
                ]

        self._filtered = names
        self._refresh_available()

    def _refresh_available(self):
        self.list_available.delete(0, tk.END)
        for name in self._filtered:
            self.list_available.insert(tk.END, name)

    def _refresh_team(self, select_name: Optional[str] = None):
        self.list_team.delete(0, tk.END)
        for name in self._team:
            self.list_team.insert(tk.END, name)
        if not self._team:
            self._current_pokemon = None
            self.list_team.selection_clear(0, tk.END)
            self._available_move_entries = []
            self._refresh_move_lists()
            self.move_hint.set("Add a Pok?mon to configure moves.")
            self._set_ability_controls(None, [])
            self._set_item_controls(None, [])
            return
        target = select_name
        if target is None:
            idxs = self.list_team.curselection()
            if idxs:
                target = self.list_team.get(idxs[0])
            elif self._current_pokemon and self._current_pokemon in self._team:
                target = self._current_pokemon
            else:
                target = self._team[0]
        if target not in self._team:
            target = self._team[0]
        self.list_team.selection_clear(0, tk.END)
        sel_idx = self._team.index(target)
        self.list_team.selection_set(sel_idx)
        self.list_team.see(sel_idx)
        self._current_pokemon = target
        self._on_team_selection()

    def _refresh_lists(self):
        self._refresh_available()
        self._refresh_team(self._current_pokemon)

    @staticmethod
    def _build_type_lookup(stats_df) -> Dict[str, List[str]]:
        lookup: Dict[str, List[str]] = {}
        try:
            type_cols = [col for col in stats_df.columns if "type" in str(col).lower()]
            records = stats_df.to_dict(orient="records")
        except Exception:
            return lookup
        if not type_cols:
            return lookup
        for record in records:
            name = str(record.get("pokemon") or record.get("name") or record.get("original_name") or "").strip()
            if not name:
                continue
            types: List[str] = []
            for col in type_cols:
                value = record.get(col)
                if value is None:
                    continue
                text = str(value).strip()
                if not text or text.lower() == "nan":
                    continue
                formatted = text.title()
                if formatted not in types:
                    types.append(formatted)
            if types:
                lookup[name] = types
        return lookup

    def _on_team_selection(self, *_):
        idx = self.list_team.curselection()
        if not idx:
            self._current_pokemon = None
            self._available_move_entries = []
            self._refresh_move_lists()
            self.move_hint.set("Select a Pok?mon to configure moves.")
            self._set_ability_controls(None, [])
            self._set_item_controls(None, [])
            return
        name = self.list_team.get(idx[0])
        self._current_pokemon = name
        self._moves_by_pokemon.setdefault(name, [])
        self._ability_by_pokemon.setdefault(name, str(self._ability_by_pokemon.get(name, "")))
        self._item_by_pokemon.setdefault(name, str(self._item_by_pokemon.get(name, "")))
        self._load_options_for(name)

    @staticmethod
    def _fetch_usage_details(name: str) -> Dict[str, List[Tuple[str, float]]]:
        try:
            det = fetch_details(name)
            if isinstance(det, dict):
                return det
        except Exception:
            pass
        return {}

    def _load_options_for(self, name: str):
        html = _load_pikalytics_html(name)
        details_cache: Optional[Dict[str, List[Tuple[str, float]]]] = None

        move_pairs = _extract_moves_from_cache(name, html=html)
        if not move_pairs:
            details_cache = details_cache or self._fetch_usage_details(name)
            move_pairs = details_cache.get("moves", []) if isinstance(details_cache, dict) else []
        entries: List[Tuple[str, str]] = []
        seen = set()
        for mv_name, pct in move_pairs:
            if not mv_name:
                continue
            key = str(mv_name).lower()
            if key in seen:
                continue
            seen.add(key)
            try:
                pct_val = float(pct)
                label = f"{mv_name} ({pct_val:.1f}%)"
            except Exception:
                label = str(mv_name)
            entries.append((label, str(mv_name)))
            if len(entries) >= 20:
                break
        if not entries and self._moves_df is not None:
            try:
                subset = self._moves_df[self._moves_df['pokemon'].astype(str).str.lower() == name.lower()]
                if not subset.empty:
                    for mv in subset.get('move', subset.get('name')).tolist():
                        mv_str = str(mv)
                        if mv_str.lower() in seen:
                            continue
                        seen.add(mv_str.lower())
                        entries.append((mv_str, mv_str))
                        if len(entries) >= 20:
                            break
            except Exception:
                pass
        self._available_move_entries = entries
        if not entries:
            self.move_hint.set('No cached moves found; defaults will be used.')
        else:
            current = len(self._moves_by_pokemon.get(name, []))
            if current == 0:
                defaults = [raw for _, raw in entries[:4]]
                self._moves_by_pokemon[name] = defaults
                current = len(defaults)
            self.move_hint.set(f'Select up to four moves ({current}/4 chosen).')
        self._refresh_move_lists()

        ability_pairs = _extract_abilities_from_cache(name, html=html)
        if not ability_pairs:
            details_cache = details_cache or self._fetch_usage_details(name)
            ability_pairs = details_cache.get("abilities", []) if isinstance(details_cache, dict) else []
        ability_names = [ab for ab, _ in ability_pairs if ab]
        self._set_ability_controls(name, ability_names)

        item_pairs = _extract_items_from_cache(name, html=html)
        if not item_pairs:
            details_cache = details_cache or self._fetch_usage_details(name)
            item_pairs = details_cache.get("items", []) if isinstance(details_cache, dict) else []
        item_names = [it for it, _ in item_pairs if it]
        self._set_item_controls(name, item_names)

    def _set_ability_controls(self, name: Optional[str], options: List[str]):
        if name is None or not options:
            self.ability_combo.config(state="disabled", values=[])
            if name is None:
                self._ability_var.set("")
            else:
                current = self._ability_by_pokemon.get(name, "")
                self._ability_var.set(current)
            return
        self.ability_combo.config(state="readonly", values=options)
        current = self._ability_by_pokemon.get(name)
        if not current or current not in options:
            current = options[0]
            self._ability_by_pokemon[name] = current
        self._ability_var.set(current)

    def _set_item_controls(self, name: Optional[str], options: List[str]):
        if name is None or not options:
            self.item_combo.config(state="disabled", values=[])
            if name is None:
                self._item_var.set("")
            else:
                current = self._item_by_pokemon.get(name, "")
                self._item_var.set(current)
            return
        self.item_combo.config(state="readonly", values=options)
        current = self._item_by_pokemon.get(name)
        if not current or current not in options:
            current = options[0]
            self._item_by_pokemon[name] = current
        self._item_var.set(current)

    def _refresh_move_lists(self):
        self.list_moves_available.delete(0, tk.END)
        for display, _ in self._available_move_entries:
            self.list_moves_available.insert(tk.END, display)
        self._refresh_selected_moves()

    def _refresh_selected_moves(self):
        self.list_moves_selected.delete(0, tk.END)
        if not self._current_pokemon:
            return
        chosen = self._moves_by_pokemon.get(self._current_pokemon, [])
        for mv_name in chosen:
            self.list_moves_selected.insert(tk.END, mv_name)
        if self._available_move_entries:
            self.move_hint.set(f'Select up to four moves ({len(chosen)}/4 chosen).')

    def _add_moves(self):
        if not self._current_pokemon or not self._available_move_entries:
            return
        indices = self.list_moves_available.curselection()
        if not indices:
            return
        chosen = self._moves_by_pokemon.setdefault(self._current_pokemon, [])
        for idx in indices:
            if len(chosen) >= 4:
                break
            _, mv_name = self._available_move_entries[idx]
            if mv_name not in chosen:
                chosen.append(mv_name)
        self._moves_by_pokemon[self._current_pokemon] = chosen[:4]
        self._refresh_selected_moves()

    def _remove_moves(self):
        if not self._current_pokemon:
            return
        indices = self.list_moves_selected.curselection()
        if not indices:
            return
        to_remove = {self.list_moves_selected.get(i) for i in indices}
        chosen = self._moves_by_pokemon.get(self._current_pokemon, [])
        self._moves_by_pokemon[self._current_pokemon] = [mv for mv in chosen if mv not in to_remove]
        self._refresh_selected_moves()

    def _clear_moves(self):
        if not self._current_pokemon:
            return
        self._moves_by_pokemon[self._current_pokemon] = []
        self._refresh_selected_moves()

    def _on_ability_selected(self, *_):
        if not self._current_pokemon:
            return
        value = (self._ability_var.get() or "").strip()
        if value:
            self._ability_by_pokemon[self._current_pokemon] = value
        else:
            self._ability_by_pokemon.pop(self._current_pokemon, None)

    def _on_item_selected(self, *_):
        if not self._current_pokemon:
            return
        value = (self._item_var.get() or "").strip()
        if value:
            self._item_by_pokemon[self._current_pokemon] = value
        else:
            self._item_by_pokemon.pop(self._current_pokemon, None)

    def _add_selected(self):
        if len(self._team) >= TEAM_SIZE:
            return
        idx = self.list_available.curselection()
        if not idx:
            return
        name = self.list_available.get(idx[0])
        if name in self._team:
            return
        self._team.append(name)
        self._moves_by_pokemon.setdefault(name, [])
        self._ability_by_pokemon.setdefault(name, "")
        self._item_by_pokemon.setdefault(name, "")
        self._refresh_team(select_name=name)

    def _remove_selected(self):
        idx = self.list_team.curselection()
        if not idx:
            return
        name = self.list_team.get(idx[0])
        self._team = [n for n in self._team if n != name]
        self._moves_by_pokemon.pop(name, None)
        self._ability_by_pokemon.pop(name, None)
        self._item_by_pokemon.pop(name, None)
        self._refresh_team()

    def _clear_team(self):
        self._team = []
        self._moves_by_pokemon.clear()
        self._ability_by_pokemon.clear()
        self._item_by_pokemon.clear()
        self._refresh_team()

    def _done(self):
        names = list(self._team)[:TEAM_SIZE]
        moves_map = {name: list(self._moves_by_pokemon.get(name, [])[:4]) for name in names}
        ability_map = {name: self._ability_by_pokemon.get(name, "") for name in names if self._ability_by_pokemon.get(name, "")}
        item_map = {name: self._item_by_pokemon.get(name, "") for name in names if self._item_by_pokemon.get(name, "")}
        self.result = {
            'names': names,
            'item_style': self._style.get() or 'balanced',
            'moves_by_pokemon': moves_map,
            'ability_by_pokemon': ability_map,
            'item_by_pokemon': item_map,
        }
        self.destroy()
