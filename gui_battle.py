"""Simple Tkinter interface for running battles and building teams."""

try:
    import tkinter as tk
    from tkinter import ttk
except Exception as e:
    raise ImportError(f"tkinter not available: {e}")

from typing import Optional

from data_loader import load_local_data, create_pokemon_from_name
from team_builder import generate_balanced_team, pick_item
from pikalytics_util import fetch_overview
from battle_ai import BattleAI, BattleState


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

        # Controls
        ctrl = ttk.Frame(root)
        ctrl.grid(row=4, column=0, sticky="ew", padx=10, pady=6)
        # Custom team builder lets the player define exact roster and item style.
        ttk.Button(ctrl, text="Build Team", command=self._open_team_builder).pack(side=tk.LEFT)
        ttk.Button(ctrl, text="New Battle", command=self._new_battle).pack(side=tk.LEFT, padx=(8,0))
        ttk.Button(ctrl, text="Quit", command=self.destroy).pack(side=tk.RIGHT)

    # ---------------- Battle Lifecycle ----------------
    def _new_battle(self):
        # Build AI allowed set from Pikalytics (>= 2.5% usage)
        allowed_ai = set()
        try:
            from pikalytics_util import load_compendium_single_csv, load_compendium_csv, CACHE_DIR
            import os
            fmt = "gen9vgc2025regh"
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

        ai_team = generate_balanced_team(
            self.stats_df,
            self.moves_df,
            self.abilities_df,
            n=6,
            item_style="aggressive",
            allowed_names=(allowed_mapped if allowed_mapped else None),
        )
        # Use custom player team if provided; otherwise generate balanced
        chosen_names = list(self.custom_player_team_names) if self.custom_player_team_names else []
        if chosen_names:
            team = []
            for nm in chosen_names[:6]:
                try:
                    p = create_pokemon_from_name(nm, self.stats_df, self.moves_df, self.abilities_df, preferred_item=pick_item(self.custom_item_style))
                    team.append(p)
                except Exception:
                    continue
            # Fill to 6 if needed
            if len(team) < 6:
                fill = generate_balanced_team(self.stats_df, self.moves_df, self.abilities_df, n=6, item_style=self.custom_item_style)
                for p in fill:
                    if p.name not in [q.name for q in team] and len(team) < 6:
                        team.append(p)
            pl_team = team
        else:
            pl_team = generate_balanced_team(self.stats_df, self.moves_df, self.abilities_df, n=6, item_style="balanced")
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

    def _ai_turn(self):
        if not self.state or self.state.is_terminal():
            self._show_outcome()
            return
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
                btn.config(text="—", state=tk.DISABLED)
            else:
                m = pl_p.moves[i]
                btn.config(text=f"{m.name}\n[{m.type}] pow {m.power}", state=tk.NORMAL)

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
        dlg = TeamBuilderDialog(self, self.stats_df, initial_names=list(self.custom_player_team_names), initial_style=self.custom_item_style)
        self.wait_window(dlg)
        if getattr(dlg, "result", None):
            self.custom_player_team_names = dlg.result.get("names", [])
            self.custom_item_style = dlg.result.get("item_style", "balanced")
            if self.custom_player_team_names:
                self.msg.set(f"Custom team set ({len(self.custom_player_team_names)}). Start a new battle to use it.")
            else:
                self.msg.set("Custom team cleared.")


class TeamBuilderDialog(tk.Toplevel):
    def __init__(self, master, stats_df, initial_names=None, initial_style: str = "balanced"):
        super().__init__(master)
        self.title("Build Your Team")
        self.resizable(False, False)
        self.result = None
        # List only Pokémon with Pikalytics usage >= 0.05%
        all_stats_names = [str(n) for n in stats_df["pokemon"].unique()] if "pokemon" in stats_df.columns else []
        allowed = set()
        try:
            # Prefer local compendium CSV if present (offline friendly)
            from pikalytics_util import load_compendium_single_csv, load_compendium_csv, CACHE_DIR
            import os
            fmt = "gen9vgc2025regh"
            single_csv = os.path.join(CACHE_DIR, f"compendium_{fmt}.csv")
            if os.path.exists(single_csv):
                comp = load_compendium_single_csv(fmt)
                allowed = {name for name, data in comp.get("pokemon", {}).items() if data.get("usage", 0) >= 0.05}
            elif os.path.exists(os.path.join(CACHE_DIR, f"compendium_{fmt}_overview.csv")):
                comp = load_compendium_csv(fmt)
                allowed = {name for name, data in comp.get("pokemon", {}).items() if data.get("usage", 0) >= 0.05}
            else:
                overview = fetch_overview(fmt)
                allowed = {n for (n, u) in overview if isinstance(u, (int, float)) and float(u) >= 0.05}
        except Exception:
            allowed = set()
        if allowed:
            allowed_lc = {n.lower() for n in allowed}
            self._all_names = sorted([n for n in all_stats_names if n.lower() in allowed_lc])
        else:
            # Fallback: show all if no usage data available
            self._all_names = sorted(all_stats_names)
        self._filtered = list(self._all_names)
        self._team = list(initial_names or [])
        self._style = tk.StringVar(value=initial_style or "balanced")

        frm = ttk.Frame(self, padding=10)
        frm.grid(row=0, column=0, sticky="nsew")
        frm.columnconfigure(1, weight=1)

        # Search
        ttk.Label(frm, text="Search:").grid(row=0, column=0, sticky="w")
        self.search_var = tk.StringVar()
        self.search_var.trace_add("write", lambda *_: self._apply_filter())
        ttk.Entry(frm, textvariable=self.search_var, width=24).grid(row=0, column=1, sticky="ew", padx=(4,0))

        # Available list
        ttk.Label(frm, text="Available").grid(row=1, column=0, sticky="w", pady=(6,0))
        self.list_available = tk.Listbox(frm, height=12, exportselection=False)
        self.list_available.grid(row=2, column=0, columnspan=2, sticky="nsew")

        # Team list
        ttk.Label(frm, text="Your Team (max 6)").grid(row=1, column=2, sticky="w", padx=(12,0), pady=(6,0))
        self.list_team = tk.Listbox(frm, height=12, exportselection=False)
        self.list_team.grid(row=2, column=2, sticky="nsew", padx=(12,0))

        # Buttons
        btns = ttk.Frame(frm)
        btns.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(8,0))
        ttk.Button(btns, text="Add", command=self._add_selected).pack(side=tk.LEFT)
        ttk.Button(btns, text="Remove", command=self._remove_selected).pack(side=tk.LEFT, padx=(6,0))
        ttk.Button(btns, text="Clear", command=self._clear_team).pack(side=tk.LEFT, padx=(6,0))

        # Item style
        ttk.Label(frm, text="Item style:").grid(row=4, column=0, sticky="w", pady=(10,0))
        style_box = ttk.Combobox(frm, values=["balanced", "aggressive", "defensive"], textvariable=self._style, state="readonly", width=18)
        style_box.grid(row=4, column=1, sticky="w", padx=(4,0), pady=(10,0))

        # Action buttons
        actions = ttk.Frame(frm)
        actions.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(10,0))
        ttk.Button(actions, text="Done", command=self._done).pack(side=tk.RIGHT)
        ttk.Button(actions, text="Cancel", command=self.destroy).pack(side=tk.RIGHT, padx=(0,6))

        self._refresh_lists()

    def _apply_filter(self):
        q = (self.search_var.get() or "").strip().lower()
        if not q:
            self._filtered = list(self._all_names)
        else:
            self._filtered = [n for n in self._all_names if q in n.lower()]
        self._refresh_available()

    def _refresh_available(self):
        self.list_available.delete(0, tk.END)
        for n in self._filtered:
            self.list_available.insert(tk.END, n)

    def _refresh_team(self):
        self.list_team.delete(0, tk.END)
        for n in self._team:
            self.list_team.insert(tk.END, n)

    def _refresh_lists(self):
        self._refresh_available()
        self._refresh_team()

    def _add_selected(self):
        if len(self._team) >= 6:
            return
        try:
            idx = self.list_available.curselection()
            if not idx:
                return
            name = self.list_available.get(idx[0])
            if name not in self._team:
                self._team.append(name)
                self._refresh_team()
        except Exception:
            pass

    def _remove_selected(self):
        try:
            idx = self.list_team.curselection()
            if not idx:
                return
            name = self.list_team.get(idx[0])
            self._team = [n for n in self._team if n != name]
            self._refresh_team()
        except Exception:
            pass

    def _clear_team(self):
        self._team = []
        self._refresh_team()

    def _done(self):
        self.result = {"names": list(self._team)[:6], "item_style": self._style.get() or "balanced"}
        self.destroy()
