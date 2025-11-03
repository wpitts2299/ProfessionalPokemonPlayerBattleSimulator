try:
    import tkinter as tk
    from tkinter import ttk
except Exception as e:
    raise ImportError(f"tkinter not available: {e}")

from typing import Optional

from data_loader import load_local_data
from team_builder import generate_balanced_team
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
        ttk.Button(ctrl, text="New Battle", command=self._new_battle).pack(side=tk.LEFT)
        ttk.Button(ctrl, text="Quit", command=self.destroy).pack(side=tk.RIGHT)

    # ---------------- Battle Lifecycle ----------------
    def _new_battle(self):
        ai_team = generate_balanced_team(self.stats_df, self.moves_df, self.abilities_df, n=6, item_style="aggressive")
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
