import tkinter as tk
from tkinter import messagebox, scrolledtext
import random, copy

# Pokemon Classes
class PokemonMove:
    def __init__(self, name, power, type_name, accuracy, pp, is_special):
        self.name = name
        self.power = power
        self.type = type_name
        self.accuracy = accuracy
        self.pp = pp
        self.max_pp = pp
        self.is_special = is_special

class TypeChart:
    effectiveness = {
        "Normal": {"Rock": 0.5, "Ghost": 0.0, "Steel": 0.5},
        "Fire": {"Fire": 0.5, "Water": 0.5, "Grass": 2, "Ice": 2, "Bug": 2, "Rock": 0.5, "Dragon": 0.5, "Steel": 2},
        "Water": {"Fire": 2, "Water": 0.5, "Grass": 0.5, "Ground": 2, "Rock": 2, "Dragon": 0.5},
        "Electric": {"Water": 2, "Electric": 0.5, "Grass": 0.5, "Ground": 0, "Flying": 2, "Dragon": 0.5},
        "Grass": {"Fire": 0.5, "Water": 2, "Grass": 0.5, "Poison": 0.5, "Ground": 2, "Flying": 0.5, "Bug": 0.5, "Rock": 2, "Dragon": 0.5, "Steel": 0.5},
        "Ice": {"Fire": 0.5, "Water": 0.5, "Grass": 2, "Ice": 0.5, "Ground": 2, "Flying": 2, "Dragon": 2, "Steel": 0.5},
        "Fighting": {"Normal": 2, "Ice": 2, "Rock": 2, "Dark": 2, "Steel": 2, "Poison": 0.5, "Flying": 0.5, "Psychic": 0.5, "Bug": 0.5, "Ghost": 0},
        "Poison": {"Grass": 2, "Fairy": 2, "Poison": 0.5, "Ground": 0.5, "Rock": 0.5, "Ghost": 0.5, "Steel": 0},
        "Ground": {"Fire": 2, "Electric": 2, "Grass": 0.5, "Poison": 2, "Flying": 0, "Bug": 0.5, "Rock": 2, "Steel": 2},
        "Flying": {"Electric": 0.5, "Grass": 2, "Fighting": 2, "Bug": 2, "Rock": 0.5, "Steel": 0.5},
        "Psychic": {"Fighting": 2, "Poison": 2, "Psychic": 0.5, "Dark": 0, "Steel": 0.5},
        "Bug": {"Fire": 0.5, "Grass": 2, "Fighting": 0.5, "Poison": 0.5, "Flying": 0.5, "Psychic": 2, "Ghost": 0.5, "Dark": 2, "Steel": 0.5, "Fairy": 0.5},
        "Rock": {"Fire": 2, "Ice": 2, "Fighting": 0.5, "Ground": 0.5, "Flying": 2, "Bug": 2, "Steel": 0.5},
        "Ghost": {"Normal": 0, "Psychic": 2, "Ghost": 2, "Dark": 0.5},
        "Dragon": {"Dragon": 2, "Steel": 0.5, "Fairy": 0},
        "Dark": {"Fighting": 0.5, "Psychic": 2, "Ghost": 2, "Dark": 0.5, "Fairy": 0.5},
        "Steel": {"Fire": 0.5, "Water": 0.5, "Electric": 0.5, "Ice": 2, "Rock": 2, "Steel": 0.5, "Fairy": 2},
        "Fairy": {"Fire": 0.5, "Fighting": 2, "Poison": 0.5, "Dragon": 2, "Dark": 2, "Steel": 0.5},
    }

    @staticmethod
    def get_effectiveness(move_type, target_type):
        return TypeChart.effectiveness.get(move_type, {}).get(target_type, 1.0)

class Pokemon:
    def __init__(self, name, type_name, hp, attack, special_attack, defense, special_defense, speed, moves):
        self.name = name
        self.type = type_name
        self.level = 50
        self.max_hp = hp
        self.hp = hp
        self.attack = attack
        self.special_attack = special_attack
        self.defense = defense
        self.special_defense = special_defense
        self.speed = speed
        self.moves = moves

    #Check if fainted, by seeing if hp is 0 or less.
    def is_fainted(self):
        return self.hp <= 0

    # Calculates damage based damage formula in main series games.
    def calculate_damage(self, move, defender):
        if move.pp <= 0:
            return 0, 1.0
        if random.randint(1,100) > move.accuracy:
            return 0, 1.0
        attack_stat = self.special_attack if move.is_special else self.attack
        defense_stat = defender.special_defense if move.is_special else defender.defense
        effectiveness = TypeChart.get_effectiveness(move.type, defender.type)
        stab = 1.5 if move.type == self.type else 1.0
        damage = int(((2*self.level/5+2)*move.power*(attack_stat/defense_stat)/50+2)*effectiveness*stab)
        if effectiveness == 0:
            damage = 0
        else:
            damage = max(1, damage)
        defender.hp = max(0, defender.hp - damage)
        move.pp -= 1
        return damage, effectiveness

# Pokemon List
all_pokemon = [
    Pokemon("Snorlax","Normal",160,110,65,65,110,30,[
        PokemonMove("Giga Impact",75,"Normal",90,10,False),
        PokemonMove("Crunch",85,"Dark",100,15,False),
        PokemonMove("High Horsepower",95,"Ground",95,10,False),
        PokemonMove("Body Slam",85,"Normal",100,15,False)
    ]),
Pokemon("Entei","Fire",115,115,100,85,75,100,[
        PokemonMove("Flamethrower",90,"Fire",100,15,True),
        PokemonMove("Crunch",80,"Dark",100,15,False),
        PokemonMove("Extrasensory",80,"Psychic",100,20,True),
        PokemonMove("Sacred Fire",100,"Fire",95,5,True)
    ]),
    Pokemon("Samurott","Water",95,100,90,80,70,70,[
        PokemonMove("Hydro Pump",110,"Water",80,5,True),
        PokemonMove("Aqua Tail",90,"Water",90,10,False),
        PokemonMove("Megahorn",120,"Bug",85,10,False),
        PokemonMove("Surf",90,"Water",100,15,True)
    ]),
    Pokemon("Sceptile","Grass",88,85,105,65,65,120,[
        PokemonMove("Leaf Blade",90,"Grass",100,15,False),
        PokemonMove("Dragon Claw",80,"Dragon",100,15,False),
        PokemonMove("X-Scissor",80,"Bug",100,15,False),
        PokemonMove("Aerial Ace",60,"Flying",100,20,False)
    ]),
    Pokemon("Jolteon","Electric",80,70,110,60,60,130,[
        PokemonMove("Thunderbolt",90,"Electric",100,15,True),
        PokemonMove("Shadow Ball",80,"Ghost",100,15,True),
        PokemonMove("Signal Beam",75,"Bug",100,15,True),
        PokemonMove("Discharge",80,"Electric",100,15,True)
    ]),
    Pokemon("Regice","Ice",110,80,95,100,100,50,[
        PokemonMove("Ice Beam",90,"Ice",100,10,True),
        PokemonMove("Thunderbolt",90,"Electric",100,15,True),
        PokemonMove("Focus Blast",120,"Fighting",70,5,True),
        PokemonMove("Blizzard",110,"Ice",70,5,True)
    ]),
    Pokemon("Machamp","Fighting",90,130,65,80,60,55,[
        PokemonMove("Dynamic Punch",100,"Fighting",50,5,False),
        PokemonMove("Stone Edge",100,"Rock",80,5,False),
        PokemonMove("Earthquake",100,"Ground",100,10,False),
        PokemonMove("Fire Punch",75,"Fire",100,15,False)
    ]),
    Pokemon("Garbodor","Poison",105,95,70,80,70,75,[
        PokemonMove("Sludge Bomb",90,"Poison",100,10,True),
        PokemonMove("Gunk Shot",120,"Poison",80,5,False),
        PokemonMove("Earthquake",100,"Ground",100,10,False),
        PokemonMove("Dark Pulse",80,"Dark",100,15,True)
    ]),
    Pokemon("Donphan","Ground",90,120,50,100,80,50,[
        PokemonMove("Earthquake",100,"Ground",100,10,False),
        PokemonMove("Stone Edge",100,"Rock",80,5,False),
        PokemonMove("Megahorn",120,"Bug",85,10,False),
        PokemonMove("Iron Tail",100,"Steel",75,15,False)
    ]),
    Pokemon("Tornadus","Flying",85,95,105,70,70,111,[
        PokemonMove("Hurricane",110,"Flying",70,10,True),
        PokemonMove("Sludge Bomb",90,"Poison",100,10,True),
        PokemonMove("Focus Blast",120,"Fighting",70,5,True),
        PokemonMove("Thunderbolt",90,"Electric",100,15,True)
    ]),
    Pokemon("Alakazam","Psychic",55,50,135,40,85,120,[
        PokemonMove("Psychic",90,"Psychic",100,10,True),
        PokemonMove("Shadow Ball",80,"Ghost",100,15,True),
        PokemonMove("Focus Blast",120,"Fighting",70,5,True),
        PokemonMove("Dazzling Gleam",80,"Fairy",100,15,True)
    ]),
    Pokemon("Pinsir","Bug",65,125,45,100,70,85,[
        PokemonMove("X-Scissor",80,"Bug",100,15,False),
        PokemonMove("Earthquake",100,"Ground",100,10,False),
        PokemonMove("Stone Edge",100,"Rock",80,5,False),
        PokemonMove("Close Combat",120,"Fighting",100,5,False)
    ]),
    Pokemon("Lycanroc","Rock",75,115,55,75,65,112,[
        PokemonMove("Stone Edge",100,"Rock",80,5,False),
        PokemonMove("Crunch",80,"Dark",100,15,False),
        PokemonMove("Earthquake",100,"Ground",100,10,False),
        PokemonMove("Rock Slide",75,"Rock",90,10,False)
    ]),
    Pokemon("Dusknoir","Ghost",45,70,45,100,100,45,[
        PokemonMove("Shadow Punch",60,"Ghost",100,20,False),
        PokemonMove("Earthquake",100,"Ground",100,10,False),
        PokemonMove("Ice Punch",75,"Ice",100,15,False),
        PokemonMove("Thunder Punch",75,"Electric",100,15,False)
    ]),
    Pokemon("Haxorus","Dragon",76,130,60,95,70,97,[
        PokemonMove("Dragon Claw",80,"Dragon",100,15,False),
        PokemonMove("Earthquake",100,"Ground",100,10,False),
        PokemonMove("Iron Tail",100,"Steel",75,15,False),
        PokemonMove("Poison Jab",80,"Poison",100,20,False)
    ]),
    Pokemon("Zoroark","Dark",60,105,60,60,60,105,[
        PokemonMove("Night Daze",85,"Dark",100,10,False),
        PokemonMove("Crunch",80,"Dark",100,15,False),
        PokemonMove("U-turn",80,"Bug",100,20,False),
        PokemonMove("Flamethrower",90,"Fire",100,15,True)
    ]),
    Pokemon("Melmetal","Steel",135,130,60,100,80,70,[
        PokemonMove("Double Iron Bash",120,"Steel",100,10,False),
        PokemonMove("Thunder Punch",75,"Electric",100,15,False),
        PokemonMove("Earthquake",100,"Ground",100,10,False),
        PokemonMove("Superpower",100,"Fighting",100,5,False)
    ]),
    Pokemon("Xerneas","Fairy",126,131,95,131,98,99,[
        PokemonMove("Moon Blast",95,"Fairy",100,15,True),
        PokemonMove("Earth Power",90,"Ground",100,10,True),
        PokemonMove("Stone Edge",100,"Rock",80,5,False),
        PokemonMove("Close Combat",120,"Fighting",100,5,False)
    ])
]

# GUI and Battle Logic
class BattleGUI:
    def __init__(self, master):
        messagebox.showinfo(
            "Welcome to Pokémon Boss Rush!",
            "Welcome to Pokémon battles!\n\n"
            "Rules:\n"
            "1. Player will select 6 different Pokémon.\n"
            "2. AI will have all 18 Pokemon.\n"
            "3. During a battle, the faster Pokémon attacks first. "
            "If both have the same speed, it's a 50/50 chance who goes first.\n"
            "4. Moves have limited PP; if a move has 0 PP, it cannot be used.\n"
            "5. Type effectiveness affects damage: super effective moves deal more, "
            "not very effective moves deal less, and some moves may have no effect.\n"
            "6. If a Pokémon defeats 4 enemy Pokémon in a row, it becomes exhausted and faints.\n"
            "7. If your active Pokémon faints, you must switch to another Pokémon.\n\n"
            "Good luck!"
)
        self.master = master
        self.master.title("Pokemon Boss Rush")
        self.player_team = []
        self.ai_team = []
        self.player_active = None
        self.ai_active = None
        self.player_move_history = []
        self.player_victories = {}  # key: player Pokémon name, value: number of AI Pokémon defeated


        # Log frame
        self.log_frame = tk.Frame(self.master)
        self.log_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        self.log = scrolledtext.ScrolledText(self.log_frame, height=15, state='disabled')
        self.log.pack(fill=tk.BOTH, expand=True)

        # Choose Pokémon
        self.choose_frame = tk.Frame(master)
        self.choose_frame.pack()
        tk.Label(self.choose_frame, text="Choose 6 Pokémon:").pack()
        for idx, pkmn in enumerate(all_pokemon):
            btn = tk.Button(self.choose_frame, text=pkmn.name,
                            command=lambda i=idx: self.add_to_team(i))
            btn.pack()

    # Makes scrolling text box for battle log
    def add_log(self, text):
        self.log['state'] = 'normal'
        self.log.insert(tk.END, text + "\n")
        self.log.yview(tk.END)
        self.log['state'] = 'disabled'

    # Makes player choose party members
    def add_to_team(self, idx):
        # Prevent selecting the same Pokémon twice
        if len(self.player_team) < 6 and all_pokemon[idx].name not in [p.name for p in self.player_team]:
            self.player_team.append(copy.deepcopy(all_pokemon[idx]))
        else:
            messagebox.showinfo("Duplicate Pokémon", f"{all_pokemon[idx].name} is already in your team!")

        # Automatically start battle once 6 different Pokémon are selected
        if len(self.player_team) == 6:
            self.start_battle()

    #Initializes AI's team, and selects starting pokemon for each side. Player's first pokemon will be the first one they selected.
    def start_battle(self):
        self.ai_team = [copy.deepcopy(p) for p in all_pokemon if p.name not in [x.name for x in self.player_team]]
        random.shuffle(self.ai_team)
        self.player_active = self.player_team[0]
        self.ai_active = self.ai_team[0]
        self.choose_frame.destroy()
        self.update_battle_ui()

    #Doea any and all updates to  the UI during battle
    def update_battle_ui(self):
        for widget in self.master.winfo_children():
            if widget not in [self.log_frame]:
                widget.destroy()
        tk.Label(self.master, text=f"Player Active: {self.player_active.name} HP: {self.player_active.hp}/{self.player_active.max_hp}").pack()
        tk.Label(self.master, text=f"AI Active: {self.ai_active.name} HP: {self.ai_active.hp}/{self.ai_active.max_hp}").pack()
        for move in self.player_active.moves:
            btn = tk.Button(self.master, text=f"{move.name} (PP {move.pp}/{move.max_pp})",
                            command=lambda m=move: self.player_attack(m))
            btn.pack()
        tk.Button(self.master, text="Switch Pokémon", command=self.switch_pokemon_ui).pack()

    # Player Turn
    #Checks if pokemon is fainted or if move has pp, then determines turn order based on speed
    def player_attack(self, move):
        # If current Pokémon is fainted, cannot move
        if self.player_active.is_fainted():
            self.add_log(f"{self.player_active.name} has fainted and cannot move!")
            return
        if move.pp <= 0:
            messagebox.showinfo("No PP", f"{move.name} has no PP left!")
            return

        player_speed = self.player_active.speed
        ai_speed = self.ai_active.speed

        # Speed comparison: player goes first, tie 50/50
        if player_speed > ai_speed or (player_speed == ai_speed and random.choice([True, False])):
            fainted = self.resolve_move(self.player_active, self.ai_active, move)
            # AI moves only if it is still alive and player did not faint
            if not fainted and not self.ai_active.is_fainted():
                self.ai_turn()
        else:
            # AI moves first
            fainted = self.ai_turn()
            # Player moves only if still alive
            if not fainted and not self.player_active.is_fainted():
                self.resolve_move(self.player_active, self.ai_active, move)

        self.update_battle_ui()

    #Executes a move by calculating damage, then checking if a Pokemon. Then it logs the results.
    def resolve_move(self, attacker, defender, move):
        damage, effectiveness = attacker.calculate_damage(move, defender)
        if damage == 0:
            self.add_log(f"{attacker.name}'s {move.name} missed!")
        else:
            self.add_log(f"{attacker.name} used {move.name}! It dealt {damage} damage.")
            if effectiveness > 1: self.add_log("It's super effective!")
            elif 0 < effectiveness < 1: self.add_log("It's not very effective...")
            elif effectiveness == 0: self.add_log("It had no effect!")

        fainted = False
        # Check if defender fainted
        if defender.is_fainted():
            fainted = True

        self.check_faint(attacker)  # pass attacker to track victories/exhaustion
        return fainted

    # AI Turn with Decision-Making
    #Handles whether to switch or attack based on several factors
    def ai_turn(self):
        if self.ai_active.is_fainted():
            return True  # nothing happens

        action, choice = self.choose_ai_action()
        if action == "switch":
            self.ai_active = choice
            self.add_log(f"AI switched to {choice.name}!")
            return True  # switching consumes the turn
        elif action == "move":
            fainted = self.resolve_move(self.ai_active, self.player_active, choice)
            return fainted
        return False

    #Implements AI decision-making logic
    def choose_ai_action(self):
        # 1. KO check if faster
        if self.ai_active.speed >= self.player_active.speed:
            for move in self.ai_active.moves:
                if move.pp > 0:
                    test_hp = self.player_active.hp
                    damage, _ = self.ai_active.calculate_damage(move, self.player_active)
                    self.player_active.hp = test_hp
                    if damage >= test_hp:
                        return "move", move

        # 2. Survivability check based on last seen player move
        if self.player_move_history:
            last_move_name = self.player_move_history[-1]
            last_move = None
            for m in self.player_active.moves:
                if m.name == last_move_name:
                    last_move = m
            if last_move:
                for candidate in self.ai_team:
                    if candidate != self.ai_active and not candidate.is_fainted():
                        test_hp = candidate.hp
                        damage, _ = self.player_active.calculate_damage(last_move, candidate)
                        candidate.hp = test_hp
                        if damage*2 < candidate.hp:  # survives 2 hits
                            return "switch", candidate

        # 3. Best damage move if staying
        best_move, best_damage = None, 0
        for move in self.ai_active.moves:
            if move.pp > 0:
                test_hp = self.player_active.hp
                damage, _ = self.ai_active.calculate_damage(move, self.player_active)
                self.player_active.hp = test_hp
                if damage > best_damage:
                    best_damage, best_move = damage, move
        if best_move:
            return "move", best_move

        # 4. Fallback random move
        moves_with_pp = [m for m in self.ai_active.moves if m.pp > 0]
        if moves_with_pp:
            return "move", random.choice(moves_with_pp)
        return "move", None

    # Switching
    #Opens a GUI for player to switch Pokemon. does not allow fainted pokemon to be used 
    def switch_pokemon_ui(self, force=False):
        top = tk.Toplevel(self.master)
        top.title("Switch Pokémon")
        tk.Label(top, text="Choose Pokémon to switch:").pack()

        # Add buttons for all available Pokémon except the active and fainted ones
        for p in self.player_team:
            if p != self.player_active and not p.is_fainted():
                btn = tk.Button(top, text=f"{p.name} HP: {p.hp}/{p.max_hp}",
                                command=lambda pk=p: self.do_switch(pk, top))
                btn.pack()

        # Add Cancel button only if not forcing a switch
        if not force:
            if not self.player_active.is_fainted():  # only allow cancel if current Pokémon is alive
                tk.Button(top, text="Cancel", command=top.destroy).pack()

    #Executes the switch and updates the UI
    def do_switch(self, new_pokemon, window):
        self.player_active = new_pokemon
        window.destroy()
        self.add_log(f"Player switched to {new_pokemon.name}")
        self.update_battle_ui()

    # Faint Handling
    #checks if either pokemon has fainted, and handles the aftermath.If player pokemon wins 4 battles, it faints from exhaustion.
    def check_faint(self, attacker=None):
        # Handle AI faint
        if self.ai_active.is_fainted():
            self.add_log(f"{self.ai_active.name} fainted!")

            # Track player Pokémon victories if attacker is player
            if attacker == self.player_active:
                p_name = self.player_active.name
                self.player_victories[p_name] = self.player_victories.get(p_name, 0) + 1

            # Remove AI Pokémon from team
            self.ai_team.remove(self.ai_active)
            if not self.ai_team:
                messagebox.showinfo("Victory", "You win!")
                self.master.destroy()
                return
            self.ai_active = self.ai_team[0]
            self.add_log(f"AI sends out {self.ai_active.name}!")
            
            # Check for exhaustion
            if self.player_victories[p_name] >= 4:
                self.add_log(f"{self.player_active.name} has fainted from exhaustion after defeating 4 AI Pokémon!")
                self.player_active.hp = 0
                self.player_team.remove(self.player_active)
                if not self.player_team:
                    messagebox.showinfo("Game Over", "AI wins!")
                    self.master.destroy()
                    return
                self.switch_pokemon_ui(force=True)
                return
        # Handle player faint normally
        if self.player_active.is_fainted():
            self.add_log(f"{self.player_active.name} fainted!")
            self.player_team.remove(self.player_active)
            if not self.player_team:
                messagebox.showinfo("Game Over", "AI wins!")
                self.master.destroy()
                return
            self.switch_pokemon_ui(force=True)

# Run Game
root = tk.Tk()
gui = BattleGUI(root)
root.mainloop()
