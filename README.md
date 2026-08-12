# Professional Pokémon Battle Simulator

A Python-based Pokémon battle simulator featuring an AI opponent, automated team construction, competitive usage-data integration, and both graphical and command-line interfaces.

The project was designed to simulate turn-based Pokémon battles while applying **game-state evaluation, decision-making heuristics, data processing, caching, and object-oriented programming**.

## Key Features

* Turn-based Pokémon battle simulation
* AI-controlled opponent using game-state scoring
* Automated team construction
* Move and Pokémon switching decisions
* Tkinter graphical user interface
* Command-line interface
* Local CSV-based Pokémon, move, and ability data
* Optional competitive usage-data integration
* Local caching for faster offline operation
* Configurable team and item-selection options

## Technologies

**Language:** Python

**Libraries / Technologies:**
Tkinter • CSV data processing • JSON • XML • Local caching

**Software Engineering Concepts:**
Object-Oriented Programming • Game-State Evaluation • Decision Heuristics • Data Processing • Modular Application Design

## Architecture

The application separates battle logic, AI decision making, data loading, team construction, external-data integration, and the user interface into individual modules.

### `main.py`

Application entry point.

* Loads game data
* Builds player and AI teams
* Initializes the battle
* Launches either the graphical or command-line interface

### `battle_ai.py`

Implements the computer-controlled opponent.

The AI evaluates the current game state to determine actions, including:

* selecting moves
* evaluating switches
* responding to the player's actions
* applying battle effects and turn logic

### `gui_battle.py`

Provides the Tkinter graphical interface for the simulator.

Players can select actions and follow battle results without interacting directly with the command line.

### `team_builder.py`

Constructs Pokémon teams using available statistics, moves, abilities, and configurable item-selection heuristics.

### `data_loader.py`

Loads Pokémon, move, and ability information from local CSV datasets and converts the data into application objects used by the battle engine.

When available, competitive usage information can also be incorporated into team construction.

### `pikalytics_util.py`

Handles optional competitive usage-data integration.

Functionality includes:

* retrieving usage information
* parsing usage tables
* caching results locally
* loading previously cached information
* generating reusable CSV, JSON, and XML data

### `generate_pikalytics_compendium.py`

Command-line utility used to pre-build competitive usage-data files so the simulator can perform team construction without repeatedly retrieving external information.

## How It Works

The simulator follows this general workflow:

1. Load Pokémon statistics, moves, and abilities from local datasets.
2. Optionally incorporate cached or retrieved competitive usage information.
3. Construct teams for the player and AI opponent.
4. Initialize the battle state.
5. Allow the player to select a move or switch Pokémon.
6. Have the AI evaluate the current game state and select its response.
7. Resolve the turn and update the battle state.
8. Continue until the battle reaches its conclusion.

## Running the Project

### Requirements

* Python 3
* Tkinter
* Project CSV datasets

Clone the repository and navigate to the project directory.

### Graphical Interface

Run:

```bash
python main.py
```

The graphical battle interface launches by default.

### Command-Line Interface

Run:

```bash
python main.py --cli
```

## Configuration Options

A specific player roster can be supplied with:

```bash
python main.py --player-team "Pokemon1,Pokemon2,Pokemon3"
```

Item-selection behavior can also be changed:

```bash
python main.py --item-style balanced
```

Available item styles include:

* `balanced`
* `aggressive`
* `defensive`

## Optional Competitive Usage Data

The simulator can use cached competitive usage information when constructing teams and selecting moves and abilities.

A local compendium can be generated with:

```bash
python generate_pikalytics_compendium.py gen9vgc2025regh --min-usage 0.5
```

Generated information is cached locally so it can be reused for faster offline team generation.

## What This Project Demonstrates

This project demonstrates practical experience with:

* Python application development
* Object-oriented programming
* AI decision-making heuristics
* Game-state evaluation
* Algorithmic decision making
* Modular software architecture
* GUI development with Tkinter
* CSV, JSON, and XML data processing
* Data caching
* Command-line interfaces
* Integrating local and externally sourced datasets

## Project Structure

```text
ProfessionalPokemonPlayerBattleSimulator/
│
├── main.py
├── battle_ai.py
├── gui_battle.py
├── team_builder.py
├── data_loader.py
├── pikalytics_util.py
├── generate_pikalytics_compendium.py
├── [game data / CSV files]
└── README.md
```

## Future Improvements

Potential areas for continued development include:

* Additional battle mechanics
* More sophisticated AI evaluation
* Expanded automated testing
* Improved GUI visualization
* Additional competitive formats
* Performance analysis of different AI strategies

---

**Developed by William Pitts**

Computer Science, Kennesaw State University — Class of 2026
