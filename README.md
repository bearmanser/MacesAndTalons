# Maces & Talons

Maces & Talons is a browser-based strategy board game with a medieval theme, local hotseat play, computer opponents, and online multiplayer rooms.

Live version: [www.grinderstudio.no/maces-and-talons](https://www.grinderstudio.no/maces-and-talons)

## What This Project Is

This repository contains the full project behind the live game:

- A frontend that renders the board, menus, rules help, and player interactions
- A backend that manages online rooms, validates moves, handles WebSocket connections, and runs the bot opponent

In simple terms, the frontend is what players see, and the backend is the server that keeps online matches working.

![Architecture Diagram](MacesAndTalonsArchitecture.png)

## How The Game Works

Maces & Talons is a two-sided tactical board game played by the Vikings and the Marauders.

The main ideas are:

- Hunters move in straight lines
- Chiefs move up to two spaces
- Ships move on water
- Maces let a piece win by reaching the enemy Chief
- The Dragon and Traitor create big turning points during a match

The app includes:

- Solo hotseat mode for two people on one device
- Bot mode with `easy`, `medium`, and `hard` difficulty
- Online multiplayer with shareable room links
- Built-in rule summaries and visual examples for new players

## Main Technology Used

The most important tech in this project is:

- React 19
- TypeScript
- Vite
- Chakra UI
- FastAPI
- Python 3.13
- WebSockets
- Docker

## Project Structure

```text
MacesAndTalons/
|- Frontend/   # React + TypeScript + Vite client
|- Backend/    # FastAPI server, room management, game engine, bot logic
|- README.md   # Project overview
```

### Frontend

The frontend lives in [Frontend](./Frontend).

It is responsible for:

- Showing the board and pieces
- Letting players select moves
- Displaying rule help and example situations
- Creating and joining multiplayer rooms
- Connecting to the backend for bot and online play

Important frontend areas:

- [Frontend/src/App.tsx](./Frontend/src/App.tsx) contains the main app flow
- [Frontend/src/components](./Frontend/src/components) contains the UI pieces
- [Frontend/src/game](./Frontend/src/game) contains shared game rules and board logic
- [Frontend/src/multiplayer.ts](./Frontend/src/multiplayer.ts) handles API and WebSocket communication

### Backend

The backend lives in [Backend](./Backend).

It is responsible for:

- Creating game rooms
- Allowing a second player to join a room
- Keeping room state synchronized over WebSockets
- Verifying legal moves
- Running the bot opponent
- Issuing seat tokens for room access

Important backend areas:

- [Backend/app/main.py](./Backend/app/main.py) starts the FastAPI app and exposes the API and WebSocket routes
- [Backend/app/rooms.py](./Backend/app/rooms.py) manages multiplayer rooms and turn handling
- [Backend/app/game](./Backend/app/game) contains the game engine and bot logic
- [Backend/app/tokens.py](./Backend/app/tokens.py) signs and verifies seat tokens

## Local Development

### 1. Start the backend

From [Backend](./Backend):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
copy .env.example .env
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### 2. Start the frontend

From [Frontend](./Frontend):

```powershell
npm install
copy .env.example .env
npm run dev
```

The frontend runs on Vite's local dev server and, by default, talks to the backend at `http://127.0.0.1:8000`.

## Environment Notes

The included example env files already cover normal local development.

Frontend options in [Frontend/.env.example](./Frontend/.env.example):

- `VITE_API_BASE_URL` sets which backend the frontend talks to
- `VITE_APP_BASE_PATH` sets the path the site is served from

Backend options in [Backend/.env.example](./Backend/.env.example):

- `MACES_TALONS_ALLOWED_ORIGINS` controls which frontend URLs may connect
- `MACES_TALONS_ALLOWED_HOSTS` controls trusted backend hostnames
- `PORT` sets the backend port

For real deployments, it is also a good idea to set `MACES_TALONS_TOKEN_SECRET` so room seat tokens are not signed with the development fallback secret.

## Deployment Notes

- The production frontend is configured to support being served from `/maces-and-talons/`
- The backend includes a [Backend/Dockerfile](./Backend/Dockerfile)
- Multiplayer rooms are stored in memory, so the backend is designed to run as a single process

That last point matters: if the backend is scaled to multiple separate processes without shared room storage, players in the same match may end up on different servers and lose sync.

## Good To Know

- Solo mode can run entirely in the frontend
- Bot mode and online multiplayer require the backend
- The bot uses rule-aware move evaluation, with stronger lookahead at higher difficulties
- The project currently keeps gameplay state in memory rather than in a database

## Why This Repo Is Split In Two

The split between `Frontend` and `Backend` keeps responsibilities clear:

- the frontend focuses on player experience and presentation
- the backend focuses on multiplayer state, validation, and bot decisions

That makes the project easier to maintain, easier to deploy, and easier for new contributors to understand.
