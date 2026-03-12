from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .game.bot import BotDifficulty
from .rooms import (
    ActionRejectedError,
    RoomManager,
    RoomNotFoundError,
    SeatUnavailableError,
)


class CreateBotRoomRequest(BaseModel):
    difficulty: BotDifficulty = "medium"


app = FastAPI(title="Maces & Talons Backend")
room_manager = RoomManager()

allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "MACES_TALONS_ALLOWED_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/rooms")
async def create_room() -> dict[str, object]:
    room, seat, seat_token = await room_manager.create_room()
    return {"room": room, "seat": seat, "seatToken": seat_token}


@app.post("/api/bot-rooms")
async def create_bot_room(request: CreateBotRoomRequest) -> dict[str, object]:
    room, seat, seat_token = await room_manager.create_bot_room(request.difficulty)
    return {"room": room, "seat": seat, "seatToken": seat_token}


@app.post("/api/rooms/{room_id}/join")
async def join_room(room_id: str) -> dict[str, object]:
    try:
        room, seat, seat_token = await room_manager.join_room(room_id)
    except RoomNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except SeatUnavailableError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    await room_manager.broadcast_room(room_id)
    return {"room": room, "seat": seat, "seatToken": seat_token}


@app.websocket("/ws/rooms/{room_id}")
async def room_socket(websocket: WebSocket, room_id: str) -> None:
    await websocket.accept()
    seat = None

    try:
        auth_message = await websocket.receive_json()

        if auth_message.get("type") != "auth" or not isinstance(auth_message.get("seatToken"), str):
            await websocket.send_json({"type": "error", "message": "Authenticate with a seat token first."})
            await websocket.close(code=4401)
            return

        token_room_id, seat = room_manager.verify_token(auth_message["seatToken"])

        if token_room_id != room_id:
            await websocket.send_json({"type": "error", "message": "That seat token is for a different room."})
            await websocket.close(code=4401)
            return

        room = await room_manager.connect_socket(room_id, seat, websocket)
        await websocket.send_json({"type": "auth_ok", "room": room, "seat": seat})
        await room_manager.broadcast_room(room_id, exclude=websocket)

        if await room_manager.maybe_run_bot_turn(room_id):
            await room_manager.broadcast_room(room_id)

        while True:
            message = await websocket.receive_json()

            try:
                await room_manager.apply_action(room_id, seat, message)
            except ActionRejectedError as exc:
                await websocket.send_json({"type": "error", "message": str(exc)})
                continue

            await room_manager.broadcast_room(room_id)

            if await room_manager.maybe_run_bot_turn(room_id):
                await room_manager.broadcast_room(room_id)
    except ActionRejectedError as exc:
        await websocket.send_json({"type": "error", "message": str(exc)})
        await websocket.close(code=4401)
    except RoomNotFoundError:
        await websocket.send_json({"type": "error", "message": "That room does not exist on this backend."})
        await websocket.close(code=4404)
    except WebSocketDisconnect:
        pass
    finally:
        if seat is not None:
            await room_manager.disconnect_socket(room_id, seat, websocket)
            try:
                await room_manager.broadcast_room(room_id)
            except RoomNotFoundError:
                return
