from __future__ import annotations

import asyncio
import copy
import secrets
import string
from dataclasses import dataclass, field

from fastapi import WebSocket

from .game import (
    create_initial_game_state,
    get_piece_controller,
    get_piece_moves,
    get_ship_moves,
    is_traitor_available,
    resolve_piece_move,
    resolve_ship_move,
    resolve_traitor_ability,
)
from .game.bot import BotDifficulty, apply_bot_action, choose_bot_action
from .game.constants import PLAYER_LABELS
from .game.types import GameState, Player, Position
from .game.utils import other_player, positions_match
from .tokens import TokenError, create_seat_token, verify_seat_token

SEATS: tuple[Player, Player] = ("vikings", "marauders")


class RoomError(Exception):
    pass


class RoomNotFoundError(RoomError):
    pass


class SeatUnavailableError(RoomError):
    pass


class ActionRejectedError(RoomError):
    pass


@dataclass(frozen=True)
class RoomBot:
    seat: Player
    difficulty: BotDifficulty


@dataclass
class Room:
    room_id: str
    game_state: GameState
    claimed_seats: dict[Player, bool]
    sockets: dict[Player, WebSocket | None]
    started: bool = False
    bot: RoomBot | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class RoomManager:
    def __init__(self) -> None:
        self.rooms: dict[str, Room] = {}
        self.rooms_lock = asyncio.Lock()

    async def create_room(self) -> tuple[dict[str, object], Player, str]:
        async with self.rooms_lock:
            room_id = self._generate_room_id()
            room = Room(
                room_id=room_id,
                game_state=create_initial_game_state(),
                claimed_seats={seat: False for seat in SEATS},
                sockets={seat: None for seat in SEATS},
            )
            room.claimed_seats["vikings"] = True
            room.game_state["status"] = "Waiting for the second player to join."
            self.rooms[room_id] = room

        seat: Player = "vikings"
        return self._snapshot_room(room), seat, create_seat_token(room_id, seat)

    async def create_bot_room(self, difficulty: BotDifficulty) -> tuple[dict[str, object], Player, str]:
        async with self.rooms_lock:
            room_id = self._generate_room_id()
            human_seat: Player = "vikings"
            bot_seat: Player = "marauders"
            room = Room(
                room_id=room_id,
                game_state=create_initial_game_state(),
                claimed_seats={seat: True for seat in SEATS},
                sockets={seat: None for seat in SEATS},
                started=True,
                bot=RoomBot(seat=bot_seat, difficulty=difficulty),
            )
            room.game_state["status"] = (
                f"{PLAYER_LABELS[room.game_state['currentTurn']]} to move against the {difficulty} bot."
            )
            self.rooms[room_id] = room

        return self._snapshot_room(room), human_seat, create_seat_token(room_id, human_seat)

    async def join_room(self, room_id: str) -> tuple[dict[str, object], Player, str]:
        room = await self.get_room(room_id)

        async with room.lock:
            seat = next((candidate for candidate in SEATS if not room.claimed_seats[candidate]), None)

            if not seat:
                raise SeatUnavailableError("This room is already full.")

            room.claimed_seats[seat] = True
            self._maybe_start_room(room)
            snapshot = self._snapshot_room(room)

        return snapshot, seat, create_seat_token(room_id, seat)

    async def get_room(self, room_id: str) -> Room:
        async with self.rooms_lock:
            room = self.rooms.get(room_id)

        if not room:
            raise RoomNotFoundError("That room does not exist on this backend.")

        return room

    async def connect_socket(self, room_id: str, seat: Player, websocket: WebSocket) -> dict[str, object]:
        room = await self.get_room(room_id)
        previous_socket: WebSocket | None = None

        async with room.lock:
            if not room.claimed_seats[seat]:
                raise ActionRejectedError("That seat has not been claimed for this room.")

            previous_socket = room.sockets[seat]
            room.sockets[seat] = websocket
            snapshot = self._snapshot_room(room)

        if previous_socket and previous_socket is not websocket:
            await self._safe_close(previous_socket, code=4001, reason="Seat reclaimed on reconnect.")

        return snapshot

    async def disconnect_socket(self, room_id: str, seat: Player, websocket: WebSocket) -> None:
        try:
            room = await self.get_room(room_id)
        except RoomNotFoundError:
            return

        async with room.lock:
            if room.sockets[seat] is websocket:
                room.sockets[seat] = None

    async def broadcast_room(self, room_id: str, exclude: WebSocket | None = None) -> None:
        room = await self.get_room(room_id)

        async with room.lock:
            snapshot = self._snapshot_room(room)
            recipients = [
                (seat, websocket)
                for seat, websocket in room.sockets.items()
                if websocket is not None and websocket is not exclude
            ]

        if not recipients:
            return

        for seat, websocket in recipients:
            try:
                await websocket.send_json({"type": "room_state", "room": snapshot})
            except Exception:
                await self.disconnect_socket(room_id, seat, websocket)

    def verify_token(self, token: str) -> tuple[str, Player]:
        try:
            room_id, seat = verify_seat_token(token)
        except TokenError as exc:
            raise ActionRejectedError(str(exc)) from exc

        return room_id, seat

    async def apply_action(self, room_id: str, seat: Player, message: dict[str, object]) -> dict[str, object]:
        room = await self.get_room(room_id)

        async with room.lock:
            if not room.claimed_seats[seat]:
                raise ActionRejectedError("That seat is not active in this room.")

            if not room.started:
                raise ActionRejectedError("The match will start once both players have joined.")

            if room.game_state["winner"]:
                raise ActionRejectedError("This match is already over.")

            if room.game_state["currentTurn"] != seat:
                raise ActionRejectedError("It is not your turn.")

            action_type = message.get("type")

            if action_type == "move_piece":
                room.game_state = self._apply_piece_move(
                    room.game_state,
                    seat,
                    str(message.get("pieceId", "")),
                    message.get("target"),
                )
            elif action_type == "move_ship":
                room.game_state = self._apply_ship_move(
                    room.game_state,
                    seat,
                    str(message.get("shipId", "")),
                    message.get("target"),
                )
            elif action_type == "use_traitor":
                room.game_state = self._apply_traitor_ability(
                    room.game_state,
                    seat,
                    str(message.get("targetHunterId", "")),
                )
            else:
                raise ActionRejectedError("Unsupported action.")

            snapshot = self._snapshot_room(room)

        return snapshot

    async def maybe_run_bot_turn(self, room_id: str) -> bool:
        room = await self.get_room(room_id)

        async with room.lock:
            if (
                room.bot is None
                or not room.started
                or room.game_state["winner"]
                or room.game_state["currentTurn"] != room.bot.seat
            ):
                return False

            action = choose_bot_action(room.game_state, room.bot.seat, room.bot.difficulty)

            if action is None:
                winner = other_player(room.bot.seat)
                room.game_state = {
                    **room.game_state,
                    "winner": winner,
                    "status": f"{PLAYER_LABELS[winner]} win because the {room.bot.difficulty} bot has no legal move.",
                }
            else:
                room.game_state = apply_bot_action(room.game_state, action)

            return True

    def _apply_piece_move(
        self, state: GameState, seat: Player, piece_id: str, target_value: object
    ) -> GameState:
        if not isinstance(target_value, dict):
            raise ActionRejectedError("Piece moves need a target square.")

        target = self._parse_position(target_value)
        piece = next((candidate for candidate in state["pieces"] if candidate["id"] == piece_id), None)

        if not piece:
            raise ActionRejectedError("That piece no longer exists.")

        if get_piece_controller(piece, state) != seat:
            raise ActionRejectedError("You cannot move that piece.")

        valid_moves = get_piece_moves(piece, state)

        if not any(positions_match(move, target) for move in valid_moves):
            raise ActionRejectedError("That square is not a legal move for the selected piece.")

        return resolve_piece_move(state, piece_id, target)

    def _apply_ship_move(
        self, state: GameState, seat: Player, ship_id: str, target_value: object
    ) -> GameState:
        if not isinstance(target_value, dict):
            raise ActionRejectedError("Ship moves need a target square.")

        target = self._parse_position(target_value)
        ship = next((candidate for candidate in state["ships"] if candidate["id"] == ship_id), None)

        if not ship:
            raise ActionRejectedError("That ship no longer exists.")

        if ship["owner"] != seat:
            raise ActionRejectedError("You cannot move that ship.")

        valid_moves = get_ship_moves(ship, state)

        if not any(positions_match(move, target) for move in valid_moves):
            raise ActionRejectedError("That square is not a legal move for the selected ship.")

        return resolve_ship_move(state, ship_id, target)

    def _apply_traitor_ability(self, state: GameState, seat: Player, target_hunter_id: str) -> GameState:
        if not is_traitor_available(state, seat):
            raise ActionRejectedError("The Traitor is not available for your side.")

        target = next((piece for piece in state["pieces"] if piece["id"] == target_hunter_id), None)

        if not target or target["kind"] != "hunter" or target["owner"] == seat:
            raise ActionRejectedError("Choose an enemy Hunter for the Traitor.")

        return resolve_traitor_ability(state, target_hunter_id)

    def _maybe_start_room(self, room: Room) -> None:
        if room.started:
            return

        if all(room.claimed_seats.values()):
            room.started = True
            room.game_state["status"] = (
                f"Both players are seated. {PLAYER_LABELS[room.game_state['currentTurn']]} to move."
            )
        else:
            room.game_state["status"] = "Waiting for the second player to join."

    def _snapshot_room(self, room: Room) -> dict[str, object]:
        return {
            "roomId": room.room_id,
            "gameState": copy.deepcopy(room.game_state),
            "seats": {
                seat: {
                    "claimed": room.claimed_seats[seat],
                    "connected": room.sockets[seat] is not None,
                }
                for seat in SEATS
            },
            "started": room.started,
            "playerCount": sum(1 for claimed in room.claimed_seats.values() if claimed),
            "bot": (
                {
                    "seat": room.bot.seat,
                    "difficulty": room.bot.difficulty,
                }
                if room.bot
                else None
            ),
        }

    def _generate_room_id(self) -> str:
        alphabet = string.ascii_lowercase + string.digits

        while True:
            room_id = "".join(secrets.choice(alphabet) for _ in range(8))

            if room_id not in self.rooms:
                return room_id

    def _parse_position(self, value: dict[str, object]) -> Position:
        row = value.get("row")
        col = value.get("col")

        if not isinstance(row, int) or not isinstance(col, int):
            raise ActionRejectedError("Board positions must include numeric row and col values.")

        return {"row": row, "col": col}

    async def _safe_close(self, websocket: WebSocket, code: int, reason: str) -> None:
        try:
            await websocket.close(code=code, reason=reason)
        except Exception:
            return
