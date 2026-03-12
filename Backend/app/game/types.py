from __future__ import annotations

from typing import Literal, TypedDict

Terrain = Literal["land", "water"]
Player = Literal["marauders", "vikings"]
PieceKind = Literal["hunter", "chief", "dragon", "traitor"]
ShipKind = Literal["longship", "chiefship"]


class Position(TypedDict):
    row: int
    col: int


class Piece(TypedDict):
    id: str
    kind: PieceKind
    owner: Player | None
    position: Position
    carriesMace: bool


class Ship(TypedDict):
    id: str
    kind: ShipKind
    owner: Player
    position: Position


class Mace(TypedDict):
    id: str
    position: Position
    carriedBy: str | None


class PlayerFlags(TypedDict):
    marauders: bool
    vikings: bool


class ActionRecord(TypedDict):
    player: Player
    type: Literal["move_piece", "move_ship", "use_traitor"]
    subjectId: str
    fromPosition: Position | None
    toPosition: Position | None


class GameState(TypedDict):
    pieces: list[Piece]
    ships: list[Ship]
    maces: list[Mace]
    currentTurn: Player
    dragonController: Player | None
    traitorTokenPosition: Position | None
    traitorClaimedBy: Player | None
    traitorAbilityUsed: PlayerFlags
    recentActions: list[ActionRecord]
    winner: Player | None
    status: str
