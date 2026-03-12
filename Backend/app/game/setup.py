from __future__ import annotations

from .constants import HUNTER_COLUMNS
from .types import GameState, Piece


def create_initial_pieces() -> list[Piece]:
    marauder_hunters: list[Piece] = [
        {
            "id": f"marauders-hunter-{col}",
            "kind": "hunter",
            "owner": "marauders",
            "position": {"row": 0, "col": col},
            "carriesMace": False,
        }
        for col in HUNTER_COLUMNS
    ]
    viking_hunters: list[Piece] = [
        {
            "id": f"vikings-hunter-{col}",
            "kind": "hunter",
            "owner": "vikings",
            "position": {"row": 12, "col": col},
            "carriesMace": False,
        }
        for col in HUNTER_COLUMNS
    ]

    return [
        {
            "id": "marauders-chief",
            "kind": "chief",
            "owner": "marauders",
            "position": {"row": 0, "col": 6},
            "carriesMace": False,
        },
        *marauder_hunters,
        {
            "id": "dragon",
            "kind": "dragon",
            "owner": None,
            "position": {"row": 6, "col": 0},
            "carriesMace": False,
        },
        {
            "id": "vikings-chief",
            "kind": "chief",
            "owner": "vikings",
            "position": {"row": 12, "col": 6},
            "carriesMace": False,
        },
        *viking_hunters,
    ]


def create_initial_game_state() -> GameState:
    return {
        "pieces": create_initial_pieces(),
        "ships": [
            {
                "id": "marauders-longship",
                "kind": "longship",
                "owner": "marauders",
                "position": {"row": 2, "col": 9},
            },
            {
                "id": "marauders-chiefship",
                "kind": "chiefship",
                "owner": "marauders",
                "position": {"row": 2, "col": 8},
            },
            {
                "id": "vikings-longship",
                "kind": "longship",
                "owner": "vikings",
                "position": {"row": 10, "col": 3},
            },
            {
                "id": "vikings-chiefship",
                "kind": "chiefship",
                "owner": "vikings",
                "position": {"row": 10, "col": 4},
            },
        ],
        "maces": [
            {
                "id": "marauders-mace",
                "position": {"row": 1, "col": 6},
                "carriedBy": None,
            },
            {
                "id": "vikings-mace",
                "position": {"row": 11, "col": 6},
                "carriedBy": None,
            },
        ],
        "currentTurn": "vikings",
        "dragonController": None,
        "traitorTokenPosition": {"row": 6, "col": 12},
        "traitorClaimedBy": None,
        "traitorAbilityUsed": {"marauders": False, "vikings": False},
        "recentActions": [],
        "winner": None,
        "status": "Vikings to move.",
    }
