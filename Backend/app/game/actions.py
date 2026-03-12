from __future__ import annotations

from .constants import BOARD_SIZE, PLAYER_LABELS
from .selectors import (
    can_be_sandwich_captured,
    get_chief,
    get_ground_mace_at,
    get_piece_at,
    get_piece_controller,
    get_piece_controller_at,
    get_piece_label,
    get_ship_at,
    get_ship_label,
    get_ship_role_label,
)
from .types import ActionRecord, GameState, Piece, Player, Position, Ship
from .utils import format_square, is_adjacent, is_in_bounds, other_player


def capture_pieces(state: GameState, piece_ids: list[str]) -> GameState:
    if not piece_ids:
        return state

    removed_ids = set(piece_ids)
    removed = [piece for piece in state["pieces"] if piece["id"] in removed_ids]
    next_maces = []

    for mace in state["maces"]:
        carrier = next((piece for piece in removed if piece["id"] == mace["carriedBy"]), None)

        if not carrier:
            next_maces.append(mace)
            continue

        next_maces.append(
            {
                **mace,
                "carriedBy": None,
                "position": {
                    "row": carrier["position"]["row"],
                    "col": carrier["position"]["col"],
                },
            }
        )

    return {
        **state,
        "pieces": [piece for piece in state["pieces"] if piece["id"] not in removed_ids],
        "maces": next_maces,
    }


def append_recent_action(state: GameState, action: ActionRecord) -> GameState:
    recent_actions = [*state["recentActions"], action]
    return {**state, "recentActions": recent_actions[-6:]}


def has_l_shape_capture_support(state: GameState, position: Position, owner: Player) -> bool:
    pairs = (
        ({"row": -1, "col": 0}, {"row": 0, "col": -1}),
        ({"row": -1, "col": 0}, {"row": 0, "col": 1}),
        ({"row": 1, "col": 0}, {"row": 0, "col": -1}),
        ({"row": 1, "col": 0}, {"row": 0, "col": 1}),
    )

    for pair in pairs:
        if all(
            is_in_bounds(position["row"] + offset["row"], position["col"] + offset["col"])
            and get_piece_controller_at(
                state,
                {
                    "row": position["row"] + offset["row"],
                    "col": position["col"] + offset["col"],
                },
            )
            == owner
            for offset in pair
        ):
            return True

    return False


def apply_sandwich_captures(state: GameState) -> tuple[GameState, list[Piece]]:
    captured_ids: set[str] = set()

    for owner in ("marauders", "vikings"):
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                start = {"row": row, "col": col}

                if get_piece_controller_at(state, start) != owner:
                    continue

                for direction in ({"row": 0, "col": 1}, {"row": 1, "col": 0}):
                    line: list[Piece] = []
                    distance = 1

                    while True:
                        target = {
                            "row": start["row"] + direction["row"] * distance,
                            "col": start["col"] + direction["col"] * distance,
                        }

                        if not is_in_bounds(target["row"], target["col"]):
                            break

                        piece = get_piece_at(state, target)

                        if (
                            piece
                            and can_be_sandwich_captured(piece)
                            and get_piece_controller(piece, state) == other_player(owner)
                        ):
                            line.append(piece)
                            distance += 1
                            continue

                        if line and get_piece_controller_at(state, target) == owner:
                            for captured_piece in line:
                                captured_ids.add(captured_piece["id"])

                        break

        for piece in state["pieces"]:
            if (
                can_be_sandwich_captured(piece)
                and get_piece_controller(piece, state) == other_player(owner)
                and has_l_shape_capture_support(state, piece["position"], owner)
            ):
                captured_ids.add(piece["id"])

    captured = [piece for piece in state["pieces"] if piece["id"] in captured_ids]
    return capture_pieces(state, list(captured_ids)), captured


def sync_carried_mace(state: GameState, piece_id: str, position: Position) -> GameState:
    return {
        **state,
        "maces": [
            {
                **mace,
                "position": {"row": position["row"], "col": position["col"]},
            }
            if mace["carriedBy"] == piece_id
            else mace
            for mace in state["maces"]
        ],
    }


def give_piece_ground_mace(state: GameState, piece_id: str) -> tuple[GameState, bool]:
    piece = next((candidate for candidate in state["pieces"] if candidate["id"] == piece_id), None)

    if not piece or piece["kind"] not in ("hunter", "traitor"):
        return state, False

    if piece["carriesMace"]:
        return sync_carried_mace(state, piece_id, piece["position"]), False

    ground_mace = get_ground_mace_at(state, piece["position"])

    if not ground_mace:
        return state, False

    return (
        {
            **state,
            "pieces": [
                {**candidate, "carriesMace": True} if candidate["id"] == piece_id else candidate
                for candidate in state["pieces"]
            ],
            "maces": [
                {
                    **mace,
                    "carriedBy": piece_id,
                    "position": {
                        "row": piece["position"]["row"],
                        "col": piece["position"]["col"],
                    },
                }
                if mace["id"] == ground_mace["id"]
                else mace
                for mace in state["maces"]
            ],
        },
        True,
    )


def claim_ship_at(state: GameState, position: Position, owner: Player) -> tuple[GameState, Ship | None]:
    ship = get_ship_at(state, position)

    if not ship or ship["owner"] == owner:
        return state, None

    return (
        {
            **state,
            "ships": [
                {**candidate, "owner": owner} if candidate["id"] == ship["id"] else candidate
                for candidate in state["ships"]
            ],
        },
        ship,
    )


def apply_chief_claims(state: GameState, chief: Piece) -> tuple[GameState, list[str]]:
    next_state = state
    notes: list[str] = []
    owner = chief["owner"]
    dragon = next((piece for piece in state["pieces"] if piece["kind"] == "dragon"), None)

    if (
        owner
        and dragon
        and is_adjacent(chief["position"], dragon["position"])
        and state["dragonController"] != owner
    ):
        next_state = {**next_state, "dragonController": owner}
        notes.append(f"The Dragon now answers to {PLAYER_LABELS[owner]}.")

    if (
        owner
        and next_state["traitorTokenPosition"]
        and is_adjacent(chief["position"], next_state["traitorTokenPosition"])
    ):
        next_state = {
            **next_state,
            "traitorClaimedBy": owner,
            "traitorTokenPosition": None,
        }
        notes.append(f"{PLAYER_LABELS[owner]} claimed the Traitor.")

    return next_state, notes


def check_mace_victory(state: GameState, piece: Piece, acting_owner: Player) -> str | None:
    if piece["kind"] not in ("hunter", "traitor") or not piece["carriesMace"]:
        return None

    enemy_chief = get_chief(state, other_player(acting_owner))

    if not enemy_chief:
        return None

    if is_adjacent(piece["position"], enemy_chief["position"]):
        return f"{PLAYER_LABELS[acting_owner]} slew the enemy Chief with a Mace."

    return None


def resolve_piece_move(state: GameState, piece_id: str, target: Position) -> GameState:
    moving_piece = next((piece for piece in state["pieces"] if piece["id"] == piece_id), None)

    if not moving_piece:
        return state

    acting_owner = get_piece_controller(moving_piece, state)

    if not acting_owner:
        return state

    origin = {
        "row": moving_piece["position"]["row"],
        "col": moving_piece["position"]["col"],
    }
    next_state = state
    notes: list[str] = []

    if moving_piece["kind"] == "dragon":
        target_piece = get_piece_at(next_state, target)

        if (
            target_piece
            and target_piece["kind"] == "hunter"
            and target_piece["owner"] == other_player(acting_owner)
        ):
            next_state = capture_pieces(next_state, [target_piece["id"]])
            notes.append(
                f"The Dragon scorched {PLAYER_LABELS[target_piece['owner']]} at {format_square(target)}."
            )

    next_state = {
        **next_state,
        "pieces": [
            {
                **piece,
                "position": {"row": target["row"], "col": target["col"]},
            }
            if piece["id"] == piece_id
            else piece
            for piece in next_state["pieces"]
        ],
    }

    next_state = sync_carried_mace(next_state, piece_id, target)
    next_state, claimed_ship = claim_ship_at(next_state, target, acting_owner)

    if claimed_ship:
        notes.append(
            f"{PLAYER_LABELS[acting_owner]} seized the {get_ship_role_label(claimed_ship)} at {format_square(target)}."
        )

    next_state, picked_up = give_piece_ground_mace(next_state, piece_id)

    if picked_up:
        updated_piece = next((piece for piece in next_state["pieces"] if piece["id"] == piece_id), None)
        if updated_piece:
            notes.append(f"{get_piece_label(updated_piece, next_state)} picked up a Mace.")

    updated_piece = next((piece for piece in next_state["pieces"] if piece["id"] == piece_id), None)

    if not updated_piece:
        return state

    if updated_piece["kind"] == "chief":
        next_state, chief_notes = apply_chief_claims(next_state, updated_piece)
        notes.extend(chief_notes)
        updated_piece = next((piece for piece in next_state["pieces"] if piece["id"] == piece_id), None) or updated_piece

    if updated_piece["kind"] == "dragon":
        enemy_chief = get_chief(next_state, other_player(acting_owner))

        if enemy_chief and is_adjacent(updated_piece["position"], enemy_chief["position"]):
            next_state = {
                **next_state,
                "dragonController": other_player(acting_owner),
            }
            notes.append(
                f"The Dragon shifted its loyalty to {PLAYER_LABELS[other_player(acting_owner)]}."
            )
            updated_piece = next((piece for piece in next_state["pieces"] if piece["id"] == piece_id), None) or updated_piece

    moved_piece_label = get_piece_label(updated_piece, next_state)
    next_state, captured = apply_sandwich_captures(next_state)

    if captured:
        notes.append(
            f"Sandwich captures removed {len(captured)} piece{'' if len(captured) == 1 else 's'}."
        )

    updated_piece = next((piece for piece in next_state["pieces"] if piece["id"] == piece_id), None)
    action_record: ActionRecord = {
        "player": acting_owner,
        "type": "move_piece",
        "subjectId": piece_id,
        "fromPosition": origin,
        "toPosition": {"row": target["row"], "col": target["col"]},
    }
    next_state = append_recent_action(next_state, action_record)

    if updated_piece:
        victory_text = check_mace_victory(next_state, updated_piece, acting_owner)

        if victory_text:
            return {
                **next_state,
                "winner": acting_owner,
                "status": victory_text,
            }

    return {
        **next_state,
        "currentTurn": other_player(acting_owner),
        "status": f"{moved_piece_label} moved to {format_square(target)}."
        + (f" {' '.join(notes)}" if notes else ""),
    }


def resolve_ship_move(state: GameState, ship_id: str, target: Position) -> GameState:
    ship = next((candidate for candidate in state["ships"] if candidate["id"] == ship_id), None)

    if not ship:
        return state

    origin = {"row": ship["position"]["row"], "col": ship["position"]["col"]}
    next_state: GameState = {
        **state,
        "ships": [
            {
                **candidate,
                "position": {"row": target["row"], "col": target["col"]},
            }
            if candidate["id"] == ship_id
            else candidate
            for candidate in state["ships"]
        ],
    }

    moved_ship = next((candidate for candidate in next_state["ships"] if candidate["id"] == ship_id), None) or ship
    next_state, captured = apply_sandwich_captures(next_state)
    next_state = append_recent_action(
        next_state,
        {
            "player": ship["owner"],
            "type": "move_ship",
            "subjectId": ship_id,
            "fromPosition": origin,
            "toPosition": {"row": target["row"], "col": target["col"]},
        },
    )

    return {
        **next_state,
        "currentTurn": other_player(ship["owner"]),
        "status": f"{get_ship_label(moved_ship)} sailed to {format_square(target)}."
        + (
            f" Sandwich captures removed {len(captured)} piece{'' if len(captured) == 1 else 's'}."
            if captured
            else ""
        ),
    }


def resolve_traitor_ability(state: GameState, target_hunter_id: str) -> GameState:
    acting_owner = state["currentTurn"]
    target_hunter = next(
        (
            piece
            for piece in state["pieces"]
            if piece["id"] == target_hunter_id
            and piece["kind"] == "hunter"
            and piece["owner"] == other_player(acting_owner)
        ),
        None,
    )

    if not target_hunter:
        return state

    next_state: GameState = {
        **state,
        "pieces": [
            {
                "id": "traitor-piece",
                "kind": "traitor",
                "owner": acting_owner,
                "position": {
                    "row": piece["position"]["row"],
                    "col": piece["position"]["col"],
                },
                "carriesMace": piece["carriesMace"],
            }
            if piece["id"] == target_hunter_id
            else piece
            for piece in state["pieces"]
        ],
        "maces": [
            {
                **mace,
                "carriedBy": "traitor-piece",
                "position": {
                    "row": target_hunter["position"]["row"],
                    "col": target_hunter["position"]["col"],
                },
            }
            if mace["carriedBy"] == target_hunter_id
            else mace
            for mace in state["maces"]
        ],
        "traitorAbilityUsed": {
            **state["traitorAbilityUsed"],
            acting_owner: True,
        },
    }

    next_state, captured = apply_sandwich_captures(next_state)
    next_state = append_recent_action(
        next_state,
        {
            "player": acting_owner,
            "type": "use_traitor",
            "subjectId": "traitor-piece",
            "fromPosition": {
                "row": target_hunter["position"]["row"],
                "col": target_hunter["position"]["col"],
            },
            "toPosition": {
                "row": target_hunter["position"]["row"],
                "col": target_hunter["position"]["col"],
            },
        },
    )

    return {
        **next_state,
        "currentTurn": other_player(acting_owner),
        "status": f"{PLAYER_LABELS[acting_owner]} activated the Traitor at {format_square(target_hunter['position'])}."
        + (
            f" Sandwich captures removed {len(captured)} piece{'' if len(captured) == 1 else 's'}."
            if captured
            else ""
        ),
    }
