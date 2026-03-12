from __future__ import annotations

from dataclasses import dataclass
from random import SystemRandom
from typing import Literal

from .actions import resolve_piece_move, resolve_ship_move, resolve_traitor_ability
from .moves import get_piece_moves, get_ship_moves
from .selectors import get_chief, get_piece_controller, is_traitor_available
from .types import GameState, Player, Position
from .utils import is_adjacent, other_player

BotDifficulty = Literal["easy", "medium", "hard"]

_RNG = SystemRandom()
_WIN_SCORE = 100_000
_SHIP_CONTROL_VALUE = 35


@dataclass(frozen=True)
class BotAction:
    type: Literal["move_piece", "move_ship", "use_traitor"]
    piece_id: str | None = None
    ship_id: str | None = None
    target: Position | None = None
    target_hunter_id: str | None = None


@dataclass(frozen=True)
class DifficultyProfile:
    depth: int
    candidate_pool: int
    score_window: int


DIFFICULTY_PROFILES: dict[BotDifficulty, DifficultyProfile] = {
    "easy": DifficultyProfile(depth=1, candidate_pool=6, score_window=180),
    "medium": DifficultyProfile(depth=1, candidate_pool=2, score_window=45),
    "hard": DifficultyProfile(depth=2, candidate_pool=1, score_window=0),
}


def choose_bot_action(state: GameState, player: Player, difficulty: BotDifficulty) -> BotAction | None:
    legal_actions = get_legal_actions(state, player)

    if not legal_actions:
        return None

    profile = DIFFICULTY_PROFILES[difficulty]
    ranked_children = _rank_children(state, legal_actions, player, reverse=True)
    scored_actions: list[tuple[BotAction, int]] = []

    for action, child_state, _ in ranked_children:
        score = _search(
            child_state,
            depth=profile.depth - 1,
            alpha=-_WIN_SCORE * 2,
            beta=_WIN_SCORE * 2,
            bot_player=player,
        )
        scored_actions.append((action, score))

    scored_actions.sort(key=lambda item: item[1], reverse=True)
    best_score = scored_actions[0][1]
    viable_actions = [
        action
        for action, score in scored_actions
        if best_score - score <= profile.score_window
    ][: profile.candidate_pool]

    return _RNG.choice(viable_actions or [scored_actions[0][0]])


def apply_bot_action(state: GameState, action: BotAction) -> GameState:
    if action.type == "move_piece" and action.piece_id and action.target:
        return resolve_piece_move(state, action.piece_id, action.target)

    if action.type == "move_ship" and action.ship_id and action.target:
        return resolve_ship_move(state, action.ship_id, action.target)

    if action.type == "use_traitor" and action.target_hunter_id:
        return resolve_traitor_ability(state, action.target_hunter_id)

    return state


def get_legal_actions(state: GameState, player: Player) -> list[BotAction]:
    actions: list[BotAction] = []

    for piece in state["pieces"]:
        if get_piece_controller(piece, state) != player:
            continue

        for target in get_piece_moves(piece, state):
            actions.append(
                BotAction(
                    type="move_piece",
                    piece_id=piece["id"],
                    target={"row": target["row"], "col": target["col"]},
                )
            )

    for ship in state["ships"]:
        if ship["owner"] != player:
            continue

        for target in get_ship_moves(ship, state):
            actions.append(
                BotAction(
                    type="move_ship",
                    ship_id=ship["id"],
                    target={"row": target["row"], "col": target["col"]},
                )
            )

    if is_traitor_available(state, player):
        enemy = other_player(player)

        for piece in state["pieces"]:
            if piece["kind"] == "hunter" and piece["owner"] == enemy:
                actions.append(BotAction(type="use_traitor", target_hunter_id=piece["id"]))

    return actions


def _search(state: GameState, depth: int, alpha: int, beta: int, bot_player: Player) -> int:
    if state["winner"] or depth == 0:
        return _evaluate_state(state, bot_player, depth)

    current_player = state["currentTurn"]
    legal_actions = get_legal_actions(state, current_player)

    if not legal_actions:
        return _evaluate_state(state, bot_player, depth)

    maximizing = current_player == bot_player
    ranked_children = _rank_children(state, legal_actions, bot_player, reverse=maximizing)

    if maximizing:
        best_score = -_WIN_SCORE * 2

        for _, child_state, _ in ranked_children:
            best_score = max(best_score, _search(child_state, depth - 1, alpha, beta, bot_player))
            alpha = max(alpha, best_score)

            if alpha >= beta:
                break

        return best_score

    best_score = _WIN_SCORE * 2

    for _, child_state, _ in ranked_children:
        best_score = min(best_score, _search(child_state, depth - 1, alpha, beta, bot_player))
        beta = min(beta, best_score)

        if beta <= alpha:
            break

    return best_score


def _rank_children(
    state: GameState,
    actions: list[BotAction],
    bot_player: Player,
    reverse: bool,
) -> list[tuple[BotAction, GameState, int]]:
    ranked_children: list[tuple[BotAction, GameState, int]] = []

    for action in actions:
        child_state = apply_bot_action(state, action)
        ranked_children.append((action, child_state, _evaluate_state(child_state, bot_player, 0)))

    ranked_children.sort(key=lambda item: item[2], reverse=reverse)
    return ranked_children


def _evaluate_state(state: GameState, bot_player: Player, depth_remaining: int) -> int:
    opponent = other_player(bot_player)

    if state["winner"] == bot_player:
        return _WIN_SCORE + depth_remaining

    if state["winner"] == opponent:
        return -_WIN_SCORE - depth_remaining

    player_chief = get_chief(state, bot_player)
    opponent_chief = get_chief(state, opponent)
    unclaimed_maces = [mace for mace in state["maces"] if not mace["carriedBy"]]
    dragon = next((piece for piece in state["pieces"] if piece["kind"] == "dragon"), None)
    score = 0

    for piece in state["pieces"]:
        controller = get_piece_controller(piece, state)

        if controller is None:
            continue

        sign = 1 if controller == bot_player else -1
        score += sign * _piece_value(piece["kind"], piece["carriesMace"])

        if piece["kind"] in ("hunter", "traitor"):
            enemy_chief = opponent_chief if controller == bot_player else player_chief

            if piece["carriesMace"] and enemy_chief:
                distance = _manhattan_distance(piece["position"], enemy_chief["position"])
                score += sign * max(0, 60 - 10 * distance)

                if is_adjacent(piece["position"], enemy_chief["position"]):
                    score += sign * 4_000
            elif unclaimed_maces:
                distance = min(
                    _manhattan_distance(piece["position"], mace["position"])
                    for mace in unclaimed_maces
                )
                score += sign * max(0, 12 - distance)

    for ship in state["ships"]:
        score += _SHIP_CONTROL_VALUE if ship["owner"] == bot_player else -_SHIP_CONTROL_VALUE

    if state["dragonController"] == bot_player:
        score += 90
    elif state["dragonController"] == opponent:
        score -= 90
    elif dragon and player_chief and opponent_chief:
        score += max(0, 7 - _manhattan_distance(player_chief["position"], dragon["position"])) * 4
        score -= max(0, 7 - _manhattan_distance(opponent_chief["position"], dragon["position"])) * 4

    if state["traitorClaimedBy"] == bot_player:
        score += 70
    elif state["traitorClaimedBy"] == opponent:
        score -= 70
    elif state["traitorTokenPosition"] and player_chief and opponent_chief:
        score += max(
            0,
            7 - _manhattan_distance(player_chief["position"], state["traitorTokenPosition"]),
        ) * 3
        score -= max(
            0,
            7 - _manhattan_distance(opponent_chief["position"], state["traitorTokenPosition"]),
        ) * 3

    if is_traitor_available(state, bot_player):
        score += 40

    if is_traitor_available(state, opponent):
        score -= 40

    return score


def _piece_value(kind: str, carries_mace: bool) -> int:
    if kind == "hunter":
        return 100 + (60 if carries_mace else 0)

    if kind == "traitor":
        return 135 + (60 if carries_mace else 0)

    if kind == "dragon":
        return 150

    return 0


def _manhattan_distance(first: Position, second: Position) -> int:
    return abs(first["row"] - second["row"]) + abs(first["col"] - second["col"])

