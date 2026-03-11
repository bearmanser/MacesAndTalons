import {
  Box,
  Button,
  Grid,
  Heading,
  HStack,
  Stack,
  Text,
  VStack,
} from "@chakra-ui/react";
import { useMemo, useState } from "react";

const BOARD_SIZE = 13;
const COLUMN_LABELS = "ABCDEFGHIJKLM".split("");
const HUNTER_COLUMNS = [2, 3, 4, 5, 7, 8, 9, 10] as const;
const TERRAIN_ROWS = [
  "WWLLLLLLLLLWW",
  "WWLLLLLLLLLWW",
  "WWLLLLLLWWLWW",
  "WWWLLLWWWWWWW",
  "LWWWWWWWWWWWL",
  "LLWWLLLLLLWLL",
  "LLWLLLLLLLWLL",
  "LLWWLLLLLLWLL",
  "LWWWWWWWWWWWL",
  "WWWWWWWLLLWWW",
  "WWLWWLLLLLLWW",
  "WWLLLLLLLLLWW",
  "WWLLLLLLLLLWW",
] as const;

const terrainMap = TERRAIN_ROWS.map((row) =>
  row.split("").map((cell) => (cell === "W" ? "water" : "land"))
);

const CARDINAL_DIRECTIONS = [
  { row: -1, col: 0 },
  { row: 1, col: 0 },
  { row: 0, col: -1 },
  { row: 0, col: 1 },
] as const;

const PLAYER_LABELS = {
  left: "Top Fleet",
  right: "Bottom Fleet",
} as const;

const PLAYER_COLORS = {
  left: "#8f2d18",
  right: "#284b63",
} as const;

const PLAYER_SURFACES = {
  left: "rgba(143, 45, 24, 0.14)",
  right: "rgba(40, 75, 99, 0.14)",
} as const;

type Terrain = "land" | "water";
type Player = "left" | "right";
type PieceKind = "hunter" | "chief" | "dragon" | "traitor";
type ShipKind = "longship" | "chiefship";

type Position = {
  row: number;
  col: number;
};

type Piece = {
  id: string;
  kind: PieceKind;
  owner: Player | null;
  position: Position;
  carriesMace: boolean;
};

type Ship = {
  id: string;
  kind: ShipKind;
  owner: Player;
  position: Position;
};

type Mace = {
  id: string;
  position: Position;
  carriedBy: string | null;
};

type GameState = {
  pieces: Piece[];
  ships: Ship[];
  maces: Mace[];
  currentTurn: Player;
  dragonController: Player | null;
  traitorTokenPosition: Position | null;
  traitorClaimedBy: Player | null;
  traitorAbilityUsed: Record<Player, boolean>;
  winner: Player | null;
  status: string;
};

type Selection =
  | { type: "piece"; id: string }
  | { type: "ship"; id: string }
  | { type: "traitorAbility" };

const positionsMatch = (first: Position, second: Position) =>
  first.row === second.row && first.col === second.col;

const isInBounds = (row: number, col: number) =>
  row >= 0 && row < BOARD_SIZE && col >= 0 && col < BOARD_SIZE;

const toPositionKey = (position: Position) => `${position.row}:${position.col}`;

const otherPlayer = (player: Player): Player =>
  player === "left" ? "right" : "left";

const isAdjacent = (first: Position, second: Position) =>
  Math.max(
    Math.abs(first.row - second.row),
    Math.abs(first.col - second.col)
  ) === 1;

const formatSquare = (position: Position) =>
  `${COLUMN_LABELS[position.col]}${position.row + 1}`;

const getShipFootprint = (_kind: ShipKind, position: Position): Position[] => [
  position,
];

const getPieceController = (piece: Piece, state: GameState): Player | null => {
  if (piece.kind === "dragon") {
    return state.dragonController;
  }

  return piece.owner;
};

const getPieceAt = (state: GameState, position: Position) =>
  state.pieces.find((piece) => positionsMatch(piece.position, position));

const getShipAt = (state: GameState, position: Position, excludedId?: string) =>
  state.ships.find(
    (ship) =>
      ship.id !== excludedId &&
      getShipFootprint(ship.kind, ship.position).some((cell) =>
        positionsMatch(cell, position)
      )
  );

const getGroundMaceAt = (state: GameState, position: Position) =>
  state.maces.find(
    (mace) => !mace.carriedBy && positionsMatch(mace.position, position)
  );

const hasTraitorToken = (state: GameState, position: Position) =>
  state.traitorTokenPosition !== null &&
  positionsMatch(state.traitorTokenPosition, position);

const getChief = (state: GameState, owner: Player) =>
  state.pieces.find((piece) => piece.kind === "chief" && piece.owner === owner);

const getOccupierOwnerAt = (
  state: GameState,
  position: Position
): Player | null => {
  const piece = getPieceAt(state, position);

  if (piece) {
    return getPieceController(piece, state);
  }

  const ship = getShipAt(state, position);
  return ship ? ship.owner : null;
};

const isSelectablePiece = (piece: Piece, state: GameState) =>
  !state.winner && getPieceController(piece, state) === state.currentTurn;

const isSelectableShip = (ship: Ship, state: GameState) =>
  !state.winner && ship.owner === state.currentTurn;

const canBeSandwichCaptured = (piece: Piece) =>
  piece.kind === "hunter" || piece.kind === "traitor";

const getPieceMarker = (piece: Piece) => {
  switch (piece.kind) {
    case "hunter":
      return "H";
    case "chief":
      return "C";
    case "dragon":
      return "D";
    case "traitor":
      return "T";
  }
};

const getPieceRoleLabel = (piece: Piece) => {
  switch (piece.kind) {
    case "hunter":
      return "Hunter";
    case "chief":
      return "Chief";
    case "dragon":
      return "Dragon";
    case "traitor":
      return "Traitor";
  }
};

const getShipRoleLabel = (ship: Ship) =>
  ship.kind === "longship" ? "Longship" : "Chiefship";

const getPieceLabel = (piece: Piece, state: GameState) => {
  const controller = getPieceController(piece, state);

  if (piece.kind === "dragon") {
    return controller ? `${PLAYER_LABELS[controller]} Dragon` : "Dragon";
  }

  if (piece.owner) {
    return `${PLAYER_LABELS[piece.owner]} ${getPieceRoleLabel(piece)}`;
  }

  return getPieceRoleLabel(piece);
};

const getShipLabel = (ship: Ship) =>
  `${PLAYER_LABELS[ship.owner]} ${getShipRoleLabel(ship)}`;

const createInitialPieces = (): Piece[] => {
  const leftHunters = HUNTER_COLUMNS.map((col) => ({
    id: `left-hunter-${col}`,
    kind: "hunter" as const,
    owner: "left" as const,
    position: { row: 0, col },
    carriesMace: false,
  }));

  const rightHunters = HUNTER_COLUMNS.map((col) => ({
    id: `right-hunter-${col}`,
    kind: "hunter" as const,
    owner: "right" as const,
    position: { row: 12, col },
    carriesMace: false,
  }));

  return [
    {
      id: "left-chief",
      kind: "chief",
      owner: "left",
      position: { row: 0, col: 6 },
      carriesMace: false,
    },
    ...leftHunters,
    {
      id: "dragon",
      kind: "dragon",
      owner: null,
      position: { row: 6, col: 0 },
      carriesMace: false,
    },
    {
      id: "right-chief",
      kind: "chief",
      owner: "right",
      position: { row: 12, col: 6 },
      carriesMace: false,
    },
    ...rightHunters,
  ];
};

const createInitialGameState = (): GameState => ({
  pieces: createInitialPieces(),
  ships: [
    {
      id: "left-longship",
      kind: "longship",
      owner: "left",
      position: { row: 2, col: 9 },
    },
    {
      id: "left-chiefship",
      kind: "chiefship",
      owner: "left",
      position: { row: 2, col: 8 },
    },
    {
      id: "right-longship",
      kind: "longship",
      owner: "right",
      position: { row: 10, col: 3 },
    },
    {
      id: "right-chiefship",
      kind: "chiefship",
      owner: "right",
      position: { row: 10, col: 4 },
    },
  ],
  maces: [
    { id: "left-mace", position: { row: 1, col: 6 }, carriedBy: null },
    { id: "right-mace", position: { row: 11, col: 6 }, carriedBy: null },
  ],
  currentTurn: "left",
  dragonController: null,
  traitorTokenPosition: { row: 6, col: 12 },
  traitorClaimedBy: null,
  traitorAbilityUsed: { left: false, right: false },
  winner: null,
  status: "Top Fleet to move.",
});
const getHunterStyleMoves = (piece: Piece, state: GameState): Position[] => {
  const validMoves: Position[] = [];

  for (const direction of CARDINAL_DIRECTIONS) {
    let distance = 1;

    while (true) {
      const target = {
        row: piece.position.row + direction.row * distance,
        col: piece.position.col + direction.col * distance,
      };

      if (!isInBounds(target.row, target.col)) {
        break;
      }

      if (hasTraitorToken(state, target)) {
        break;
      }

      if (getPieceAt(state, target)) {
        break;
      }

      const terrain = terrainMap[target.row][target.col] as Terrain;
      const ship = getShipAt(state, target);

      if (
        terrain === "water" &&
        (!ship || ship.kind !== "longship" || ship.owner !== piece.owner)
      ) {
        break;
      }

      validMoves.push(target);
      distance += 1;
    }
  }

  return validMoves;
};

const getChiefMoves = (piece: Piece, state: GameState): Position[] => {
  const owner = piece.owner;

  if (!owner) {
    return [];
  }

  return CARDINAL_DIRECTIONS.flatMap((direction) => {
    const target = {
      row: piece.position.row + direction.row,
      col: piece.position.col + direction.col,
    };

    if (!isInBounds(target.row, target.col)) {
      return [];
    }

    if (hasTraitorToken(state, target) || getPieceAt(state, target)) {
      return [];
    }

    if (terrainMap[target.row][target.col] === "water") {
      const ship = getShipAt(state, target);

      if (!ship || ship.kind !== "chiefship" || ship.owner !== owner) {
        return [];
      }
    }

    return [target];
  });
};

const getDragonMoves = (piece: Piece, state: GameState): Position[] => {
  const controller = state.dragonController;

  if (!controller) {
    return [];
  }

  const validMoves: Position[] = [];

  for (const direction of CARDINAL_DIRECTIONS) {
    for (let distance = 1; distance <= 3; distance += 1) {
      const target = {
        row: piece.position.row + direction.row * distance,
        col: piece.position.col + direction.col * distance,
      };

      if (!isInBounds(target.row, target.col)) {
        break;
      }

      if (hasTraitorToken(state, target)) {
        continue;
      }

      const targetPiece = getPieceAt(state, target);

      if (!targetPiece) {
        validMoves.push(target);
        continue;
      }

      if (
        targetPiece.kind === "hunter" &&
        targetPiece.owner === otherPlayer(controller)
      ) {
        validMoves.push(target);
      }
    }
  }

  return validMoves;
};

const getPieceMoves = (piece: Piece, state: GameState): Position[] => {
  switch (piece.kind) {
    case "hunter":
    case "traitor":
      return getHunterStyleMoves(piece, state);
    case "chief":
      return getChiefMoves(piece, state);
    case "dragon":
      return getDragonMoves(piece, state);
  }
};

const getShipMoves = (ship: Ship, state: GameState): Position[] => {
  const occupied = getShipFootprint(ship.kind, ship.position).some((cell) =>
    getPieceAt(state, cell)
  );

  if (occupied) {
    return [];
  }

  const validMoves: Position[] = [];

  for (const direction of CARDINAL_DIRECTIONS) {
    let distance = 1;

    while (true) {
      const target = {
        row: ship.position.row + direction.row * distance,
        col: ship.position.col + direction.col * distance,
      };

      const footprint = getShipFootprint(ship.kind, target);

      const blocked = footprint.some((cell) => {
        if (!isInBounds(cell.row, cell.col)) {
          return true;
        }

        if (terrainMap[cell.row][cell.col] !== "water") {
          return true;
        }

        if (getPieceAt(state, cell)) {
          return true;
        }

        return Boolean(getShipAt(state, cell, ship.id));
      });

      if (blocked) {
        break;
      }

      validMoves.push(target);
      distance += 1;
    }
  }

  return validMoves;
};

const capturePieces = (state: GameState, pieceIds: string[]): GameState => {
  if (pieceIds.length === 0) {
    return state;
  }

  const removed = state.pieces.filter((piece) => pieceIds.includes(piece.id));
  const nextMaces = state.maces.map((mace) => {
    const carrier = removed.find((piece) => piece.id === mace.carriedBy);

    if (!carrier) {
      return mace;
    }

    return {
      ...mace,
      carriedBy: null,
      position: carrier.position,
    };
  });

  return {
    ...state,
    pieces: state.pieces.filter((piece) => !pieceIds.includes(piece.id)),
    maces: nextMaces,
  };
};

const applySandwichCaptures = (state: GameState) => {
  const capturedIds = new Set<string>();

  for (const owner of ["left", "right"] as const) {
    for (let row = 0; row < BOARD_SIZE; row += 1) {
      for (let col = 0; col < BOARD_SIZE; col += 1) {
        const start = { row, col };

        if (getOccupierOwnerAt(state, start) !== owner) {
          continue;
        }

        for (const direction of [
          { row: 0, col: 1 },
          { row: 1, col: 0 },
        ]) {
          const line: Piece[] = [];
          let distance = 1;

          while (true) {
            const target = {
              row: start.row + direction.row * distance,
              col: start.col + direction.col * distance,
            };

            if (!isInBounds(target.row, target.col)) {
              break;
            }

            const piece = getPieceAt(state, target);

            if (
              piece &&
              canBeSandwichCaptured(piece) &&
              getPieceController(piece, state) === otherPlayer(owner)
            ) {
              line.push(piece);
              distance += 1;
              continue;
            }

            if (
              line.length > 0 &&
              getOccupierOwnerAt(state, target) === owner
            ) {
              line.forEach((captured) => capturedIds.add(captured.id));
            }

            break;
          }
        }
      }
    }
  }

  const captured = state.pieces.filter((piece) => capturedIds.has(piece.id));

  return {
    state: capturePieces(state, [...capturedIds]),
    captured,
  };
};

const syncCarriedMace = (
  state: GameState,
  pieceId: string,
  position: Position
): GameState => ({
  ...state,
  maces: state.maces.map((mace) =>
    mace.carriedBy === pieceId ? { ...mace, position } : mace
  ),
});

const givePieceGroundMace = (
  state: GameState,
  pieceId: string
): { state: GameState; pickedUp: boolean } => {
  const piece = state.pieces.find((candidate) => candidate.id === pieceId);

  if (!piece || (piece.kind !== "hunter" && piece.kind !== "traitor")) {
    return { state, pickedUp: false };
  }

  if (piece.carriesMace) {
    return {
      state: syncCarriedMace(state, pieceId, piece.position),
      pickedUp: false,
    };
  }

  const groundMace = getGroundMaceAt(state, piece.position);

  if (!groundMace) {
    return { state, pickedUp: false };
  }

  return {
    pickedUp: true,
    state: {
      ...state,
      pieces: state.pieces.map((candidate) =>
        candidate.id === pieceId
          ? { ...candidate, carriesMace: true }
          : candidate
      ),
      maces: state.maces.map((mace) =>
        mace.id === groundMace.id
          ? { ...mace, carriedBy: pieceId, position: piece.position }
          : mace
      ),
    },
  };
};
const applyChiefClaims = (state: GameState, chief: Piece) => {
  let nextState = state;
  const notes: string[] = [];
  const owner = chief.owner;
  const dragon = state.pieces.find((piece) => piece.kind === "dragon");

  if (
    owner &&
    dragon &&
    isAdjacent(chief.position, dragon.position) &&
    state.dragonController !== owner
  ) {
    nextState = {
      ...nextState,
      dragonController: owner,
    };
    notes.push(`The Dragon now answers to ${PLAYER_LABELS[owner]}.`);
  }

  if (
    owner &&
    nextState.traitorTokenPosition &&
    isAdjacent(chief.position, nextState.traitorTokenPosition)
  ) {
    nextState = {
      ...nextState,
      traitorClaimedBy: owner,
      traitorTokenPosition: null,
    };
    notes.push(`${PLAYER_LABELS[owner]} claimed the Traitor.`);
  }

  return { state: nextState, notes };
};

const checkMaceVictory = (
  state: GameState,
  piece: Piece,
  actingOwner: Player
) => {
  if (
    (piece.kind !== "hunter" && piece.kind !== "traitor") ||
    !piece.carriesMace
  ) {
    return null;
  }

  const enemyChief = getChief(state, otherPlayer(actingOwner));

  if (!enemyChief) {
    return null;
  }

  return isAdjacent(piece.position, enemyChief.position)
    ? `${PLAYER_LABELS[actingOwner]} slew the enemy Chief with a Mace.`
    : null;
};

const resolvePieceMove = (
  state: GameState,
  pieceId: string,
  target: Position
) => {
  const movingPiece = state.pieces.find((piece) => piece.id === pieceId);

  if (!movingPiece) {
    return state;
  }

  const actingOwner = getPieceController(movingPiece, state);

  if (!actingOwner) {
    return state;
  }

  let nextState = state;
  const notes: string[] = [];

  if (movingPiece.kind === "dragon") {
    const targetPiece = getPieceAt(nextState, target);

    if (
      targetPiece &&
      targetPiece.kind === "hunter" &&
      targetPiece.owner === otherPlayer(actingOwner)
    ) {
      nextState = capturePieces(nextState, [targetPiece.id]);
      notes.push(
        `The Dragon scorched ${
          PLAYER_LABELS[targetPiece.owner]
        } at ${formatSquare(target)}.`
      );
    }
  }

  nextState = {
    ...nextState,
    pieces: nextState.pieces.map((piece) =>
      piece.id === pieceId ? { ...piece, position: target } : piece
    ),
  };

  nextState = syncCarriedMace(nextState, pieceId, target);

  const pickupResult = givePieceGroundMace(nextState, pieceId);
  nextState = pickupResult.state;

  if (pickupResult.pickedUp) {
    const updatedPiece = nextState.pieces.find((piece) => piece.id === pieceId);

    if (updatedPiece) {
      notes.push(`${getPieceLabel(updatedPiece, nextState)} picked up a Mace.`);
    }
  }

  let updatedPiece = nextState.pieces.find((piece) => piece.id === pieceId);

  if (!updatedPiece) {
    return state;
  }

  if (updatedPiece.kind === "chief") {
    const claimResult = applyChiefClaims(nextState, updatedPiece);
    nextState = claimResult.state;
    notes.push(...claimResult.notes);
    updatedPiece =
      nextState.pieces.find((piece) => piece.id === pieceId) ?? updatedPiece;
  }

  const victoryText = checkMaceVictory(nextState, updatedPiece, actingOwner);

  if (victoryText) {
    return {
      ...nextState,
      winner: actingOwner,
      status: victoryText,
    };
  }

  const captureResult = applySandwichCaptures(nextState);
  nextState = captureResult.state;

  if (captureResult.captured.length > 0) {
    notes.push(
      `Sandwich captures removed ${captureResult.captured.length} piece${
        captureResult.captured.length === 1 ? "" : "s"
      }.`
    );
  }

  return {
    ...nextState,
    currentTurn: otherPlayer(actingOwner),
    status: `${getPieceLabel(updatedPiece, nextState)} moved to ${formatSquare(
      target
    )}.${notes.length > 0 ? ` ${notes.join(" ")}` : ""}`,
  };
};

const resolveShipMove = (
  state: GameState,
  shipId: string,
  target: Position
) => {
  const ship = state.ships.find((candidate) => candidate.id === shipId);

  if (!ship) {
    return state;
  }

  let nextState: GameState = {
    ...state,
    ships: state.ships.map((candidate) =>
      candidate.id === shipId ? { ...candidate, position: target } : candidate
    ),
  };

  const movedShip =
    nextState.ships.find((candidate) => candidate.id === shipId) ?? ship;
  const captureResult = applySandwichCaptures(nextState);

  nextState = captureResult.state;

  return {
    ...nextState,
    currentTurn: otherPlayer(ship.owner),
    status: `${getShipLabel(movedShip)} sailed to ${formatSquare(target)}.${
      captureResult.captured.length > 0
        ? ` Sandwich captures removed ${captureResult.captured.length} piece${
            captureResult.captured.length === 1 ? "" : "s"
          }.`
        : ""
    }`,
  };
};

const resolveTraitorAbility = (state: GameState, targetHunterId: string) => {
  const actingOwner = state.currentTurn;
  const targetHunter = state.pieces.find(
    (piece) =>
      piece.id === targetHunterId &&
      piece.kind === "hunter" &&
      piece.owner === otherPlayer(actingOwner)
  );

  if (!targetHunter) {
    return state;
  }

  let nextState: GameState = {
    ...state,
    pieces: state.pieces.map((piece) =>
      piece.id === targetHunterId
        ? {
            id: "traitor-piece",
            kind: "traitor",
            owner: actingOwner,
            position: piece.position,
            carriesMace: piece.carriesMace,
          }
        : piece
    ),
    maces: state.maces.map((mace) =>
      mace.carriedBy === targetHunterId
        ? {
            ...mace,
            carriedBy: "traitor-piece",
            position: targetHunter.position,
          }
        : mace
    ),
    traitorAbilityUsed: {
      ...state.traitorAbilityUsed,
      [actingOwner]: true,
    },
  };

  const captureResult = applySandwichCaptures(nextState);
  nextState = captureResult.state;

  return {
    ...nextState,
    currentTurn: otherPlayer(actingOwner),
    status: `${
      PLAYER_LABELS[actingOwner]
    } activated the Traitor at ${formatSquare(targetHunter.position)}.${
      captureResult.captured.length > 0
        ? ` Sandwich captures removed ${captureResult.captured.length} piece${
            captureResult.captured.length === 1 ? "" : "s"
          }.`
        : ""
    }`,
  };
};

function App() {
  const [gameState, setGameState] = useState<GameState>(() =>
    createInitialGameState()
  );
  const [selection, setSelection] = useState<Selection | null>(null);

  const selectedPiece =
    selection?.type === "piece"
      ? gameState.pieces.find((piece) => piece.id === selection.id) ?? null
      : null;

  const selectedShip =
    selection?.type === "ship"
      ? gameState.ships.find((ship) => ship.id === selection.id) ?? null
      : null;

  const pieceTargets = useMemo(
    () => (selectedPiece ? getPieceMoves(selectedPiece, gameState) : []),
    [gameState, selectedPiece]
  );

  const shipTargets = useMemo(
    () => (selectedShip ? getShipMoves(selectedShip, gameState) : []),
    [gameState, selectedShip]
  );

  const traitorTargets = useMemo(
    () =>
      selection?.type === "traitorAbility"
        ? gameState.pieces.filter(
            (piece) =>
              piece.kind === "hunter" &&
              piece.owner === otherPlayer(gameState.currentTurn)
          )
        : [],
    [gameState, selection]
  );

  const pieceTargetKeys = useMemo(
    () => new Set(pieceTargets.map(toPositionKey)),
    [pieceTargets]
  );

  const shipTargetKeys = useMemo(
    () => new Set(shipTargets.map(toPositionKey)),
    [shipTargets]
  );

  const shipFootprintKeys = useMemo(() => {
    if (!selectedShip) {
      return new Set<string>();
    }

    const keys = new Set<string>();

    shipTargets.forEach((target) => {
      getShipFootprint(selectedShip.kind, target).forEach((cell) => {
        keys.add(toPositionKey(cell));
      });
    });

    return keys;
  }, [selectedShip, shipTargets]);

  const traitorTargetMap = useMemo(
    () =>
      new Map(
        traitorTargets.map((piece) => [toPositionKey(piece.position), piece.id])
      ),
    [traitorTargets]
  );
  const traitorAvailable =
    gameState.traitorClaimedBy === gameState.currentTurn &&
    !gameState.traitorAbilityUsed[gameState.currentTurn] &&
    gameState.pieces.some(
      (piece) =>
        piece.kind === "hunter" &&
        piece.owner === otherPlayer(gameState.currentTurn)
    );

  const dragonPiece = gameState.pieces.find((piece) => piece.kind === "dragon");

  const selectionHint = (() => {
    if (gameState.winner) {
      return `${
        PLAYER_LABELS[gameState.winner]
      } won the match. Reset to play the setup again.`;
    }

    if (selection?.type === "traitorAbility") {
      return "Choose one enemy Hunter to replace with the Traitor.";
    }

    if (selectedShip?.kind === "longship") {
      return "Click a highlighted water square to move the Longship.";
    }

    if (selectedShip?.kind === "chiefship") {
      return "Click a highlighted water square to move the Chiefship.";
    }

    if (selectedPiece) {
      return `Click a highlighted square to move the ${getPieceRoleLabel(
        selectedPiece
      ).toLowerCase()}.`;
    }

    if (traitorAvailable) {
      return "You can move a piece, move a ship, or activate the Traitor this turn.";
    }

    return "Select one of your pieces or ships to take a turn.";
  })();

  const playerStats = (player: Player) => {
    const hunters = gameState.pieces.filter(
      (piece) => piece.kind === "hunter" && piece.owner === player
    ).length;
    const maceBearers = gameState.pieces.filter(
      (piece) => piece.owner === player && piece.carriesMace
    ).length;
    const hasTraitorPiece = gameState.pieces.some(
      (piece) => piece.kind === "traitor" && piece.owner === player
    );

    return {
      hunters,
      maceBearers,
      hasTraitorPiece,
    };
  };

  const leftStats = playerStats("left");
  const rightStats = playerStats("right");

  const handleReset = () => {
    setGameState(createInitialGameState());
    setSelection(null);
  };

  const handleTraitorToggle = () => {
    if (!traitorAvailable) {
      return;
    }

    setSelection((current) =>
      current?.type === "traitorAbility" ? null : { type: "traitorAbility" }
    );
  };

  const handleSquareClick = (row: number, col: number) => {
    if (gameState.winner) {
      return;
    }

    const position = { row, col };
    const positionKey = toPositionKey(position);

    if (selection?.type === "traitorAbility") {
      const targetHunterId = traitorTargetMap.get(positionKey);

      if (targetHunterId) {
        setGameState((current) =>
          resolveTraitorAbility(current, targetHunterId)
        );
        setSelection(null);
        return;
      }
    }

    if (selection?.type === "piece" && pieceTargetKeys.has(positionKey)) {
      setGameState((current) =>
        resolvePieceMove(current, selection.id, position)
      );
      setSelection(null);
      return;
    }

    if (selection?.type === "ship" && shipTargetKeys.has(positionKey)) {
      setGameState((current) =>
        resolveShipMove(current, selection.id, position)
      );
      setSelection(null);
      return;
    }

    const clickedPiece = getPieceAt(gameState, position);
    const clickedShip = clickedPiece ? null : getShipAt(gameState, position);

    if (clickedPiece && isSelectablePiece(clickedPiece, gameState)) {
      setSelection((current) =>
        current?.type === "piece" && current.id === clickedPiece.id
          ? null
          : { type: "piece", id: clickedPiece.id }
      );
      return;
    }

    if (
      !clickedPiece &&
      clickedShip &&
      isSelectableShip(clickedShip, gameState)
    ) {
      setSelection((current) =>
        current?.type === "ship" && current.id === clickedShip.id
          ? null
          : { type: "ship", id: clickedShip.id }
      );
      return;
    }

    setSelection(null);
  };

  return (
    <Box
      minH="100vh"
      bg="linear-gradient(180deg, #efe3c1 0%, #d8c39a 45%, #8b6a44 100%)"
      color="#1d1a16"
      px={{ base: 4, md: 8 }}
      py={{ base: 6, md: 10 }}
      css={{
        "--panel-bg": "rgba(244, 236, 213, 0.9)",
        "--panel-border": "rgba(86, 60, 32, 0.28)",
        "--land-light": "#d8ba78",
        "--land-dark": "#b89154",
        "--water-light": "#5d8494",
        "--water-dark": "#2d5565",
        "--highlight": "#f5dd63",
        "--ink-soft": "#4e3b22",
      }}
    >
      <VStack gap={6} maxW="1200px" mx="auto" align="stretch">
        <VStack
          gap={4}
          align="stretch"
          bg="var(--panel-bg)"
          border="1px solid"
          borderColor="var(--panel-border)"
          borderRadius="28px"
          p={{ base: 5, md: 7 }}
          boxShadow="0 22px 50px rgba(58, 34, 11, 0.16)"
          backdropFilter="blur(8px)"
        >
          <HStack justify="space-between" align="start" flexWrap="wrap" gap={4}>
            <VStack gap={3} align="start" maxW="760px">
              <Text
                textTransform="uppercase"
                letterSpacing="0.28em"
                fontSize="xs"
                color="#705633"
                fontWeight="700"
              >
                Maces & Talons
              </Text>
              <Heading
                as="h1"
                fontSize={{ base: "3xl", md: "5xl" }}
                lineHeight="1"
                fontFamily="'Palatino Linotype', 'Book Antiqua', serif"
              >
                Opening rules prototype
              </Heading>
              <Text
                maxW="760px"
                fontSize={{ base: "sm", md: "md" }}
                color="var(--ink-soft)"
              >
                The full opening setup is on the board now: Chiefs, Hunters,
                ships, Maces, the Dragon, and the Traitor. Win by moving a
                Mace-bearer adjacent to the enemy Chief.
              </Text>
            </VStack>

            <VStack
              gap={2}
              align="stretch"
              minW={{ base: "100%", md: "260px" }}
              bg={PLAYER_SURFACES[gameState.currentTurn]}
              border="1px solid rgba(86, 60, 32, 0.18)"
              borderRadius="20px"
              p={4}
            >
              <Text
                fontSize="xs"
                textTransform="uppercase"
                letterSpacing="0.18em"
                color="#705633"
              >
                Turn
              </Text>
              <Text
                fontSize="xl"
                fontWeight="800"
                color={PLAYER_COLORS[gameState.currentTurn]}
              >
                {gameState.winner
                  ? `${PLAYER_LABELS[gameState.winner]} Won`
                  : PLAYER_LABELS[gameState.currentTurn]}
              </Text>
              <Text fontSize="sm" color="var(--ink-soft)">
                {gameState.status}
              </Text>
            </VStack>
          </HStack>

          <HStack gap={4} flexWrap="wrap" fontSize="sm" color="var(--ink-soft)">
            <HStack gap={2}>
              <Box boxSize="14px" borderRadius="4px" bg="var(--land-dark)" />
              <Text>Land</Text>
            </HStack>
            <HStack gap={2}>
              <Box boxSize="14px" borderRadius="4px" bg="var(--water-dark)" />
              <Text>Water</Text>
            </HStack>
            <HStack gap={2}>
              <Box
                boxSize="14px"
                borderRadius="4px"
                bg="rgba(143, 45, 24, 0.42)"
              />
              <Text>Ship squares</Text>
            </HStack>
            <HStack gap={2}>
              <Box
                boxSize="14px"
                borderRadius="999px"
                bg="#f1ca5e"
                border="1px solid rgba(68, 44, 8, 0.45)"
              />
              <Text>Mace</Text>
            </HStack>
          </HStack>

          <HStack gap={3} flexWrap="wrap">
            <Button
              onClick={handleTraitorToggle}
              disabled={!traitorAvailable || Boolean(gameState.winner)}
              bg={selection?.type === "traitorAbility" ? "#f0b66a" : "#eedbb3"}
              color="#3b2814"
              border="1px solid rgba(86, 60, 32, 0.2)"
              _hover={{
                bg:
                  selection?.type === "traitorAbility" ? "#ecaa54" : "#e8d0a0",
              }}
            >
              {selection?.type === "traitorAbility"
                ? "Cancel Traitor"
                : "Activate Traitor"}
            </Button>
            <Button
              onClick={handleReset}
              bg="#3f2f21"
              color="#f7ecd7"
              _hover={{ bg: "#2f2217" }}
            >
              Reset Match
            </Button>
            <Text fontSize="sm" color="var(--ink-soft)">
              {selectionHint}
            </Text>
          </HStack>
        </VStack>
        <Box
          bg="rgba(58, 34, 11, 0.92)"
          borderRadius="30px"
          p={{ base: 3, md: 5 }}
          boxShadow="0 28px 65px rgba(42, 24, 8, 0.35)"
          overflowX="auto"
        >
          <Box minW="fit-content" mx="auto">
            <Grid
              templateColumns={`36px repeat(${BOARD_SIZE}, 42px)`}
              gap={2}
              alignItems="center"
            >
              <Box />
              {COLUMN_LABELS.map((label) => (
                <Text
                  key={label}
                  textAlign="center"
                  fontSize="sm"
                  fontWeight="700"
                  color="#f1dfb8"
                >
                  {label}
                </Text>
              ))}

              {terrainMap.map((terrainRow, rowIndex) => (
                <Box key={`row-${rowIndex}`} display="contents">
                  <Text
                    textAlign="center"
                    fontSize="sm"
                    fontWeight="700"
                    color="#f1dfb8"
                  >
                    {rowIndex + 1}
                  </Text>

                  {terrainRow.map((terrain, colIndex) => {
                    const position = { row: rowIndex, col: colIndex };
                    const key = toPositionKey(position);
                    const piece = getPieceAt(gameState, position);
                    const ship = getShipAt(gameState, position);
                    const mace = getGroundMaceAt(gameState, position);
                    const hasTraitor = hasTraitorToken(gameState, position);
                    const isSelectedPiece =
                      selection?.type === "piece" && piece?.id === selection.id;
                    const isSelectedShip =
                      selection?.type === "ship" && ship?.id === selection.id;
                    const isPieceTarget = pieceTargetKeys.has(key);
                    const isShipTarget = shipTargetKeys.has(key);
                    const isShipFootprint = shipFootprintKeys.has(key);
                    const isTraitorTarget = traitorTargetMap.has(key);
                    const clickable =
                      !gameState.winner &&
                      (isPieceTarget ||
                        isShipTarget ||
                        isTraitorTarget ||
                        (piece ? isSelectablePiece(piece, gameState) : false) ||
                        (!piece && ship
                          ? isSelectableShip(ship, gameState)
                          : false));

                    const pieceOwner = piece
                      ? getPieceController(piece, gameState)
                      : null;
                    const pieceColor =
                      piece?.kind === "dragon"
                        ? pieceOwner
                          ? PLAYER_COLORS[pieceOwner]
                          : "#6b572a"
                        : pieceOwner
                        ? PLAYER_COLORS[pieceOwner]
                        : "#5c4a32";

                    const cellBorderColor =
                      isSelectedPiece || isSelectedShip
                        ? "var(--highlight)"
                        : isTraitorTarget
                        ? "#f0b66a"
                        : isShipTarget
                        ? "#f5dd63"
                        : isPieceTarget
                        ? "rgba(245, 221, 99, 0.85)"
                        : "rgba(255, 244, 219, 0.12)";

                    return (
                      <Box
                        key={key}
                        as="button"
                        onClick={() => handleSquareClick(rowIndex, colIndex)}
                        aria-label={`Square ${COLUMN_LABELS[colIndex]}${
                          rowIndex + 1
                        }, ${terrain}`}
                        position="relative"
                        display="flex"
                        alignItems="center"
                        justifyContent="center"
                        boxSize="42px"
                        borderRadius="14px"
                        border="2px solid"
                        borderColor={cellBorderColor}
                        bg={
                          terrain === "water"
                            ? "linear-gradient(145deg, var(--water-light), var(--water-dark))"
                            : "linear-gradient(145deg, var(--land-light), var(--land-dark))"
                        }
                        cursor={clickable ? "pointer" : "default"}
                        transition="transform 120ms ease, border-color 120ms ease, box-shadow 120ms ease"
                        boxShadow={
                          isSelectedPiece || isSelectedShip
                            ? "0 0 0 3px rgba(245, 221, 99, 0.18), inset 0 0 0 1px rgba(255, 251, 227, 0.3)"
                            : isTraitorTarget
                            ? "0 0 0 3px rgba(240, 182, 106, 0.22)"
                            : isPieceTarget || isShipTarget
                            ? "0 0 0 3px rgba(245, 221, 99, 0.14)"
                            : "inset 0 1px 0 rgba(255, 245, 218, 0.2)"
                        }
                        _hover={
                          clickable
                            ? {
                                transform: "translateY(-1px)",
                                boxShadow: "0 8px 16px rgba(20, 16, 10, 0.2)",
                              }
                            : undefined
                        }
                        _focusVisible={{
                          outline: "3px solid #fff4cf",
                          outlineOffset: "2px",
                        }}
                      >
                        {ship ? (
                          <Box
                            position="absolute"
                            inset="4px"
                            borderRadius={
                              ship.kind === "chiefship" ? "999px" : "10px"
                            }
                            bg={
                              ship.kind === "chiefship"
                                ? `${PLAYER_COLORS[ship.owner]}cc`
                                : `${PLAYER_COLORS[ship.owner]}88`
                            }
                            border="1px solid rgba(247, 239, 219, 0.7)"
                          />
                        ) : null}

                        {isShipFootprint && !ship ? (
                          <Box
                            position="absolute"
                            inset="5px"
                            borderRadius="10px"
                            bg="rgba(245, 221, 99, 0.18)"
                          />
                        ) : null}

                        {isPieceTarget && !piece && !ship ? (
                          <Box
                            position="absolute"
                            boxSize="11px"
                            borderRadius="999px"
                            bg="rgba(255, 247, 196, 0.92)"
                          />
                        ) : null}

                        {isShipTarget ? (
                          <Box
                            position="absolute"
                            boxSize="14px"
                            rotate="45deg"
                            bg="rgba(255, 247, 196, 0.9)"
                            borderRadius="3px"
                          />
                        ) : null}

                        {ship ? (
                          <Text
                            position="absolute"
                            insetInlineStart="5px"
                            insetBlockEnd="3px"
                            fontSize="9px"
                            fontWeight="900"
                            color="#fff6df"
                            zIndex={1}
                          >
                            {ship.kind === "longship" ? "LS" : "CS"}
                          </Text>
                        ) : null}

                        {mace ? (
                          <Box
                            position="absolute"
                            insetInlineEnd="3px"
                            insetBlockEnd="3px"
                            boxSize="13px"
                            borderRadius="4px"
                            bg="#f1ca5e"
                            color="#4e3208"
                            border="1px solid rgba(68, 44, 8, 0.45)"
                            display="grid"
                            placeItems="center"
                            fontSize="9px"
                            fontWeight="900"
                            zIndex={2}
                          >
                            M
                          </Box>
                        ) : null}

                        {hasTraitor ? (
                          <Box
                            position="relative"
                            zIndex={3}
                            display="grid"
                            placeItems="center"
                            boxSize="28px"
                            borderRadius="999px"
                            bg="#2f2418"
                            color="#f3e4c0"
                            border="2px solid #d8bd82"
                            fontWeight="900"
                            fontSize="lg"
                          >
                            ?
                          </Box>
                        ) : null}

                        {piece ? (
                          <Box
                            position="relative"
                            zIndex={4}
                            display="grid"
                            placeItems="center"
                            boxSize={piece.kind === "chief" ? "31px" : "29px"}
                            borderRadius={
                              piece.kind === "chief" ? "10px" : "999px"
                            }
                            bg={pieceColor}
                            color="#fff7e7"
                            fontWeight="900"
                            fontSize="lg"
                            border={
                              piece.kind === "dragon"
                                ? "2px solid #f1ca5e"
                                : "2px solid #f7efdb"
                            }
                            boxShadow="0 10px 18px rgba(50, 17, 8, 0.35)"
                          >
                            {getPieceMarker(piece)}
                            {piece.carriesMace ? (
                              <Box
                                position="absolute"
                                insetInlineEnd="-4px"
                                insetBlockStart="-4px"
                                boxSize="14px"
                                borderRadius="4px"
                                bg="#f1ca5e"
                                color="#4e3208"
                                border="1px solid rgba(68, 44, 8, 0.45)"
                                display="grid"
                                placeItems="center"
                                fontSize="9px"
                                fontWeight="900"
                              >
                                M
                              </Box>
                            ) : null}
                          </Box>
                        ) : null}
                      </Box>
                    );
                  })}
                </Box>
              ))}
            </Grid>
          </Box>
        </Box>

        <Stack
          direction={{ base: "column", lg: "row" }}
          gap={4}
          align="stretch"
        >
          <VStack
            flex={1}
            align="stretch"
            gap={3}
            bg="var(--panel-bg)"
            border="1px solid"
            borderColor="var(--panel-border)"
            borderRadius="24px"
            p={5}
          >
            <Text
              fontSize="xs"
              textTransform="uppercase"
              letterSpacing="0.18em"
              color="#705633"
            >
              Top Fleet
            </Text>
            <Text fontSize="2xl" fontWeight="800" color={PLAYER_COLORS.left}>
              Hunters: {leftStats.hunters}
            </Text>
            <Text fontSize="sm" color="var(--ink-soft)">
              Mace bearers: {leftStats.maceBearers}
            </Text>
            <Text fontSize="sm" color="var(--ink-soft)">
              Dragon control:{" "}
              {gameState.dragonController === "left" ? "Yes" : "No"}
            </Text>
            <Text fontSize="sm" color="var(--ink-soft)">
              Traitor:{" "}
              {leftStats.hasTraitorPiece
                ? "On the board"
                : gameState.traitorClaimedBy === "left"
                ? gameState.traitorAbilityUsed.left
                  ? "Claimed and spent"
                  : "Claimed and ready"
                : "Unclaimed"}
            </Text>
          </VStack>

          <VStack
            flex={1.4}
            align="stretch"
            gap={3}
            bg="var(--panel-bg)"
            border="1px solid"
            borderColor="var(--panel-border)"
            borderRadius="24px"
            p={5}
          >
            <Text
              fontSize="xs"
              textTransform="uppercase"
              letterSpacing="0.18em"
              color="#705633"
            >
              Board Notes
            </Text>
            <Text fontSize="sm" color="var(--ink-soft)">
              Hunters and the Traitor move like rooks, Chiefs move one square in
              orthogonally, and the Dragon moves up to three squares
              horizontally or vertically.
            </Text>
            <Text fontSize="sm" color="var(--ink-soft)">
              Longships and Chiefships move only on water, and every ship now
              occupies one square. Hunters may cross water only on friendly
              Longships, while Chiefs may stand on water only by using their own
              Chiefship.
            </Text>
            <Text fontSize="sm" color="var(--ink-soft)">
              The Dragon starts on{" "}
              {dragonPiece ? formatSquare(dragonPiece.position) : "the board"}.
              The Traitor begins at{" "}
              {gameState.traitorTokenPosition
                ? formatSquare(gameState.traitorTokenPosition)
                : "a claimed position"}
              .
            </Text>
            <Text fontSize="sm" color="var(--ink-soft)">
              Sandwich captures remove Hunters and the Traitor in continuous
              horizontal or vertical lines between two enemy-controlled squares.
            </Text>
          </VStack>

          <VStack
            flex={1}
            align="stretch"
            gap={3}
            bg="var(--panel-bg)"
            border="1px solid"
            borderColor="var(--panel-border)"
            borderRadius="24px"
            p={5}
          >
            <Text
              fontSize="xs"
              textTransform="uppercase"
              letterSpacing="0.18em"
              color="#705633"
            >
              Bottom Fleet
            </Text>
            <Text fontSize="2xl" fontWeight="800" color={PLAYER_COLORS.right}>
              Hunters: {rightStats.hunters}
            </Text>
            <Text fontSize="sm" color="var(--ink-soft)">
              Mace bearers: {rightStats.maceBearers}
            </Text>
            <Text fontSize="sm" color="var(--ink-soft)">
              Dragon control:{" "}
              {gameState.dragonController === "right" ? "Yes" : "No"}
            </Text>
            <Text fontSize="sm" color="var(--ink-soft)">
              Traitor:{" "}
              {rightStats.hasTraitorPiece
                ? "On the board"
                : gameState.traitorClaimedBy === "right"
                ? gameState.traitorAbilityUsed.right
                  ? "Claimed and spent"
                  : "Claimed and ready"
                : "Unclaimed"}
            </Text>
          </VStack>
        </Stack>
      </VStack>
    </Box>
  );
}

export default App;
