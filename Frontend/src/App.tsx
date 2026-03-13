import { Box, Button, HStack, Stack, Text, VStack } from "@chakra-ui/react";
import { useEffect, useMemo, useRef, useState } from "react";
import { BotSetupScreen } from "./components/BotSetupScreen.tsx";
import { CaptureExamples } from "./components/CaptureExamples.tsx";
import { GameBoard } from "./components/GameBoard.tsx";
import { LandingMenu } from "./components/LandingMenu.tsx";
import { MultiplayerWaitingRoom } from "./components/MultiplayerWaitingRoom.tsx";
import { RulesSummaryCard } from "./components/RulesSummaryCard.tsx";
import { TraitorPanel } from "./components/TraitorPanel.tsx";
import {
  resolvePieceMove,
  resolveShipMove,
  resolveTraitorAbility,
} from "./game/actions.ts";
import { PLAYER_LABELS } from "./game/constants.ts";
import { getPieceMoves, getShipMoves } from "./game/moves.ts";
import {
  getPieceAt,
  getPieceRoleLabel,
  getShipAt,
  getShipFootprint,
  isSelectablePiece,
  isSelectableShip,
} from "./game/selectors.ts";
import { createInitialGameState } from "./game/setup.ts";
import type { GameState, Player, Selection } from "./game/types.ts";
import { formatSquare, otherPlayer, toPositionKey } from "./game/utils.ts";
import {
  buildJoinUrl,
  clearSeatToken,
  createBotRoom,
  createRoom,
  getRoomWebSocketUrl,
  getStoredSeatToken,
  joinRoom,
  readRoomIdFromUrl,
  storeSeatToken,
  writeRoomIdToUrl,
  type BotDifficulty,
  type ClientMessage,
  type RoomSnapshot,
  type ServerMessage,
} from "./multiplayer.ts";

type AppMode = "solo" | "bot" | "multiplayer";
type Screen = "menu" | "botSetup" | "multiplayerLobby" | "game";
type ConnectionState =
  | "idle"
  | "creating"
  | "joining"
  | "connecting"
  | "connected"
  | "reconnecting"
  | "error";
type PendingClientMessage = Exclude<ClientMessage, { type: "auth" }>;

const BOT_DIFFICULTY_LABELS: Record<BotDifficulty, string> = {
  easy: "Easy",
  medium: "Medium",
  hard: "Hard",
};

const BOT_DIFFICULTY_OPTIONS: Array<{
  value: BotDifficulty;
  label: string;
  description: string;
}> = [
  {
    value: "easy",
    label: "Easy",
    description:
      "Shallow lookahead with more variety. Good for learning the rules and openings.",
  },
  {
    value: "medium",
    label: "Medium",
    description:
      "Shallow minimax with tighter move selection, so it feels steadier than easy without long waits.",
  },
  {
    value: "hard",
    label: "Hard",
    description:
      "Two-ply minimax that looks ahead to your reply and plays much more deliberately.",
  },
];

const isOnlineMode = (mode: AppMode) => mode !== "solo";

const describeSeatStatus = (room: RoomSnapshot | null, player: Player) => {
  if (!room) {
    return "open";
  }

  if (room.bot?.seat === player) {
    return `${BOT_DIFFICULTY_LABELS[room.bot.difficulty]} bot`;
  }

  const seat = room.seats[player];

  if (!seat.claimed) {
    return "open";
  }

  return seat.connected ? "claimed and connected" : "claimed";
};

const getSelectionHint = ({
  gameState,
  selection,
  selectedPieceLabel,
  selectedShipKind,
  traitorAvailable,
}: {
  gameState: GameState;
  selection: Selection | null;
  selectedPieceLabel: string | null;
  selectedShipKind: "longship" | "chiefship" | null;
  traitorAvailable: boolean;
}) => {
  if (gameState.winner) {
    return `${
      PLAYER_LABELS[gameState.winner]
    } won the match. Reset to play the setup again.`;
  }

  if (selection?.type === "traitorAbility") {
    return "Choose one enemy Hunter to replace with the Traitor.";
  }

  if (selectedShipKind === "longship") {
    return "Click a highlighted water square to move the Longship.";
  }

  if (selectedShipKind === "chiefship") {
    return "Click a highlighted water square to move the Chiefship.";
  }

  if (selectedPieceLabel) {
    return `Click a highlighted square to move the ${selectedPieceLabel.toLowerCase()}.`;
  }

  if (traitorAvailable) {
    return "You can move a piece, move a ship, or activate the Traitor this turn.";
  }

  return "Select one of your pieces or ships to take a turn.";
};

const describeConnection = (connectionState: ConnectionState) => {
  switch (connectionState) {
    case "creating":
      return "Creating room";
    case "joining":
      return "Joining room";
    case "connecting":
      return "Connecting";
    case "connected":
      return "Connected";
    case "reconnecting":
      return "Reconnecting";
    case "error":
      return "Needs attention";
    default:
      return "Idle";
  }
};

const getTraitorPanelCopy = (gameState: GameState) => {
  const deployedTraitor = gameState.pieces.find(
    (piece) => piece.kind === "traitor" && piece.owner
  );

  if (deployedTraitor?.owner) {
    return {
      status: "Traitor Deployed",
      description: `${PLAYER_LABELS[deployedTraitor.owner]} already has the Traitor on the board.`,
      instruction:
        "The Traitor moves like a Hunter, can carry a mace, and still threatens the enemy Chief.",
    };
  }

  if (!gameState.traitorClaimedBy && gameState.traitorTokenPosition) {
    return {
      status: "Token Unclaimed",
      description: `Move a Chief adjacent to the Traitor token at ${formatSquare(
        gameState.traitorTokenPosition
      )} to secure it.`,
      instruction:
        "After claiming it, that side may replace one enemy Hunter with the Traitor on a later turn.",
    };
  }

  if (
    gameState.traitorClaimedBy &&
    !gameState.traitorAbilityUsed[gameState.traitorClaimedBy]
  ) {
    return {
      status: `${PLAYER_LABELS[gameState.traitorClaimedBy]} Ready`,
      description: `${PLAYER_LABELS[gameState.traitorClaimedBy]} controls the token and can turn one enemy Hunter into the Traitor.`,
      instruction:
        gameState.currentTurn === gameState.traitorClaimedBy
          ? "Use the button, then click an enemy Hunter to perform the conversion."
          : `The ability is waiting for ${PLAYER_LABELS[gameState.traitorClaimedBy]}'s turn.`,
    };
  }

  if (gameState.traitorClaimedBy) {
    return {
      status: "Ability Spent",
      description: `${PLAYER_LABELS[gameState.traitorClaimedBy]} already used the once-per-side Traitor conversion.`,
      instruction:
        "The token is exhausted, so the rest of the match is about piece pressure, Dragon control, and mace routes.",
    };
  }

  return {
    status: "Watching The Field",
    description: "The Traitor remains neutral until a Chief claims the token.",
    instruction:
      "Chief positioning is the trigger. Move next to the token first, then plan the conversion turn.",
  };
};

function App() {
  const initialRoomId = readRoomIdFromUrl();
  const [mode, setMode] = useState<AppMode>(() =>
    initialRoomId ? "multiplayer" : "solo"
  );
  const [screen, setScreen] = useState<Screen>(() =>
    initialRoomId ? "multiplayerLobby" : "menu"
  );
  const [botDifficulty, setBotDifficulty] = useState<BotDifficulty>("medium");
  const [soloGameState, setSoloGameState] = useState<GameState>(() =>
    createInitialGameState()
  );
  const [selection, setSelection] = useState<Selection | null>(null);
  const [roomId, setRoomId] = useState<string | null>(() => initialRoomId);
  const [seatToken, setSeatToken] = useState<string | null>(() =>
    initialRoomId ? getStoredSeatToken(initialRoomId) : null
  );
  const [multiplayerRoom, setMultiplayerRoom] = useState<RoomSnapshot | null>(
    null
  );
  const [multiplayerSeat, setMultiplayerSeat] = useState<Player | null>(null);
  const [connectionState, setConnectionState] =
    useState<ConnectionState>(() => {
      if (!initialRoomId) {
        return "idle";
      }

      return getStoredSeatToken(initialRoomId) ? "connecting" : "idle";
    });
  const [multiplayerError, setMultiplayerError] = useState<string | null>(null);
  const [multiplayerNotice, setMultiplayerNotice] = useState<string | null>(
    null
  );
  const [optimisticGameState, setOptimisticGameState] =
    useState<GameState | null>(null);

  const socketRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<number | null>(null);
  const authenticatedRef = useRef(false);
  const reconnectAllowedRef = useRef(true);

  const roomBot = multiplayerRoom?.bot ?? null;
  const activeBotDifficulty = roomBot?.difficulty ?? botDifficulty;
  const botDifficultyDescription =
    BOT_DIFFICULTY_OPTIONS.find((option) => option.value === activeBotDifficulty)
      ?.description ?? "";

  const gameState =
    mode === "solo"
      ? soloGameState
      : optimisticGameState ?? multiplayerRoom?.gameState ?? null;

  const canInteract =
    gameState !== null &&
    !gameState.winner &&
    (mode === "solo"
      ? true
      : multiplayerRoom?.started === true &&
        multiplayerSeat === gameState.currentTurn &&
        connectionState === "connected");

  const activeSelection = useMemo(() => {
    if (!gameState || !canInteract || !selection) {
      return null;
    }

    if (
      selection.type === "piece" &&
      !gameState.pieces.some((piece) => piece.id === selection.id)
    ) {
      return null;
    }

    if (
      selection.type === "ship" &&
      !gameState.ships.some((ship) => ship.id === selection.id)
    ) {
      return null;
    }

    return selection;
  }, [canInteract, gameState, selection]);

  const selectedPiece =
    activeSelection?.type === "piece" && gameState
      ? gameState.pieces.find((piece) => piece.id === activeSelection.id) ?? null
      : null;

  const selectedShip =
    activeSelection?.type === "ship" && gameState
      ? gameState.ships.find((ship) => ship.id === activeSelection.id) ?? null
      : null;

  const pieceTargets = useMemo(
    () => (selectedPiece && gameState ? getPieceMoves(selectedPiece, gameState) : []),
    [gameState, selectedPiece]
  );

  const shipTargets = useMemo(
    () => (selectedShip && gameState ? getShipMoves(selectedShip, gameState) : []),
    [gameState, selectedShip]
  );

  const traitorTargets = useMemo(
    () =>
      activeSelection?.type === "traitorAbility" && gameState
        ? gameState.pieces.filter(
            (piece) =>
              piece.kind === "hunter" &&
              piece.owner === otherPlayer(gameState.currentTurn)
          )
        : [],
    [activeSelection, gameState]
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

  const traitorTargetKeys = useMemo(
    () => new Set(traitorTargetMap.keys()),
    [traitorTargetMap]
  );

  const traitorAvailable =
    gameState !== null &&
    gameState.traitorClaimedBy === gameState.currentTurn &&
    !gameState.traitorAbilityUsed[gameState.currentTurn] &&
    gameState.pieces.some(
      (piece) =>
        piece.kind === "hunter" &&
        piece.owner === otherPlayer(gameState.currentTurn)
    );

  const selectionHint = useMemo(() => {
    if (!gameState) {
      if (mode === "bot") {
        return "Choose a difficulty and start a backend minimax match. You play Vikings and the bot controls Marauders.";
      }

      return roomId
        ? "Open the room, claim a seat, and the board will sync from the backend."
        : "Choose solo for hotseat play, challenge the bot, or create a multiplayer room.";
    }

    if (mode !== "solo") {
      if (!multiplayerRoom?.started) {
        return mode === "bot"
          ? "The bot room will unlock as soon as the backend connection is ready."
          : "The match stays locked until both seats are filled.";
      }

      if (connectionState !== "connected") {
        return "The board is temporarily locked while the client reconnects to the room.";
      }

      if (
        multiplayerSeat &&
        multiplayerSeat !== gameState.currentTurn &&
        !gameState.winner
      ) {
        if (mode === "bot") {
          return `The ${BOT_DIFFICULTY_LABELS[activeBotDifficulty].toLowerCase()} bot is choosing a move.`;
        }

        return `Waiting for ${PLAYER_LABELS[gameState.currentTurn]} to move.`;
      }
    }

    return getSelectionHint({
      gameState,
      selection: activeSelection,
      selectedPieceLabel: selectedPiece ? getPieceRoleLabel(selectedPiece) : null,
      selectedShipKind: selectedShip?.kind ?? null,
      traitorAvailable,
    });
  }, [
    activeBotDifficulty,
    activeSelection,
    connectionState,
    gameState,
    mode,
    multiplayerRoom?.started,
    multiplayerSeat,
    roomId,
    selectedPiece,
    selectedShip,
    traitorAvailable,
  ]);

  const joinLink = roomId && !roomBot ? buildJoinUrl(roomId) : null;
  const modeLabel =
    mode === "solo"
      ? "Solo hotseat"
      : mode === "bot"
      ? `${BOT_DIFFICULTY_LABELS[activeBotDifficulty]} bot match`
      : roomId
      ? `Multiplayer room ${roomId.toUpperCase()}`
      : "Multiplayer room";
  const turnLabel = gameState
    ? gameState.winner
      ? `${PLAYER_LABELS[gameState.winner]} victory`
      : `${PLAYER_LABELS[gameState.currentTurn]} to move`
    : "Waiting for board";
  const traitorPanelCopy = gameState ? getTraitorPanelCopy(gameState) : null;
  const onlineConnectionLabel =
    mode === "solo" ? null : describeConnection(connectionState);
  const activeScreen: Screen =
    mode === "multiplayer"
      ? multiplayerRoom?.started
        ? "game"
        : "multiplayerLobby"
      : mode === "bot" && multiplayerRoom
      ? "game"
      : screen;

  const syncRoomSnapshot = (room: RoomSnapshot) => {
    setOptimisticGameState(null);
    setMultiplayerRoom(room);
    setMode((current) =>
      current === "solo" ? current : room.bot ? "bot" : "multiplayer"
    );

    if (room.bot) {
      setBotDifficulty(room.bot.difficulty);
    }
  };

  const clearNetworkSession = () => {
    if (roomId) {
      clearSeatToken(roomId);
    }

    writeRoomIdToUrl(null);
    setRoomId(null);
    setSeatToken(null);

    setMultiplayerRoom(null);
    setMultiplayerSeat(null);
    setOptimisticGameState(null);
    setConnectionState("idle");
    setMultiplayerError(null);
    setMultiplayerNotice(null);
  };

  useEffect(() => {
    if (!isOnlineMode(mode) || !roomId || !seatToken) {
      authenticatedRef.current = false;
      reconnectAllowedRef.current = false;


      if (reconnectTimerRef.current !== null) {
        window.clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }

      if (socketRef.current) {
        const socket = socketRef.current;
        socketRef.current = null;
        socket.close();
      }

      return;
    }

    let cancelled = false;
    authenticatedRef.current = false;
    reconnectAllowedRef.current = true;

    const connect = () => {
      if (cancelled) {
        return;
      }

      setConnectionState((current) =>
        current === "connected" ? "reconnecting" : "connecting"
      );

      const socket = new WebSocket(getRoomWebSocketUrl(roomId));
      socketRef.current = socket;

      socket.addEventListener("open", () => {
        socket.send(JSON.stringify({ type: "auth", seatToken } satisfies ClientMessage));
      });

      socket.addEventListener("message", (event) => {
        const message = JSON.parse(event.data) as ServerMessage;

        if (message.type === "auth_ok") {
          authenticatedRef.current = true;
          syncRoomSnapshot(message.room);
          setMultiplayerSeat(message.seat);
          setConnectionState("connected");
          setMultiplayerError(null);
          setMultiplayerNotice(null);
          return;
        }

        if (message.type === "room_state") {
          syncRoomSnapshot(message.room);

          if (authenticatedRef.current) {
            setConnectionState("connected");
          }

          return;
        }

        setMultiplayerError(message.message);
        setMultiplayerNotice(null);
        setOptimisticGameState(null);

        if (!authenticatedRef.current) {
          reconnectAllowedRef.current = false;
          setConnectionState("error");
          clearSeatToken(roomId);
          setSeatToken(null);
          setMultiplayerSeat(null);
        }
      });

      socket.addEventListener("close", () => {
        if (socketRef.current === socket) {
          socketRef.current = null;
        }


        if (cancelled || !reconnectAllowedRef.current) {
          return;
        }

        setConnectionState((current) =>
          current === "error" ? current : "reconnecting"
        );
        reconnectTimerRef.current = window.setTimeout(connect, 1500);
      });
    };

    connect();

    return () => {
      cancelled = true;
      reconnectAllowedRef.current = false;

      if (reconnectTimerRef.current !== null) {
        window.clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }

      if (socketRef.current) {
        const socket = socketRef.current;
        socketRef.current = null;
        socket.close();
      }
    };
  }, [mode, roomId, seatToken]);

  const sendMultiplayerMessage = (message: PendingClientMessage) => {
    const socket = socketRef.current;

    if (!socket || socket.readyState !== WebSocket.OPEN) {
      setMultiplayerError("The room connection is not ready yet. Please wait a moment.");
      setMultiplayerNotice(null);
      setConnectionState("reconnecting");
      return false;
    }

    try {
      socket.send(JSON.stringify(message));
      return true;
    } catch {
      setMultiplayerError("The room connection dropped before the move could be sent.");
      setMultiplayerNotice(null);
      setConnectionState("reconnecting");
      return false;
    }
  };

  const submitOnlineAction = (
    nextState: GameState,
    message: PendingClientMessage
  ) => {
    if (!sendMultiplayerMessage(message)) {
      setOptimisticGameState(null);
      return false;
    }

    setOptimisticGameState(nextState);
    setMultiplayerError(null);
    setMultiplayerNotice(null);
    return true;
  };

  const handleResetSoloMatch = () => {
    setSoloGameState(createInitialGameState());
    setSelection(null);
  };

  const handleSelectBotDifficulty = (difficulty: BotDifficulty) => {
    setBotDifficulty(difficulty);
  };

  const handleCopyJoinLink = async () => {
    if (!joinLink || !navigator.clipboard) {
      return;
    }

    try {
      await navigator.clipboard.writeText(joinLink);
      setMultiplayerNotice("Share link copied to the clipboard.");
      setMultiplayerError(null);
    } catch {
      setMultiplayerError("Could not copy the join link automatically.");
      setMultiplayerNotice(null);
    }
  };

  const handleReturnToMenu = () => {
    clearNetworkSession();
    setMode("solo");
    setScreen("menu");
    setSelection(null);
  };

  const handleStartSolo = () => {
    clearNetworkSession();
    setMode("solo");
    setSoloGameState(createInitialGameState());
    setScreen("game");
    setSelection(null);
  };

  const handleOpenBotSetup = () => {
    clearNetworkSession();
    setMode("bot");
    setScreen("botSetup");
    setSelection(null);
  };

  const handleCreateRoom = async () => {
    if (roomId) {
      clearSeatToken(roomId);
    }

    setOptimisticGameState(null);
    setMultiplayerRoom(null);
    setMultiplayerSeat(null);
    setMode("multiplayer");
    setScreen("multiplayerLobby");

    setConnectionState("creating");
    setMultiplayerError(null);
    setMultiplayerNotice(null);

    try {
      const session = await createRoom();
      const nextRoomId = session.room.roomId;

      storeSeatToken(nextRoomId, session.seatToken);
      writeRoomIdToUrl(nextRoomId);
      setRoomId(nextRoomId);
      setSeatToken(session.seatToken);
      setMultiplayerRoom(session.room);
      setMultiplayerSeat(session.seat);
      setConnectionState("connecting");
      setSelection(null);
    } catch (error) {
      setConnectionState("error");
      setMultiplayerError(
        error instanceof Error ? error.message : "Could not create a room."
      );
      setMultiplayerNotice(null);
    }
  };

  const handleCreateBotRoom = async () => {
    if (roomId) {
      clearSeatToken(roomId);
    }

    setOptimisticGameState(null);
    setMultiplayerRoom(null);
    setMultiplayerSeat(null);
    setMode("bot");
    setScreen("botSetup");

    setConnectionState("creating");
    setMultiplayerError(null);
    setMultiplayerNotice(null);

    try {
      const session = await createBotRoom(botDifficulty);
      const nextRoomId = session.room.roomId;

      storeSeatToken(nextRoomId, session.seatToken);
      writeRoomIdToUrl(nextRoomId);
      setRoomId(nextRoomId);
      setSeatToken(session.seatToken);
      setMultiplayerRoom(session.room);
      setMultiplayerSeat(session.seat);
      setConnectionState("connecting");
      setSelection(null);
    } catch (error) {
      setConnectionState("error");
      setMultiplayerError(
        error instanceof Error ? error.message : "Could not create a bot match."
      );
      setMultiplayerNotice(null);
    }
  };

  const handleJoinRoom = async () => {
    if (!roomId) {
      return;
    }

    setOptimisticGameState(null);
    setMode("multiplayer");

    setConnectionState("joining");
    setMultiplayerError(null);
    setMultiplayerNotice(null);

    try {
      const session = await joinRoom(roomId);

      storeSeatToken(roomId, session.seatToken);
      setSeatToken(session.seatToken);
      setMultiplayerRoom(session.room);
      setMultiplayerSeat(session.seat);
      setConnectionState("connecting");
      setSelection(null);
    } catch (error) {
      setConnectionState("error");
      setMultiplayerError(
        error instanceof Error ? error.message : "Could not join that room."
      );
      setMultiplayerNotice(null);
    }
  };

  const handleStartMultiplayer = () => {
    clearNetworkSession();
    setMode("multiplayer");
    setScreen("multiplayerLobby");
    setSelection(null);
    void handleCreateRoom();
  };

  const handleTraitorToggle = () => {
    if (!gameState || !traitorAvailable || !canInteract) {
      return;
    }

    setSelection((current) =>
      current?.type === "traitorAbility" ? null : { type: "traitorAbility" }
    );
  };

  const handleSquareClick = (row: number, col: number) => {
    if (!gameState || !canInteract || gameState.winner) {
      return;
    }

    const position = { row, col };
    const positionKey = toPositionKey(position);

    if (activeSelection?.type === "traitorAbility") {
      const targetHunterId = traitorTargetMap.get(positionKey);

      if (targetHunterId) {
        if (mode === "solo") {
          setSoloGameState((current) =>
            resolveTraitorAbility(current, targetHunterId)
          );
          setSelection(null);
          return;
        }

        const nextState = resolveTraitorAbility(gameState, targetHunterId);

        if (submitOnlineAction(nextState, { type: "use_traitor", targetHunterId })) {
          setSelection(null);
        }

        return;
      }
    }

    if (activeSelection?.type === "piece" && pieceTargetKeys.has(positionKey)) {
      if (mode === "solo") {
        setSoloGameState((current) =>
          resolvePieceMove(current, activeSelection.id, position)
        );
        setSelection(null);
        return;
      }

      const nextState = resolvePieceMove(gameState, activeSelection.id, position);

      if (
        submitOnlineAction(nextState, {
          type: "move_piece",
          pieceId: activeSelection.id,
          target: position,
        })
      ) {
        setSelection(null);
      }

      return;
    }

    if (activeSelection?.type === "ship" && shipTargetKeys.has(positionKey)) {
      if (mode === "solo") {
        setSoloGameState((current) =>
          resolveShipMove(current, activeSelection.id, position)
        );
        setSelection(null);
        return;
      }

      const nextState = resolveShipMove(gameState, activeSelection.id, position);

      if (
        submitOnlineAction(nextState, {
          type: "move_ship",
          shipId: activeSelection.id,
          target: position,
        })
      ) {
        setSelection(null);
      }

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

    if (!clickedPiece && clickedShip && isSelectableShip(clickedShip, gameState)) {
      setSelection((current) =>
        current?.type === "ship" && current.id === clickedShip.id
          ? null
          : { type: "ship", id: clickedShip.id }
      );
      return;
    }

    setSelection(null);
  };

  const isBotSetupBusy =
    connectionState === "creating" ||
    connectionState === "connecting" ||
    connectionState === "joining";
  const isMultiplayerLobbyBusy =
    connectionState === "creating" ||
    connectionState === "joining" ||
    connectionState === "connecting";

  const renderScreen = () => {
    if (activeScreen === "menu") {
      return (
        <LandingMenu
          onPlaySolo={handleStartSolo}
          onPlayBot={handleOpenBotSetup}
          onPlayMultiplayer={handleStartMultiplayer}
        />
      );
    }

    if (activeScreen === "botSetup") {
      return (
        <BotSetupScreen
          difficulty={botDifficulty}
          options={BOT_DIFFICULTY_OPTIONS}
          busy={isBotSetupBusy}
          error={multiplayerError}
          onSelectDifficulty={handleSelectBotDifficulty}
          onStart={handleCreateBotRoom}
          onBack={handleReturnToMenu}
        />
      );
    }

    if (activeScreen === "multiplayerLobby") {
      return (
        <MultiplayerWaitingRoom
          roomId={roomId}
          joinLink={joinLink}
          seatLabel={multiplayerSeat ? PLAYER_LABELS[multiplayerSeat] : null}
          connectionLabel={describeConnection(connectionState)}
          roomReady={multiplayerRoom?.started ?? false}
          vikingsSeatStatus={describeSeatStatus(multiplayerRoom, "vikings")}
          maraudersSeatStatus={describeSeatStatus(multiplayerRoom, "marauders")}
          canJoinRoom={Boolean(roomId && !seatToken)}
          busy={isMultiplayerLobbyBusy}
          error={multiplayerError}
          notice={multiplayerNotice}
          onCopyLink={handleCopyJoinLink}
          onJoinRoom={handleJoinRoom}
          onCreateRoom={() => {
            void handleCreateRoom();
          }}
          onBackToMenu={handleReturnToMenu}
        />
      );
    }

    if (!gameState || !traitorPanelCopy) {
      return (
        <VStack
          maxW="900px"
          mx="auto"
          align="stretch"
          gap={4}
          bg="var(--panel-bg)"
          border="1px solid"
          borderColor="var(--panel-border)"
          borderRadius="28px"
          p={{ base: 6, md: 7 }}
        >
          <Text
            fontSize="xs"
            textTransform="uppercase"
            letterSpacing="0.18em"
            color="var(--text-muted)"
            fontWeight="700"
          >
            Loading Match
          </Text>
          <Text color="var(--text-muted)">
            Preparing the board and reconnecting to the room.
          </Text>
        </VStack>
      );
    }

    return (
      <VStack maxW="1320px" mx="auto" align="stretch" gap={6}>
        <Stack
          direction={{ base: "column", lg: "row" }}
          justify="space-between"
          align={{ base: "start", lg: "center" }}
          gap={4}
        >
          <VStack align="start" gap={1}>
            <Text
              fontSize="xs"
              textTransform="uppercase"
              letterSpacing="0.24em"
              color="var(--text-muted)"
              fontWeight="700"
            >
              Maces & Talons
            </Text>
            <Text fontSize="lg" color="var(--text-muted)">
              {modeLabel}
              {mode === "bot" ? ` - ${botDifficultyDescription}` : ""}
            </Text>
          </VStack>

          <HStack gap={3} flexWrap="wrap">
            {mode === "solo" ? (
              <Button
                onClick={handleResetSoloMatch}
                bg="var(--button-ghost)"
                color="#f5efe1"
                border="1px solid rgba(219, 189, 131, 0.2)"
                _hover={{ bg: "var(--button-ghost-hover)" }}
              >
                Reset Match
              </Button>
            ) : null}
            {mode === "bot" ? (
              <Button
                onClick={handleOpenBotSetup}
                bg="var(--button-ghost)"
                color="#f5efe1"
                border="1px solid rgba(219, 189, 131, 0.2)"
                _hover={{ bg: "var(--button-ghost-hover)" }}
              >
                New Bot Match
              </Button>
            ) : null}
            {mode === "multiplayer" && joinLink ? (
              <Button
                onClick={handleCopyJoinLink}
                bg="var(--button-ghost)"
                color="#f5efe1"
                border="1px solid rgba(219, 189, 131, 0.2)"
                _hover={{ bg: "var(--button-ghost-hover)" }}
              >
                Copy Share Link
              </Button>
            ) : null}
            <Button
              onClick={handleReturnToMenu}
              bg="#dbc08b"
              color="#18120d"
              fontWeight="800"
              _hover={{ bg: "#e7cc97" }}
            >
              Main Menu
            </Button>
          </HStack>
        </Stack>

        <RulesSummaryCard />

        <Stack direction={{ base: "column", xl: "row" }} gap={5} align="stretch">
          <GameBoard
            gameState={gameState}
            selection={activeSelection}
            pieceTargetKeys={pieceTargetKeys}
            shipTargetKeys={shipTargetKeys}
            shipFootprintKeys={shipFootprintKeys}
            traitorTargetKeys={traitorTargetKeys}
            canInteract={canInteract}
            onSquareClick={handleSquareClick}
          />
          <TraitorPanel
            modeLabel={modeLabel}
            turnLabel={turnLabel}
            connectionLabel={onlineConnectionLabel}
            matchStatus={gameState.status}
            selectionHint={selectionHint}
            traitorStatus={traitorPanelCopy.status}
            traitorDescription={traitorPanelCopy.description}
            traitorInstruction={traitorPanelCopy.instruction}
            traitorButtonLabel={
              activeSelection?.type === "traitorAbility"
                ? "Cancel Traitor"
                : "Activate Traitor"
            }
            traitorButtonDisabled={
              !traitorAvailable || !canInteract || Boolean(gameState.winner)
            }
            traitorButtonActive={activeSelection?.type === "traitorAbility"}
            onTraitorToggle={handleTraitorToggle}
          />
        </Stack>

        <CaptureExamples />

        {multiplayerNotice ? (
          <Text color="#d8f2d0" fontSize="sm">
            {multiplayerNotice}
          </Text>
        ) : null}

        {multiplayerError ? (
          <Text fontSize="sm" color="#ffb39a">
            {multiplayerError}
          </Text>
        ) : null}
      </VStack>
    );
  };

  return (
    <Box
      minH="100vh"
      bg="linear-gradient(180deg, #4a311d 0%, #1d140d 38%, #5b4329 100%)"
      color="#f5efe1"
      px={{ base: 4, md: 8 }}
      py={{ base: 6, md: 10 }}
      position="relative"
      overflow="hidden"
      css={{
        "--panel-bg": "rgba(42, 28, 17, 0.84)",
        "--panel-strong": "linear-gradient(180deg, rgba(72, 51, 31, 0.96) 0%, rgba(24, 15, 10, 0.98) 100%)",
        "--panel-soft": "rgba(244, 232, 208, 0.05)",
        "--button-ghost": "rgba(95, 70, 47, 0.76)",
        "--button-ghost-hover": "rgba(120, 89, 60, 0.86)",
        "--panel-border": "rgba(219, 189, 131, 0.18)",
        "--land-light": "#b79258",
        "--land-dark": "#876137",
        "--water-light": "#6a7f88",
        "--water-dark": "#394d57",
        "--highlight": "#e4c46d",
        "--ink-soft": "rgba(241, 230, 206, 0.8)",
        "--text-muted": "rgba(241, 230, 206, 0.72)",
      }}
    >
      <Box
        position="absolute"
        top="-120px"
        left="-80px"
        w="380px"
        h="380px"
        borderRadius="999px"
        bg="rgba(168, 122, 69, 0.18)"
        filter="blur(80px)"
        pointerEvents="none"
      />
      <Box
        position="absolute"
        right="-60px"
        bottom="-140px"
        w="420px"
        h="420px"
        borderRadius="999px"
        bg="rgba(92, 110, 121, 0.14)"
        filter="blur(100px)"
        pointerEvents="none"
      />
      <Box position="relative" zIndex={1}>
        {renderScreen()}
      </Box>
    </Box>
  );
}

export default App;

