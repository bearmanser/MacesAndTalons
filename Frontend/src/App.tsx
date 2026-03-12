import {
  Box,
  Button,
  Heading,
  HStack,
  Stack,
  Text,
  VStack,
} from "@chakra-ui/react";
import { type ChangeEvent, useEffect, useMemo, useRef, useState } from "react";
import { BoardNotesCard } from "./components/BoardNotesCard.tsx";
import { GameBoard } from "./components/GameBoard.tsx";
import { PlayerSummaryCard } from "./components/PlayerSummaryCard.tsx";
import {
  resolvePieceMove,
  resolveShipMove,
  resolveTraitorAbility,
} from "./game/actions.ts";
import {
  PLAYER_COLORS,
  PLAYER_LABELS,
  PLAYER_SURFACES,
} from "./game/constants.ts";
import { getPieceMoves, getShipMoves } from "./game/moves.ts";
import {
  getPieceAt,
  getPieceRoleLabel,
  getPlayerStats,
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
type ConnectionState =
  | "idle"
  | "creating"
  | "joining"
  | "connecting"
  | "connected"
  | "reconnecting"
  | "error";

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
    description: "Shallow lookahead with more variety. Good for learning the rules and openings.",
  },
  {
    value: "medium",
    label: "Medium",
    description: "Shallow minimax with tighter move selection, so it feels steadier than easy without long waits.",
  },
  {
    value: "hard",
    label: "Hard",
    description: "Two-ply minimax that looks ahead to your reply and plays much more deliberately.",
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

const getTraitorStatus = (
  player: Player,
  gameState: GameState,
  hasTraitorPiece: boolean
) => {
  if (hasTraitorPiece) {
    return "On the board";
  }

  if (gameState.traitorClaimedBy !== player) {
    return "Unclaimed";
  }

  return gameState.traitorAbilityUsed[player]
    ? "Claimed and spent"
    : "Claimed and ready";
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

function App() {
  const [mode, setMode] = useState<AppMode>(() =>
    readRoomIdFromUrl() ? "multiplayer" : "solo"
  );
  const [botDifficulty, setBotDifficulty] = useState<BotDifficulty>("medium");
  const [soloGameState, setSoloGameState] = useState<GameState>(() =>
    createInitialGameState()
  );
  const [selection, setSelection] = useState<Selection | null>(null);
  const [roomId, setRoomId] = useState<string | null>(() => readRoomIdFromUrl());
  const [seatToken, setSeatToken] = useState<string | null>(() => {
    const initialRoomId = readRoomIdFromUrl();
    return initialRoomId ? getStoredSeatToken(initialRoomId) : null;
  });
  const [multiplayerRoom, setMultiplayerRoom] = useState<RoomSnapshot | null>(
    null
  );
  const [multiplayerSeat, setMultiplayerSeat] = useState<Player | null>(null);
  const [connectionState, setConnectionState] =
    useState<ConnectionState>(() => {
      if (!readRoomIdFromUrl()) {
        return "idle";
      }

      return getStoredSeatToken(readRoomIdFromUrl()!) ? "connecting" : "idle";
    });
  const [multiplayerError, setMultiplayerError] = useState<string | null>(null);

  const socketRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<number | null>(null);
  const authenticatedRef = useRef(false);
  const reconnectAllowedRef = useRef(true);

  const roomBot = multiplayerRoom?.bot ?? null;
  const activeBotDifficulty = roomBot?.difficulty ?? botDifficulty;
  const botDifficultyDescription =
    BOT_DIFFICULTY_OPTIONS.find((option) => option.value === activeBotDifficulty)
      ?.description ?? "";

  const gameState = mode === "solo" ? soloGameState : multiplayerRoom?.gameState ?? null;

  const selectedPiece =
    selection?.type === "piece" && gameState
      ? gameState.pieces.find((piece) => piece.id === selection.id) ?? null
      : null;

  const selectedShip =
    selection?.type === "ship" && gameState
      ? gameState.ships.find((ship) => ship.id === selection.id) ?? null
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
      selection?.type === "traitorAbility" && gameState
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

  const canInteract =
    gameState !== null &&
    !gameState.winner &&
    (mode === "solo"
      ? true
      : multiplayerRoom?.started === true &&
        multiplayerSeat === gameState.currentTurn &&
        connectionState === "connected");

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

      if (multiplayerSeat && multiplayerSeat !== gameState.currentTurn && !gameState.winner) {
        if (mode === "bot") {
          return `The ${BOT_DIFFICULTY_LABELS[activeBotDifficulty].toLowerCase()} bot is choosing a move.`;
        }

        return `Waiting for ${PLAYER_LABELS[gameState.currentTurn]} to move.`;
      }
    }

    return getSelectionHint({
      gameState,
      selection,
      selectedPieceLabel: selectedPiece ? getPieceRoleLabel(selectedPiece) : null,
      selectedShipKind: selectedShip?.kind ?? null,
      traitorAvailable,
    });
  }, [
    activeBotDifficulty,
    connectionState,
    gameState,
    mode,
    multiplayerRoom?.started,
    multiplayerSeat,
    roomId,
    selectedPiece,
    selectedShip,
    selection,
    traitorAvailable,
  ]);

  const dragonPiece = gameState?.pieces.find((piece) => piece.kind === "dragon") ?? null;
  const marauderStats = gameState ? getPlayerStats(gameState, "marauders") : null;
  const vikingStats = gameState ? getPlayerStats(gameState, "vikings") : null;
  const joinLink = roomId && !roomBot ? buildJoinUrl(roomId) : null;

  const syncRoomSnapshot = (room: RoomSnapshot) => {
    setMultiplayerRoom(room);
    setMode((current) => (current === "solo" ? current : room.bot ? "bot" : "multiplayer"));

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
    setConnectionState("idle");
    setMultiplayerError(null);
  };

  useEffect(() => {
    if (!gameState) {
      setSelection(null);
      return;
    }

    if (!canInteract) {
      setSelection(null);
      return;
    }

    if (
      selection?.type === "piece" &&
      !gameState.pieces.some((piece) => piece.id === selection.id)
    ) {
      setSelection(null);
      return;
    }

    if (
      selection?.type === "ship" &&
      !gameState.ships.some((ship) => ship.id === selection.id)
    ) {
      setSelection(null);
    }
  }, [canInteract, gameState, selection]);

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

  const sendMultiplayerMessage = (
    message: Exclude<ClientMessage, { type: "auth" }>
  ) => {
    const socket = socketRef.current;

    if (!socket || socket.readyState !== WebSocket.OPEN) {
      setMultiplayerError("The room connection is not ready yet. Please wait a moment.");
      setConnectionState("reconnecting");
      return false;
    }

    socket.send(JSON.stringify(message));
    return true;
  };

  const handleReset = () => {
    setSoloGameState(createInitialGameState());
    setSelection(null);
  };

  const handleBotDifficultyChange = (event: ChangeEvent<HTMLSelectElement>) => {
    setBotDifficulty(event.target.value as BotDifficulty);
  };

  const handleCopyJoinLink = async () => {
    if (!joinLink || !navigator.clipboard) {
      return;
    }

    try {
      await navigator.clipboard.writeText(joinLink);
      setMultiplayerError("Join link copied to the clipboard.");
    } catch {
      setMultiplayerError("Could not copy the join link automatically.");
    }
  };

  const handleSwitchToSolo = () => {
    clearNetworkSession();
    setMode("solo");
    setSelection(null);
  };

  const handleSwitchToMultiplayer = () => {
    if (mode !== "multiplayer" || roomBot) {
      clearNetworkSession();
    }

    setMode("multiplayer");
    setSelection(null);
  };

  const handleSwitchToBot = () => {
    if (mode !== "bot" || !roomBot) {
      clearNetworkSession();
    }

    setMode("bot");
    setSelection(null);
  };

  const handleCreateRoom = async () => {
    if (roomId) {
      clearSeatToken(roomId);
    }

    setMode("multiplayer");
    setConnectionState("creating");
    setMultiplayerError(null);

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
    }
  };

  const handleCreateBotRoom = async () => {
    if (roomId) {
      clearSeatToken(roomId);
    }

    setMode("bot");
    setConnectionState("creating");
    setMultiplayerError(null);

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
    }
  };

  const handleJoinRoom = async () => {
    if (!roomId) {
      return;
    }

    setMode("multiplayer");
    setConnectionState("joining");
    setMultiplayerError(null);

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
    }
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

    if (selection?.type === "traitorAbility") {
      const targetHunterId = traitorTargetMap.get(positionKey);

      if (targetHunterId) {
        if (mode === "solo") {
          setSoloGameState((current) =>
            resolveTraitorAbility(current, targetHunterId)
          );
        } else {
          sendMultiplayerMessage({ type: "use_traitor", targetHunterId });
        }

        setSelection(null);
        return;
      }
    }

    if (selection?.type === "piece" && pieceTargetKeys.has(positionKey)) {
      if (mode === "solo") {
        setSoloGameState((current) =>
          resolvePieceMove(current, selection.id, position)
        );
      } else {
        sendMultiplayerMessage({
          type: "move_piece",
          pieceId: selection.id,
          target: position,
        });
      }

      setSelection(null);
      return;
    }

    if (selection?.type === "ship" && shipTargetKeys.has(positionKey)) {
      if (mode === "solo") {
        setSoloGameState((current) => resolveShipMove(current, selection.id, position));
      } else {
        sendMultiplayerMessage({
          type: "move_ship",
          shipId: selection.id,
          target: position,
        });
      }

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
                Play the opening as local hotseat, challenge a backend minimax bot,
                or move the trusted game state to the backend and share a join link
                for a live match. Multiplayer seats reconnect with signed tokens,
                and bot matches run through the same room flow.
              </Text>
            </VStack>

            <VStack
              gap={2}
              align="stretch"
              minW={{ base: "100%", md: "280px" }}
              bg={
                gameState
                  ? PLAYER_SURFACES[gameState.currentTurn]
                  : "rgba(40, 75, 99, 0.12)"
              }
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
                {mode === "solo" ? "Mode" : mode === "bot" ? "Opponent" : "Room"}
              </Text>
              <Text fontSize="xl" fontWeight="800" color="#3f2f21">
                {mode === "solo"
                  ? "Solo Hotseat"
                  : mode === "bot"
                  ? roomBot
                    ? `${BOT_DIFFICULTY_LABELS[roomBot.difficulty]} Bot Match`
                    : "Bot Lobby"
                  : roomId
                  ? `Room ${roomId.toUpperCase()}`
                  : "Multiplayer Lobby"}
              </Text>
              <Text fontSize="sm" color="var(--ink-soft)">
                {gameState
                  ? gameState.winner
                    ? `${PLAYER_LABELS[gameState.winner]} Won`
                    : PLAYER_LABELS[gameState.currentTurn]
                  : mode === "bot"
                  ? "Create a bot match"
                  : "Create or join a room"}
              </Text>
              {mode === "multiplayer" ? (
                <Text fontSize="sm" color="var(--ink-soft)">
                  {describeConnection(connectionState)}
                  {multiplayerSeat ? ` as ${PLAYER_LABELS[multiplayerSeat]}` : ""}
                </Text>
              ) : mode === "bot" ? (
                <Text fontSize="sm" color="var(--ink-soft)">
                  {roomBot
                    ? `${BOT_DIFFICULTY_LABELS[roomBot.difficulty]} minimax bot controls ${PLAYER_LABELS[roomBot.seat]}. ${describeConnection(connectionState)}.`
                    : "Choose a difficulty and start a backend minimax match."}
                </Text>
              ) : (
                <Text fontSize="sm" color="var(--ink-soft)">
                  Move both sides locally on the same board.
                </Text>
              )}
            </VStack>
          </HStack>

          <HStack gap={3} flexWrap="wrap" align="center">
            <Button
              onClick={handleSwitchToSolo}
              bg={mode === "solo" ? "#3f2f21" : "#eedbb3"}
              color={mode === "solo" ? "#f7ecd7" : "#3b2814"}
              border="1px solid rgba(86, 60, 32, 0.2)"
              _hover={{ bg: mode === "solo" ? "#2f2217" : "#e8d0a0" }}
            >
              Play Solo
            </Button>
            <Button
              onClick={handleSwitchToBot}
              bg={mode === "bot" ? "#3f2f21" : "#eedbb3"}
              color={mode === "bot" ? "#f7ecd7" : "#3b2814"}
              border="1px solid rgba(86, 60, 32, 0.2)"
              _hover={{ bg: mode === "bot" ? "#2f2217" : "#e8d0a0" }}
            >
              Play Bot
            </Button>
            <Button
              onClick={handleSwitchToMultiplayer}
              bg={mode === "multiplayer" ? "#3f2f21" : "#eedbb3"}
              color={mode === "multiplayer" ? "#f7ecd7" : "#3b2814"}
              border="1px solid rgba(86, 60, 32, 0.2)"
              _hover={{ bg: mode === "multiplayer" ? "#2f2217" : "#e8d0a0" }}
            >
              Play Multiplayer
            </Button>
            {mode === "bot" ? (
              <select
                value={botDifficulty}
                onChange={handleBotDifficultyChange}
                style={{
                  background: "rgba(255, 248, 232, 0.92)",
                  color: "#3b2814",
                  border: "1px solid rgba(86, 60, 32, 0.2)",
                  borderRadius: "12px",
                  padding: "0.5rem 0.75rem",
                  minWidth: "160px",
                  fontWeight: 700,
                  fontFamily: "inherit",
                }}
              >
                {BOT_DIFFICULTY_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            ) : null}
            {mode === "bot" ? (
              <Button
                onClick={handleCreateBotRoom}
                bg="#8f2d18"
                color="#f8edda"
                _hover={{ bg: "#742515" }}
              >
                {roomId && roomBot ? "New Bot Match" : "Start Bot Match"}
              </Button>
            ) : null}
            {mode === "multiplayer" && !roomId ? (
              <Button
                onClick={handleCreateRoom}
                bg="#8f2d18"
                color="#f8edda"
                _hover={{ bg: "#742515" }}
              >
                Create Game
              </Button>
            ) : null}
            {mode === "multiplayer" && roomId && !seatToken ? (
              <Button
                onClick={handleJoinRoom}
                bg="#284b63"
                color="#f3eadb"
                _hover={{ bg: "#203b4e" }}
              >
                Join Game
              </Button>
            ) : null}
            {mode === "multiplayer" && joinLink ? (
              <Button
                onClick={handleCopyJoinLink}
                bg="#eedbb3"
                color="#3b2814"
                border="1px solid rgba(86, 60, 32, 0.2)"
                _hover={{ bg: "#e8d0a0" }}
              >
                Copy Join Link
              </Button>
            ) : null}
            {mode === "solo" ? (
              <Button
                onClick={handleReset}
                bg="#3f2f21"
                color="#f7ecd7"
                _hover={{ bg: "#2f2217" }}
              >
                Reset Match
              </Button>
            ) : null}
          </HStack>

          {mode === "bot" ? (
            <Stack direction={{ base: "column", lg: "row" }} gap={4}>
              <VStack
                flex={1}
                align="stretch"
                gap={2}
                bg="rgba(255, 248, 232, 0.56)"
                border="1px solid rgba(86, 60, 32, 0.16)"
                borderRadius="20px"
                p={4}
              >
                <Text fontSize="xs" textTransform="uppercase" letterSpacing="0.16em" color="#705633">
                  Bot Flow
                </Text>
                <Text fontSize="sm" color="var(--ink-soft)">
                  {roomBot
                    ? `You play ${multiplayerSeat ? PLAYER_LABELS[multiplayerSeat] : PLAYER_LABELS.vikings}. The backend controls ${PLAYER_LABELS[roomBot.seat]} and calculates replies with minimax search.`
                    : "Start a bot room to play Vikings against a backend-controlled Marauders side."}
                </Text>
                {roomId && roomBot ? (
                  <Text fontSize="sm" color="var(--ink-soft)" fontFamily="mono">
                    Room {roomId.toUpperCase()} stays local to this session for reconnects.
                  </Text>
                ) : null}
              </VStack>
              <VStack
                flex={1}
                align="stretch"
                gap={2}
                bg="rgba(255, 248, 232, 0.56)"
                border="1px solid rgba(86, 60, 32, 0.16)"
                borderRadius="20px"
                p={4}
              >
                <Text fontSize="xs" textTransform="uppercase" letterSpacing="0.16em" color="#705633">
                  Difficulty
                </Text>
                <Text fontSize="sm" color="var(--ink-soft)">
                  {BOT_DIFFICULTY_LABELS[activeBotDifficulty]}: {botDifficultyDescription}
                </Text>
                <Text fontSize="sm" color="var(--ink-soft)">
                  Connection: {describeConnection(connectionState)}
                </Text>
              </VStack>
            </Stack>
          ) : mode === "multiplayer" ? (
            <Stack direction={{ base: "column", lg: "row" }} gap={4}>
              <VStack
                flex={1}
                align="stretch"
                gap={2}
                bg="rgba(255, 248, 232, 0.56)"
                border="1px solid rgba(86, 60, 32, 0.16)"
                borderRadius="20px"
                p={4}
              >
                <Text fontSize="xs" textTransform="uppercase" letterSpacing="0.16em" color="#705633">
                  Room Flow
                </Text>
                <Text fontSize="sm" color="var(--ink-soft)">
                  {roomId
                    ? `Share ${joinLink ?? "the room link"} and the backend will hold the trusted state.`
                    : "Create a room to become Vikings, then share the generated join link with the second player."}
                </Text>
                {roomId ? (
                  <Text fontSize="sm" color="var(--ink-soft)" fontFamily="mono">
                    {joinLink}
                  </Text>
                ) : null}
              </VStack>
              <VStack
                flex={1}
                align="stretch"
                gap={2}
                bg="rgba(255, 248, 232, 0.56)"
                border="1px solid rgba(86, 60, 32, 0.16)"
                borderRadius="20px"
                p={4}
              >
                <Text fontSize="xs" textTransform="uppercase" letterSpacing="0.16em" color="#705633">
                  Seats
                </Text>
                <Text fontSize="sm" color="var(--ink-soft)">
                  Vikings: {describeSeatStatus(multiplayerRoom, "vikings")}
                </Text>
                <Text fontSize="sm" color="var(--ink-soft)">
                  Marauders: {describeSeatStatus(multiplayerRoom, "marauders")}
                </Text>
                <Text fontSize="sm" color="var(--ink-soft)">
                  Game start: {multiplayerRoom?.started ? "ready" : "waiting for both players"}
                </Text>
              </VStack>
            </Stack>
          ) : null}

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

          {gameState ? (
            <HStack gap={3} flexWrap="wrap">
              <Button
                onClick={handleTraitorToggle}
                disabled={!traitorAvailable || !canInteract || Boolean(gameState.winner)}
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
              <Text fontSize="sm" color="var(--ink-soft)">
                {selectionHint}
              </Text>
            </HStack>
          ) : null}

          {multiplayerError ? (
            <Text fontSize="sm" color="#8f2d18">
              {multiplayerError}
            </Text>
          ) : null}
        </VStack>

        {gameState ? (
          <GameBoard
            gameState={gameState}
            selection={selection}
            pieceTargetKeys={pieceTargetKeys}
            shipTargetKeys={shipTargetKeys}
            shipFootprintKeys={shipFootprintKeys}
            traitorTargetKeys={traitorTargetKeys}
            canInteract={canInteract}
            onSquareClick={handleSquareClick}
          />
        ) : (
          <VStack
            align="stretch"
            gap={3}
            bg="rgba(58, 34, 11, 0.92)"
            color="#f3e8cb"
            borderRadius="30px"
            p={{ base: 5, md: 6 }}
            boxShadow="0 28px 65px rgba(42, 24, 8, 0.35)"
          >
            <Heading as="h2" fontSize="2xl" fontFamily="'Palatino Linotype', 'Book Antiqua', serif">
              {mode === "bot" ? "Bot Lobby" : "Multiplayer Lobby"}
            </Heading>
            <Text color="rgba(243, 232, 203, 0.82)">
              {mode === "bot"
                ? "Choose a difficulty and start a backend minimax match. You play Vikings and the bot controls Marauders."
                : "Create a room to seat Vikings, or open a shared join link to claim the other side."}
            </Text>
          </VStack>
        )}

        {gameState && marauderStats && vikingStats ? (
          <Stack direction={{ base: "column", lg: "row" }} gap={4} align="stretch">
            <PlayerSummaryCard
              title="Marauders"
              color={PLAYER_COLORS.marauders}
              hunters={marauderStats.hunters}
              maceBearers={marauderStats.maceBearers}
              dragonControlled={gameState.dragonController === "marauders"}
              traitorStatus={getTraitorStatus(
                "marauders",
                gameState,
                marauderStats.hasTraitorPiece
              )}
            />
            <BoardNotesCard
              dragonSquare={dragonPiece ? formatSquare(dragonPiece.position) : "the board"}
              traitorSquare={
                gameState.traitorTokenPosition
                  ? formatSquare(gameState.traitorTokenPosition)
                  : "a claimed position"
              }
            />
            <PlayerSummaryCard
              title="Vikings"
              color={PLAYER_COLORS.vikings}
              hunters={vikingStats.hunters}
              maceBearers={vikingStats.maceBearers}
              dragonControlled={gameState.dragonController === "vikings"}
              traitorStatus={getTraitorStatus(
                "vikings",
                gameState,
                vikingStats.hasTraitorPiece
              )}
            />
          </Stack>
        ) : null}
      </VStack>
    </Box>
  );
}

export default App;


