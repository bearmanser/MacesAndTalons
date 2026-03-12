import type { GameState, Player } from "./game/types.ts";

export type BotDifficulty = "easy" | "medium" | "hard";

export type SeatPresence = {
  claimed: boolean;
  connected: boolean;
};

export type BotMatchSnapshot = {
  seat: Player;
  difficulty: BotDifficulty;
};

export type RoomSnapshot = {
  roomId: string;
  gameState: GameState;
  seats: Record<Player, SeatPresence>;
  started: boolean;
  playerCount: number;
  bot: BotMatchSnapshot | null;
};

export type RoomSessionResponse = {
  room: RoomSnapshot;
  seat: Player;
  seatToken: string;
};

export type ServerMessage =
  | { type: "auth_ok"; room: RoomSnapshot; seat: Player }
  | { type: "room_state"; room: RoomSnapshot }
  | { type: "error"; message: string };

export type ClientMessage =
  | { type: "auth"; seatToken: string }
  | { type: "move_piece"; pieceId: string; target: { row: number; col: number } }
  | { type: "move_ship"; shipId: string; target: { row: number; col: number } }
  | { type: "use_traitor"; targetHunterId: string };

const apiBaseUrl =
  (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, "") ??
  "http://127.0.0.1:8000";

const toWsBaseUrl = (baseUrl: string) => {
  if (baseUrl.startsWith("https://")) {
    return `wss://${baseUrl.slice("https://".length)}`;
  }

  if (baseUrl.startsWith("http://")) {
    return `ws://${baseUrl.slice("http://".length)}`;
  }

  return baseUrl;
};

const wsBaseUrl = toWsBaseUrl(apiBaseUrl);

const readErrorMessage = async (response: Response) => {
  try {
    const data = (await response.json()) as { detail?: string };
    return data.detail ?? `Request failed with status ${response.status}.`;
  } catch {
    return `Request failed with status ${response.status}.`;
  }
};

export const buildJoinUrl = (roomId: string) => {
  if (typeof window === "undefined") {
    return `?room=${roomId}`;
  }

  const url = new URL(window.location.href);
  url.searchParams.set("room", roomId);
  return url.toString();
};

export const readRoomIdFromUrl = () => {
  if (typeof window === "undefined") {
    return null;
  }

  const roomId = new URL(window.location.href).searchParams.get("room");
  return roomId?.trim() || null;
};

export const writeRoomIdToUrl = (roomId: string | null) => {
  if (typeof window === "undefined") {
    return;
  }

  const url = new URL(window.location.href);

  if (roomId) {
    url.searchParams.set("room", roomId);
  } else {
    url.searchParams.delete("room");
  }

  window.history.replaceState({}, "", url);
};

export const seatTokenStorageKey = (roomId: string) =>
  `maces-and-talons-seat:${roomId}`;

export const getStoredSeatToken = (roomId: string) => {
  if (typeof window === "undefined") {
    return null;
  }

  return window.localStorage.getItem(seatTokenStorageKey(roomId));
};

export const storeSeatToken = (roomId: string, seatToken: string) => {
  if (typeof window === "undefined") {
    return;
  }

  window.localStorage.setItem(seatTokenStorageKey(roomId), seatToken);
};

export const clearSeatToken = (roomId: string) => {
  if (typeof window === "undefined") {
    return;
  }

  window.localStorage.removeItem(seatTokenStorageKey(roomId));
};

export const createRoom = async (): Promise<RoomSessionResponse> => {
  const response = await fetch(`${apiBaseUrl}/api/rooms`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
  });

  if (!response.ok) {
    throw new Error(await readErrorMessage(response));
  }

  return (await response.json()) as RoomSessionResponse;
};

export const createBotRoom = async (
  difficulty: BotDifficulty
): Promise<RoomSessionResponse> => {
  const response = await fetch(`${apiBaseUrl}/api/bot-rooms`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ difficulty }),
  });

  if (!response.ok) {
    throw new Error(await readErrorMessage(response));
  }

  return (await response.json()) as RoomSessionResponse;
};

export const joinRoom = async (roomId: string): Promise<RoomSessionResponse> => {
  const response = await fetch(`${apiBaseUrl}/api/rooms/${roomId}/join`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
  });

  if (!response.ok) {
    throw new Error(await readErrorMessage(response));
  }

  return (await response.json()) as RoomSessionResponse;
};

export const getRoomWebSocketUrl = (roomId: string) =>
  `${wsBaseUrl}/ws/rooms/${roomId}`;
