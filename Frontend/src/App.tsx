import { Box, Grid, Heading, HStack, Text, VStack } from "@chakra-ui/react";
import { useMemo, useState } from "react";

const BOARD_SIZE = 13;
const COLUMN_LABELS = "ABCDEFGHIJKLM".split("");
const TERRAIN_ROWS = [
  "WWLLLLLLLLLWW",
  "WWLLLLLLLLLWW",
  "WWLLLLLLWLLWW",
  "WWWLLLLWWWWWW",
  "LWWWWWWWWWWWL",
  "LLWWLLLLLWWLL",
  "LLWWLLLLLWWLL",
  "LLWWLLLLLWWLL",
  "LWWWWWWWWWWWL",
  "WWWWWWLLLLWWW",
  "WWLLWLLLLLLWW",
  "WWLLLLLLLLLWW",
  "WWLLLLLLLLLWW",
] as const;

type Terrain = "land" | "water";

type Position = {
  row: number;
  col: number;
};

const terrainMap: Terrain[][] = TERRAIN_ROWS.map((row) =>
  row.split("").map((cell) => (cell === "W" ? "water" : "land"))
);

const positionsMatch = (first: Position, second: Position) =>
  first.row === second.row && first.col === second.col;

const isInBounds = (row: number, col: number) =>
  row >= 0 && row < BOARD_SIZE && col >= 0 && col < BOARD_SIZE;

const getValidMoves = (piece: Position): Position[] => {
  const validMoves: Position[] = [];

  for (let rowOffset = -1; rowOffset <= 1; rowOffset += 1) {
    for (let colOffset = -1; colOffset <= 1; colOffset += 1) {
      if (rowOffset === 0 && colOffset === 0) {
        continue;
      }

      const targetRow = piece.row + rowOffset;
      const targetCol = piece.col + colOffset;

      if (!isInBounds(targetRow, targetCol)) {
        continue;
      }

      if (terrainMap[targetRow][targetCol] === "land") {
        validMoves.push({ row: targetRow, col: targetCol });
      }
    }
  }

  return validMoves;
};

function App() {
  const [piecePosition, setPiecePosition] = useState<Position>({
    row: 6,
    col: 5,
  });
  const [selected, setSelected] = useState(false);

  const validMoves = useMemo(
    () => (selected ? getValidMoves(piecePosition) : []),
    [piecePosition, selected]
  );

  const handleSquareClick = (row: number, col: number) => {
    const clickedPosition = { row, col };
    const clickedPiece = positionsMatch(clickedPosition, piecePosition);

    if (clickedPiece) {
      setSelected((current) => !current);
      return;
    }

    const isValidMove = validMoves.some((move) =>
      positionsMatch(move, clickedPosition)
    );

    if (isValidMove) {
      setPiecePosition(clickedPosition);
      setSelected(false);
      return;
    }

    setSelected(false);
  };

  return (
    <Box
      minH="100vh"
      bg="linear-gradient(180deg, #efe3c1 0%, #d8c39a 45%, #8b6a44 100%)"
      color="#1d1a16"
      px={{ base: 4, md: 8 }}
      py={{ base: 6, md: 10 }}
      css={{
        "--panel-bg": "rgba(244, 236, 213, 0.86)",
        "--panel-border": "rgba(86, 60, 32, 0.3)",
        "--land-light": "#d8ba78",
        "--land-dark": "#b89154",
        "--water-light": "#4f7c8d",
        "--water-dark": "#2d5565",
        "--highlight": "#f5dd63",
        "--piece": "#8f2d18",
      }}
    >
      <VStack gap={6} maxW="1100px" mx="auto" align="stretch">
        <VStack
          gap={3}
          align="start"
          bg="var(--panel-bg)"
          border="1px solid"
          borderColor="var(--panel-border)"
          borderRadius="28px"
          p={{ base: 5, md: 7 }}
          boxShadow="0 22px 50px rgba(58, 34, 11, 0.16)"
          backdropFilter="blur(8px)"
        >
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
            Chess-inspired movement demo
          </Heading>
          <Text
            maxW="720px"
            fontSize={{ base: "sm", md: "md" }}
            color="#4e3b22"
          >
            Click the scout to reveal valid moves. It can step exactly one
            square in any direction, but only onto land.
          </Text>
          <HStack gap={4} wrap="wrap" fontSize="sm" color="#4e3b22">
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
                borderRadius="999px"
                bg="var(--piece)"
                border="2px solid #f7efdb"
              />
              <Text>Scout</Text>
            </HStack>
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
              templateColumns={`52px repeat(${BOARD_SIZE}, minmax(0, 1fr))`}
              gap={2}
              alignItems="center"
            >
              <Box />
              {COLUMN_LABELS.map((label) => (
                <Text
                  key={label}
                  textAlign="center"
                  fontSize={{ base: "xs", md: "sm" }}
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
                    fontSize={{ base: "xs", md: "sm" }}
                    fontWeight="700"
                    color="#f1dfb8"
                  >
                    {rowIndex + 1}
                  </Text>

                  {terrainRow.map((terrain, colIndex) => {
                    const tilePosition = { row: rowIndex, col: colIndex };
                    const hasPiece = positionsMatch(
                      tilePosition,
                      piecePosition
                    );
                    const isValidMove = validMoves.some((move) =>
                      positionsMatch(move, tilePosition)
                    );
                    const isSelectedPiece = selected && hasPiece;
                    const isWater = terrain === "water";

                    return (
                      <Box
                        key={`${rowIndex}-${colIndex}`}
                        as="button"
                        onClick={() => handleSquareClick(rowIndex, colIndex)}
                        aria-label={`Square ${COLUMN_LABELS[colIndex]}${
                          rowIndex + 1
                        }, ${terrain}`}
                        position="relative"
                        display="flex"
                        alignItems="center"
                        justifyContent="center"
                        boxSize={{ base: "42px", md: "52px" }}
                        borderRadius="14px"
                        border="2px solid"
                        borderColor={
                          isSelectedPiece
                            ? "var(--highlight)"
                            : isValidMove
                            ? "rgba(245, 221, 99, 0.9)"
                            : "rgba(255, 244, 219, 0.12)"
                        }
                        bg={
                          isWater
                            ? "linear-gradient(145deg, var(--water-light), var(--water-dark))"
                            : "linear-gradient(145deg, var(--land-light), var(--land-dark))"
                        }
                        cursor={
                          isWater && !hasPiece ? "not-allowed" : "pointer"
                        }
                        transition="transform 120ms ease, border-color 120ms ease, box-shadow 120ms ease"
                        boxShadow={
                          isValidMove
                            ? "inset 0 0 0 2px rgba(255, 251, 227, 0.3), 0 0 0 3px rgba(245, 221, 99, 0.18)"
                            : "inset 0 1px 0 rgba(255, 245, 218, 0.22)"
                        }
                        _hover={{
                          transform: "translateY(-1px)",
                          boxShadow: "0 8px 16px rgba(20, 16, 10, 0.2)",
                        }}
                        _focusVisible={{
                          outline: "3px solid #fff4cf",
                          outlineOffset: "2px",
                        }}
                      >
                        {isValidMove && !hasPiece ? (
                          <Box
                            position="absolute"
                            boxSize="12px"
                            borderRadius="999px"
                            bg="rgba(255, 247, 196, 0.9)"
                          />
                        ) : null}

                        {hasPiece ? (
                          <Box
                            position="relative"
                            zIndex={1}
                            display="grid"
                            placeItems="center"
                            boxSize={{ base: "28px", md: "34px" }}
                            borderRadius="999px"
                            bg="var(--piece)"
                            color="#fff7e7"
                            fontWeight="900"
                            fontSize={{ base: "lg", md: "xl" }}
                            border="2px solid #f7efdb"
                            boxShadow="0 10px 18px rgba(50, 17, 8, 0.35)"
                          >
                            S
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
      </VStack>
    </Box>
  );
}

export default App;
