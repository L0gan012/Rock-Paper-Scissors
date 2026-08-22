export type ModelMove = 'rock' | 'paper' | 'scissors';
export type ModelOutcome = 'win' | 'tie' | 'loss';

export interface GameRecord {
  playerMove: ModelMove;
  outcome: ModelOutcome;
  previousMove?: ModelMove;
  previousOutcome?: ModelOutcome;
  repeated: boolean;
}

const moves: ModelMove[] = ['rock', 'paper', 'scissors'];
const counters: Record<ModelMove, ModelMove> = {
  rock: 'paper',
  paper: 'scissors',
  scissors: 'rock',
};

function categoricalLikelihood<T>(
  games: GameRecord[],
  move: ModelMove,
  feature: (game: GameRecord) => T | undefined,
  value: T,
  possibleValues: number,
): number {
  const gamesWithMove = games.filter((game) => game.playerMove === move);
  const matchingGames = gamesWithMove.filter((game) => feature(game) === value).length;
  return (matchingGames + 1) / (gamesWithMove.length + possibleValues);
}

export function chooseNaiveBayesMove(games: GameRecord[]): ModelMove {
  if (games.length === 0) {
    return moves[Math.floor(Math.random() * moves.length)];
  }

  const latestGame = games[games.length - 1];
  const posterior = (move: ModelMove): number => {
    const prior = (games.filter((game) => game.playerMove === move).length + 1) / (games.length + moves.length);
    const previousMoveLikelihood = categoricalLikelihood(games, move, (game) => game.previousMove, latestGame.playerMove, moves.length);
    const previousOutcomeLikelihood = categoricalLikelihood(games, move, (game) => game.previousOutcome, latestGame.outcome, 3);
    const repeatedLikelihood = categoricalLikelihood(games, move, (game) => game.repeated, move === latestGame.playerMove, 2);
    return prior * previousMoveLikelihood * previousOutcomeLikelihood * repeatedLikelihood;
  };

  const predictedPlayerMove = moves.reduce((mostLikely, move) =>
    posterior(move) > posterior(mostLikely) ? move : mostLikely, 'rock');

  return counters[predictedPlayerMove];
}
