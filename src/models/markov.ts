import type { GameRecord, ModelMove } from './naiveBayes';

const moves: ModelMove[] = ['rock', 'paper', 'scissors'];
const counters: Record<ModelMove, ModelMove> = {
  rock: 'paper',
  paper: 'scissors',
  scissors: 'rock',
};

export function chooseMarkovMove(games: GameRecord[]): ModelMove {
  if (games.length < 2) return moves[Math.floor(Math.random() * moves.length)];

  const previousMove = games[games.length - 1].playerMove;
  const transitionCounts = { rock: 1, paper: 1, scissors: 1 } as Record<ModelMove, number>;
  games.forEach((game) => {
    if (game.previousMove === previousMove) transitionCounts[game.playerMove] += 1;
  });

  const predictedPlayerMove = moves.reduce((mostLikely, move) =>
    transitionCounts[move] > transitionCounts[mostLikely] ? move : mostLikely, 'rock');
  return counters[predictedPlayerMove];
}
