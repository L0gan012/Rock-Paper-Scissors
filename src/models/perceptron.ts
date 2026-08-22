import type { GameRecord, ModelMove } from './naiveBayes';

const moves: ModelMove[] = ['rock', 'paper', 'scissors'];
const counters: Record<ModelMove, ModelMove> = {
  rock: 'paper',
  paper: 'scissors',
  scissors: 'rock',
};

function createFeatures(game: GameRecord | undefined): number[] {
  const features = [1, 0, 0, 0, 0, 0, 0];
  if (!game) return features;

  if (game.previousMove) features[moves.indexOf(game.previousMove) + 1] = 1;
  if (game.previousOutcome) {
    const outcomeIndex = ['win', 'tie', 'loss'].indexOf(game.previousOutcome);
    features[outcomeIndex + 4] = 1;
  }
  return features;
}

function score(features: number[], weights: number[][], classIndex: number): number {
  return features.reduce((total, feature, index) => total + feature * weights[classIndex][index], 0);
}

export function choosePerceptronMove(games: GameRecord[]): ModelMove {
  if (games.length < 2) return moves[Math.floor(Math.random() * moves.length)];

  const weights = moves.map(() => Array(7).fill(0));
  for (let pass = 0; pass < 20; pass += 1) {
    for (let index = 0; index < games.length; index += 1) {
      const features = createFeatures(games[index]);
      const scores = moves.map((_, classIndex) => score(features, weights, classIndex));
      const predicted = scores.indexOf(Math.max(...scores));
      const actual = moves.indexOf(games[index].playerMove);
      if (predicted !== actual) {
        features.forEach((feature, featureIndex) => {
          weights[predicted][featureIndex] -= feature;
          weights[actual][featureIndex] += feature;
        });
      }
    }
  }

  const currentContext = createFeatures(games[games.length - 1]);
  const predictedPlayerMove = moves[moves.map((_, classIndex) => score(currentContext, weights, classIndex))
    .indexOf(Math.max(...moves.map((_, classIndex) => score(currentContext, weights, classIndex))))];
  return counters[predictedPlayerMove];
}
