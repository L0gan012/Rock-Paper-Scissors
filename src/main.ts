import '../style.css';
import { chooseNaiveBayesMove, type GameRecord } from './models/naiveBayes';

type Move = 'rock' | 'paper' | 'scissors';
type Outcome = 'win' | 'tie' | 'loss';
type Model = 'frequency' | 'naive-bayes' | 'perceptron';

const moves: Move[] = ['rock', 'paper', 'scissors'];
const counters: Record<Move, Move> = { rock: 'paper', paper: 'scissors', scissors: 'rock' };
const beats: Record<Move, Move> = { rock: 'scissors', paper: 'rock', scissors: 'paper' };
const imagePaths: Record<Move, string> = {
  rock: '/Images/player-rock-hand.jpg',
  paper: '/Images/player-paper-hand.jpg',
  scissors: '/Images/player-scissors-hand.jpg',
};

const handButtons = document.querySelectorAll<HTMLButtonElement>('.choice-button');
const roundStatus = document.querySelector<HTMLElement>('#round-status');
const roundMessage = document.querySelector<HTMLElement>('#round-message');
const playerDisplay = document.querySelector<HTMLElement>('#player-display');
const computerDisplay = document.querySelector<HTMLElement>('#computer-display');
const roundNumber = document.querySelector<HTMLElement>('#round-number');
const modelSelect = document.querySelector<HTMLSelectElement>('#model-select');
const modelName = document.querySelector<HTMLElement>('#model-name');
const modelDescription = document.querySelector<HTMLElement>('#model-description');
const themeToggle = document.querySelector<HTMLButtonElement>('#theme-toggle');
const themeLabel = document.querySelector<HTMLElement>('#theme-label');
const themeMeta = document.querySelector<HTMLMetaElement>('meta[name="theme-color"]');
const scoreElements: Record<Outcome, HTMLElement | null> = {
  win: document.querySelector('#wins'), tie: document.querySelector('#ties'), loss: document.querySelector('#losses'),
};

const history: Move[] = [];
const games: GameRecord[] = [];
const scores: Record<Outcome, number> = { win: 0, tie: 0, loss: 0 };
let selectedModel: Model = 'frequency';
const modelDescriptions: Record<Model, string> = {
  frequency: 'Counters the move you use most often.',
  'naive-bayes': 'Combines move, outcome, and repeat patterns to estimate your next move.',
  perceptron: 'Trains a lightweight classifier to recognize your sequences.',
};

function setTheme(theme: 'dark' | 'light'): void {
  document.documentElement.dataset.theme = theme;
  const isLight = theme === 'light';
  themeToggle?.setAttribute('aria-pressed', String(isLight));
  if (themeLabel) themeLabel.textContent = isLight ? 'DARK MODE' : 'LIGHT MODE';
  themeMeta?.setAttribute('content', isLight ? '#f4f1ea' : '#121416');
}

const savedTheme = localStorage.getItem('rps-theme');
const preferredTheme = window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
setTheme(savedTheme === 'light' || savedTheme === 'dark' ? savedTheme : preferredTheme);

themeToggle?.addEventListener('click', () => {
  const nextTheme = document.documentElement.dataset.theme === 'light' ? 'dark' : 'light';
  setTheme(nextTheme);
  localStorage.setItem('rps-theme', nextTheme);
});

modelSelect?.addEventListener('change', () => {
  selectedModel = modelSelect.value as Model;
  const label = modelSelect.options[modelSelect.selectedIndex].text.toUpperCase();
  if (modelName) modelName.textContent = label;
  if (modelDescription) modelDescription.textContent = modelDescriptions[selectedModel];
  if (roundStatus) roundStatus.textContent = `${label} READY`;
  if (roundMessage) roundMessage.textContent = 'Model selected - choose a hand to continue';
});

function randomMove(): Move {
  return moves[Math.floor(Math.random() * moves.length)];
}

function mostFrequentMove(): Move {
  const counts = history.reduce<Record<Move, number>>((total, move) => {
    total[move] += 1;
    return total;
  }, { rock: 0, paper: 0, scissors: 0 });
  return moves.reduce((mostCommon, move) => counts[move] > counts[mostCommon] ? move : mostCommon, 'rock');
}

function perceptronMove(): Move {
  if (history.length < 2) return randomMove();
  const weights = Array.from({ length: 4 }, () => [0, 0, 0]);
  const score = (features: number[], classIndex: number) => features.reduce((sum, feature, index) => sum + feature * weights[index][classIndex], 0);
  for (let pass = 0; pass < 12; pass += 1) {
    for (let index = 1; index < history.length; index += 1) {
      const features = [1, 0, 0, 0];
      features[moves.indexOf(history[index - 1]) + 1] = 1;
      const classScores = moves.map((_, classIndex) => score(features, classIndex));
      const predicted = classScores.indexOf(Math.max(...classScores));
      const actual = moves.indexOf(history[index]);
      if (predicted !== actual) features.forEach((feature, featureIndex) => {
        weights[featureIndex][predicted] -= feature;
        weights[featureIndex][actual] += feature;
      });
    }
  }
  const features = [1, 0, 0, 0];
  features[moves.indexOf(history[history.length - 1]) + 1] = 1;
  return moves[ moves.map((_, classIndex) => score(features, classIndex)).indexOf(Math.max(...moves.map((_, classIndex) => score(features, classIndex)))) ];
}

function chooseComputerMove(): Move {
  if (selectedModel === 'naive-bayes') return chooseNaiveBayesMove(games);
  if (history.length === 0) return randomMove();
  const predictedMove = selectedModel === 'perceptron' ? perceptronMove() : mostFrequentMove();
  return counters[predictedMove];
}

function getOutcome(playerMove: Move, computerMove: Move): Outcome {
  if (playerMove === computerMove) return 'tie';
  return beats[playerMove] === computerMove ? 'win' : 'loss';
}

function showMove(element: HTMLElement | null, move: Move): void {
  if (element) element.innerHTML = `<img src="${imagePaths[move]}" alt="${move}" />`;
}

handButtons.forEach((button) => {
  button.addEventListener('click', () => {
    const playerMove = button.dataset.choice as Move;
    const computerMove = chooseComputerMove();
    const outcome = getOutcome(playerMove, computerMove);

    handButtons.forEach((currentButton) => currentButton.classList.remove('selected'));
    button.classList.add('selected');
    showMove(playerDisplay, playerMove);
    showMove(computerDisplay, computerMove);
    history.push(playerMove);
    const previousGame = games[games.length - 1];
    games.push({
      playerMove,
      outcome,
      previousMove: previousGame?.playerMove,
      previousOutcome: previousGame?.outcome,
      repeated: previousGame?.playerMove === playerMove,
    });
    scores[outcome] += 1;

    if (roundStatus) roundStatus.textContent = outcome === 'tie' ? 'ROUND DRAW' : outcome === 'win' ? 'ROUND WON' : 'ROUND LOST';
    if (roundMessage) roundMessage.textContent = outcome === 'tie'
      ? `Both players chose ${playerMove}`
      : outcome === 'win' ? `${playerMove} beats ${computerMove}` : `${computerMove} beats ${playerMove}`;
    if (roundNumber) roundNumber.textContent = String(history.length + 1).padStart(2, '0');
    const scoreElement = scoreElements[outcome];
    if (scoreElement) scoreElement.textContent = String(scores[outcome]);
  });
});
