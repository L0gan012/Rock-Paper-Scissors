import '../style.css';
import { chooseNaiveBayesMove, getNaiveBayesProbabilities, type GameRecord } from './models/naiveBayes';
import { choosePerceptronMove } from './models/perceptron';
import { chooseMarkovMove } from './models/markov';

type Move = 'rock' | 'paper' | 'scissors';
type Outcome = 'win' | 'tie' | 'loss';
type Model = 'frequency' | 'naive-bayes' | 'perceptron' | 'markov';
type BenchmarkOpponent = 'always-rock' | 'always-paper' | 'always-scissors' | 'repeat-last' | 'repeating' | 'random-mode' | 'countering' | 'aggressive' | 'cycle' | 'alternate' | 'random';
interface RoundRecord extends GameRecord {
  computerMove: Move;
  model: string;
  playerLikelihood: number;
  computerLikelihood: number;
}

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
const benchmarkButton = document.querySelector<HTMLButtonElement>('#benchmark-run');
const benchmarkResults = document.querySelector<HTMLElement>('#benchmark-results');
const benchmarkModelSelect = document.querySelector<HTMLSelectElement>('#benchmark-model');
const benchmarkOpponentSelect = document.querySelector<HTMLSelectElement>('#benchmark-opponent');
const viewTabs = document.querySelectorAll<HTMLButtonElement>('.view-tab');
const viewSections = document.querySelectorAll<HTMLElement>('.view-section');
const historyResults = document.querySelector<HTMLElement>('#history-results');
const historyCount = document.querySelector<HTMLElement>('#history-count');
const historyPageSizeSelect = document.querySelector<HTMLSelectElement>('#history-page-size');
const historyPrevious = document.querySelector<HTMLButtonElement>('#history-previous');
const historyNext = document.querySelector<HTMLButtonElement>('#history-next');
const historyPageLabel = document.querySelector<HTMLElement>('#history-page-label');
const themeToggle = document.querySelector<HTMLButtonElement>('#theme-toggle');
const themeLabel = document.querySelector<HTMLElement>('#theme-label');
const themeMeta = document.querySelector<HTMLMetaElement>('meta[name="theme-color"]');
const scoreElements: Record<Outcome, HTMLElement | null> = {
  win: document.querySelector('#wins'), tie: document.querySelector('#ties'), loss: document.querySelector('#losses'),
};

const history: Move[] = [];
const games: RoundRecord[] = [];
const scores: Record<Outcome, number> = { win: 0, tie: 0, loss: 0 };
let selectedModel: Model = 'frequency';
let historyPage = 1;
let historyPageSize = 10;
const modelDescriptions: Record<Model, string> = {
  frequency: 'Counters the move you use most often.',
  'naive-bayes': 'Combines move, outcome, and repeat patterns to estimate your next move.',
  perceptron: 'Trains a lightweight classifier to recognize your sequences.',
  markov: 'Learns which move most often follows your previous move.',
};

viewTabs.forEach((tab) => {
  tab.addEventListener('click', () => {
    const targetView = tab.dataset.viewTarget;
    if (!targetView) return;
    viewTabs.forEach((currentTab) => {
      const isActive = currentTab === tab;
      currentTab.classList.toggle('is-active', isActive);
      currentTab.setAttribute('aria-selected', String(isActive));
    });
    viewSections.forEach((section) => section.classList.toggle('is-active', section.dataset.view === targetView));
  });
});

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

function chooseComputerMove(): Move {
  if (selectedModel === 'naive-bayes') return chooseNaiveBayesMove(games);
  if (selectedModel === 'perceptron') return choosePerceptronMove(games);
  if (selectedModel === 'markov') return chooseMarkovMove(games);
  if (history.length === 0) return randomMove();
  const predictedMove = mostFrequentMove();
  return counters[predictedMove];
}

function chooseBenchmarkMove(model: Model, history: Move[], games: GameRecord[]): Move {
  if (model === 'naive-bayes') return chooseNaiveBayesMove(games);
  if (model === 'perceptron') return choosePerceptronMove(games);
  if (model === 'markov') return chooseMarkovMove(games);
  if (history.length === 0) return randomMove();
  const counts = history.reduce<Record<Move, number>>((total, move) => {
    total[move] += 1;
    return total;
  }, { rock: 0, paper: 0, scissors: 0 });
  const predictedMove = moves.reduce((mostCommon, move) => counts[move] > counts[mostCommon] ? move : mostCommon, 'rock');
  return counters[predictedMove];
}

function chooseBenchmarkPlayerMove(opponent: BenchmarkOpponent, history: Move[], round: number): Move {
  if (opponent === 'always-paper') return 'paper';
  if (opponent === 'always-scissors') return 'scissors';
  if ((opponent === 'repeat-last' || opponent === 'repeating') && history.length > 0) return history[history.length - 1];
  if (opponent === 'random-mode') return randomMove();
  if (opponent === 'countering' && history.length > 0) return counters[history[history.length - 1]];
  if (opponent === 'aggressive' && history.length > 0) return counters[mostFrequentMoveFrom(history)];
  if (opponent === 'cycle') return moves[round % moves.length];
  if (opponent === 'alternate') return round % 2 === 0 ? 'rock' : 'paper';
  if (opponent === 'random') return randomMove();
  return 'rock';
}

function mostFrequentMoveFrom(moveHistory: Move[]): Move {
  const counts = moveHistory.reduce<Record<Move, number>>((total, move) => {
    total[move] += 1;
    return total;
  }, { rock: 0, paper: 0, scissors: 0 });
  return moves.reduce((mostCommon, move) => counts[move] > counts[mostCommon] ? move : mostCommon, 'rock');
}

function runBenchmark(): void {
  const modelLabels: Record<Model, string> = { frequency: 'ADAPTIVE FREQUENCY', 'naive-bayes': 'NAIVE BAYES', perceptron: 'PERCEPTRON', markov: 'MARKOV CHAIN' };
  const model = (benchmarkModelSelect?.value ?? 'frequency') as Model;
  const opponent = (benchmarkOpponentSelect?.value ?? 'always-rock') as BenchmarkOpponent;
  const opponentLabel = benchmarkOpponentSelect?.options[benchmarkOpponentSelect.selectedIndex].text ?? 'Always Rock';
  let result = { wins: 0, ties: 0, losses: 0 };
  {
      const playerHistory: Move[] = [];
      const benchmarkGames: GameRecord[] = [];
      let aiWins = 0;
      let ties = 0;
      let playerWins = 0;
      for (let round = 0; round < 100; round += 1) {
        const playerMove = chooseBenchmarkPlayerMove(opponent, playerHistory, round);
        const computerMove = chooseBenchmarkMove(model, playerHistory, benchmarkGames);
        const outcome = getOutcome(playerMove, computerMove);
        if (outcome === 'loss') aiWins += 1;
        if (outcome === 'tie') ties += 1;
        if (outcome === 'win') playerWins += 1;
        const previousGame = benchmarkGames[benchmarkGames.length - 1];
        benchmarkGames.push({ playerMove, outcome, previousMove: previousGame?.playerMove, previousOutcome: previousGame?.outcome, repeated: previousGame?.playerMove === playerMove });
        playerHistory.push(playerMove);
      }
      result = { wins: aiWins, ties, losses: playerWins };
  }
  if (benchmarkResults) benchmarkResults.innerHTML = `<strong>${result.wins}%</strong><span>AI WIN RATE</span><div class="benchmark-bar" role="img" aria-label="${result.wins} AI wins, ${result.ties} draws, ${result.losses} player wins"><i class="benchmark-bar-ai" style="width: ${result.wins}%"></i><i class="benchmark-bar-draw" style="width: ${result.ties}%"></i><i class="benchmark-bar-player" style="width: ${result.losses}%"></i></div><div class="benchmark-legend"><span><i class="legend-ai"></i>AI WINS ${result.wins}</span><span><i class="legend-draw"></i>DRAWS ${result.ties}</span><span><i class="legend-player"></i>PLAYER WINS ${result.losses}</span></div><small>${modelLabels[model]} VS ${opponentLabel.toUpperCase()} · 100 ROUNDS</small>`;
}

benchmarkButton?.addEventListener('click', runBenchmark);

function getOutcome(playerMove: Move, computerMove: Move): Outcome {
  if (playerMove === computerMove) return 'tie';
  return beats[playerMove] === computerMove ? 'win' : 'loss';
}

function showMove(element: HTMLElement | null, move: Move): void {
  if (element) element.innerHTML = `<img src="${imagePaths[move]}" alt="${move}" />`;
}

function historicalLikelihood(move: Move, values: Move[]): number {
  const count = values.filter((value) => value === move).length;
  return Math.round(((count + 1) / (values.length + moves.length)) * 100);
}

function getModelLikelihood(move: Move, games: GameRecord[], fallbackValues: Move[]): number {
  if (selectedModel !== 'naive-bayes') return historicalLikelihood(move, fallbackValues) / 100;
  return getNaiveBayesProbabilities(games)[move];
}

function renderHistory(): void {
  if (historyCount) historyCount.textContent = `${games.length} ${games.length === 1 ? 'ROUND' : 'ROUNDS'}`;
  if (!historyResults) return;
  if (games.length === 0) {
    historyResults.innerHTML = '<tr><td colspan="7">Play a round to start your session log</td></tr>';
    if (historyPageLabel) historyPageLabel.textContent = 'PAGE 1 OF 1';
    if (historyPrevious) historyPrevious.disabled = true;
    if (historyNext) historyNext.disabled = true;
    return;
  }
  const totalPages = Math.ceil(games.length / historyPageSize);
  historyPage = Math.min(historyPage, totalPages);
  const start = (historyPage - 1) * historyPageSize;
  const pageGames = games.slice().reverse().slice(start, start + historyPageSize);
  historyResults.innerHTML = pageGames.map((game, index) =>
    `<tr><td>${games.length - start - index}</td><td>${game.playerMove.toUpperCase()}</td><td>${game.playerLikelihood}%</td><td>${game.computerMove.toUpperCase()}</td><td>${game.computerLikelihood}%</td><td>${game.model}</td><td class="result-${game.outcome}">${game.outcome.toUpperCase()}</td></tr>`).join('');
  if (historyPageLabel) historyPageLabel.textContent = `PAGE ${historyPage} OF ${totalPages}`;
  if (historyPrevious) historyPrevious.disabled = historyPage === 1;
  if (historyNext) historyNext.disabled = historyPage === totalPages;
}

historyPageSizeSelect?.addEventListener('change', () => {
  historyPageSize = Number(historyPageSizeSelect.value);
  historyPage = 1;
  renderHistory();
});
historyPrevious?.addEventListener('click', () => { historyPage -= 1; renderHistory(); });
historyNext?.addEventListener('click', () => { historyPage += 1; renderHistory(); });
renderHistory();

handButtons.forEach((button) => {
  button.addEventListener('click', () => {
    const playerMove = button.dataset.choice as Move;
    const computerMove = chooseComputerMove();
    const outcome = getOutcome(playerMove, computerMove);

    handButtons.forEach((currentButton) => currentButton.classList.remove('selected'));
    button.classList.add('selected');
    showMove(playerDisplay, playerMove);
    showMove(computerDisplay, computerMove);
    const previousGame = games[games.length - 1];
    const probabilities = selectedModel === 'naive-bayes' ? getNaiveBayesProbabilities(games) : null;
    const predictedPlayerMove = moves.find((move) => counters[move] === computerMove) ?? computerMove;
    games.push({
      playerMove,
      outcome,
      computerMove,
      model: modelSelect?.options[modelSelect.selectedIndex].text ?? 'Adaptive Frequency',
      playerLikelihood: Math.round((probabilities?.[playerMove] ?? getModelLikelihood(playerMove, games, history)) * 100),
      computerLikelihood: Math.round((probabilities?.[predictedPlayerMove] ?? getModelLikelihood(computerMove, games, games.map((game) => game.computerMove))) * 100),
      previousMove: previousGame?.playerMove,
      previousOutcome: previousGame?.outcome,
      repeated: previousGame?.playerMove === playerMove,
    });
    history.push(playerMove);
    scores[outcome] += 1;

    if (roundStatus) roundStatus.textContent = outcome === 'tie' ? 'ROUND DRAW' : outcome === 'win' ? 'ROUND WON' : 'ROUND LOST';
    if (roundMessage) roundMessage.textContent = outcome === 'tie'
      ? `Both players chose ${playerMove}`
      : outcome === 'win' ? `${playerMove} beats ${computerMove}` : `${computerMove} beats ${playerMove}`;
    if (roundNumber) roundNumber.textContent = String(history.length + 1).padStart(2, '0');
    const scoreElement = scoreElements[outcome];
    if (scoreElement) scoreElement.textContent = String(scores[outcome]);
    renderHistory();
  });
});
