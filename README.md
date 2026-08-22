# Rock Paper Scissors

A browser-based Rock Paper Scissors game with multiple opponent strategies and predictive AI models. Players choose a move, the computer predicts the next move using different models, and the game tracks scores, history, and benchmark performance.

## Features

- Play against multiple AI strategies:
  - Adaptive Frequency
  - Naive Bayes
  - Perceptron
  - Markov Chain
- Live match history with paginated round log
- Score tracking for wins, ties, and losses
- Theme toggle with dark/light mode support
- Benchmark runner to test a model against common opponent behaviors
- Responsive single-page interface built with Vite + TypeScript

## Project structure

- `index.html` – app shell and UI layout
- `style.css` – styling and theme definitions
- `src/main.ts` – game logic, UI wiring, model selection, and benchmark simulation
- `src/models/naiveBayes.ts` – Naive Bayes predictive logic
- `src/models/perceptron.ts` – lightweight perceptron classifier
- `src/models/markov.ts` – Markov-based move prediction
- `public/Images/` – hand images used in the UI

## Getting started

1. Install dependencies:

   ```bash
   npm install
   ```

2. Start the development server:

   ```bash
   npm run dev
   ```

3. Open the local Vite URL shown in the terminal to play.

## Available scripts

```bash
npm run dev
npm run build
npm run preview
```

## How the models work

- Frequency model: predicts the move you have used most often in recent history.
- Naive Bayes model: estimates the probability of your next move based on prior move, previous outcome, and repetition patterns.
- Perceptron model: learns a simple weighted classifier from earlier game states.
- Markov model: looks at the last move and estimates the most likely follow-up pattern.

## Notes

This project is intentionally educational and designed to compare different lightweight prediction approaches in a simple game context.
