import numpy as np
import pandas as pd
import random as rd

from enum import Enum

class Inputs(Enum):
    r = 0
    p = 1
    s = 2

# gets the users selection
def get_input():
    return input("Select 'r', 'p', 's' or 'x' to exit: ")

# returns the feature vector of the current user selection
def get_features(games, pick):
    win, lose = get_prev_win_lose(games)
    return [1 if games.iloc[-1]['pick'] == pick else 0, 1 if pick == win else 0, 1 if pick == lose else 0]

# returns the winning and losing hand of the previous match
def get_prev_win_lose(games):
    prev_game = games.iloc[-1]
    win = 0
    lose = 0
    if prev_game['outcome'] == 1:
        win = prev_game['pick']
        lose = find_loser(win)
    elif prev_game['outcome'] == -1:
        lose = prev_game['pick']
        win = find_winner(lose)
    else:
        win = find_winner(prev_game['pick'])
        lose = find_loser(prev_game['pick'])
    return win, lose

# determines the loser given a winner
def find_loser(winner):
    if winner == 0:
        loser = 2
    elif winner == 1:
        loser = 0
    else:
        loser = 1
    return loser

# determines the winner given a loser
def find_winner(loser):
    if loser == 0:
        winner = 1
    elif loser == 1:
        winner = 2
    else:
        winner = 0
    return winner

# determines the outcome of the game given the user pick and the agents prediction
def determine_outcome(player_pick, agent_pick):
    if find_loser(agent_pick) == player_pick:
        outcome = -1
    else:
        if find_winner(agent_pick) != player_pick:
            outcome = 0
        else:
            outcome = 1
    return outcome

# returns a vector corresponding to the probability the instance is a member of each class
def softmax(z):
    return np.exp(z)/np.sum(np.exp(z), axis=0)

# algorithm used to update the weight matrix and predict a classification of the data instance
def perceptron(x, y, classes, learning_rate=0.01, max_iter=1):
    feature_size = x.columns.size
    game_history = len(x.index)
    # creates a weight matrix where the columns correspond to the class weight vector
    W = np.zeros((feature_size, classes))
    for t in range(max_iter):
        for n in range(game_history):
            x_n = x.iloc[n].to_numpy()
            actual_y = y[n]
            predicted_y = np.argmax(softmax(np.matmul(x_n,W).astype(float)))
            # update the weight matrix if the perdicted value does not match the actual
            if(predicted_y != actual_y):
                W[:,predicted_y] = W[:,predicted_y] - learning_rate*(actual_y*x_n)
                W[:,actual_y] = W[:,actual_y] + learning_rate*(actual_y*x_n)
    return W, predicted_y

# displays the results of the game to the console
def show_scoreboard(games):
    score = games['outcome'].value_counts()
    try:
        wins = score.at[1]
    except KeyError:
        wins = 0
    try:
        ties = score.at[0]
    except KeyError:
        ties = 0
    try:
        losses = score.at[-1]
    except KeyError:
        losses = 0
    print('\nWins: ', wins, " Ties: ", ties, " Losses: ", losses)

# main funtion
def main():
    games = pd.DataFrame(columns=['pick', 'same', 'what_won', 'what_lost', 'outcome'])
    memory_depth = 7

    pick = ''
    while pick != 'x':
        pick = get_input()
        if pick == 'x':
            continue
        elif pick not in Inputs.__members__:
            print("Please input a valid selection")
            continue
        # checks to see if its the first game
        if games.size > 0:
            features = get_features(games, Inputs[pick].value)
            x = games.tail(memory_depth).drop(columns=['pick', 'outcome']).reset_index(drop=True)
            y = games.tail(memory_depth)['pick'].reset_index(drop=True)
            w, prediction = perceptron(x, y, 3)
            agent_pick = find_winner(prediction)
        else:
            features = [0, 0, 0]
            # random prediction for first game
            agent_pick = rd.choice(list(Inputs)).value
        game = pd.Series({'pick':Inputs[pick].value, 'same':features[0], 'what_won':features[1], 'what_lost':features[2], 'outcome':determine_outcome(Inputs[pick].value, agent_pick)})
        games = games.append(game, ignore_index=True)
        show_scoreboard(games)
    print("Thanks for playing!")

main()
