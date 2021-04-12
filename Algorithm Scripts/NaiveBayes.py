import pandas as pd
import random

#valid user inputs
inputs = ['r','p','s']
#creates a dataframe that stores the user input, the computer decision, and the outcome of the game
games = pd.DataFrame(columns=['User', 'Prediction', 'Outcome'])

#gets the users choice
def get_input():
    return input("Select 'r', 'p', 's', or 'x' to exit: ")

#predicts what the users input will be using naive bayes, default choice will be random
def predict_input(games):
    #random default choice
    predict = random.choice(['r','p','s'])
    #total number of games played
    total_games = len(games.index)
    #gets the number of wins(for the user)
    try:
        total_wins = games['Outcome'].value_counts().loc['lose']
    except KeyError:
        total_wins = 0
    #calculates the win probability
    try:
        win_prob = total_wins/total_games
    except ZeroDivisionError:
        win_prob = 0
    #uses the naive bayes algorithm to make a prediction of what the user will input
    win_prob_given_r = naive_bayes('r', total_wins, win_prob, games)
    win_prob_given_p = naive_bayes('p', total_wins, win_prob, games)
    win_prob_given_s = naive_bayes('s', total_wins, win_prob, games)

    if win_prob_given_r > win_prob_given_p and win_prob_given_r > win_prob_given_s:
        predict = 'r'
    elif win_prob_given_p > win_prob_given_r and win_prob_given_p > win_prob_given_s:
        predict = 'p'
    elif win_prob_given_s > win_prob_given_r and win_prob_given_s > win_prob_given_p:
        predict = 's'
    return predict

#determines the probability of a win given the a user input
def naive_bayes(guess, total_wins, win_prob, games):
    #tries to get the probability a guess given a win, zero if no wins are found
    try:
        prob_guess_given_win = games.loc[games['User'] == guess, ['Outcome']].value_counts().at['lose']/total_wins
    except KeyError:
        prob_guess_given_win = 0
    #total number of user inputs
    total_guesses = len(games.index)
    #tries to get the probability a user will select a given guess, zero if that guess has never been choosen
    try:
        prob_guess = games.loc[:,'User'].value_counts().at[guess]/total_guesses
    except KeyError:
        prob_guess = 0
    #calculates the probability a given class will result in a win
    if prob_guess != 0:
        prob = (prob_guess_given_win*win_prob)/prob_guess
    else:
        prob = 0
    return prob

#returns the computers choice based on the predicted user input
def guess(predict_input):
    if predict_input == 'r':
        guess = 'p'
    elif predict_input == 'p':
        guess = 's'
    elif predict_input == 's':
        guess = 'r'
    return guess

#returns the outcome of the game given the users pick and the computers choice
def get_outcome(user_pick, comp_pick):
    if user_pick == comp_pick:
        outcome = 'tie'
    elif user_pick == 'r':
        if comp_pick == 's':
            outcome = 'lose'
        else:
            outcome = 'win'
    elif user_pick == 'p':
        if comp_pick == 'r':
            outcome = 'lose'
        else:
            outcome = 'win'
    elif user_pick == 's':
        if comp_pick == 'p':
            outcome = 'lose'
        else:
            outcome = 'win'
    return outcome

#main method
def main(games):
    #continually plays games with the user until they choose to exit
    pick = ''
    while pick != 'x':
        #get the users pick
        pick = get_input()
        #pick = random.choice(['r','p','s'])
        #if the users input is invaild prompt them to input again
        if pick == 'x':
            continue
        elif inputs.count(pick) == 0:
            print("Invaild input, please try again.")
            continue
        #computer makes a prediction
        predict = guess(predict_input(games))
        #outcome of the game
        outcome = get_outcome(pick, predict)
        #creates the game and adds it to the knowledge base
        game = pd.Series({'User':pick, 'Prediction':predict, 'Outcome':outcome})
        games = games.append(game, ignore_index=True)
        score = games['Outcome'].value_counts()
        try:
            wins = score.at['lose']
        except KeyError:
            wins = 0
        try:
            ties = score.at['tie']
        except KeyError:
            ties = 0
        try:
            losses = score.at['win']
        except KeyError:
            losses = 0
        print('\nWins: ', wins, " Ties: ", ties, " Losses: ", losses)
    print("Thanks for playing!")

main(games)
