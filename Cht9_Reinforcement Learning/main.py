import numpy as np
import pandas as pd
import time

np.random.seed(2)

#  Initialization
N_STATES = 6  # 6 states
ACTIONS = ['left', 'right']
EPSILON = 0.9  # The probability of choosing the optimal strategy
ALPHA = 0.1  # learning rate
GAMMA = 0.9  # decay ratio
# Find 13 time (epoch = 13)
MAX_EPISODES = 13  # Maximum number of rounds
# Here can let you know the o is moving.
FRESH_TIME = 0.3  # Travel interval time

def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),
        columns=actions,  # actions's name
    )
    return table


tmp_table = build_q_table(N_STATES, ACTIONS)
print('tmp_table = \n', tmp_table)  # At first, we will get the table with all 0.

# Implements the ε-greedy policy to select an action.
def choose_action(state, q_table):
    # This is how to choose an action
    state_actions = q_table.iloc[state, :]  # Select all the action values of this state
    # 10% chance, or the action value of the state is 0 -> Choose the random action
    if (np.random.uniform() > EPSILON) or ((state_actions == 0).all()):
        action_name = np.random.choice(ACTIONS)  # Randomly pick an action
    else:
        action_name = state_actions.idxmax()
    return action_name


import numpy as np
import pandas as pd

a = pd.DataFrame([[1, 0], [1, 1]])
b = a.iloc[1, :]
print((a == 1).all())
print("-" * 10)
print((b == 1).all())

def get_env_feedback(S, A):
    '''
    move right
    '''
    # This is how agent will interact with the environment
    if A == 'right':  # move right
        if S == N_STATES - 2:  # terminate
            S_ = 'terminal'
            R = 1  # If we get to the terminal, we will have the instant reward.
        else:
            S_ = S + 1
            R = 0
    else:  # move left
        # The agent receives no reward (R = 0) for moving left.
        R = 0
        if S == 0:
            # o position: If the agent is at the leftmost state (S == 0), it remains there (S_ = S).
            S_ = S  # reach the wall
        else:
            # Otherwise, it moves one state back (S - 1).
            S_ = S - 1
    #         R will be only 0 or 1 or -1.
    return S_, R


# Visualizes the agent's progress
def update_env(S, episode, step_counter):
    # This is how environment be updated
    env_list = ['-'] * (N_STATES - 1) + ['T']  # - - - - -T is the environment, set up our environment
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode + 1, step_counter)
        print('\r{}'.format(interaction), end='')  # /r : Return the cursor to the beginning
        # 休息 2 秒
        time.sleep(2)
        print('\n                                ', end='')  # end='':No line breaks after output
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)  # Make the list to be a string
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)


# Implements the Q-learning algorithm
def rl():
    q_table = build_q_table(N_STATES, ACTIONS)

    # Iterate Through Episodes: every time the Q-table will be updated from learning.
    for episode in range(MAX_EPISODES):
        step_counter = 0
        '''
        S: State
        A: Action
        '''
        S = 0
        is_terminated = False
        update_env(S, episode, step_counter)  # Call update_env to display the agent’s initial position
        # Main Episode Loop
        while not is_terminated:
            A = choose_action(S, q_table)
            print('\n S, A = ', S, A)
            # Go right (reward) or left (penalty).
            S_, R = get_env_feedback(S, A)
            q_predict = q_table.loc[S, A]  # 下一個狀態與 reward
            # Calculate Q-Target
            if S_ != 'terminal':
                q_target = R + GAMMA * q_table.iloc[S_, :].max()  # next state is not terminal
            else:
                # If the o reach to terminate, we will stop and break from for-loop.
                q_target = R  # next state is terminal
                is_terminated = True
            q_table.loc[S, A] += ALPHA * (q_target - q_predict)

            # Transition to the Next State
            S = S_
            update_env(S, episode, step_counter + 1)
            step_counter += 1
    # Return the Learned Q-Table
    return q_table


if __name__ == "__main__":
    q_table = rl()
    # Output
    print('\r\nQ-table:\n')
    print(q_table)
