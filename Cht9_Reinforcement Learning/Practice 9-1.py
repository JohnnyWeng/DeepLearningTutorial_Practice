import numpy as np
import pandas as pd
import time

np.random.seed(2)
ACTIONS = ['left', 'right']


N_STATES = 7  # Adding a new state for the Cliff
env_list = ['C'] + ['-']*(N_STATES-2) + ['T']  # Add 'C' for cliff and 'T' for treasure
EPSILON = 0.9  # The probability of choosing the optimal strategy
ALPHA = 0.1  # learning rate
GAMMA = 0.9  # decay ratio
MAX_EPISODES = 13  # Maximum number of rounds
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


# Models the environment's response
# def get_env_feedback(S, A):
#     # # Agent moves right
#         # This is how agent will interact with the environment
#     if A == 'right':  # move right
#         if S == N_STATES - 2:  # terminate
#             S_ = 'terminal'
#             R = 1
#         else:
#             S_ = S + 1
#             R = 0
#     # Penalty
#     else:  # move left
#         # R = 0
#         # if S == 0:
#         #     S_ = S  # reach the wall
#         # else:
#         #     S_ = S - 1
#         if S == 1:  # Cliff state
#             S_ = 0  # Reset to initial state after penalty
#             R = -1  # Penalty for falling into the cliff
#         elif S == 0:  # Already at cliff
#             S_ = S  # Stay in place
#             R = -1  # Penalty
#         else:
#             S_ = S - 1
#             R = 0  # Neutral reward for moving
#     return S_, R
def get_env_feedback(S, A):
    if A == 'right':  # move right
        if S == N_STATES - 2:  # terminate
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0
    # Penalty
    # I only have 2 things to do: move and penalty when reaching to the edge.
    else:  # move left
        if S == 1:  # Already at cliff
            S_ = S  # Stay in place
            R = -1  # Penalty
        else:
            S_ = S - 1
            R = 0  # Neutral reward for moving
    return S_, R

# Visualizes the agent's progress
# def update_env(S, episode, step_counter):
#     # This is how environment be updated
#     env_list = ['-'] * (N_STATES - 1) + ['T']  # - - - - -T is the environment, set up our environment
#     # print('env_list = ', env_list)
#
#     # Here we update the "visually progress" to see where the 'o' is in the progress. for example: --o--T
#     if S == 'terminal':
#         interaction = 'Episode %s: total_steps = %s' % (episode + 1, step_counter)
#         print('\r{}'.format(interaction), end='')  # /r : Return the cursor to the beginning
#         # 休息 2 秒
#         time.sleep(2)
#         print('\r                                ', end='')  # end='':No line breaks after output
#     else:
#         env_list[S] = 'o'
#         interaction = ''.join(env_list)  # Make the list to be a string
#         print('\r{}'.format(interaction), end='')
#         time.sleep(FRESH_TIME)

# Updated update_env for visualization
def update_env(S, episode, step_counter):
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\n                                ', end='')  # Clear line
    else:
        env_display = env_list[:]  # Copy environment list
        env_display[S] = 'o'  # Mark agent's position
        print('\r{}'.format(''.join(env_display)), end='')  # Display the environment
        time.sleep(FRESH_TIME)


# Implements the Q-learning algorithm
def rl():
    # main part of RL loop
    # Initialize the Q-Table
    q_table = build_q_table(N_STATES, ACTIONS)

    # Iterate Through Episodes
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = N_STATES // 2  # Start in the middle of the environment
        is_terminated = False
        update_env(S, episode, step_counter)  # Call update_env to display the agent’s initial position
        # Main Episode Loop
        while not is_terminated:
            A = choose_action(S, q_table)
            print('\n S, A = ', S, A)
            S_, R = get_env_feedback(S, A)  # Achieve the next status and score
            if S_ == 1 and R == -1:  # Agent falls into the cliff
                print("\nAgent fell into the cliff! Restarting episode...")
                is_terminated = True  # End the current episode
                S = N_STATES // 2
                continue  # Skip the rest of the loop
            print('R = ', R)
            q_predict = q_table.loc[S, A]
            # Calculate Q-Target
            if S_ != 'terminal':
                q_target = R + GAMMA * q_table.iloc[S_, :].max()  # next state is not terminal
            else:
                q_target = R  # next state is terminal
                is_terminated = True
            # Update Q-Value
            q_table.loc[S, A] += ALPHA * (q_target - q_predict)

            # Transition to the Next State
            S = S_
            update_env(S, episode, step_counter + 1)
            step_counter += 1
            print('\nR2 = ', R)
    # Return the Learned Q-Table
    return q_table


if __name__ == "__main__":
    q_table = rl()
    # Output
    print('\r\nQ-table:\n')
    print(q_table)

