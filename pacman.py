import pandas as pd
import numpy as np
import random
import dataframe_image as dfi
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

ALPHA = 0.5
DISCOUNT_FACTOR = 0.5
EPSILON = 0.1
ENVIRONMENT = [
    [2, 0, 25], # GREEN
    [2, 2, 80],
    [0, 2, 100],

    [0, 1, -80], # RED
    [2, 1, -100]
]

def not_terminal(state):
    for i in ENVIRONMENT:
        if state[0] == i[0] and state[1] == i[1]:
            return False
    return True

def get_reward(state):
    for i in ENVIRONMENT:
        if state[0] == i[0] and state[1] == i[1]:
            return i[2]
    return 0

def update_state_with_action(state, action):
    if action == 0:
        state[0] -= 1
    if action == 1:
        state[0] += 1
    if action == 2:
        state[1] -= 1
    if action == 3:
        state[1] += 1

    if state[0] < 0:
        state[0] = 0
        print("WARNING TOP")
    if state[0] > 2:
        state[0] = 2
        print("WARNING BOT")
    if state[1] < 0:
        state[1] = 0
        print("WARNING WEST")
    if state[1] > 2:
        state[1] = 2
        print("WARNING EST")
    return state

def translate_qtable_to_pacman(nb):
    if nb == 0:
        return [0, 0]
    if nb == 1:
        return [0, 1]
    if nb == 2:
        return [0, 2]
    if nb == 3:
        return [1, 0]
    if nb == 4:
        return [1, 1]
    if nb == 5:
        return [1, 2]
    if nb == 6:
        return [2, 0]
    if nb == 7:
        return [2, 1]
    if nb == 8:
        return [2, 2]

def translate_pacman_world_to_qtable(state):
    if state == [0, 0]:
        return 0
    if state == [0, 1]:
        return 1
    if state ==  [0,2]:
        return 2
    if state == [1, 0]:
        return 3
    if state == [1, 1]:
        return 4
    if state ==  [1,2]:
        return 5
    if state == [2, 0]:
        return 6
    if state == [2, 1]:
        return 7
    if state == [2, 2]:
        return 8
    return "ERROR"
    
def generate_table_with_zeros(w, h):
    return np.zeros((w, h))

# Initializing table
nb_actions = 4  # North [0], South [1], West [2], Est [3]
nb_states = 3 * 3
# [0, 0] = [0], [0, 1] = [1], [0,2] = [2]
# [1, 0] = [3], [1, 1] = [4], [1,2] = [5]
# [2, 0] = [6], [2, 1] = [7], [2,2] = [8]
Qtable = generate_table_with_zeros(nb_states, nb_actions)
print(Qtable)

try:
    episodes = int(input("How many episodes do you want ?"))
except:
    print("Please give an integer.")
    exit(84)

def get_random_state():
    while 1:
        tmp = True
        row = random.randint(0, 2)
        col = random.randint(0, 2)
        for i in ENVIRONMENT:
            if row == i[0] and col == i[1]:
                tmp = False
                break
        if tmp is True:
            return [row, col]

def select_action(Qtable, state, EPSILON):
    action = None
    n = np.random.uniform(0,1)

    if n < EPSILON:
        action = np.random.randint(Qtable.shape[1])
    else:
        action = np.argmax(Qtable[state])
    return action
                
# For each step in each episode, we calculate the Q-value
# and update the Q table
for i in range(episodes):
    print("Generation N°", i)
    # Initialize state S
    state = get_random_state()
    qtable_state = translate_pacman_world_to_qtable(state)
    while not_terminal(state):
        action = select_action(Qtable, qtable_state, EPSILON)
        next_state = update_state_with_action(state, action)
        qtable_nextstate = translate_pacman_world_to_qtable(next_state)

        reward = get_reward(next_state)
        td_delta = reward + DISCOUNT_FACTOR * np.max(Qtable[qtable_nextstate]) - Qtable[qtable_state][action]
        Qtable[qtable_state, action] = Qtable[qtable_state, action] + ALPHA * td_delta

        state = next_state
        qtable_state = qtable_nextstate

################## DRAW RESULTS ##################################


print(Qtable)


row_label = [   "[0, 0]", "[0, 1]", "[0, 2]",
                "[1, 0]", "[1, 1]", "[1, 2]",
                "[2, 0]", "[2, 1]", "[2, 2]"]
col_label = ["N", "S", "W", "E"]
df = pd.DataFrame(Qtable, index=row_label, columns=col_label)
dfi.export(df, 'dataframe.png')


myFont = ImageFont.truetype('./FreeMono.ttf', 40)

img = Image.open('./template.jpg')
I1 = ImageDraw.Draw(img)
green = (0, 255, 0)
red = (255, 0, 0)

def get_row_col(index):
    row = int(index / 3)
    col = index % 3
    return [row, col]

def get_color(value):
    if value > 0:
        return green
    elif value == 0:
        return (0, 0, 0)
    else:
        return red

point_1 = (640, 530) # North
point_2 = (640, 740) # South
point_3 = (530, 640) # West
point_4 = (740, 640) # Est
cube_diff = 350

for i in range(len(Qtable)):
    current = get_row_col(i)
    distx = (current[1] - 1) * cube_diff
    disty = (current[0] - 1) * cube_diff
    I1.text((point_1[0] + distx, point_1[1] + disty), f'{Qtable[i][0]:.2f}', anchor="mm", font=myFont, fill=get_color(Qtable[i][0]))
    I1.text((point_2[0] + distx, point_2[1] + disty), f'{Qtable[i][1]:.2f}', anchor="mm", font=myFont, fill=get_color(Qtable[i][1]))
    I1.text((point_3[0] + distx, point_3[1] + disty), f'{Qtable[i][2]:.2f}', anchor="mm", font=myFont, fill=get_color(Qtable[i][2]))
    I1.text((point_4[0] + distx, point_4[1] + disty), f'{Qtable[i][3]:.2f}', anchor="mm", font=myFont, fill=get_color(Qtable[i][3]))

img.save("./qlearning_results.png")
