#   SNAKE GAME
#   Author : Apaar Gupta (@apaar97)
#   Python 3.5.2 Pygame

import pygame
import sys
import time
import random
import numpy as np
from random import sample

# Pygame Init
init_status = pygame.init()
if init_status[1] > 0:
    print("(!) Had {0} initialising errors, exiting... ".format(init_status[1]))
    sys.exit()
else:
    print("(+) Pygame initialised successfully ")

# Play Surface
# 640 x 320 (width x height)
size = width, height = 320, 320
playSurface = pygame.display.set_mode(size)
pygame.display.set_caption("Snake Game")
delta = 10

# Colors
red = pygame.Color(255, 0, 0)
green = pygame.Color(0, 255, 0)
black = pygame.Color(0, 0, 0)
white = pygame.Color(255, 255, 255)
brown = pygame.Color(165, 42, 42)

# FPS controller
fpsController = pygame.time.Clock()

#################################################################
# AI 
 
import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Input, concatenate
from keras.utils import plot_model
from keras.optimizers import Adam
from keras.models import load_model
from keras import regularizers

def e_greedy_policy(Q_values , e = 0.9):
    
    #print(Q_values)
    p = np.random.uniform()
    if p<e:
        return np.random.randint(len(Q_values))
    return np.argmax(Q_values)
    


# Input layers
direction_input_layer = Input(shape = (4,))
matrix_input_layer = Input(shape = (int(height/delta), int(width/delta),1))
raw_classes_input_layer = Input(shape = (int(height/delta)* int(width/delta)*3, ))

"""
conv1 = Convolution2D(4, (3, 3), activation = 'relu', kernel_regularizer=regularizers.l2(0.01))(matrix_input_layer)
max_pool1 = MaxPooling2D(pool_size = (2,2))(conv1)
conv2 = Convolution2D(4, (3, 3), activation = 'relu', kernel_regularizer=regularizers.l2(0.01))(max_pool1)
max_pool2 = MaxPooling2D(pool_size = (2,2))(conv2)
flatten_matrix = Flatten()(max_pool2)


dropout_layer = Dropout(0.1)(flatten_matrix)

dense1 = Dense(100, activation = 'relu', kernel_regularizer=regularizers.l2(0.01))(dropout_layer)
concatenated_layer = concatenate([dense1, direction_input_layer])
dense2 = Dense(32, activation = 'relu', kernel_regularizer=regularizers.l2(0.01))(concatenated_layer)
out_layer = Dense(4, activation = 'linear', kernel_regularizer=regularizers.l2(0.01))(dense2)
"""
concatenated_layer = concatenate([raw_classes_input_layer, direction_input_layer])
dense1 = Dense(300 , activation = 'relu', kernel_regularizer = regularizers.l2(0.01)) (concatenated_layer)
dropout1 = Dropout(0.1)(dense1)
dense2 = Dense(100 , activation = 'relu', kernel_regularizer = regularizers.l2(0.01)) (dropout1)
dropout2 = Dropout(0.1)(dense2)
dense3 = Dense(4 , activation = 'relu', kernel_regularizer = regularizers.l2(0.01)) (dropout2)



Q_network = Model(inputs = [raw_classes_input_layer, direction_input_layer], outputs = [dense3])
Q_network.summary()
adam = Adam(lr = 0.001)
Q_network.compile(loss='mse', optimizer=adam)
plot_model(Q_network, to_file='model.png')

print('loading pretrained model')
Q_network = load_model('Qnet_v3.h5')

Q_network.compile(loss='mse', optimizer=adam)


def choose_action(state, direction):
    #print("['RIGHT','LEFT', 'UP', 'DOWN']")
    
    Q_values = Q_network.predict([np.array([state]), np.array([direction])])[0]
    #print(Q_values)
    return e_greedy_policy(Q_values)


##################################################################

# Game settings

snakePos = [100, 50]
snakeBody = [[100, 50], [90, 50], [80, 50]]
foodPos = [400, 50]
foodSpawn = True
direction = 'RIGHT'
changeto = ''
score = 0

directions_list = ['RIGHT','LEFT', 'UP', 'DOWN']
directions_dict = {}

for d in directions_list:
    directions_dict[d] = len(directions_dict)

def string_to_hotform(direction):

    hotform = np.zeros(len(directions_list))
    hotform[directions_dict[direction]] = 1

    return hotform

def get_snake_state():

    global snakePos, snakeBody, foodPos
    # Note that the dimensions are inverted
    state = np.zeros((int(height/delta), int(width/delta),1))

    for pos in snakeBody[1:]:
        i = int(pos[1]/delta)
        j = int(pos[0]/delta)
        state[i][j][0] = 200

    ii = int(foodPos[1]/delta)
    jj = int(foodPos[0]/delta)
    state[ii][jj][0] = 300

    ii = int(snakeBody[0][1]/delta)
    jj = int(snakeBody[0][0]/delta)
    
    state[ii][jj][0] = 100
    
    return state


def get_snake_state_v2():

    global snakePos, snakeBody, foodPos
    # Note that the dimensions are inverted
    state = np.zeros((int(height/delta), int(width/delta),3))

    for pos in snakeBody[1:]:
        i = int(pos[1]/delta)
        j = int(pos[0]/delta)
        state[i][j][0] = 1

    ii = int(foodPos[1]/delta)
    jj = int(foodPos[0]/delta)
    state[ii][jj][1] = 1

    ii = int(snakeBody[0][1]/delta)
    jj = int(snakeBody[0][0]/delta)
    
    state[ii][jj][2] = 1

    state = state.flatten()
    
    return state


def spawn_snake():


    random_init_pos_x = random.randrange(3, (width // 10) -3 ) * delta
    random_init_pos_y = random.randrange(3, (width // 10) -3 ) * delta
    snakePos = [random_init_pos_x, random_init_pos_y]
    snakeBody = []
    
    if random.randrange(0,2)==1:
        ii = sample([-1,1],1)[0]
        snakeBody = [[random_init_pos_x, random_init_pos_y], [random_init_pos_x +delta*ii, random_init_pos_y], [random_init_pos_x  +2*delta*ii, random_init_pos_y]]
    else:
        ii = sample([-1,1],1)[0]
        snakeBody = [[random_init_pos_x, random_init_pos_y], [random_init_pos_x, random_init_pos_y +delta*ii], [random_init_pos_x, random_init_pos_y +2*delta*ii ]]
    
    return snakePos, snakeBody
def restartGame():
    global snakePos, snakeBody, foodPos, foodSpawn, direction, changeto, score
    
    """
    random_init_pos_x = max(random.randrange(1, width // 10), 3) * delta
    random_init_pos_y = max(random.randrange(1, height // 10), 3) * delta
    snakePos = [random_init_pos_x, random_init_pos_y]
    snakeBody = [[random_init_pos_x, random_init_pos_y], [random_init_pos_x -delta, random_init_pos_y], [random_init_pos_x - 2*delta, random_init_pos_y]]
    """

    snakePos, snakeBody = spawn_snake()
    foodPos = [random.randrange(1, width // 10) * delta, random.randrange(1, height // 10) * delta]
    foodSpawn = True
    direction = sample(directions_list,1)[0]
    #print (direction)
    changeto = ''
    score = 0


# Game Over
def gameOver():
    myFont = pygame.font.SysFont('monaco', 72)
    GOsurf = myFont.render("Game Over", True, red)
    GOrect = GOsurf.get_rect()
    GOrect.midtop = (320, 25)
    #playSurface.blit(GOsurf, GOrect)
    #showScore(0)
    #pygame.display.flip()
    #time.sleep(1)
    #pygame.quit()
    #sys.exit()
    restartGame()


# Show Score
def showScore(choice=1):
    SFont = pygame.font.SysFont('monaco', 32)
    Ssurf = SFont.render("Score  :  {0}".format(score), True, black)
    Srect = Ssurf.get_rect()
    if choice == 1:
        Srect.midtop = (80, 10)
    else:
        Srect.midtop = (320, 100)
    playSurface.blit(Ssurf, Srect)


restartGame()
episode_count = 0
accumulated_reward = 0

experience_buffer = []

def store_experience(state, direction,action, reward, next_state, next_direction, flag):

    global experience_buffer
    experience_buffer.append((state, direction ,action, reward, next_state, next_direction, flag))
    experience_buffer = experience_buffer[0:20000]

def remember_experience():

    global experience_buffer, Q_network

    gamma = 0.99

    memories = sample(experience_buffer, min(2004, len(experience_buffer)))
    next_states = [[experience[4], experience[5]] for experience in memories] 
    current_states = [[experience[0], experience[1]] for experience in memories]
    actions = [experience[2] for experience in memories]
    rewards = [experience[3] for experience in memories]

    flags = [experience[6] for experience in memories]
    next_states_s = np.array([state[0] for state in next_states])
    next_states_d = np.array([state[1] for state in next_states])

    rewards = np.array(rewards)

    Q_next = Q_network.predict([next_states_s, next_states_d])
    Q_next = np.max(Q_next, axis = 1)
    
    ##########################################
    Q_next_aux = []
    for i in range(len(memories)):
        if flags[i]:
            Q_next_aux.append(0)
        else:
            Q_next_aux.append(Q_next[i])
    ##########################################
    
    Q_next = np.array(Q_next_aux)

    Q_next = rewards + Q_next * gamma 

    current_states_s = np.array([state[0] for state in current_states])
    current_states_d = np.array([state[1] for state in current_states])

    Q_target = Q_network.predict([current_states_s, current_states_d])

    for i, action in enumerate(actions):
        Q_target[i][action] = Q_next[i]

    verb = 0
    if np.random.uniform()<0.1:
        verb = 1
    Q_network.fit([current_states_s, current_states_d], Q_target, verbose = verb)



reward_hist = []
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                changeto = 'RIGHT'
            if event.key == pygame.K_LEFT or event.key == pygame.K_a:
                changeto = 'LEFT'
            if event.key == pygame.K_UP or event.key == pygame.K_w:
                changeto = 'UP'
            if event.key == pygame.K_DOWN or event.key == pygame.K_s:
                changeto = 'DOWN'
            if event.key == pygame.K_ESCAPE:
                pygame.event.post(pygame.event.Event(pygame.QUIT))

    # Validate direction
    if changeto == 'RIGHT' and direction != 'LEFT':
        direction = changeto
    if changeto == 'LEFT' and direction != 'RIGHT':
        direction = changeto
    if changeto == 'UP' and direction != 'DOWN':
        direction = changeto
    if changeto == 'DOWN' and direction != 'UP':
        direction = changeto
    
    # AI choses the direction
     
    current_state = get_snake_state_v2()
    current_direction = string_to_hotform(direction)
    action = choose_action(current_state, current_direction)
    direction = directions_list[action]
    #print('direction ', direction)

    # Update snake position
    if direction == 'RIGHT':
        snakePos[0] += delta
    if direction == 'LEFT':
        snakePos[0] -= delta
    if direction == 'DOWN':
        snakePos[1] += delta
    if direction == 'UP':
        snakePos[1] -= delta

    # Snake body mechanism
    reward = 1
    snakeBody.insert(0, list(snakePos))

    if snakePos == foodPos:
        foodSpawn = False
        reward+=1000
        score += 1
    else:
        snakeBody.pop()

    if foodSpawn == False:
        foodPos = [random.randrange(1, width // 10) * delta, random.randrange(1, height // 10) * delta]
        foodSpawn = True

    playSurface.fill(white)

    for pos in snakeBody:
        pygame.draw.rect(playSurface, green, pygame.Rect(pos[0], pos[1], delta, delta))
    pygame.draw.rect(playSurface, brown, pygame.Rect(foodPos[0], foodPos[1], delta, delta))

    flag = False
    # Bounds
    if snakePos[0] >= width or snakePos[0] < 0:
        gameOver()
        flag = True
    if snakePos[1] >= height or snakePos[1] < 0:
        gameOver()
        flag = True

    # Self hit
    for block in snakeBody[1:]:
        if snakePos == block:
            gameOver()
            flag = True

    showScore()
    pygame.display.flip()
    ###########################################################################################
    # Q- learning
    """
    gamma = 0.99

    Q_next_state = Q_network.predict( [np.array([get_snake_state()]), np.array([string_to_hotform(direction)])]) [0]
    Q_max = Q_next_state.max()
    Q_target = Q_network.predict( [np.array([current_state]), np.array([string_to_hotform(current_direction)])]) [0]
    """
    accumulated_reward += reward
    
    if flag:

        #Q_max = 0
        remember_experience()
        reward = -1000
        episode_count+=1
        accumulated_reward += reward
        reward_hist.append(accumulated_reward)
        accumulated_reward = 0
        if episode_count % 1000 == 0:
            print('saving, episodes passed', episode_count)
            mean = np.array(reward_hist).mean()
            print('mean reward ', mean)
            reward_hist = []
            Q_network.save('Qnet_v3.h5')



    
    next_state = get_snake_state_v2()
    next_direction = string_to_hotform(direction)
    store_experience(current_state, current_direction, action, reward, next_state, next_direction, flag)
    #Q_target[action] = reward + gamma * Q_max
    
    #Q_network.fit([np.array([get_snake_state()]), np.array([string_to_hotform(direction)])], np.array([Q_target]), verbose =0) 
    


    ###########################################################################################


    #fpsController.tick(5)
