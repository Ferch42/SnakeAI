"""
Snake Eater
Made with PyGame
"""

import pygame, sys, time, random
import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Lambda, Conv2D, Flatten, Dropout, Subtract, Add
import keras.backend as K
from keras import optimizers
from keras.models import load_model
import pickle
# Difficulty settings
# Easy      ->  10
# Medium    ->  25
# Hard      ->  40
# Harder    ->  60
# Impossible->  120
difficulty = 25

# Window size
frame_size_x = 500
frame_size_y = 500

frame_size_x_simplified = int(frame_size_x/10)
frame_size_y_simplified = int(frame_size_y/10)

# Checks for errors encountered
check_errors = pygame.init()
# pygame.init() example output -> (6, 0)
# second number in tuple gives number of errors
if check_errors[1] > 0:
	print(f'[!] Had {check_errors[1]} errors when initialising game, exiting...')
	sys.exit(-1)
else:
	print('[+] Game successfully initialised')


# Initialise game window
pygame.display.set_caption('Snake Eater')
game_window = pygame.display.set_mode((frame_size_x, frame_size_y))


# Colors (R, G, B)
black = pygame.Color(0, 0, 0)
white = pygame.Color(255, 255, 255)
red = pygame.Color(255, 0, 0)
green = pygame.Color(0, 255, 0)
blue = pygame.Color(0, 0, 255)


# FPS (frames per second) controller
fps_controller = pygame.time.Clock()


# Game variables
snake_pos = [100, 50]
snake_body = [[100, 50], [100-10, 50], [100-(2*10), 50]]

food_pos = [random.randrange(1, (frame_size_x//10)) * 10, random.randrange(1, (frame_size_y//10)) * 10]
food_spawn = True

direction = 'RIGHT'
change_to = direction

score = 0

#####################################################################

def restart_game():
	global snake_pos, snake_body, food_pos, food_spawn, direction, change_to, score
	snake_pos = [100, 50]
	snake_body = [[100, 50], [100-10, 50], [100-(2*10), 50]]

	food_pos = [int(frame_size_x/2), int(frame_size_y/2)]
	food_spawn = True

	direction = 'RIGHT'
	change_to = direction

	score = 0


#####################################################################
# Game Over
def game_over():
	my_font = pygame.font.SysFont('times new roman', 90)
	game_over_surface = my_font.render('YOU DIED', True, red)
	game_over_rect = game_over_surface.get_rect()
	game_over_rect.midtop = (frame_size_x/2, frame_size_y/4)
	game_window.fill(black)
	game_window.blit(game_over_surface, game_over_rect)
	show_score(0, red, 'times', 20)
	pygame.display.flip()
	time.sleep(3)
	pygame.quit()
	sys.exit()


# Score
def show_score(choice, color, font, size):
	score_font = pygame.font.SysFont(font, size)
	score_surface = score_font.render('Score : ' + str(score), True, color)
	score_rect = score_surface.get_rect()
	if choice == 1:
		score_rect.midtop = (frame_size_x/10, 15)
	else:
		score_rect.midtop = (frame_size_x/2, frame_size_y/1.25)
	game_window.blit(score_surface, score_rect)
	# pygame.display.flip()


#############################################################################
# Building the NN
direction_list = ['UP', 'DOWN', 'RIGHT', 'LEFT']

def reshape_layer(in_layer, out_dim):
    
    reshaping_matrix = K.ones((1, out_dim))
    return Lambda(lambda x: K.dot(x, reshaping_matrix))(in_layer)


###################################################
#model = Sequential()
#model.add(Conv2D(64, kernel_size=3, activation= 'relu', input_shape= (frame_size_x_simplified , frame_size_y_simplified ,1)))
#model.add(Conv2D(32, kernel_size=3, activation='relu'))
#model.add(Flatten())
#model.add(Dense(4, activation='linear'))
##################################################

input_layer = Input(shape = (frame_size_x_simplified , frame_size_y_simplified ,1))
conv1 = Conv2D(8, kernel_size = 3, activation = 'relu')(input_layer)
conv2 = Conv2D(8, kernel_size = 3, activation = 'relu')(conv1)
flatten = Flatten()(conv2)

V_stream_dense1 = Dense(4, activation = 'relu')(flatten)
#V_stream_dense2 = Dense(16, activation = 'relu')(V_stream_dense1)
V_stream_out = Dense(1, activation = 'linear')(V_stream_dense1)
V_stream = reshape_layer(V_stream_out, 4)

A_stream_dense1 = Dense(4, activation = 'relu')(flatten)
#A_stream_dense2 = Dense(16, activation = 'relu')(A_stream_dense1)
A_stream_out = Dense(4, activation = 'linear')(A_stream_dense1)

A_mean = Lambda(lambda x: K.mean(x, axis=1, keepdims = True))(A_stream_out)
#A_mean = reshape_layer(A_mean, 4)

A_stream = Subtract()([A_stream_out, A_mean])

Q_stream = Add()([V_stream, A_stream])

sgd = optimizers.SGD(lr=0.00001, clipnorm = 10)
model = Model(inputs= [input_layer], outputs = [Q_stream])
model.load_weights('snake.h5')
model.compile(loss='mean_squared_error', optimizer=sgd)
print(model.summary())


model_copy= keras.models.clone_model(model)
model_copy.compile(optimizer=sgd, loss='mse')
model_copy.set_weights(model.get_weights())


# e greedy policy

def e_greedy(Q_values, e = 0.1):

	p = np.random.uniform()

	if p<e:
		return np.random.randint(len(Q_values))

	else:
		return Q_values.argmax()
def softmax_sample(Q_values):

	p = np.random.uniform()

	exp_q = np.exp(Q_values)
	exp_sum = exp_q.sum()
	probs = exp_q/exp_sum

	action = 0
	accumulative_density = 0

	while(accumulative_density<1):
		accumulative_density = accumulative_density + probs[action]

		if accumulative_density > p:
			break
		action+=1

	return action

# Predict action 

def predict_action(state):

	global model
	
	Q_values = model.predict(np.array([state]))[0]
	
	#action =  e_greedy(Q_values)
	action = softmax_sample(Q_values)
	#print(action)
	return action


# Observate state

def get_state():

	global snake_body, snake_pos , food_pos
	
	state = np.zeros((frame_size_x_simplified, frame_size_y_simplified, 1))
	
	# head
	state[int(snake_pos[0]/10)][int(snake_pos[1]/10)][0] = 1

	# body
	for snake_x, snake_y in snake_body:
		snake_x = min(max(0, snake_x), frame_size_x-10)
		snake_y = min(max(0,snake_y ), frame_size_y-10)

		state[int(snake_x/10)][int(snake_y/10)][0] = 2

	# food
	state[int(food_pos[0]/10)][int(food_pos[1]/10)][0] = 3


	return state



# REPLAY BUFFER            
replay_buffer_size = 100000
replay_buffer = set()
state_dict = dict()
aux_value_dict = dict()
# Train

def train_nn(training_batch_size = 32, gamma = 0.99):
	global model, model_copy , replay_buffer, state_dict, aux_value_dict

	training_batch = random.sample(replay_buffer, min(training_batch_size, len(replay_buffer)))
	Q_target = list()

	s_list = []
	q_next_list =  model.predict(np.array([state_dict[x[0]] for x in training_batch]))
	counter = 0

	action_argmax = np.argmax(q_next_list, axis =1)
	#print(q_next_list)
	#print(action_argmax)

	for s,a,r,ss,terminal in training_batch:

		s_list.append(state_dict[s])

		if ss not in aux_value_dict.keys():
			ss_matrix = state_dict[ss]
			target = model_copy.predict(np.array([ss_matrix]))[0]
			aux_value_dict[ss] = target

		#q_next = model.predict(np.array([state_dict[ss]]))[0]
		q_next = q_next_list[counter]
		
		#q_aux = aux_value_dict[ss].max()
		q_aux = aux_value_dict[ss][action_argmax[counter]]
		counter+=1

		if terminal:
			q_aux = 0
		
		q_next[a] = r+ gamma * q_aux
		
		Q_target.append(q_next)


	s_list = np.array(s_list)
	Q_target = np.array(Q_target)
	model.fit(s_list, Q_target, verbose= 0)

def update_parameters():

	global model, model_copy, aux_value_dict, replay_buffer, state_dict
	

	model_copy.set_weights(model.get_weights())
	model.save_weights('snake.h5')

	#pickle.dump(replay_buffer, open('replay_buffer.pkl', 'wb'))
	#pickle.dump(state_dict, open('state_dict.pkl', 'wb'))
	aux_value_dict = dict()


#############################################################################
# 
episode_count = 0 
number_of_steps = 0 
average_score = 0 
scores = []
restart_game()

# Main logic
while True:
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			pygame.quit()
			sys.exit()
		# Whenever a key is pressed down
		elif event.type == pygame.KEYDOWN:
			# W -> Up; S -> Down; A -> Left; D -> Right
			if event.key == pygame.K_UP or event.key == ord('w'):
				change_to = 'UP'
			if event.key == pygame.K_DOWN or event.key == ord('s'):
				change_to = 'DOWN'
			if event.key == pygame.K_LEFT or event.key == ord('a'):
				change_to = 'LEFT'
			if event.key == pygame.K_RIGHT or event.key == ord('d'):
				change_to = 'RIGHT'
			# Esc -> Create event to quit the game
			if event.key == pygame.K_ESCAPE:
				pygame.event.post(pygame.event.Event(pygame.QUIT))

	s = get_state()
	a = predict_action(s)
	change_to = direction_list[a]
	#print(change_to)
	# Making sure the snake cannot move in the opposite direction instantaneously
	if change_to == 'UP' and direction != 'DOWN':
		direction = 'UP'
	if change_to == 'DOWN' and direction != 'UP':
		direction = 'DOWN'
	if change_to == 'LEFT' and direction != 'RIGHT':
		direction = 'LEFT'
	if change_to == 'RIGHT' and direction != 'LEFT':
		direction = 'RIGHT'

	# Moving the snake
	if direction == 'UP':
		snake_pos[1] -= 10
	if direction == 'DOWN':
		snake_pos[1] += 10
	if direction == 'LEFT':
		snake_pos[0] -= 10
	if direction == 'RIGHT':
		snake_pos[0] += 10

	reward = 0 
	# Snake body growing mechanism
	snake_body.insert(0, list(snake_pos))
	if snake_pos[0] == food_pos[0] and snake_pos[1] == food_pos[1]:
		score += 1
		reward += 1
		food_spawn = False
	else:
		snake_body.pop()

	# Spawning food on the screen
	if not food_spawn:
		#food_pos = [random.randrange(1, (frame_size_x//10)) * 10, random.randrange(1, (frame_size_y//10)) * 10]
		food_pos = [int(frame_size_x/2), int(frame_size_y/2)]
	food_spawn = True

	# GFX
	game_window.fill(black)
	for pos in snake_body:
		# Snake body
		# .draw.rect(play_surface, color, xy-coordinate)
		# xy-coordinate -> .Rect(x, y, size_x, size_y)
		pygame.draw.rect(game_window, green, pygame.Rect(pos[0], pos[1], 10, 10))

	# Snake food
	pygame.draw.rect(game_window, white, pygame.Rect(food_pos[0], food_pos[1], 10, 10))

	# Game Over conditions
	# Getting out of bounds
	terminal_state = False
	if snake_pos[0] < 0 or snake_pos[0] > frame_size_x-10:
		#game_over()
		terminal_state = True	
	if snake_pos[1] < 0 or snake_pos[1] > frame_size_y-10:
		#game_over()
		terminal_state = True	
	# Touching the snake body
	for block in snake_body[1:]:
		if snake_pos[0] == block[0] and snake_pos[1] == block[1]:
			#game_over()
			terminal_state = True

	snake_pos[0] = min(max(0, snake_pos[0] ), frame_size_x-10)
	snake_pos[1] = min(max(0, snake_pos[1] ), frame_size_y-10)

	ss = get_state()

	#global replay_buffer, state_dict

	s_hash = hash(s.tostring())
	ss_hash = hash(ss.tostring())
	

	state_dict[s_hash] = s
	state_dict[ss_hash] = ss

	replay_buffer.add((s_hash, a, reward , ss_hash, terminal_state))

	
	if len(replay_buffer) > replay_buffer_size:
		replay_buffer.pop()

	number_of_steps +=1
	train_nn()
	
	if terminal_state:

		scores.append(score)
		restart_game()
		episode_count+=1
		

		#average_score = (average_score*2+score)/3

		if episode_count%20==0:
			print('-----------------------------------------------')
			print('average rewardz ', np.array(scores).mean())
			#print('len replay_buffer ', len(replay_buffer))
			scores = []
		
	if number_of_steps %1000==0:
		update_parameters()
		print('number of steps ', number_of_steps)
		print('updating parameters')

	show_score(1, white, 'consolas', 20)
	# Refresh game screen
	pygame.display.update()
	# Refresh rate
	#fps_controller.tick(difficulty)