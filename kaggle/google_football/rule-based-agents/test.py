from kaggle_environments.envs.football.helpers import *
# from v9_1_combine_everything import *
import math
import random
import numpy as np

last_ball_owner = 0
RUN_SPEED = 0.015

LONG_STEPS = 8.2
HIGH_STEPS = 6.9
SHORT_STEPS = 8.2
SHOT_STEPS = 7.5

RUN_SPEED = 0.015
PASS_SPEED = 0.025
SHOT_SPEED = 0.031
PASS_Y_SPEED = 0.265

ACCELERATION = 0.001
UPPER_ACCELERATION = 0.0015

# def environment_fits(obs, player_x, player_y):
#     """ environment fits constraints """
#     # if anticipating: return False

#     right_score = 0
#     top_right_score = 0
#     bottom_right_score = 0
#     top_score = 0
#     bottom_score = 0
#     top_left_score = 0
#     bottom_left_score = 0
#     left_score = 0

#     for step in range(10):
#         future_x = player_x + obs["left_team_direction"][obs["active"]][0]*step
#         future_y = player_y + obs["left_team_direction"][obs["active"]][1]*step

#         horizontal_acceleration = ACCELERATION/2 * (step**2)
#         diagonal_acceleration = math.sqrt(((horizontal_acceleration)**2)/2)

#         # right_score += 1/10 * get_player_score(obs, future_x + horizontal_acceleration, future_y, steps=-2/6)
#         # top_right_score += 1/10 * get_player_score(obs, future_x + diagonal_acceleration, future_y - diagonal_acceleration, steps=-1/6)
#         # bottom_right_score += 1/10 * get_player_score(obs, future_x + diagonal_acceleration, future_y + diagonal_acceleration, steps=-1/6)
#         # top_score += 1/10 * get_player_score(obs, future_x, future_y - horizontal_acceleration, steps=0)
#         # bottom_score += 1/10 * get_player_score(obs, future_x, future_y + horizontal_acceleration, steps=0)
#         # top_left_score += 1/10 * get_player_score(obs, future_x - diagonal_acceleration, future_y - diagonal_acceleration, steps=1/6)
#         # bottom_left_score += 1/10 * get_player_score(obs, future_x - diagonal_acceleration, future_y + diagonal_acceleration, steps=1/6)
#         # left_score += 1/10 * get_player_score(obs, future_x - horizontal_acceleration, future_y, steps=2/6)

#         right_score += 1/10 * get_player_score(obs, future_x + horizontal_acceleration, future_y, steps=-2/6, routing=True)
#         top_right_score += 1/10 * get_player_score(obs, future_x + diagonal_acceleration, future_y - diagonal_acceleration, steps=-1/6, routing=True)
#         bottom_right_score += 1/10 * get_player_score(obs, future_x + diagonal_acceleration, future_y + diagonal_acceleration, steps=-1/6, routing=True)
#         top_score += 1/10 * get_player_score(obs, future_x, future_y - horizontal_acceleration, steps=0, routing=True)
#         bottom_score += 1/10 * get_player_score(obs, future_x, future_y + horizontal_acceleration, steps=0, routing=True)
#         top_left_score += 1/10 * get_player_score(obs, future_x - diagonal_acceleration, future_y - diagonal_acceleration, steps=1/6, routing=True)
#         bottom_left_score += 1/10 * get_player_score(obs, future_x - diagonal_acceleration, future_y + diagonal_acceleration, steps=1/6, routing=True)
#         left_score += 1/10 * get_player_score(obs, future_x - horizontal_acceleration, future_y, steps=2/6, routing=True)
#     possible_movement = [-999, -999, -999, -999, -999, -999, -999, -999]
    

#     print(right_score)
#     print(top_right_score)
#     print(bottom_right_score)
#     print(top_score)
#     print(bottom_score)
#     print(top_left_score)
#     print(bottom_left_score)
#     print(left_score)


#     if is_within_border(obs, future_x, future_y, xborder=1, yborder=None): possible_movement[0] = right_score
#     if is_within_border(obs, future_x, future_y, xborder=1, yborder=-0.42): possible_movement[1] = top_right_score
#     if is_within_border(obs, future_x, future_y, xborder=1, yborder=0.42): possible_movement[2] = bottom_right_score
#     if is_within_border(obs, future_x, future_y, xborder=None, yborder=-0.42): possible_movement[3] = top_score
#     if is_within_border(obs, future_x, future_y, xborder=None, yborder=0.42): possible_movement[4] = bottom_score
#     if is_within_border(obs, future_x, future_y, xborder=-1, yborder=-0.42): possible_movement[5] = top_left_score
#     if is_within_border(obs, future_x, future_y, xborder=-1, yborder=0.42): possible_movement[6] = bottom_left_score
#     if is_within_border(obs, future_x, future_y, xborder=-1, yborder=None): possible_movement[7] = left_score

#     if max(possible_movement) == -999: return False

#     best_movement = np.argmax(possible_movement)

#     if best_movement == 0: obs["memory_patterns"]["go_around_opponent"] = Action.Right
#     if best_movement == 1: obs["memory_patterns"]["go_around_opponent"] = Action.TopRight
#     if best_movement == 2: obs["memory_patterns"]["go_around_opponent"] = Action.BottomRight
#     if best_movement == 3: obs["memory_patterns"]["go_around_opponent"] = Action.Top
#     if best_movement == 4: obs["memory_patterns"]["go_around_opponent"] = Action.Bottom
#     if best_movement == 5: obs["memory_patterns"]["go_around_opponent"] = Action.TopLeft
#     if best_movement == 6: obs["memory_patterns"]["go_around_opponent"] = Action.BottomLeft
#     if best_movement == 7: obs["memory_patterns"]["go_around_opponent"] = Action.Left

#     if not is_within_border(obs, player_x + obs["left_team_direction"][obs["active"]][0]*3, 0, xborder=-1, yborder=None) or not is_within_border(obs, obs["ball"][0] + obs["ball_direction"][0]*3, 0, xborder=1, yborder=None): 
#         if obs["left_team_direction"][obs["active"]][1] < 0: obs["memory_patterns"]["go_around_opponent"] = Action.TopRight
#         if obs["left_team_direction"][obs["active"]][1] > 0: obs["memory_patterns"]["go_around_opponent"] = Action.BottomRight
#     if not is_within_border(obs, player_x + obs["left_team_direction"][obs["active"]][0]*3, 0, xborder=1, yborder=None) or not is_within_border(obs, obs["ball"][0] + obs["ball_direction"][0]*3, 0, xborder=1, yborder=None):
#         if obs["left_team_direction"][obs["active"]][1] < 0: obs["memory_patterns"]["go_around_opponent"] = Action.TopLeft
#         if obs["left_team_direction"][obs["active"]][1] > 0: obs["memory_patterns"]["go_around_opponent"] = Action.BottomLeft
#     if not is_within_border(obs, 0, player_y + obs["left_team_direction"][obs["active"]][1]*3, xborder=None, yborder=0.42) or not is_within_border(obs, 0, obs["ball"][1] + obs["ball_direction"][1]*3, xborder=None, yborder=0.42): 
#         if obs["left_team_direction"][obs["active"]][0] < 0: obs["memory_patterns"]["go_around_opponent"] = Action.TopLeft
#         if obs["left_team_direction"][obs["active"]][0] > 0: obs["memory_patterns"]["go_around_opponent"] = Action.TopRight
#     if not is_within_border(obs, 0, player_y + obs["left_team_direction"][obs["active"]][1]*3, xborder=None, yborder=-0.42) or not is_within_border(obs, 0, obs["ball"][1] + obs["ball_direction"][1]*3, xborder=None, yborder=-0.42): 
#         if obs["left_team_direction"][obs["active"]][0] < 0: obs["memory_patterns"]["go_around_opponent"] = Action.BottomLeft
#         if obs["left_team_direction"][obs["active"]][0] > 0: obs["memory_patterns"]["go_around_opponent"] = Action.BottomRight

#     return True
a = [-88 for i in range(100)]
print(a)
a[0] = 1
print(a)

# obs = {}
# obs["memory_patterns"] = {}
# obs["active"] = 0
# obs["left_team_direction"] = [[0,-0.1]]
# obs["ball"] = [0,0]
# obs["ball_direction"] = [0,0.1]
# obs["right_team"] = [[0,0.1],[0,0.1],[0.01,0.11],[0.2,0.13]]
# print(environment_fits(obs, 0, 0.41))
# print(obs["memory_patterns"]["go_around_opponent"])



