# import unittest
# from v5_3_high_pass_landing_function import height_to_ball, quadratic_equation_solver
# import math

# obs = {}

# obs["ball_direction"] = [0,0,0]
# obs["ball"] = [0,0,2]

# steps = height_to_ball(obs, 0, 0)

# final_pos = obs["ball"][2] - obs["ball_direction"][2] * steps - 0.0981 / 2  * steps**2 

# step2 = quadratic_equation_solver(-0.0981/2, 0, 0.9)

# final_pos2 = 0.9 - 0.0981 / 2  * step2**2 
# # print(final_pos2)

# #%%
# from v5_4_check_pass_is_blocked import check_block

# obs = {}

# obs["right_team"] = [[0, 0.1],[0, 0.2],[0.3,0.3]]

# # print(check_block(obs, 0,0, 0.2,0.4))

# #%%
# from v5_8_is_1plus_4plus_5 import check_block, check_steps, close_to_goalkeeper_shot

# obs = {}

# obs["right_team"] = [[0, 0.1],[0, 0.2],[0.3,0.3]]

# # print(check_block(obs, 0,0, 0.2,0.4))
#%%

# from v7_3_bad_angle_pass_function import bad_angle_short_pass

# def environment_fits(obs, player_x, player_y):
#     """ environment fits constraints """
#     # player is at bad angle to opponent's goal
#     upper_bound = lambda x: (x * 0.752 - 0.796)
#     lower_bound = lambda x: (-x * 0.752 + 0.796)
#     if player_y < upper_bound(player_x): return True
#     elif player_y > lower_bound(player_x): return True
#     else: return False

# obs = {}


# print(environment_fits(obs,1,0))

#%%
# import math
# def quadratic_equation_solver(a, b, c):
#     sol1 = (-b+math.sqrt(b**2 - 4*a*c))/2/a
#     sol2 = (-b-math.sqrt(b**2 - 4*a*c))/2/a
#     return max(sol1, sol2)

# print(0.015*15-0.001*(15)**2/2)

# from v8_1_combined_short_pass_with_others import close_to_opponent_pass


# obs = {}
# obs["active"] = 0
# obs["left_team_direction"] = [[1,1]]
# obs["left_team"] = [[1,1]]

# print(close_to_opponent_pass(obs, 0, 0)["environment_fits"])
# print(close_to_opponent_pass(obs, 0, 0)["get_action"])

# import math

# print(10**-1/10)

from v8 import is_blocked

# print(0.265**2 - 4*0.0981/2)
# print(quadratic_equation_solver(-0.0981/2, 0.265, 1))

# def is_blocked(obs, player_x, player_y, teammate_x, teammate_y, speed=PASS_SPEED, opponent_speed=1/2):
#     ball_pos = [player_x, player_y]
#     steps = check_steps(player_x, player_y, teammate_x, teammate_y, speed)
#     if steps == 0: return True
#     if steps != 0: unit_direction = [(teammate_x - player_x)/steps, (teammate_y - player_y)/steps]
#     for step in range(steps):
#         ball_pos[0] += unit_direction[0]
#         ball_pos[1] += unit_direction[1]
#         for opponent in range(0, len(obs["right_team"])):
#             distance_to_opponent = normal_get_distance(ball_pos[0], ball_pos[1], obs["right_team"][opponent][0], obs["right_team"][opponent][1]) + RUN_SPEED*step*opponent_speed
#             if distance_to_opponent < RUN_SPEED:
#                 return True
#     return False

# obs = {}
# obs["right_team"] = [[1, 0]]

# print(is_blocked(obs, 0.75, 0, 1, 0, opponent_speed=1))

from v9 import get_distance
import numpy as np
import math
# obs = {}
# obs["active"] = 0
# obs["left_team_direction"] = [[0.015, 0]]
# print(is_within_border(obs, 0.9, 0, 1, None))

RUN_SPEED = 0.015
ACCELERATION = 0.001

def is_within_border(obs, player_x, player_y, xborder = None, yborder = None):
    if xborder: 
        distance_to_xborder = abs(xborder - player_x)
        if xborder*obs["left_team_direction"][obs["active"]][0] < 0: distance_to_xborder = 999
        if xborder*player_x < 0: distance_to_xborder = 999
    else: distance_to_xborder = 999
    if yborder: 
        distance_to_yborder = abs(yborder - player_y)
        if yborder*obs["left_team_direction"][obs["active"]][1] < 0: distance_to_yborder = 999
        if yborder*player_y < 0: distance_to_yborder = 999
    else: distance_to_yborder = 999

    if xborder and yborder: acceleration_ = math.sqrt(((ACCELERATION)**2)/2)
    safe_steps_to_xborder = abs(obs["left_team_direction"][obs["active"]][0] / ACCELERATION)
    safe_steps_to_yborder = abs(obs["left_team_direction"][obs["active"]][1] / ACCELERATION)
    safe_distance_to_xborder = abs(obs["left_team_direction"][obs["active"]][0]*safe_steps_to_xborder - np.sign(obs["left_team_direction"][obs["active"]][0])*ACCELERATION/2*(safe_steps_to_xborder)**2)
    safe_distance_to_yborder = abs(obs["left_team_direction"][obs["active"]][1]*safe_steps_to_yborder - np.sign(obs["left_team_direction"][obs["active"]][1])*ACCELERATION/2*(safe_steps_to_yborder)**2)
    if distance_to_xborder < safe_distance_to_xborder + RUN_SPEED*2: 
        if xborder != None: return False
    if distance_to_yborder < safe_distance_to_yborder + RUN_SPEED*2: 
        if yborder != None: return False
    return True

obs = {}
obs["active"] = 0
obs["left_team_direction"] = [[1,-0.015]]

print(is_within_border(obs, 1, 0.41, xborder=1, yborder=None))
print(is_within_border(obs, -0.99, 0.41, xborder=1, yborder=-0.42))
print(is_within_border(obs, -0.99, 0.41, xborder=1, yborder=0.42))
print(is_within_border(obs, -0.99, 0.41, xborder=None, yborder=-0.42))
print(is_within_border(obs, -0.99, 0.41, xborder=None, yborder=0.42))
print(is_within_border(obs, -0.99, 0.41, xborder=-1, yborder=-0.42))
print(is_within_border(obs, -0.99, 0.41, xborder=-1, yborder=0.42))
print(is_within_border(obs, -0.99, 0.41, xborder=-1, yborder=None))


# is_within_border(obs, 0,0,1,0.42)