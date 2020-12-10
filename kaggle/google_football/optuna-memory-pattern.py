# start executing cells from here to rewrite submission.py

from kaggle_environments.envs.football.helpers import *
import math
import random
import numpy as np

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

last_ball_owner = -1

OPPONENT_SPEED = 3/4
GET_DISTANCE_SCALING2 = 1
GET_DISTANCE_SCALING = 2.38
RATIO = 1/6

def find_patterns(obs, player_x, player_y):
    """ find list of appropriate patterns in groups of memory patterns """
    for get_group in groups_of_memory_patterns:
        group = get_group(obs, player_x, player_y)
        if group["environment_fits"](obs, player_x, player_y):
            return group["get_memory_patterns"](obs, player_x, player_y)

def get_action_of_agent(obs, player_x, player_y):
    """ get action of appropriate pattern in agent's memory """
    memory_patterns = find_patterns(obs, player_x, player_y)
    # find appropriate pattern in list of memory patterns
    for get_pattern in memory_patterns:
        pattern = get_pattern(obs, player_x, player_y)
        if pattern["environment_fits"](obs, player_x, player_y):
            return pattern["get_action"](obs, player_x, player_y)

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

def get_average_distance_to_opponents(obs, player_x, player_y):
    """ get average distance to closest opponents """
    distances_sum = 0
    distances_amount = 0
    for i in range(1, len(obs["right_team"])):
        # if opponent is ahead of player
        if obs["right_team"][i][0] > (player_x - 0.02):
            distance_to_opponent = get_distance(player_x, player_y, obs["right_team"][i][0], obs["right_team"][i][1])
            if distance_to_opponent < 0.03:
                distances_sum += distance_to_opponent
                distances_amount += 1
    # if there is no opponents close around
    if distances_amount == 0:
        return 2, distances_amount
    return distances_sum / distances_amount, distances_amount

def distance_to_opponent(obs, player_x, player_y, steps=0):
    """ get average distance to closest opponents """
    distances_score = 0
    for i in range(1, len(obs["right_team"])):
        damping_factor = 1
        if obs["right_team"][i][0] < (player_x): damping_factor *= 0.8
        if abs(player_y) < abs(obs["right_team"][i][1]): damping_factor *= 0.8
        distance_to_opponent = get_distance(player_x, player_y, obs["right_team"][i][0], obs["right_team"][i][1])

        step_to_opponent = distance_to_opponent/RUN_SPEED
        distances_score -= 10**((10-step_to_opponent+steps*OPPONENT_SPEED)*damping_factor/10)

    return distances_score

def distance_to_goal(obs, player_x, player_y, steps=0):
    distance_to_goal = get_distance(player_x, player_y, 0.9, 0)
    # opponent_center_inverse_score = get_distance(player_x, player_y, 0.5, 0)
    distance_to_center = get_distance(player_x, player_y, 0, 0)

    unit_steps = (steps*RUN_SPEED)
    goal_score = 10**(1-distance_to_goal+unit_steps)
    center_inverse_score = 10**(-1+distance_to_center+unit_steps)
    distances_score = goal_score/3+center_inverse_score*2/3
    return distances_score

def distance_to_corner(obs, player_x, player_y, steps=0):
    distance_to_corner_up = get_distance(player_x, player_y, 1, -0.42)
    distance_to_goal_mid = get_distance(player_x, player_y, 0.9, 0)
    distance_to_corner_down = get_distance(player_x, player_y, 1, 0.42)
    unit_steps = (steps*RUN_SPEED)

    future_y = player_y + obs["left_team_direction"][obs["active"]][1]
    
    if future_y < -0.42/6: 
        distance_to_corner = distance_to_corner_up
        distances_score = 10**(1-distance_to_corner+unit_steps)
    if future_y > 0.42/6: 
        distance_to_corner = distance_to_corner_down
        distances_score = 10**(1-distance_to_corner+unit_steps)
    else: 
        distance_to_corner = distance_to_goal_mid
        distances_score = 10**(1-distance_to_corner+unit_steps)
    return distances_score

def get_player_score(obs, player_x, player_y, steps=0, routing=False):
    player_score = 0
    player_score += distance_to_opponent(obs, player_x, player_y, steps)
    if not routing: player_score += distance_to_goal(obs, player_x, player_y, steps)
    if routing: player_score += distance_to_corner(obs, player_x, player_y, steps)
    return player_score

def evaluation(obs, player_x, player_y, teammate_x, teammate_y, passing_steps=SHORT_STEPS):
    if is_blocked(obs, player_x, player_y, teammate_x, teammate_y, speed=passing_steps): return False
    if is_illigal_move(obs, teammate_x, teammate_y): return False
    
    player_steps_to_teammate = check_steps(player_x, player_y, teammate_x, teammate_y)

    player_score = get_player_score(obs, player_x, player_y, steps=0)
    teammate_score = get_player_score(obs, teammate_x, teammate_y, steps=player_steps_to_teammate)

    if teammate_score > player_score: return teammate_score
    else: return False

def is_illigal_move(obs, player_x, player_y):
    player_is_rightest = True
    for i in range(1, len(obs["right_team"])):
        if obs["right_team"][i][0] > player_x + RUN_SPEED:
            player_is_rightest = False
    if player_is_rightest: return True
    else: return False

def check_steps(player_x, player_y, teammate_x, teammate_y, speed=PASS_SPEED):
    distance = normal_get_distance(player_x, player_y, teammate_x, teammate_y)
    steps = int(round(distance / speed))
    return steps

def is_blocked(obs, player_x, player_y, teammate_x, teammate_y, speed=PASS_SPEED, opponent_speed=OPPONENT_SPEED, high_pass=False):
    ball_pos = [player_x, player_y]
    if high_pass: 
        height_steps = int(round(1/PASS_Y_SPEED))
        steps = check_steps(player_x, player_y, teammate_x, teammate_y, speed)
        if steps == 0: return True
        if steps != 0: unit_direction = [(teammate_x - player_x)/steps, (teammate_y - player_y)/steps]
        ball_pos_ = ball_pos[:]
        for step in range(height_steps+1):
            ball_pos[0] += unit_direction[0]
            ball_pos[1] += unit_direction[1]
            for opponent in range(0, len(obs["right_team"])):
                distance_to_opponent = normal_get_distance(ball_pos[0], ball_pos[1], obs["right_team"][opponent][0], obs["right_team"][opponent][1]) - RUN_SPEED*step*opponent_speed
                if distance_to_opponent < RUN_SPEED:
                    return True
        for step in range(steps-height_steps,steps+1):
            ball_pos[0] = ball_pos_[0] + step * unit_direction[0]
            ball_pos[1] = ball_pos_[1] + step * unit_direction[1]
            for opponent in range(0, len(obs["right_team"])):
                distance_to_opponent = normal_get_distance(ball_pos[0], ball_pos[1], obs["right_team"][opponent][0], obs["right_team"][opponent][1]) - RUN_SPEED*step*opponent_speed
                if distance_to_opponent < RUN_SPEED:
                    return True

    if not high_pass:
        steps = check_steps(player_x, player_y, teammate_x, teammate_y, speed)
        if steps == 0: return True
        if steps != 0: unit_direction = [(teammate_x - player_x)/steps, (teammate_y - player_y)/steps]
        for step in range(steps+1):
            ball_pos[0] += unit_direction[0]
            ball_pos[1] += unit_direction[1]
            for opponent in range(0, len(obs["right_team"])):
                distance_to_opponent = normal_get_distance(ball_pos[0], ball_pos[1], obs["right_team"][opponent][0], obs["right_team"][opponent][1]) - RUN_SPEED*step*opponent_speed
                if distance_to_opponent < RUN_SPEED:
                    return True
    return False

def normal_get_distance(x1, y1, x2, y2):
    """ get two-dimensional Euclidean distance, considering y size of the field """
    return math.sqrt((x1 - x2) ** 2 + (y1 * GET_DISTANCE_SCALING2 - y2 * GET_DISTANCE_SCALING2) ** 2)

def get_distance(x1, y1, x2, y2):
    """ get two-dimensional Euclidean distance, considering y size of the field """
    return math.sqrt((x1 - x2) ** 2 + (y1 * GET_DISTANCE_SCALING - y2 * GET_DISTANCE_SCALING) ** 2)
    
def quadratic_equation_solver(a, b, c):
    sol1 = (-b+math.sqrt(b**2 - 4*a*c))/2/a
    sol2 = (-b-math.sqrt(b**2 - 4*a*c))/2/a
    return max(sol1, sol2)

def height_to_ball(obs, player_x, player_y):
    steps = quadratic_equation_solver(-0.0981/2, obs["ball_direction"][2], (obs["ball"][2] - 1.1))
    return steps

def bad_angle_short_pass(obs, player_x, player_y):
    """ perform a short pass, if player is at bad angle to opponent's goal """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # player is at bad angle to opponent's goal
        if player_x > (1-0.1125) and abs(player_y) > (0.42-0.1125): return True
        else: return False
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        if player_y > 0:
            if Action.Top not in obs["sticky_actions"]:
                return Action.Top
        else:
            if Action.Bottom not in obs["sticky_actions"]:
                return Action.Bottom
        if Action.Sprint in obs["sticky_actions"]:
            return Action.ReleaseSprint
        if is_blocked(obs, player_x, player_y, 1, 0, SHOT_SPEED) and is_blocked(obs, player_x, player_y, 1, 0.044, SHOT_SPEED) and is_blocked(obs, player_x, player_y, 1, -0.044, SHOT_SPEED): return Action.HighPass
        else: return Action.LongPass
    
    return {"environment_fits": environment_fits, "get_action": get_action}

def close_to_goalkeeper_shot(obs, player_x, player_y):
    """ shot if close to the goalkeeper """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        player_speed = math.sqrt(obs["left_team_direction"][obs["active"]][0]**2 + obs["left_team_direction"][obs["active"]][1]**2)
        goalkeeper_speed = math.sqrt(obs["right_team_direction"][0][0]**2 + obs["right_team_direction"][0][1]**2)
        for step in range(9):
            goalkeeper_x = obs["right_team"][0][0] + obs["right_team_direction"][0][0] * step
            goalkeeper_y = obs["right_team"][0][1] + obs["right_team_direction"][0][1] * step
            player_x_ = player_x + obs["left_team_direction"][obs["active"]][0] * step
            player_y_ = player_y + obs["left_team_direction"][obs["active"]][1] * step
            # player located close to the goalkeeper
            if get_distance(player_x_, player_y_, goalkeeper_x, goalkeeper_y) < 0.25:
                return True
        if get_distance(player_x, player_y, 1, 0) <= 0.5:
            if not is_blocked(obs, player_x, player_y, 1, 0, SHOT_SPEED):
                return True
            elif not is_blocked(obs, player_x, player_y, 1, 0.044, SHOT_SPEED):
                return True
            elif not is_blocked(obs, player_x, player_y, 1, -0.044, SHOT_SPEED):
                return True
        if get_distance(player_x, player_y, 1, 0) <= 0.25: return True
        return False
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        if player_y > 0 and player_y < 0.03:
            if Action.BottomRight not in obs["sticky_actions"]:
                return Action.BottomRight
        if player_y < 0 and player_y > -0.03:
            if Action.TopRight not in obs["sticky_actions"]:
                return Action.TopRight
        if Action.Sprint in obs["sticky_actions"]:
            return Action.ReleaseSprint
        return Action.Shot
    
    return {"environment_fits": environment_fits, "get_action": get_action}

def close_to_opponent_pass(obs, player_x, player_y):
    """ perform a short pass, if close to opponent's player and close to teammate """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        short_pass = [-999]
        long_pass = [-999]
        high_pass = [-999]
        for j in range(0, len(obs["left_team"])):
            check1 = check2 = check3 = check4 = False
            if obs["left_team_direction"][obs["active"]][0] > 0 and obs["left_team"][j][0] > player_x: check1 = True
            if obs["left_team_direction"][obs["active"]][0] < 0 and obs["left_team"][j][0] < player_x: check2 = True
            if obs["left_team_direction"][obs["active"]][1] > 0 and obs["left_team"][j][1] > player_y: check3 = True
            if obs["left_team_direction"][obs["active"]][1] < 0 and obs["left_team"][j][1] < player_y: check4 = True
            if check1 or check2 or check3 or check4:
                distance_to_teammate = get_distance(player_x, player_y, obs["left_team"][j][0], obs["left_team"][j][1])
                if distance_to_teammate < 1/2:
                    evaluation_score = evaluation(obs, player_x, player_y, obs["left_team"][j][0], obs["left_team"][j][1], passing_steps=SHORT_STEPS)
                    if evaluation_score: short_pass.append(evaluation_score)

                if distance_to_teammate < 1 and distance_to_teammate > 1/2:
                    evaluation_score = evaluation(obs, player_x, player_y, obs["left_team"][j][0], obs["left_team"][j][1], passing_steps=LONG_STEPS)
                    if evaluation_score: 
                        if j != 0: long_pass.append(evaluation_score)
                    
                if distance_to_teammate < 1.5 and distance_to_teammate > 1/2:
                    evaluation_score =  evaluation(obs, player_x, player_y, obs["left_team"][j][0], obs["left_team"][j][1], passing_steps=HIGH_STEPS)
                    if evaluation_score: 
                        if j != 0: high_pass.append(evaluation_score)
        comparison_list = [max(short_pass), max(long_pass), max(high_pass)]
        
        if max(comparison_list) == -999: return False
        
        best_action = np.argmax(comparison_list)
        if best_action == 0: obs["memory_patterns"]["pass"] = Action.ShortPass
        if best_action == 1: obs["memory_patterns"]["pass"] = Action.LongPass
        if best_action == 2: obs["memory_patterns"]["pass"] = Action.HighPass
        return True
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        if Action.Sprint in obs["sticky_actions"]:
            return Action.ReleaseSprint
        return obs["memory_patterns"]["pass"]

    return {"environment_fits": environment_fits, "get_action": get_action}

def far_from_goal_shot(obs, player_x, player_y):
    """ perform a shot, if far from opponent's goal """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # player is far from opponent's goal or it's the goalkeeper
        if player_x < -0.6 or obs["ball_owned_player"] == 0:
            return True
        return False
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        if Action.Sprint in obs["sticky_actions"]:
            return Action.ReleaseSprint
        return Action.Shot
    
    return {"environment_fits": environment_fits, "get_action": get_action}

def far_from_goal_high_pass(obs, player_x, player_y):
    """ perform a high pass, if far from opponent's goal """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # player is far from opponent's goal or it's the goalkeeper
        if player_x < -0.3 or obs["ball_owned_player"] == 0:
            return True
        return False
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        if Action.Right not in obs["sticky_actions"]:
            return Action.Right
        if Action.Sprint in obs["sticky_actions"]:
            return Action.ReleaseSprint
        return Action.HighPass
    
    return {"environment_fits": environment_fits, "get_action": get_action}

def go_through_opponents(obs, player_x, player_y):
    """ avoid closest opponents by going around them """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # if anticipating: return False

        right_score = 0
        top_right_score = 0
        bottom_right_score = 0
        top_score = 0
        bottom_score = 0
        top_left_score = 0
        bottom_left_score = 0
        left_score = 0

        for step in range(10):
            future_x = player_x + obs["left_team_direction"][obs["active"]][0]*step
            future_y = player_y + obs["left_team_direction"][obs["active"]][1]*step

            horizontal_acceleration = ACCELERATION/2 * (step**2)
            diagonal_acceleration = math.sqrt(((horizontal_acceleration)**2)/2)

            right_score += 1/10 * get_player_score(obs, future_x + horizontal_acceleration, future_y, steps=-2*RATIO, routing=True)
            top_right_score += 1/10 * get_player_score(obs, future_x + diagonal_acceleration, future_y - diagonal_acceleration, steps=-1*RATIO, routing=True)
            bottom_right_score += 1/10 * get_player_score(obs, future_x + diagonal_acceleration, future_y + diagonal_acceleration, steps=-1*RATIO, routing=True)
            top_score += 1/10 * get_player_score(obs, future_x, future_y - horizontal_acceleration, steps=0, routing=True)
            bottom_score += 1/10 * get_player_score(obs, future_x, future_y + horizontal_acceleration, steps=0, routing=True)
            top_left_score += 1/10 * get_player_score(obs, future_x - diagonal_acceleration, future_y - diagonal_acceleration, steps=1*RATIO, routing=True)
            bottom_left_score += 1/10 * get_player_score(obs, future_x - diagonal_acceleration, future_y + diagonal_acceleration, steps=1*RATIO, routing=True)
            left_score += 1/10 * get_player_score(obs, future_x - horizontal_acceleration, future_y, steps=2*RATIO, routing=True)

        possible_movement = [-999, -999, -999, -999, -999, -999, -999, -999]
        
        if is_within_border(obs, future_x, future_y, xborder=1, yborder=None): possible_movement[0] = right_score
        if is_within_border(obs, future_x, future_y, xborder=1, yborder=-0.42): possible_movement[1] = top_right_score
        if is_within_border(obs, future_x, future_y, xborder=1, yborder=0.42): possible_movement[2] = bottom_right_score
        if is_within_border(obs, future_x, future_y, xborder=None, yborder=-0.42): possible_movement[3] = top_score
        if is_within_border(obs, future_x, future_y, xborder=None, yborder=0.42): possible_movement[4] = bottom_score
        if is_within_border(obs, future_x, future_y, xborder=-1, yborder=-0.42): possible_movement[5] = top_left_score
        if is_within_border(obs, future_x, future_y, xborder=-1, yborder=0.42): possible_movement[6] = bottom_left_score
        if is_within_border(obs, future_x, future_y, xborder=-1, yborder=None): possible_movement[7] = left_score

        if max(possible_movement) == -999: return False

        best_movement = np.argmax(possible_movement)

        if best_movement == 0: obs["memory_patterns"]["go_around_opponent"] = Action.Right
        if best_movement == 1: obs["memory_patterns"]["go_around_opponent"] = Action.TopRight
        if best_movement == 2: obs["memory_patterns"]["go_around_opponent"] = Action.BottomRight
        if best_movement == 3: obs["memory_patterns"]["go_around_opponent"] = Action.Top
        if best_movement == 4: obs["memory_patterns"]["go_around_opponent"] = Action.Bottom
        if best_movement == 5: obs["memory_patterns"]["go_around_opponent"] = Action.TopLeft
        if best_movement == 6: obs["memory_patterns"]["go_around_opponent"] = Action.BottomLeft
        if best_movement == 7: obs["memory_patterns"]["go_around_opponent"] = Action.Left

        if not is_within_border(obs, player_x + obs["left_team_direction"][obs["active"]][0]*3, 0, xborder=-1, yborder=None) or not is_within_border(obs, obs["ball"][0] + obs["ball_direction"][0]*3, 0, xborder=1, yborder=None): 
            if obs["left_team_direction"][obs["active"]][1] < 0: obs["memory_patterns"]["go_around_opponent"] = Action.TopRight
            if obs["left_team_direction"][obs["active"]][1] > 0: obs["memory_patterns"]["go_around_opponent"] = Action.BottomRight
        if not is_within_border(obs, player_x + obs["left_team_direction"][obs["active"]][0]*3, 0, xborder=1, yborder=None) or not is_within_border(obs, obs["ball"][0] + obs["ball_direction"][0]*3, 0, xborder=1, yborder=None):
            if obs["left_team_direction"][obs["active"]][1] < 0: obs["memory_patterns"]["go_around_opponent"] = Action.TopLeft
            if obs["left_team_direction"][obs["active"]][1] > 0: obs["memory_patterns"]["go_around_opponent"] = Action.BottomLeft
        if not is_within_border(obs, 0, player_y + obs["left_team_direction"][obs["active"]][1]*3, xborder=None, yborder=0.42) or not is_within_border(obs, 0, obs["ball"][1] + obs["ball_direction"][1]*3, xborder=None, yborder=0.42): 
            if obs["left_team_direction"][obs["active"]][0] < 0: obs["memory_patterns"]["go_around_opponent"] = Action.TopLeft
            if obs["left_team_direction"][obs["active"]][0] > 0: obs["memory_patterns"]["go_around_opponent"] = Action.TopRight
        if not is_within_border(obs, 0, player_y + obs["left_team_direction"][obs["active"]][1]*3, xborder=None, yborder=-0.42) or not is_within_border(obs, 0, obs["ball"][1] + obs["ball_direction"][1]*3, xborder=None, yborder=-0.42): 
            if obs["left_team_direction"][obs["active"]][0] < 0: obs["memory_patterns"]["go_around_opponent"] = Action.BottomLeft
            if obs["left_team_direction"][obs["active"]][0] > 0: obs["memory_patterns"]["go_around_opponent"] = Action.BottomRight

        return True
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        if Action.Sprint not in obs["sticky_actions"]:
            return Action.Sprint
        return obs["memory_patterns"]["go_around_opponent"]
    
    return {"environment_fits": environment_fits, "get_action": get_action}

def khorne_slide(obs, berzerker_x, berzerker_y):
    """ BLOOD FOR THE BLOOD GOD!!! SKULLS FOR THE SKULL THRONE!!! """
    def environment_fits(obs, berzerker_x, berzerker_y):
        """ environment fits constraints """
        # if prey has the ball
        if obs["left_team_yellow_card"][obs["active"]] != 1:
            if obs["ball_owned_team"] == 1:
                prey_x = obs["right_team"][obs["ball_owned_player"]][0]
                prey_y = obs["right_team"][obs["ball_owned_player"]][1]
                # by x position, amount of berzerker's team players between prey and goal of berzerker's team
                players_amount = 0
                nearby_players = 0
                for i in range(1, len(obs["left_team"])):
                    if obs["left_team"][i][0] < prey_x:
                        players_amount += 1
                    if get_distance(obs["left_team"][i][0], obs["left_team"][i][1], prey_x, prey_y) < RUN_SPEED*10: nearby_players += 1
                prey_x_direction = obs["right_team_direction"][obs["ball_owned_player"]][0]
                future_prey_x = prey_x + obs["right_team_direction"][obs["ball_owned_player"]][0]
                future_prey_y = prey_y + obs["right_team_direction"][obs["ball_owned_player"]][1]
                future_berzerker_x = berzerker_x + obs["left_team_direction"][obs["active"]][0]
                future_berzerker_y = berzerker_y + obs["left_team_direction"][obs["active"]][1]
                distance_to_prey = get_distance(berzerker_x, berzerker_y, prey_x, prey_y)
                future_distance_to_prey = get_distance(future_berzerker_x, future_berzerker_y, future_prey_x, future_prey_y)
                # if berzerker is not close to his own penalty zone
                # and prey is beyond x position of too many players of berzerker's team
                # and berzerker is close enough to prey
                # and berzerker is running in direction of prey
                if ((berzerker_x > -0.65 or abs(berzerker_y) > 0.3) and
                        players_amount <= 7 and
                        nearby_players < 2 and
                        future_distance_to_prey < 0.015 and
                        distance_to_prey > future_distance_to_prey):
                    return True
            return False
        
    def get_action(obs, berzerker_x, berzerker_y):
        """ get action of this memory pattern """
        return Action.Slide
    
    return {"environment_fits": environment_fits, "get_action": get_action}

def run_to_ball_bottom(obs, player_x, player_y):
    """ run to the ball if it is to the bottom from player's position """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # ball is to the bottom from player's position
        if (obs["ball"][1] > player_y and
                abs(obs["ball"][0] - player_x) < 0.01):
            return True
        return False
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        return Action.Bottom
    
    return {"environment_fits": environment_fits, "get_action": get_action}

def run_to_ball_bottom_left(obs, player_x, player_y):
    """ run to the ball if it is to the bottom left from player's position """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # ball is to the bottom left from player's position
        if (obs["ball"][0] < player_x and
                obs["ball"][1] > player_y):
            return True
        return False
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        return Action.BottomLeft
    
    return {"environment_fits": environment_fits, "get_action": get_action}

def run_to_ball_bottom_right(obs, player_x, player_y):
    """ run to the ball if it is to the bottom right from player's position """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # ball is to the bottom right from player's position
        if (obs["ball"][0] > player_x and
                obs["ball"][1] > player_y):
            return True
        return False
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        return Action.BottomRight
    
    return {"environment_fits": environment_fits, "get_action": get_action}

def run_to_ball_left(obs, player_x, player_y):
    """ run to the ball if it is to the left from player's position """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # ball is to the left from player's position
        if (obs["ball"][0] < player_x and
                abs(obs["ball"][1] - player_y) < 0.01):
            return True
        return False
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        return Action.Left
    
    return {"environment_fits": environment_fits, "get_action": get_action}

def run_to_ball_right(obs, player_x, player_y):
    """ run to the ball if it is to the right from player's position """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # ball is to the right from player's position
        if (obs["ball"][0] > player_x and
                abs(obs["ball"][1] - player_y) < 0.01):
            return True
        return False
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        return Action.Right
    
    return {"environment_fits": environment_fits, "get_action": get_action}

def run_to_ball_top(obs, player_x, player_y):
    """ run to the ball if it is to the top from player's position """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # ball is to the top from player's position
        if (obs["ball"][1] < player_y and
                abs(obs["ball"][0] - player_x) < 0.01):
            return True
        return False
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        return Action.Top
    
    return {"environment_fits": environment_fits, "get_action": get_action}

def run_to_ball_top_left(obs, player_x, player_y):
    """ run to the ball if it is to the top left from player's position """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # ball is to the top left from player's position
        if (obs["ball"][0] < player_x and
                obs["ball"][1] < player_y):
            return True
        return False
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        return Action.TopLeft
    
    return {"environment_fits": environment_fits, "get_action": get_action}

def run_to_ball_top_right(obs, player_x, player_y):
    """ run to the ball if it is to the top right from player's position """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # ball is to the top right from player's position
        if (obs["ball"][0] > player_x and
                obs["ball"][1] < player_y):
            return True
        return False
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        return Action.TopRight
    
    return {"environment_fits": environment_fits, "get_action": get_action}

def idle(obs, player_x, player_y):
    """ do nothing, release all sticky actions """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        return True
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        return Action.Idle
    
    return {"environment_fits": environment_fits, "get_action": get_action}

def start_sprinting(obs, player_x, player_y):
    """ make sure player is sprinting """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        if Action.Sprint not in obs["sticky_actions"]:
            return True
        return False
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        return Action.Sprint
    
    return {"environment_fits": environment_fits, "get_action": get_action}
# "%%writefile -a submission.py" will append the code below to submission.py,
# it WILL NOT rewrite submission.py

def corner(obs, player_x, player_y):
    """ perform a shot in corner game mode """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # it is corner game mode
        if obs['game_mode'] == GameMode.Corner:
            return True
        return False
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        if player_y > 0:
            if Action.TopRight not in obs["sticky_actions"]:
                return Action.TopRight
        else:
            if Action.BottomRight not in obs["sticky_actions"]:
                return Action.BottomRight
        return Action.HighPass # check if it's not blocked 
    
    return {"environment_fits": environment_fits, "get_action": get_action}

def free_kick(obs, player_x, player_y):
    """ perform a high pass or a shot in free kick game mode """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # it is free kick game mode
        if obs['game_mode'] == GameMode.FreeKick:
            return True
        return False
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        # shot if player close to goal
        if player_x > 0.75:
            if player_y > 0:
                if Action.TopRight not in obs["sticky_actions"]:
                    return Action.TopRight
            else:
                if Action.BottomRight not in obs["sticky_actions"]:
                    return Action.BottomRight
            return Action.Shot
        # high pass if player far from goal
        else:
            if player_y > 0:
                if Action.TopRight not in obs["sticky_actions"]:
                    return Action.TopRight
            else:
                if Action.BottomRight not in obs["sticky_actions"]:
                    return Action.BottomRight
            return Action.ShortPass
    
    return {"environment_fits": environment_fits, "get_action": get_action}

def goal_kick(obs, player_x, player_y):
    """ perform a short pass in goal kick game mode """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # it is goal kick game mode
        if obs['game_mode'] == GameMode.GoalKick:
            return True
        return False
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        if (random.random() < 0.5 and
                Action.TopRight not in obs["sticky_actions"] and
                Action.BottomRight not in obs["sticky_actions"]):
            return Action.TopRight
        else:
            if Action.BottomRight not in obs["sticky_actions"]:
                return Action.BottomRight
        return Action.ShortPass
    
    return {"environment_fits": environment_fits, "get_action": get_action}

def kick_off(obs, player_x, player_y):
    """ perform a short pass in kick off game mode """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # it is kick off game mode
        if obs['game_mode'] == GameMode.KickOff:
            return True
        return False
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        if player_y > 0:
            if Action.Top not in obs["sticky_actions"]:
                return Action.Top
        else:
            if Action.Bottom not in obs["sticky_actions"]:
                return Action.Bottom
        return Action.ShortPass
    
    return {"environment_fits": environment_fits, "get_action": get_action}

def penalty(obs, player_x, player_y):
    """ perform a shot in penalty game mode """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # it is penalty game mode
        if obs['game_mode'] == GameMode.Penalty:
            return True
        return False
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        if (random.random() < 0.5 and
                Action.TopRight not in obs["sticky_actions"] and
                Action.BottomRight not in obs["sticky_actions"]):
            return Action.TopRight
        else:
            if Action.BottomRight not in obs["sticky_actions"]:
                return Action.BottomRight
        return Action.Shot
    
    return {"environment_fits": environment_fits, "get_action": get_action}

def throw_in(obs, player_x, player_y):
    """ perform a short pass in throw in game mode """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # it is throw in game mode
        if obs['game_mode'] == GameMode.ThrowIn:
            return True
        return False
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        if Action.Right not in obs["sticky_actions"]:
            return Action.Right
        elif Action.TopRight not in obs["sticky_actions"]:
            return Action.TopRight
        elif Action.Top not in obs["sticky_actions"]:
            return Action.Top
        return Action.ShortPass
    
    return {"environment_fits": environment_fits, "get_action": get_action}
# "%%writefile -a submission.py" will append the code below to submission.py,
# it WILL NOT rewrite submission.py

def defence_memory_patterns(obs, player_x, player_y):
    """ group of memory patterns for environments in which opponent's team has the ball """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # player don't have the ball
        if obs["ball_owned_team"] != 0:
            return True
        return False
        
    def get_memory_patterns(obs, player_x, player_y):
        """ get list of memory patterns """
        # shift ball position
        # global anticipating
        # anticipating = False
        if obs["ball"][2] > 1.1 and obs["ball_owned_team"] == -1:
            steps = height_to_ball(obs, player_x, player_y)
            obs["ball"][0] += obs["ball_direction"][0] * steps
            obs["ball"][1] += obs["ball_direction"][1] * steps
        else:
            obs["ball"][0] += obs["ball_direction"][0] * 10/2.84*2
            obs["ball"][1] += obs["ball_direction"][1] * 10/2.84*0.84
            # if opponent has the ball and is far from y axis center
            if abs(obs["ball"][1]) > 0.07 and obs["ball_owned_team"] == 1:
                obs["ball"][0] -= 0.01
                if obs["ball"][1] > 0:
                    obs["ball"][1] -= 0.01
                else:
                    obs["ball"][1] += 0.01
            
        memory_patterns = [
            start_sprinting,
            # khorne_slide,
            run_to_ball_right,
            run_to_ball_left,
            run_to_ball_bottom,
            run_to_ball_top,
            run_to_ball_top_right,
            run_to_ball_top_left,
            run_to_ball_bottom_right,
            run_to_ball_bottom_left,
            idle
        ]
        return memory_patterns
        
    return {"environment_fits": environment_fits, "get_memory_patterns": get_memory_patterns}

def goalkeeper_memory_patterns(obs, player_x, player_y):
    """ group of memory patterns for goalkeeper """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # player is a goalkeeper have the ball
        opponent_close = True
        if obs["ball_owned_player"] == obs["active"] and obs["ball_owned_player"] == 0:
            for step in range(10):
                player_distance = normal_get_distance(obs["ball"][0]+step*obs["ball_direction"][0], obs["ball"][1]+step*obs["ball_direction"][1], obs["left_team"][0][0]+step*obs["left_team_direction"][0][0], obs["right_team"][0][1]+step*obs["left_team_direction"][0][1])
                if player_distance < RUN_SPEED*2:
                    opponent_close = False
                    for i in range(0, len(obs["right_team"])):
                        distance_to_opponent = normal_get_distance(obs["ball"][0]+step*obs["ball_direction"][0], obs["ball"][1]+step*obs["ball_direction"][1], obs["right_team"][i][0]+step*obs["right_team_direction"][i][0], obs["right_team"][i][1]+step*obs["right_team_direction"][i][1])
                        if distance_to_opponent < player_distance*2: opponent_close = True
                if opponent_close: return False
            if not opponent_close: return True
        if (obs["ball_owned_player"] == obs["active"] and
                obs["ball_owned_team"] == 0 and
                obs["ball_owned_player"] == 0):
            return True
        return False

    def get_memory_patterns(obs, player_x, player_y):
        """ get list of memory patterns """
        memory_patterns = [
            close_to_opponent_pass,
            far_from_goal_high_pass,
            idle
        ]
        return memory_patterns
        
    return {"environment_fits": environment_fits, "get_memory_patterns": get_memory_patterns}

def offence_memory_patterns(obs, player_x, player_y):
    """ group of memory patterns for environments in which player's team has the ball """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # player have the ball
        if obs["ball_owned_player"] == obs["active"] and obs["ball_owned_team"] == 0:
            return True
        opponent_close = True
        for step in range(10):
            player_distance = normal_get_distance(obs["ball"][0]+step*obs["ball_direction"][0], obs["ball"][1]+step*obs["ball_direction"][1], obs["left_team"][0][0]+step*obs["left_team_direction"][0][0], obs["right_team"][0][1]+step*obs["left_team_direction"][0][1])
            if player_distance < RUN_SPEED*2:
                opponent_close = False
                for i in range(0, len(obs["right_team"])):
                    distance_to_opponent = normal_get_distance(obs["ball"][0]+step*obs["ball_direction"][0], obs["ball"][1]+step*obs["ball_direction"][1], obs["right_team"][i][0]+step*obs["right_team_direction"][i][0], obs["right_team"][i][1]+step*obs["right_team_direction"][i][1])
                    if distance_to_opponent < player_distance*2: opponent_close = True
            if opponent_close: return False
        if not opponent_close: return True
        return False
            

        # if last_ball_owner == 0:
        #     distance_to_player = normal_get_distance(obs["ball"][0], obs["ball"][1], player_x, player_y)
        #     ball_speed = math.sqrt(obs["ball"][0]**2 + obs["ball"][1]**2)
        #     step_to_player = distance_to_player/ball_speed
        #     ball_pos = [obs["ball"][0], obs["ball"][1]]
        #     opponent_close = False
        #     player_close = False
        #     for step in range(step_to_player+1):
        #         ball_pos[0] += obs["ball_direction"][0]
        #         ball_pos[1] += obs["ball_direction"][1]
        #         distance_to_player = normal_get_distance(ball_pos[0], ball_pos[1], player_x, player_y)
        #         if distance_to_player < RUN_SPEED/2: player_close = True
        #         for i in range(0, len(obs["right_team"])):
        #             distance_to_opponent = normal_get_distance(ball_pos[0], ball_pos[1], obs["right_team"][i][0], obs["right_team"][i][1])
        #             if distance_to_opponent < distance_to_player + RUN_SPEED * step: opponent_close = True
        #     if player_close and not opponent_close: 
        #         global anticipating
        #         anticipating = True
        #         return True

    def get_memory_patterns(obs, player_x, player_y):
        """ get list of memory patterns """
        memory_patterns = [
            far_from_goal_high_pass,
            bad_angle_short_pass,
            close_to_goalkeeper_shot,
            close_to_opponent_pass,
            go_through_opponents,
            idle
        ]
        return memory_patterns


    return {"environment_fits": environment_fits, "get_memory_patterns": get_memory_patterns}

def other_memory_patterns(obs, player_x, player_y):
    """ group of memory patterns for all other environments """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        return True
        
    def get_memory_patterns(obs, player_x, player_y):
        """ get list of memory patterns """
        memory_patterns = [
            idle
        ]
        return memory_patterns
        
    return {"environment_fits": environment_fits, "get_memory_patterns": get_memory_patterns}

def special_game_modes_memory_patterns(obs, player_x, player_y):
    """ group of memory patterns for special game mode environments """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # if game mode is not normal
        if obs['game_mode'] != GameMode.Normal:
            return True
        return False
        
    def get_memory_patterns(obs, player_x, player_y):
        """ get list of memory patterns """
        memory_patterns = [
            corner,
            free_kick,
            goal_kick,
            kick_off,
            penalty,
            throw_in,
            idle
        ]
        return memory_patterns
        
    return {"environment_fits": environment_fits, "get_memory_patterns": get_memory_patterns}
# "%%writefile -a submission.py" will append the code below to submission.py,
# it WILL NOT rewrite submission.py

# list of groups of memory patterns
groups_of_memory_patterns = [
    special_game_modes_memory_patterns,
    goalkeeper_memory_patterns,
    offence_memory_patterns,
    defence_memory_patterns,
    other_memory_patterns
]

@human_readable_agent
def agent(obs):
    """ Ole ole ole ole """
    # dictionary for Memory Patterns data
    obs["memory_patterns"] = {}
    # We always control left team (observations and actions
    # are mirrored appropriately by the environment).
    controlled_player_pos = obs["left_team"][obs["active"]]
    global last_ball_owner
    if obs["ball_owned_team"] != last_ball_owner and obs["ball_owned_team"] != -1: last_ball_owner = obs["ball_owned_team"]
    # get action of appropriate pattern in agent's memory
    action = get_action_of_agent(obs, controlled_player_pos[0], controlled_player_pos[1])
    # return action
    return action


#%%
import optuna
import joblib
from kaggle_environments import make

env = make("football", configuration={"save_video": False, "scenario_name": "11_vs_11_kaggle"})

def objective(trial):

    joblib.dump(study, "study.pkl")

    global OPPONENT_SPEED
#     PLAYER_GOAL_RANGE = trial.suggest_float("PLAYER_GOAL_RANGE", -0.9, 0.3)
#     PASS_DISTANCE_OPPONENT = trial.suggest_float("PASS_DISTANCE_OPPONENT", -0.06, 0.06)
    OPPONENT_SPEED = trial.suggest_float("OPPONENT_SPEED", 0.5, 1.5, step=0.25)
#     ROUNTING_DISTANCE = trial.suggest_float("ROUNTING_DISTANCE", -0.17, 0.13)

    objective_value = 0
    for step in range(25):
        
        intermediate_value = 0
                
        for repetition in range(5): # to reduce variance
            output = env.run([agent, "../input/gfootball/submission.py"])[-1]
            # intermediate_value += output[0]["reward"]
            
            # for check_point in range(7):
            #     output = env_list[check_point].run([agent, "submission.py"])[-1]
            #     if output[0]["reward"] == 0: output[0]["reward"] = -1
                
            #     intermediate_value += output[0]["reward"]/7

        print('Left player: reward = %s, total reward = %s' % (output[0]["reward"], objective_value))        

        # Report intermediate objective value.
        trial.report(intermediate_value/5, step)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.TrialPruned()

        objective_value += intermediate_value


    return objective_value/2


if __name__ == "__main__":

    study = optuna.create_study(direction="maximize")
    # study = joblib.load('study.pkl')
    study.optimize(objective, n_trials=20)

    pruned_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

optuna.visualization.plot_intermediate_values(study)
optuna.visualization.plot_optimization_history(study)



# %%
