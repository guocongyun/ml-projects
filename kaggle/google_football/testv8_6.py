import numpy as np
import math
from v9_1_combine_everything import get_player_score, RUN_SPEED, is_within_border, Action, get_distance, evaluation, SHORT_STEPS,PASS_SPEED,normal_get_distance,check_steps,PASS_Y_SPEED

def is_blocked(obs, player_x, player_y, teammate_x, teammate_y, teammate_speed=[0,0], speed=PASS_SPEED, opponent_speed=1, high_pass=False):
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
                distance_to_opponent = normal_get_distance(ball_pos[0], ball_pos[1], obs["right_team"][opponent][0]+step*obs["right_team_direction"][opponent][0], obs["right_team"][opponent][1]+step*obs["right_team_direction"][opponent][1])
                distance_to_teammate = normal_get_distance(ball_pos[0], ball_pos[1], teammate_x+step*teammate_speed[0], teammate_y+step*teammate_speed[1])
                if distance_to_opponent < RUN_SPEED*2:
                    return True
                elif distance_to_opponent < distance_to_teammate + RUN_SPEED*2:
                    return True
        
        for step in range(steps-height_steps,steps+1):
            ball_pos[0] = ball_pos_[0] + step * unit_direction[0]
            ball_pos[1] = ball_pos_[1] + step * unit_direction[1]
            for opponent in range(0, len(obs["right_team"])):
                distance_to_opponent = normal_get_distance(ball_pos[0], ball_pos[1], obs["right_team"][opponent][0]+step*obs["right_team_direction"][opponent][0], obs["right_team"][opponent][1]+step*obs["right_team_direction"][opponent][1])
                distance_to_teammate = normal_get_distance(ball_pos[0], ball_pos[1], teammate_x+step*teammate_speed[0], teammate_y+step*teammate_speed[1])
                if distance_to_opponent < RUN_SPEED*2:
                    return True
                elif distance_to_opponent < distance_to_teammate + RUN_SPEED*2:
                    return True

    if not high_pass:
        steps = check_steps(player_x, player_y, teammate_x, teammate_y, speed)
        if steps == 0: return True
        if steps != 0: unit_direction = [(teammate_x - player_x)/steps, (teammate_y - player_y)/steps]
        for step in range(steps+1):
            ball_pos[0] += unit_direction[0]
            ball_pos[1] += unit_direction[1]
            for opponent in range(0, len(obs["right_team"])):
                distance_to_opponent = normal_get_distance(ball_pos[0], ball_pos[1], obs["right_team"][opponent][0]+step*obs["right_team_direction"][opponent][0], obs["right_team"][opponent][1]+step*obs["right_team_direction"][opponent][1])
                distance_to_teammate = normal_get_distance(ball_pos[0], ball_pos[1], teammate_x+step*teammate_speed[0], teammate_y+step*teammate_speed[1])
                if distance_to_opponent < RUN_SPEED*2:
                    return True
                elif distance_to_opponent < distance_to_teammate + RUN_SPEED*2:
                    return True
    return False

obs = {}
obs["active"] = 0
obs["memory_patterns"] = {}
obs["left_team_direction"] = [[0,-0.015]]
obs["left_team"] = [[0,-0.015]]
obs["right_team"] = [[0,0], [0,0.1], [0, 0.2],[0.2, 0.2], [0, -0.1]]
print(is_blocked(obs, 0,0,0,0,[1,2],1,1,True))