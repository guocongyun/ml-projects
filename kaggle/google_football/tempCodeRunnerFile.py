ayer_x, player_y):
    """ environment fits constraints """
    short_pass = [-999 for i in range(len(obs["left_team"]))]
    long_pass = [-999 for i in range(len(obs["left_team"]))]
    high_pass = [-999 for i in range(len(obs["left_team"]))]
    distance_to_opponent = []
    for i in range(1, len(obs["right_team"])):
        distance_to_opponent.append(get_distance(player_x, player_y, obs["right_team"][i][0], obs["right_team"][i][1]))
    min_distance_to_opponent = min(distance_to_opponent)
    for j in range(0, len(obs["left_team"])):
        check1 = False
        check2 = False
        check3 = False
        check4 = False
        if obs["left_team_direction"][obs["active"]][0] > 0 and obs["left_team"][j][0] > player_x: check1 = True
        if obs["left_team_direction"][obs["active"]][0] < 0 and obs["left_team"][j][0] < player_x: check2 = True
        if obs["left_team_direction"][obs["active"]][1] > 0 and obs["left_team"][j][1] > player_y: check3 = True
        if obs["left_team_direction"][obs["active"]][1] < 0 and obs["left_team"][j][1] < player_y: check4 = True
        if min_distance_to_opponent > RUN_SPEED*2: check5 = True
        if check1 or check2 or check3 or check4 or check5:
            distance_to_teammate = get_distance(player_x, player_y, obs["left_team"][j][0], obs["left_team"][j][1])
            if distance_to_teammate < 1/2:
                evaluation_score = evaluation(obs, player_x, player_y, obs["left_team"][j][0], obs["left_team"][j][1], teammate_speed=[obs["left_team_direction"][j][0], obs["left_team_direction"][j][1]], passing_steps=SHORT_STEPS)
                if evaluation_score: short_pass[j] = evaluation_score

            if distance_to_teammate < 1 and distance_to_teammate > 1/2 and min_distance_to_opponent > RUN_SPEED:
                evaluation_score = evaluation(obs, player_x, player_y, obs["left_team"][j][0], obs["left_team"][j][1], teammate_speed=[obs["left_team_direction"][j][0], obs["left_team_direction"][j][1]], passing_steps=LONG_STEPS)
                if evaluation_score: 
                    if j != 0: long_pass[j] = evaluation_score
                
            if distance_to_teammate < 1.5 and distance_to_teammate > 1/2 and min_distance_to_opponent > RUN_SPEED:
                evaluation_score =  evaluation(obs, player_x, player_y, obs["left_team"][j][0], obs["left_team"][j][1], teammate_speed=[obs["left_team_direction"][j][0], obs["left_team_direction"][j][1]], passing_steps=HIGH_STEPS)
                if evaluation_score: 
                    if j != 0: high_pass[j] = evaluation_score
    comparison_list = [max(short_pass), max(long_pass), max(high_pass)]
    
    if max(comparison_list) == -999: return False
    
    best_action = np.argmax(comparison_list)
    if best_action == 0: obs["memory_patterns"]["pass"] = Action.ShortPass
    if best_action == 1: obs["memory_patterns"]["pass"] = Action.LongPass
    if best_action == 2: obs["memory_patterns"]["pass"] = Action.HighPass
    if check5: 
        if best_action == 0: 
            teammate_idx = np.argmax(short_pass)
        if best_action == 1: 
            teammate_idx = np.argmax(long_pass)
        if best_action == 2: 
            teammate_idx = np.argmax(high_pass)
        if obs["left_team_direction"][obs["active"]][0] > 0 and obs["left_team"][teammate_idx][0] < player_x: obs["memory_patterns"]["pass"] = Action.Left
        if obs["left_team_direction"][obs["active"]][0] < 0 and obs["left_team"][teammate_idx][0] > player_x: obs["memory_patterns"]["pass"] = Action.Right
        if obs["left_team_direction"][obs["active"]][1] > 0 and obs["left_team"][teammate_idx][1] < player_y: obs["memory_patterns"]["pass"] = Action.Top
        if obs["left_team_direction"][obs["active"]][1] < 0 and obs["left_team"][teammate_idx][1] > player_y: obs["memory_patterns"]["pass"] = Action.Bottom
    return True