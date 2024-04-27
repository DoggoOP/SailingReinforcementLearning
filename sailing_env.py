import numpy as np
import gym
from gym import spaces
from math import sqrt, radians, cos, sin, pi, degrees, e
from collections import defaultdict



lift_calcs = [
    (5, 0., -52.5, 12031.3), (5, 0., -50., 11129.5), (5, 0., -47.5, 10462.4),
    (5, 0., -45., 9340.7), (5, 0., -42.5, 8479.94), (5, 0., -40., 8070.49),
    (5, 0., -37.5, 7690.91), (6, 0., -52.5, 17162.2), (6, 0., -50., 15840.8),
    (6, 0., -47.5, 14875.9), (6, 0., -45., 13326.4), (6, 0., -42.5, 12136.6),
    (6, 0., -40., 11556.), (6, 0., -37.5, 10986.6), (7, 0., -52.5, 23401.8),
    (7, 0., -50., 21509.6), (7, 0., -47.5, 20156.2), (7, 0., -45., 18067.8),
    (7, 0., -42.5, 16450.1), (7, 0., -40., 15640.2), (7, 0., -37.5, 14855.9),
    (8, 0., -52.5, 31109.4), (8, 0., -50., 28270.4), (8, 0., -47.5, 26284.6),
    (8, 0., -45., 23448.3), (8, 0., -42.5, 21268.9), (8, 0., -40., 20169.1),
    (8, 0., -37.5, 19125.2), (9, 0., -52.5, 39846.5), (9, 0., -50., 35989.1),
    (9, 0., -47.5, 33349.1), (9, 0., -45., 29711.4), (9, 0., -42.5, 26931.7),
    (9, 0., -40., 25541.9), (9, 0., -37.5, 24216.6), (10, 0., -52.5, 49639.5),
    (10, 0., -50., 44517.2), (10, 0., -47.5, 41011.), (10, 0., -45., 36445.4),
    (10, 2.5, -42.5, 33056.6), (10, 0., -40., 31398.8), (10, 0., -37.5, 29725.7),
    (11, 0., -52.5, 61171.2), (11, 0., -50., 54488.8), (11, 0., -47.5, 49925.2),
    (11, 0., -45., 44347.8), (11, 2.5, -42.5, 40881.9), (11, 0., -40., 38670.1),
    (11, 0., -37.5, 36768.9), (12, 0., -52.5, 73011.7), (12, 0., -50., 64410.5),
    (12, 0., -47.5, 58818.2), (12, 0., -45., 52388.4), (12, 2.5, -42.5, 49076.5),
    (12, 0., -40., 46826.), (12, 0., -37.5, 44951.7), (13, 0., -52.5, 87897.3),
    (13, 0., -50., 76656.1), (13, 0., -47.5, 69262.4), (13, 0., -45., 61417.6),
    (13, 2.5, -42.5, 57313.), (13, 0., -40., 55434.), (13, 0., -37.5, 53175.9),
    (14, 0., -52.5, 103769.), (14, 0., -50., 90294.6), (14, 0., -47.5, 81518.3),
    (14, 0., -45., 72610.6), (14, 2.5, -42.5, 68211.6), (14, 0., -40., 66569.1),
    (14, 0., -37.5, 63384.4), (15, 0., -52.5, 122107.), (15, 0., -50., 105593.),
    (15, 0., -47.5, 94531.9), (15, 0., -45., 83796.8), (15, 0., -42.5, 78133.2),
    (15, 0., -40., 76585.1), (15, 0., -37.5, 71821.6), (16, 0., -52.5, 141331.),
    (16, 0., -50., 122169.), (16, 0., -47.5, 108838.), (16, 0., -45., 95833.9),
    (16, 0., -42.5, 89222.8), (16, 0., -40., 86768.9), (16, 0., -37.5, 80731.1),
    (17, 0., -52.5, 163972.), (17, 0., -50., 141997.), (17, 0., -47.5, 126153.),
    (17, 0., -45., 110687.), (17, 0., -42.5, 102952.), (17, 0., -40., 99210.6),
    (17, 0., -37.5, 90954.), (18, 0., -52.5, 184522.), (18, 0., -50., 160730.),
    (18, 0., -47.5, 142999.), (18, 0., -45., 124960.), (18, 0., -42.5, 115956.),
    (18, 0., -40., 110351.), (18, 0., -37.5, 100106.), (19, 0., -52.5, 203456.),
    (19, 0., -50., 179002.), (19, 0., -47.5, 160248.), (19, 0., -45., 139945.),
    (19, 0., -42.5, 129431.), (19, 0., -40., 122221.), (19, 0., -37.5, 109978.),
    (20, 0., -52.5, 232600.), (20, 0., -50., 207188.), (20, 0., -47.5, 186096.),
    (20, 0., -45., 161270.), (20, 0., -42.5, 146233.), (20, 0., -40., 134972.),
    (20, 0., -37.5, 119316.)
]

class SailingEnv(gym.Env):
    metadata = {'render.modes': ['console']}
    
    def __init__(self, lift_calcs, goal_position, initial_position=[500, 20], boundary=[0, 1000]):
        super(SailingEnv, self).__init__()
        self.lift_calcs = lift_calcs
        self.lift_calcs_by_wind = self.organize_lift_calcs(self.lift_calcs)
        self.goal_position = np.array(goal_position, dtype=np.float32)
        self.initial_position = np.array(initial_position, dtype=np.float32)
        self.boundary = boundary
        self.consecutive_tacks = 0
        self.total_tacks = 0
        self.on_starboard = True
        self.grid_size = 10
        self.current_position = self.initial_position.copy()
        self.base_direction = radians(270)
        self.wind_field = None
        self.been_on_layline = False
        self.base_wind_speed = None
        self.on_layline = False  # Track whether the layline has been reached
        self.generate_wind_field()
        self.action_space = spaces.Discrete(8)
        # Expanded observation space to include full wind field data
        self.observation_space = spaces.Box(
        low=np.append(np.zeros(250),np.array([1,1])),  # Adding one boolean value
        high=np.append(np.concatenate([
            np.array([1000, 1000, 20, 2*np.pi, 2*np.pi, 1000, 1, 8, 8, 1000]),
            np.full((40,), 1000),
            np.full((200,), max(20, 2*np.pi))
        ]), np.array([1,1]))
    )


    def compute_observation(self):
        wind_speed, wind_direction = self.get_wind_at_position(self.current_position)
        distance_to_goal = np.linalg.norm(self.goal_position - self.current_position)
        boundary_penalty_gradient = max(0, (100 - min(self.current_position[0], self.boundary[1] - self.current_position[0],
                                                        self.current_position[1], self.boundary[1] - self.current_position[1])) / 100)
        angle_to_goal = np.arctan2(self.goal_position[1] - self.current_position[1], self.goal_position[0] - self.current_position[0])
        relative_angle = (wind_direction - angle_to_goal) % (2 * np.pi)
        tack_needed = self.check_if_tack_needed()

        # Updated to include 'on_layline' status directly in the observation
        future_positions, future_directions, estimated_rewards, time_taken_predictions = self.predict_and_evaluate_actions(wind_speed, wind_direction)
        tack_penalty = self.calculate_tack_penalty()

        flat_wind_field = self.wind_field.flatten()

        obs = np.concatenate([
            np.array([self.current_position[0], self.current_position[1], wind_speed, wind_direction, relative_angle, distance_to_goal,
                    boundary_penalty_gradient, self.consecutive_tacks, self.total_tacks, tack_penalty, tack_needed, self.on_layline]),
            future_positions.flatten(),
            future_directions.flatten(),
            estimated_rewards.flatten(),
            time_taken_predictions.flatten(),
            flat_wind_field
        ])
        return obs



    def check_if_tack_needed(self):
        angle_to_goal = np.arctan2(self.goal_position[1] - self.current_position[1],
                                self.goal_position[0] - self.current_position[0]) % (2 * np.pi)
        wind_speed, wind_direction = self.get_wind_at_position(self.current_position)
        tacking_angle_starboard = (wind_direction - radians(40)) % (2 * np.pi)
        tacking_angle_port = (wind_direction + radians(40)) % (2 * np.pi)

        # Calculate if on layline
        starboard_layline = np.abs((tacking_angle_starboard - angle_to_goal + 2 * np.pi) % (2 * np.pi)) < radians(10)
        port_layline = np.abs((tacking_angle_port - angle_to_goal + 2 * np.pi) % (2 * np.pi)) < radians(10)

        # Determine if the boat has passed the layline
        past_layline = False
        if self.on_starboard and starboard_layline:
            past_layline = angle_to_goal > tacking_angle_starboard
        elif not self.on_starboard and port_layline:
            past_layline = angle_to_goal < tacking_angle_port

        self.on_layline = starboard_layline or port_layline

        # Recommend tacking if past the layline
        return past_layline or self.on_layline

    
    def calculate_tack_penalty(self):
        # Exponential penalty as an example, can be adjusted as necessary
        return -1000 * (2 ** self.consecutive_tacks) if self.consecutive_tacks > 0 else 0

    def predict_and_evaluate_actions(self, wind_speed, wind_direction):
        current_distance_to_goal = np.linalg.norm(self.goal_position - self.current_position)
        future_positions = []
        future_directions = []
        estimated_rewards = []
        time_taken_for_movements = []
        tack_needed = self.check_if_tack_needed()

        for action in range(self.action_space.n):
            if action == 7:  # If the action is tacking
                new_position = self.current_position.copy()
                new_direction = wind_direction
                time_taken_for_movement = 10
                if tack_needed:
                    simulated_reward = 1000  # Large reward if tacking is needed and chosen
                else:
                    simulated_reward = 500-100*self.total_tacks  # Penalty if tacking when not needed
            else:
                result = self.simulate_action(wind_speed, wind_direction, action, current_distance_to_goal)
                new_position = result['position']
                new_direction = result['direction']
                simulated_reward = result['reward']
                time_taken_for_movement = result['time_taken']

            future_positions.append(new_position)
            future_directions.append(new_direction)
            estimated_rewards.append(simulated_reward-time_taken_for_movement**2)
            time_taken_for_movements.append(time_taken_for_movement)

        return np.array(future_positions), np.array(future_directions), np.array(estimated_rewards), np.array(time_taken_for_movements)


    def simulate_action(self, wind_speed, wind_direction, action, current_distance_to_goal):
        boat_angle = self.lift_calcs[action % len(self.lift_calcs)][2]
        lift = self.lift_calcs[action % len(self.lift_calcs)][3]
        effective_angle = wind_direction + radians(-boat_angle - 180 if self.on_starboard else boat_angle + 180)
        move_distance = 0.1 * sqrt(lift)  # Assuming movement distance is proportional to sqrt of lift for simplicity

        # Calculate the new position
        new_x = self.current_position[0] + cos(effective_angle) * move_distance
        new_y = self.current_position[1] + sin(effective_angle) * move_distance
        new_position = np.array([new_x, new_y])

        # Vector towards the goal from the new position
        goal_vector = self.goal_position - new_position
        goal_distance = np.linalg.norm(goal_vector)
        goal_direction = goal_vector / goal_distance if goal_distance != 0 else np.zeros_like(goal_vector)

        # Movement vector
        movement_vector = np.array([cos(effective_angle), sin(effective_angle)]) * move_distance
        # Calculate VMG as the dot product of the normalized movement vector and the goal direction vector
        vmg = np.dot(movement_vector, goal_direction)

        # Time penalty for movement
        time_taken_for_movement = 500000 / lift

        # Compute reward based on VMG and distance to the goal
        reward = vmg * 1000  # Scale VMG contribution
        if new_position[0] < 0 or new_position[0] > 1000 or new_position[1] < 0 or new_position[1] > 1000:
            reward -= 5000  # Penalty for going out of bounds

        if goal_distance <= 5:
            reward += 10000  # Large reward for reaching the goal

        return {
            'position': new_position,
            'direction': effective_angle,
            'reward': reward,
            'time_taken': time_taken_for_movement
        }

    def estimate_future_rewards(self, future_position):
        # Simple reward estimation based on distance to goal
        simulated_reward = np.linalg.norm(self.current_position - future_position)
        return simulated_reward  # Reward inversely proportional to distance
        
    def organize_lift_calcs(self, lift_calcs):
        wind_dict = defaultdict(list)
        for calc in lift_calcs:
            wind_speed = int(calc[0])  # Ensure the wind speed is an integer if not already
            wind_dict[wind_speed].append(calc)
        return dict(wind_dict)  # Convert to a regular dictionary for fixed keys
    
        
    def generate_wind_field(self, min_strength=7, max_strength=18):
        self.base_wind_speed = np.random.randint(min_strength, max_strength)  # Base wind speed
        self.wind_field = np.zeros((self.grid_size, self.grid_size, 2))  # Initialize wind field
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                variation = radians(np.random.randint(-20, 20))
                speed_variation = np.random.randint(-3, 4)
                self.wind_field[i, j] = [self.base_wind_speed + speed_variation, self.base_direction + variation]
    
    
    
    def update_wind_field(self):
        # Optionally re-generate or adjust the existing wind field
        self.wind_field = self.generate_wind_field()  # Re-generate the entire field

    def reset(self, new_goal_position=None):
        self.current_position = self.validate_position(self.initial_position.copy())
        self.time_taken = 0
        self.current_step = 0
        self.consecutive_tacks = 0
        self.total_tacks = 0
        self.been_on_layline = False
        self.incorrect_actions = 0
        self.generate_wind_field()
        self.on_starboard = True
        # Set new goal position if provided, otherwise keep existing one
        if new_goal_position is None:
            self.goal_position = np.array([np.random.randint(50,950), np.random.randint(850,950)], dtype=np.float32)
        elif new_goal_position is not None:
            self.goal_position = np.array(new_goal_position, dtype=np.float32)
        return self.compute_observation()


    def validate_position(self, position):
        if position.shape != (2,):
            raise ValueError("Position must be a 1D array with two elements.")
        return position
    
    def get_valid_actions(self, wind_speed):
        valid_actions = [i for i, calc in enumerate(self.lift_calcs) if abs(calc[0] - wind_speed) <= 0.5]
        valid_actions.append(len(self.lift_calcs))  # Tacking is always valid
        return valid_actions

    def step(self, action):
        reward = 0
        done = False
        info = {}

        if action == 7:
            self.total_tacks += 1
            self.consecutive_tacks += 1
            self.on_starboard = not self.on_starboard
            self.time_taken += 10
            #reward= 500-50*self.time_taken
            if self.consecutive_tacks > 1:
                reward = -5000
                done = True
                info['episode_terminated'] = "Too many consecutive tacks!"
                print("Too many consecutive tacks")
            elif self.on_layline:
                if not self.check_if_tack_needed():  # Layline passed or no longer valid
                    reward = -5000
                    info['tacked_away_from_layline'] = True
                    print("Tacked away from layline")
                    self.on_layline = False
                elif self.been_on_layline == False:
                    reward = 1000
                    print("Tacked on layline")
                    self.been_on_layline = True
                else:
                    reward = 0
            
            # elif self.total_tacks > 15:
            #     reward = -1000
            #     done = True
            #     info['episode_terminated'] = "Too many tacks!"
            #     print("Too many tacks")

            info['tacked'] = True
        else:
            # If not tacking, reset consecutive tacks
            self.consecutive_tacks = 0
            angle_index = action
            angles = np.array([-52.5, -50, -47.5, -45, -42.5, -40, -37.5])
            angle = angles[angle_index]
            wind_speed, wind_direction = self.get_wind_at_position(self.current_position)
            wind_speed_key = min(self.lift_calcs_by_wind.keys(), key=lambda x: abs(x - wind_speed))
            valid_actions = self.lift_calcs_by_wind.get(wind_speed_key, [])
            chosen_action = next((act for act in valid_actions if act[2] == angle), None)
            if chosen_action:
                reward += self.perform_movement(chosen_action, wind_speed)
                info['angle'] = angle
            else:
                reward -= 100  # Penalty for invalid action
                info['error'] = "No valid action found for the chosen angle."

        self.current_step += 1
        obs = self.compute_observation()
        done, more_info, reward = self.check_completion(reward, done)
        info.update(more_info)
        return obs, reward, done, info

    
    def update_environment(self):
        # Example: Update wind field periodically or based on some condition
        if self.current_step % 50000 == 0:  # Every 50,000 steps, update the wind field
            self.update_wind_field()

    def check_completion(self, reward, done):
        info = {}
        # Calculate the Euclidean distance to the goal
        distance_to_goal = np.linalg.norm(self.current_position - self.goal_position)
        

        if distance_to_goal <= 5:
            info['goal_reached'] = True
            goal_reward = 50000 - 15*self.time_taken  # Example scoring mechanism
            print("Goal reached at position", self.goal_position, "in", self.time_taken, "time steps", "with reward", goal_reward, "wind speed", self.base_wind_speed)
            return True, info, goal_reward

        # Check if out of bounds and determine which boundary was crossed
        out_of_bounds, boundary_type = self.is_out_of_bounds()
        if out_of_bounds:
            info['out_of_bounds'] = True
            print("Out of bounds at position", self.current_position)
            if boundary_type == 'side':
                penalty = -10000  # More severe penalty for side boundaries
            else:
                penalty = -2000  # Lesser penalty for top boundary
            return True, info, penalty

        return done, info, reward

    def is_out_of_bounds(self):
        x_out_of_bounds = not (self.boundary[0] <= self.current_position[0] <= self.boundary[1])
        y_out_of_bounds = not (self.boundary[0] <= self.current_position[1] <= self.boundary[1])

        if x_out_of_bounds:
            return True, 'side'
        elif y_out_of_bounds:
            return True, 'top_bottom'
        return False, None

    def perform_movement(self, action, wind_speed):
        previous_distance = np.linalg.norm(self.current_position - self.goal_position)
        sail_size, sail_angle, boat_angle, lift = action
        _, wind_direction = self.get_wind_at_position(self.current_position)
        effective_angle = wind_direction + radians(-boat_angle - 180 if self.on_starboard else boat_angle - 180)
        

        move_distance = lift/5000
        new_x = self.current_position[0] + cos(effective_angle) * move_distance
        new_y = self.current_position[1] + sin(effective_angle) * move_distance
        self.current_position = self.validate_position(np.array([new_x, new_y]))
        new_position = np.array([new_x, new_y])
        #print(f"Moved to {self.current_position}")

        current_distance = np.linalg.norm(self.current_position - self.goal_position)
        

        # Vector towards the goal from the new position
        goal_vector = self.goal_position - new_position
        goal_distance = np.linalg.norm(goal_vector)
        goal_direction = goal_vector / goal_distance if goal_distance != 0 else np.zeros_like(goal_vector)

        # Movement vector
        movement_vector = np.array([cos(effective_angle), sin(effective_angle)]) * move_distance
        # Calculate VMG as the dot product of the normalized movement vector and the goal direction vector
        vmg = np.dot(movement_vector, goal_direction)

        time_taken_for_movement = 100000 / lift
        self.time_taken += time_taken_for_movement
        
        return vmg * 10 if current_distance < previous_distance else -500


    def get_wind_at_position(self, position):
        if self.wind_field is None:
            raise Exception("Wind field is not initialized.")
        grid_x = int(position[0] // (1000 // self.grid_size))
        grid_y = int(position[1] // (1000 // self.grid_size))
        grid_x = max(min(grid_x, self.wind_field.shape[0] - 1), 0)
        grid_y = max(min(grid_y, self.wind_field.shape[1] - 1), 0)
        return self.wind_field[grid_x, grid_y]


    
    def set_goal_position(self, new_goal_position):
        """Set a new goal position for the environment and reset relevant attributes."""
        self.goal_position = np.array(new_goal_position, dtype=np.float32)
        # Optionally reset the environment state
        self.reset()

    def render(self, mode='console'):
        if mode == 'console':
            print(f'Position: {self.current_position}, Time Taken: {self.time_taken}')
        else:
            raise NotImplementedError("Only 'console' mode is supported.")

    def close(self):
        pass
