import numpy as np
import gym, random
from gym import spaces
from math import sqrt, radians, cos, sin, pi, degrees, e
from collections import defaultdict



lift_calcs = [
    (5, 0., 37.5, 7485.78), (5, 0., 40., 8274.08), (5, 0., 42.5, 9352.42), 
    (5, 0., 45., 10428.8), (5, 0., 47.5, 11073.6), (5, 0., 50., 11957.), 
    (5, 0., 52.5, 13074.), (6, 0., 37.5, 10653.5), (6, 0.,40., 11843.2), 
    (6, 0., 42.5, 13348.2), (6, 0., 45., 14771.4), (6, 0., 47.5, 15664.1), 
    (6, 0., 50., 16941.2), (6, 0., 52.5, 18641.7), (7, 0., 37.5, 14318.7), 
    (7, 0., 40., 15988.5), (7, 0., 42.5, 17993.1), (7, 0., 45., 19818.1), 
    (7, 0., 47.5, 21012.3), (7, 0., 50., 22781.2), (7, 0., 52.5, 25265.9), 
    (8, 0., 37.5, 18443.6), (8, 0., 40., 20665.8), (8, 0., 42.5, 23252.1), 
    (8, 0., 45., 25557.7), (8, 0., 47.5, 27123.2), (8, 0., 50., 29486.6), 
    (8, 0., 52.5, 32930.6), (9, 0., 37.5, 22974.7), (9, 0., 40., 25848.), 
    (9, 0., 42.5, 29125.3), (9, 0., 45., 31983.7), (9, 0., 47.5, 33958.2), 
    (9, 0., 50., 36961.8), (9, 0., 52.5, 41399.5), (10, 0., 37.5, 27847.7), 
    (10, 0., 40., 31479.5), (10, 0., 42.5, 35541.9), (10, 0., 45., 39026.6), 
    (10, 0., 47.5, 41414.7), (10, 0., 50., 44979.7), (10, 0., 52.5, 50105.8), 
    (11, 0., 37.5, 33085.9), (11, 0., 40., 37597.4), (11, 0., 42.5, 42622.4), 
    (11, 0., 45., 46886.9), (11, 0., 47.5, 49732.), (11, 0., 50., 53754.8), 
    (11, 0., 52.5, 59094.7), (12, 0., 37.5, 38708.4), (12, 0., 40., 44220.3), 
    (12, 0., 42.5, 50386.2), (12, 0., 45., 55628.5), (12, 0., 47.5, 58798.5), 
    (12, 0., 50., 62810.1), (12, 0., 52.5, 68231.6), (13, 0., 37.5, 45025.1), 
    (13, 0., 40., 51727.1), (13, 0., 42.5, 59273.), (13, 0., 45., 65731.9), 
    (13, 0., 47.5, 69042.8), (13,0., 50., 72515.9), (13, 0., 52.5, 78266.4), 
    (14, 0., 37.5, 52787.9), (14, 0., 40., 60429.6), (14, 0., 42.5, 69276.1), 
    (14, 0., 45., 76632.6), (14, 0., 47.5, 79309.2), (14, 0., 50., 81540.2), 
    (14,0., 52.5, 87744.7), (15, 0., 37.5, 61523.5), (15, 0., 40., 69676.8), 
    (15, 0., 42.5, 79872.7), (15, 0., 45., 87876.7), (15, 0., 47.5, 89201.1), 
    (15, 2.5, 50., 91307.2), (15, 0., 52.5, 96974.4), (16, 0., 37.5, 71732.3), 
    (16, 0., 40., 80533.7), (16, 0., 42.5, 92057.4), (16, 0., 45., 99726.2), 
    (16, 0., 47.5, 100037.), (16, 2.5, 50., 104215.), (16, 0., 52.5, 108167.), 
    (17, 0.,37.5, 80946.9), (17, 0., 40., 90891.), (17, 0., 42.5, 103431.), 
    (17, 0., 45., 110028.), (17, 0., 47.5, 110657.), (17, 2.5,50., 116122.), 
    (17, 5., 52.5, 119354.), (18, 0., 37.5, 90666.9), (18, 0., 40., 102188.), 
    (18, 0., 42.5, 115258.), (18, 0., 45., 122306.), (18, 0., 47.5, 123303.), 
    (18, 2.5, 50., 130305.), (18, 5., 52.5, 134771.), (19, 0., 37.5, 98984.9), 
    (19, 0.,40., 112911.), (19, 0., 42.5, 127089.), (19, 0., 45., 135654.), 
    (19, 0., 47.5, 134869.), (19, 2.5, 50., 143496.), (19, 5.,52.5, 149019.), 
    (20, 0., 37.5, 105839.), (20, 0., 40., 122876.), (20, 0., 42.5, 139076.), 
    (20, 0., 45., 148172.), (20, 2.5,47.5, 145756.), (20, 2.5, 50., 153960.), 
    (20, 5., 52.5, 160799.)
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
        self.wind_chunk_size = 5
        self.been_on_layline = False
        self.base_wind_speed = None
        self.current_step = 0
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
        effective_angle = wind_direction + radians(boat_angle - 180 if self.on_starboard else -boat_angle - 180)
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
        time_taken_for_movement = 100000 / lift

        # Compute reward based on VMG and distance to the goal
        reward = vmg * 1000 - 5*time_taken_for_movement # Scale VMG contribution
        if new_position[0] < 0 or new_position[0] > 1000 or new_position[1] < 0 or new_position[1] > 1000:
            reward -= 10000  # Penalty for going out of bounds

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
        for i in range(self.wind_chunk_size):
            for j in range(self.wind_chunk_size):
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
        self.on_starboard = random.choice([True, False])
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

        # if self.current_step % 10 == 0:
        #     action = self.action_space.sample()
        
        if action == 7:
            self.total_tacks += 1
            self.consecutive_tacks += 1
            self.on_starboard = not self.on_starboard
            self.time_taken += 10
            #reward= 500-50*self.time_taken
            if self.consecutive_tacks > 1:
                reward = -10000
                done = True
                info['episode_terminated'] = "Too many consecutive tacks!"
                print("Too many consecutive tacks")
            elif self.on_layline:
                if not self.check_if_tack_needed():  # Layline passed or no longer valid
                    reward = -1000
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
            angles = np.array([37.5, 40, 42.5, 45, 52.5, 50, 52.5])
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
            goal_reward = 50000 - 1.5*self.base_wind_speed*self.time_taken  # Example scoring mechanism
            print("Goal reached at position", self.goal_position, "in", self.time_taken, "time steps", "with reward", goal_reward, "wind speed", self.base_wind_speed)
            return True, info, goal_reward

        # Check if out of bounds and determine which boundary was crossed
        out_of_bounds, boundary_type = self.is_out_of_bounds()
        if out_of_bounds:
            info['out_of_bounds'] = True
            print("Out of bounds at position", self.current_position)
            if boundary_type == 'side':
                penalty = -50000  # More severe penalty for side boundaries
            else:
                penalty = -10000  # Lesser penalty for top boundary
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
        effective_angle = wind_direction + radians(boat_angle - 180 if self.on_starboard else -boat_angle - 180)
        

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
        
        return vmg * 10 - 5*time_taken_for_movement if current_distance < previous_distance else -500


    def get_wind_at_position(self, position):
        if self.wind_field is None:
            raise Exception("Wind field is not initialized.")
        grid_x = int(position[0] // (1000 // self.wind_chunk_size))
        grid_y = int(position[1] // (1000 // self.wind_chunk_size))
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
