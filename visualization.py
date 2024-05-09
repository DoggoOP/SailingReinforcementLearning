import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from sailing_env import SailingEnv
from stable_baselines3 import PPO
from matplotlib.cm import viridis, plasma
from matplotlib.colors import Normalize

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


def run_visualization():
    env = SailingEnv(lift_calcs=lift_calcs, goal_position=[500, 950])
    model = PPO.load("ppo_sailing")
    steps=0

    obs = env.reset()
    path = [env.current_position.copy()]

    fig, ax = plt.subplots(figsize=(10, 10))
    line, = ax.plot([], [], '-o', color='blue', label='Path', linewidth=2, markersize=5)
    agent_marker, = ax.plot([], [], 'o', color='red', label='Agent', markersize=10)
    start_marker, = ax.plot(env.initial_position[0], env.initial_position[1], 'go', markersize=10, label='Start')
    goal_marker, = ax.plot(env.goal_position[0], env.goal_position[1], 'ro', markersize=10, label='Goal')
    ax.set_xlim([0, 1000])
    ax.set_ylim([0, 1000])
    ax.set_title('Sailing Path Visualization')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.legend(loc='upper right')

    norm = Normalize(vmin=5, vmax=20)
    grid_size = 50
    quivers = []
    for x in range(0, 1000, grid_size):
        for y in range(0, 1000, grid_size):
            wind_speed, wind_direction = env.get_wind_at_position([x + grid_size/2, y + grid_size/2])
            dx = wind_speed * np.cos(wind_direction) * 10
            dy = wind_speed * np.sin(wind_direction) * 10
            color = plasma(norm(wind_speed))
            quiver = ax.quiver(x + grid_size/2, y + grid_size, dx, dy, color=color, alpha=0.5)
            quivers.append(quiver)

    def update(frame):
        nonlocal obs, path, steps
        steps+=1
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        path.append(env.current_position.copy())
        line.set_data([p[0] for p in path], [p[1] for p in path])
        agent_marker.set_data(env.current_position[0], env.current_position[1])
        wind_speed, wind_direction = env.get_wind_at_position(env.current_position)
        print(f"Step: {steps}, Position: {env.current_position}, Action: {action}, Reward: {rewards}, Done: {dones}, Info: {info}, Wind Speed: {wind_speed}, Wind Direction: {wind_direction}")
        if dones:
            # Reset the environment and path
            obs = env.reset()
            path = [env.current_position.copy()]
            line.set_data([], [])
            agent_marker.set_data([], [])
            # Reset goal and start positions
            start_marker.set_data(env.initial_position[0], env.initial_position[1])
            goal_marker.set_data(env.goal_position[0], env.goal_position[1])
            # Reset wind field
            for quiver in quivers:
                quiver.remove()
            quivers.clear()
            for x in range(0, 1000, grid_size):
                for y in range(0, 1000, grid_size):
                    wind_speed, wind_direction = env.get_wind_at_position([x + grid_size/2, y + grid_size/2])
                    dx = wind_speed * np.cos(wind_direction) * 10
                    dy = wind_speed * np.sin(wind_direction) * 10
                    color = plasma(norm(wind_speed))
                    quiver = ax.quiver(x + grid_size/2, y + grid_size, dx, dy, color=color, alpha=0.5)
                    quivers.append(quiver)
        return line, agent_marker, goal_marker, start_marker, quivers

    anim = FuncAnimation(fig, update, frames=np.arange(500), repeat=True, interval=0.01)  # 50 ms per frame
    plt.show()

run_visualization()