import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from sailing_env import SailingEnv
from stable_baselines3 import PPO
from matplotlib.cm import viridis, plasma
from matplotlib.colors import Normalize

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