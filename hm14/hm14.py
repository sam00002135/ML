import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset(seed=42)

total_steps = 0  # 紀錄總共撐了幾步

for _ in range(1000):  # 可以增加最大步數
    env.render()

    # 使用 pole angle 來控制方向
    pole_angle = observation[2]
    if pole_angle > 0:
        action = 1  # 向右推
    else:
        action = 0  # 向左推

    observation, reward, terminated, truncated, info = env.step(action)
    total_steps += 1

    print('observation =', observation)

    if terminated or truncated:
        print(f'Done! Survived for {total_steps} steps.')
        observation, info = env.reset()
        total_steps = 0  # 重設步數

env.close()
