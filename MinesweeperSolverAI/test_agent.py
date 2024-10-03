from stable_baselines3 import DQN
from MinesweeperEnv import MinesweeperEnv
from ActionMaskEnv import ActionMaskEnv
import time


def test_agent():
    # Load the trained model
    model = DQN.load("minesweeper_dqn_agent_9*9")

    # Create the environment
    env = MinesweeperEnv()
    env = ActionMaskEnv(env)

    # Initialize variables
    obs, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        # Get action from the trained agent
        action, _states = model.predict(obs, deterministic=False)
        # Take a step in the environment
        obs, reward, terminated, truncated, info = env.step(action)
        # print("Updated Observation:", obs)
        done = terminated or truncated
        total_reward += reward
        # Render the environment
        env.render()
        # Optionally, add a delay to see the steps
        # time.sleep(1)

    print(f"Total Reward: {total_reward}")

if __name__ == "__main__":
    test_agent()
