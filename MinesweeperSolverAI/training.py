from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env

from ActionMaskEnv import ActionMaskEnv
from MinesweeperEnv import MinesweeperEnv

def train_agent():
    # Create the environment
    env = MinesweeperEnv()
    env = ActionMaskEnv(env)
    # Check if the environment follows the Gym API
    check_env(env, warn=True)

    # Initialize the agent
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1e-3,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=32,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        max_grad_norm=10,
    )

    # Train the agent
    model.learn(total_timesteps=100000, log_interval=10)

    # Save the model
    model.save("minesweeper_dqn_agent_9*9")

if __name__ == "__main__":
    train_agent()
