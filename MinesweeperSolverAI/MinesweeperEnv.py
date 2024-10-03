import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from typing import Optional, Tuple, Dict

class MinesweeperEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, board_size=(9, 9), num_mines=5):
        super(MinesweeperEnv, self).__init__()
        self.board_size = board_size
        self.num_mines = num_mines
        self.action_space = spaces.Discrete(board_size[0] * board_size[1])
        self.observation_space = spaces.Box(
            low=-1, high=8, shape=(self.board_size[0] * self.board_size[1],), dtype=np.float32
        )
        self.reset()

    def seed(self, seed: Optional[int] = None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        random.seed(seed)
        return [seed]

    def reset(self):
        # Initialize the board, place mines, and reset game state
        self._generate_board()
        self.done = False
        return self._get_observation()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self.seed(seed)
        self._generate_board()
        self.done = False
        observation = self._get_observation()
        info = {}
        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # Convert action to board coordinates
        x, y = divmod(action, self.board_size[1])
        reward = self._reveal_cell(x, y)
        observation = self._get_observation()
        info = {}
        terminated = self.done
        truncated = False  # You can implement truncation logic if needed
        return observation, reward, terminated, truncated, info

    def _generate_board(self):
        # Generate the mines and numbers on the board
        self.mines = np.zeros(self.board_size, dtype=bool)
        self.revealed = np.zeros(self.board_size, dtype=bool)
        # Place mines randomly
        mine_positions = self.np_random.choice(
            self.board_size[0] * self.board_size[1], self.num_mines, replace=False
        )
        for pos in mine_positions:
            x, y = divmod(pos, self.board_size[1])
            self.mines[x, y] = True
        # Calculate numbers for each cell
        self.numbers = np.zeros(self.board_size, dtype=np.int32)
        for x in range(self.board_size[0]):
            for y in range(self.board_size[1]):
                if not self.mines[x, y]:
                    self.numbers[x, y] = np.sum(
                        self.mines[max(0, x - 1):min(x + 2, self.board_size[0]),
                        max(0, y - 1):min(y + 2, self.board_size[1])]
                    )

    def _reveal_cell(self, x, y):
        if self.revealed[x, y]:
            # Already revealed
            return 0
        self.revealed[x, y] = True
        if self.mines[x, y]:
            self.done = True
            return -1  # Negative reward for hitting a mine
        else:
            # Positive reward for safe cell
            reward = 1
            # Optionally, implement auto-reveal of adjacent cells if number is zero
            if self.numbers[x, y] == 0:
                self._reveal_adjacent_cells(x, y)
            if self._check_win():
                self.done = True
                reward += 10  # Additional reward for winning
            return reward

    def _reveal_adjacent_cells(self, x, y):
        # Recursively reveal adjacent cells
        for i in range(max(0, x - 1), min(x + 2, self.board_size[0])):
            for j in range(max(0, y - 1), min(y + 2, self.board_size[1])):
                if not self.revealed[i, j] and not self.mines[i, j]:
                    self.revealed[i, j] = True
                    if self.numbers[i, j] == 0:
                        self._reveal_adjacent_cells(i, j)

    def _check_win(self):
        # Check if all non-mine cells are revealed
        return np.all(self.revealed | self.mines)

    def _get_observation(self):
        # Return the current state of the board
        obs = np.where(
            self.revealed,
            self.numbers,
            -1  # Use -1 to represent unrevealed cells
        )
        obs = obs.astype(np.float32).flatten()
        return obs

    def render(self, mode='human'):
        # Simple text-based rendering
        display_board = np.where(
            self.revealed,
            self.numbers.astype(str),
            '■'  # Use a character to represent unrevealed cells
        )
        # Replace mines with '*' if the game is over
        if self.done:
            display_board = np.where(
                self.mines & ~self.revealed,
                '*',
                display_board
            )
        # Print the board
        for row in display_board:
            print(' '.join(row))
        print()  # Add an empty line between steps
