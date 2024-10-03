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
