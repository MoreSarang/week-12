import numpy as np
from IPython.display import clear_output
import time
import seaborn as sns
import matplotlib.pyplot as plt


def update_board(current_board):
    """
    Execute one step of Conway's Game of Life on a finite grid
    without wrapping at the edges.

    Parameters
    ----------
    current_board : numpy.ndarray
        Binary array where 1 represents living cells and 0 represents dead cells.

    Returns
    -------
    numpy.ndarray
        Updated board state after one generation.
    """
    rows, cols = current_board.shape
    updated_board = np.zeros_like(current_board)

    for i in range(rows):
        for j in range(cols):
            # count live neighbours, skipping out-of-bounds indices
            neighbour_count = 0
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    ni = i + di
                    nj = j + dj
                    if 0 <= ni < rows and 0 <= nj < cols:
                        neighbour_count += current_board[ni, nj]

            if current_board[i, j] == 1:
                # live cell survives with 2 or 3 neighbours
                updated_board[i, j] = 1 if neighbour_count in (2, 3) else 0
            else:
                # dead cell becomes live with exactly 3 neighbours
                updated_board[i, j] = 1 if neighbour_count == 3 else 0

    return updated_board


def show_game(game_board, n_steps=10, pause=0.5):
    """
    Show `n_steps` of Conway's Game of Life, given the `update_board` function.

    Parameters
    ----------
    game_board : numpy.ndarray
        A binary array representing the initial starting conditions for Conway's Game of Life. In this array, ` represents a "living" cell and 0 represents a "dead" cell.
    n_steps : int, optional
        Number of game steps to run through, by default 10
    pause : float, optional
        Number of seconds to wait between steps, by default 0.5
    """
    for step in range(n_steps):
        clear_output(wait=True)

        # update board
        game_board = update_board(game_board)

        # show board
        sns.heatmap(game_board, cmap='plasma', cbar=False, square=True)
        plt.title(f'Board State at Step {step + 1}')
        plt.show()

        # wait for the next step
        if step + 1 < n_steps:
            time.sleep(pause)
