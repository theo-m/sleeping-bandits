
import numpy as np


def flood_fill(board, x, y):
  n, m = board.shape
  if x >= 0 and x < n and y >= 0 and y < m and board[x, y] == 1:
    board[x, y] = 42
    flood_fill(board, x + 1, y)
    flood_fill(board, x - 1, y)
    flood_fill(board, x    , y + 1)
    flood_fill(board, x    , y - 1)
    flood_fill(board, x + 1, y + 1)
    flood_fill(board, x - 1, y - 1)
    flood_fill(board, x - 1, y + 1)
    flood_fill(board, x + 1, y - 1)


class Environment:

  def __init__(self):
    self.nbarms
    pass

  def step(self):
    raise NotImplementedError

  def available_arms(self):
    raise NotImplementedError

  def reinit(self):
    raise NotImplementedError


class GameOfLife(Environment):
  def __init__(self, arms, p=.5):
    """ Instantiate a board
    """
    size = int(len(arms)**.5)
    self.nbarms = size**2
    self.size = size
    self.p = p
    self.arms = arms

    self.board = np.zeros((size, size))
    self.board[np.random.rand(size, size) > p] = 1

  def step(self):
    counts = (
      self.board[0:-2, 0:-2] + self.board[0:-2, 1:-1] + self.board[0:-2, 2:] +
      self.board[1:-1, 0:-2] + self.board[1:-1, 2:] + self.board[2:, 0:-2] +
      self.board[2:, 1:-1] + self.board[2:, 2:]
    )

    birth = (counts == 3) & (self.board[1:-1, 1:-1] == 0)
    survive = ((counts == 2) | (counts == 3)) & (self.board[1:-1, 1:-1] == 1)
    self.board[...] = 0
    self.board[1:-1, 1:-1][birth | survive] = 1
    if np.all(self.board == 0):
      raise ValueError

  def available_arms(self):
    xx, yy = np.nonzero(self.board)
    return xx * self.size + yy

  def plot(self, ax):
    ax.imshow(self.board, cmap="gray", interpolation="none")
    ax.grid(which='major', axis='both', linestyle='-', color='c', linewidth=2)
    ax.axis("off")

  def get_friends(self, x, y):
    filled = self.board.copy()
    flood_fill(filled, x, y)
    xx, yy = np.where(filled == 42)
    return xx * self.size + yy

  def reinit(self):
    self.board = np.zeros((self.size, self.size))
    self.board[np.random.rand(self.size, self.size) > self.p] = 1
