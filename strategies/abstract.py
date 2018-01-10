
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm


class AbstractStrategy:

  def __init__(self, env):
    """ Instantiate the strategy
    env -- a class with a board, arms and a step method
    """
    self.name = "Abstract class"
    self.env = env
    self.mem = {i: [] for i in range(env.nbarms)}

  def choice(self, armlist):
    """ Return the choice of the arm, and its reward
    armlist -- list of indices of available arm
    """
    raise NotImplementedError

  def store(self, armlist):
    """ Keep in memory the rewards
    rewards -- a dictionary key=arm_id, value=reward
    """
    raise NotImplementedError

  def observe(self, armlist):
    """ Collect feedback from a list of arms
    armlist -- list of indices of arms to observe
    """
    raise NotImplementedError

  def run(self, horizon, render=False):
    """ Run the strategy, return the accumulated decisions and rewards

    horizon -- number of iterations
    render -- bool, save the trajectory in images
    """

    if render:
      self.folder = "../runs/{}".format(np.random.randint(1e3, 1e4))
      os.makedirs(self.folder, exist_ok=True)
      print("Saving renders in {}".format(self.folder))

    rewards = np.zeros((horizon, 2))  # pull, reward columns
    history = list()

    for t in tqdm(range(horizon), desc=self.name):
      arms = self.env.available_arms()
      history.append(arms)
      pull, reward = self.choice(arms, t)
      rewards[t] = pull, reward

      if render: self.render(arms, pull, t)

      try: self.env.step()
      except ValueError:
        print("No more life")
        break

    return rewards[:t + 1, 0], rewards[:t + 1, 1], self.mem, history

  def regret(self, history, pulls):
    """ Compute all regrets at times t
    history -- the list of indices of available arms for all times t
    pulls -- the list of actual pulls
    """
    real_means = [arm.mean for arm in self.env.arms]
    benchmark = np.argsort(real_means)[::-1]

    regrets = []
    for t, (pull, available_arms) in enumerate(zip(pulls, history)):
        for arm in benchmark:
            if arm in available_arms:
                break
        best_reward = self.env.arms[arm].mean
        regrets.append(best_reward - self.env.arms[int(pull)].mean)
    return regrets

  def render(self, arms, pull, t):
    """ Save a visualization of the path
    Can be saved to a video with
    `ffmpeg -start_number 0 -i %d.jpg -vcodec mpeg4 test.avi`

    arms -- list of indices of currently available arms
    pull -- index of the chosen arm
    t -- time step
    """
    if t == 0:
      os.makedirs(self.folder, exist_ok=True)

    fig = plt.figure(figsize=(15, 4))

    ax = fig.add_subplot(121)
    ax.imshow(self.env.board.T, cmap="gray", interpolation="none")
    x, y = np.unravel_index(pull, self.env.board.shape)
    ax.scatter(x, y, c="r")

    ax.set_title("time={}, pull={} ({}) {}".format(t, pull, (x, y), self.env.board[x, y]))

    ax = fig.add_subplot(122)

    means = self.means()
    if self.env.size > 10:
      cax = ax.imshow(self.env.board.T * means.reshape(self.env.board.shape).T, cmap=cm.coolwarm)
      ax.imshow(self.env.board.T, cmap="gray", interpolation="none", alpha=.4)
      cbar = fig.colorbar(cax, ticks=[means.min(), means.mean(), means.max()])
      ax.scatter(x, y, c="c")
    else:
      ax.bar(np.arange(self.env.nbarms), means, color="blue")
      ax.bar(arms, means[arms], label="available", color="green")
      ax.bar(pull, means[pull], label="chosen", color="red")
      ax.bar(np.arange(self.env.nbarms), [arm.mean for arm in self.env.arms], alpha=.5, color="white", label="real", lw=1, edgecolor="k")
      ax.legend()
    ax.set_title("Means recovered, current pull: {:.3f}".format(means[pull]))

    fig.tight_layout()
    fig.savefig("{}/{}.png".format(self.folder, t))
    plt.close(fig)
    plt.clf()

  def reinit(self):
    """ Erase the memory so that the agent can start anew without instantiating a new one
    """
    self.mem = {i: [] for i in range(self.env.nbarms)}


class RandomStrategy(AbstractStrategy):

  def choice(self, armlist):
    pull = np.random.choice(armlist)
    return pull, self.env.arms[pull].sample()

  def run(self, horizon):
    pulls = list()
    rewards = list()
    history = list()
    for t in tqdm(range(horizon), desc="random"):
      arms = self.env.available_arms()
      history.append(arms)
      pull, reward = self.choice(arms)
      pulls.append(pull)
      rewards.append(reward)
    return np.array(pulls), np.array(rewards), None, history
