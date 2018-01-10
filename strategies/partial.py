
import numpy as np

from strategies.abstract import AbstractStrategy


class EpsilonGreedy(AbstractStrategy):

  def __init__(self, env, epsilon):
    super(EpsilonGreedy, self).__init__(env)
    self.eps = epsilon
    self.name = "Epsilon {:.1f}".format(self.eps)

  def store(self, rewards):
    for arm, reward in rewards.items():
      self.mem[arm].append(reward)

  def observe(self, x, y):
    return {i: self.env.arms[i].sample() for i in self.env.get_friends(x, y)}

  def means(self, fill=0):
    return np.array([
      np.mean(self.mem[a])
      if len(self.mem[a]) > 0 else fill
      for a in range(self.env.nbarms)])

  def choice(self, armlist, t):

    if np.random.rand() < self.eps:
      pull = np.random.choice(armlist)
      reward = self.env.arms[pull].sample()
      self.mem[pull].append(reward)
      return pull, reward

    pull = np.argmax(self.means()[armlist])
    pull = armlist[pull]

    x, y = np.unravel_index(pull, self.env.board.shape)
    rewards = self.observe(x, y)
    reward = rewards[pull]
    self.store(rewards)

    return pull, reward


class AnnealingEpsilonGreedy(EpsilonGreedy):

  def choice(self, armlist, t):

    if np.random.rand() < self.eps / np.sqrt(1 + t):
      pull = np.random.choice(armlist)
      reward = self.env.arms[pull].sample()
      self.mem[pull].append(reward)
      return pull, reward

    pull = np.argmax(self.means()[armlist])
    pull = armlist[pull]

    x, y = np.unravel_index(pull, self.env.board.shape)
    rewards = self.observe(x, y)
    reward = rewards[pull]
    self.store(rewards)

    return pull, reward


class FTAL(EpsilonGreedy):

  def __init__(self, env):
    super(FTAL, self).__init__(env, epsilon=0)
    self.name = "FTAL"

  def means(self, t):
    return super(FTAL, self).means(fill=np.inf)

  def choice(self, armlist, t):
    pull = np.argmax(self.means(t)[armlist])
    pull = armlist[pull]

    x, y = np.unravel_index(pull, self.env.board.shape)
    rewards = self.observe(x, y)
    self.store(rewards)

    reward = rewards[pull]
    return pull, reward


class AUER(FTAL):

  def __init__(self, env):
    super(AUER, self).__init__(env)
    self.name = "AUER"

  def means(self, t):
    return np.array([
      np.mean(self.mem[a]) + np.sqrt(8 * np.log(t) / len(self.mem[a]))
      if len(self.mem[a]) > 0 else np.inf
      for a in range(self.env.nbarms)
    ])
