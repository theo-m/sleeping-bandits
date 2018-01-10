
import numpy as np

from strategies.abstract import AbstractStrategy


def oracle(env):
    real_means = [arm.mean for arm in env.arms]
    benchmark = np.argsort(real_means)[::-1]
    ordered_means = np.array(real_means)[benchmark]
    return 32 * sum([1 / (ordered_means[i] - ordered_means[i + 1]) for i in range(len(real_means) - 1)])


class EpsilonGreedy(AbstractStrategy):

  def __init__(self, env, epsilon):
    super(EpsilonGreedy, self).__init__(env)
    self.eps = epsilon
    self.name = "Epsilon {:.1f}".format(self.eps)

  def store(self, rewards):
    for arm, reward in rewards.items():
      self.mem[arm].append(reward)

  def observe(self, armlist):
    return {i: self.env.arms[i].sample() for i in armlist}

  def means(self, fill=0):
    return np.array([
      np.mean(self.mem[a])
      if len(self.mem[a]) > 0 else fill
      for a in range(self.env.nbarms)])

  def choice(self, armlist, t):
    rewards = self.observe(armlist)
    self.store(rewards)
    pull = np.argmax(self.means()[armlist])
    pull = armlist[pull]

    if np.random.rand() < self.eps:
      pull = np.random.choice(armlist)
    return pull, rewards[pull]


class AnnealingEpsilonGreedy(EpsilonGreedy):

  def choice(self, armlist, t):
    rewards = self.observe(armlist)
    self.store(rewards)
    pull = np.argmax(self.means()[armlist])
    pull = armlist[pull]

    if np.random.rand() < self.eps / np.sqrt(1 + t):
      pull = np.random.choice(armlist)
    return pull, rewards[pull]


class FTAL(EpsilonGreedy):

  def __init__(self, env):
    super(FTAL, self).__init__(env, epsilon=0)
    self.name = "FTAL"

  def means(self, t):
    return np.array([
      np.mean(self.mem[a])
      if len(self.mem[a]) > 0 else np.inf
      for a in range(self.env.nbarms)
    ])

  def choice(self, armlist, t):
    """ Return the choice of the arm, and its reward

    armlist -- list of indices of available arm
    t -- time step for annealing epsilon
    """

    armlist = self.env.available_arms()
    rewards = self.observe(armlist)
    self.store(rewards)

    pull = np.argmax(self.means(t)[armlist])
    pull = armlist[pull]
    reward = rewards[pull]
    return pull, reward


class AUER(FTAL):

  def __init__(self, env):
    super(AUER, self).__init__(env, epsilon=0)
    self.name = "AUER"

  def means(self, t):
    return np.array([
      np.mean(self.mem[a]) + np.sqrt(8 * np.log(t) / len(self.mem[a]))
      if len(self.mem[a]) > 0 else np.inf
      for a in range(self.env.nbarms)
    ])
