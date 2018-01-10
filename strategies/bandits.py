
import numpy as np

from strategies.abstract import AbstractStrategy


def oracle(env, T):
    real_means = [arm.mean for arm in env.arms]
    benchmark = np.argsort(real_means)[::-1]
    ordered_means = np.array(real_means)[benchmark]
    return (66*np.log(T) + 1) * sum([1/(ordered_means[i] - ordered_means[i+1]) for i in range(len(real_means)-1)])


class EpsilonGreedy(AbstractStrategy):

  def __init__(self, env, epsilon):
    super(EpsilonGreedy, self).__init__(env)
    self.eps = epsilon
    self.name = "Epsilon {:.1f}".format(self.eps)

  def store(self, pull, reward):
    self.mem[pull].append(reward)

  def means(self, fill=0):
    return np.array([
      np.mean(self.mem[a])
      if len(self.mem[a]) > 0 else fill
      for a in range(self.env.nbarms)])

  def choice(self, armlist, t):

    if np.random.rand() < self.eps:
      pull = np.random.choice(armlist)
    else:
      pull = np.argmax(self.means()[armlist])
      pull = armlist[pull]

    reward = self.env.arms[pull].sample()
    self.store(pull, reward)
    return pull, reward


class AUER(EpsilonGreedy):

  def __init__(self, env):
    super(AUER, self).__init__(env, epsilon=0)
    self.name = "AUER"

  def means(self, t):
    return np.array([
      np.mean(self.mem[a]) + np.sqrt(8 * np.log(t) / len(self.mem[a]))
      if len(self.mem[a]) > 0 else np.inf
      for a in range(self.env.nbarms)
    ])

  def choice(self, armlist, t):
    pull = np.argmax(self.means(t)[armlist])
    pull = armlist[pull]

    reward = self.env.arms[pull].sample()
    self.store(pull, reward)

    return pull, reward
