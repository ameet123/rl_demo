import numpy as np
from GridEnv import GridEnv


class GridWorld:
    def __init__(self):
        self.env = GridEnv()
        self.env.no_calc_states()

    def value_iteration(self):
        for i in range(30):
            prev_v = np.copy(self.env.V)
            for s in range(12):
                if s in self.env.no_calc_states():
                    continue
                q_sa = [0 for _ in range(4)]  # this is for 4 actions.

                for a in range(4):
                    for pr, s_next in self.env.next_tuple(s, a):
                        q_sa[a] = q_sa[a] + pr * prev_v[s_next]

                self.env.update_v(s, max(q_sa))
                self.env.PI[s] = np.argmax(q_sa)
            if np.allclose(prev_v, self.env.V):
                print("!!! Convergence at:{}".format(i))
                break

        pi_str = np.vectorize(self.env.action_depict)(self.env.PI).reshape((3, 4))
        print("PI==>\n{}".format(pi_str))


if __name__ == '__main__':
    world = GridWorld()
    world.value_iteration()
