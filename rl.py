import numpy as np
import random

class WumpusEnv:
    ACTIONS = ["TurnLeft", "TurnRight", "Forward", "Grab", "Climb"]

    def __init__(self):
        self.width, self.height = 4, 4
        self.start = (1, 1)
        self.start_dir = 1  # 0=N,1=E,2=S,3=W
        self.wumpus = (1, 3)
        self.gold = (2, 3)
        self.pits = {(3, 1), (3, 3), (4, 4)}
        
        self.step_penalty, self.death_penalty = -1, -1000
        self.grab_reward, self.climb_reward = +1000, +1000
        self.reset()

    def reset(self):
        self.pos = self.start
        self.dir = self.start_dir
        self.has_gold = False
        self.done = False
        self.bump = False
        return self._get_percept()

    def _neighbors(self, x, y):
        nb = []
        for nx, ny in [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]:
            if 1 <= nx <= self.width and 1 <= ny <= self.height:
                nb.append((nx, ny))
        return nb

    def _in_bounds(self, x, y):
        return 1 <= x <= self.width and 1 <= y <= self.height

    def _get_percept(self):
        x, y = self.pos
        breeze = any(n in self.pits for n in self._neighbors(x, y))
        stench = any(n == self.wumpus for n in self._neighbors(x, y))
        at_start = (self.pos == self.start)
        glitter = (self.pos == self.gold and not self.has_gold)
        bump = self.bump
        return (
            int(at_start),
            int(breeze),
            int(stench),
            int(glitter),
            int(bump),
            int(self.has_gold),
            int(self.dir),
        )

    def step(self, action_idx):
        if self.done:
            return self._get_percept(), 0, True

        reward = self.step_penalty
        x, y = self.pos
        self.bump = False

        if action_idx == 0:  # TurnLeft
            self.dir = (self.dir - 1) % 4
        elif action_idx == 1:  # TurnRight
            self.dir = (self.dir + 1) % 4
        elif action_idx == 2:  # Forward
            dx, dy = [(0,1),(1,0),(0,-1),(-1,0)][self.dir]
            nx, ny = x + dx, y + dy
            if self._in_bounds(nx, ny):
                self.pos = (nx, ny)
            else:
                self.bump = True
        elif action_idx == 3:  # Grab
            if self.pos == self.gold and not self.has_gold:
                self.has_gold = True
                reward += self.grab_reward
            else:
                reward -= 100 
        elif action_idx == 4:  # Climb
            if self.pos == self.start and self.has_gold:
                reward += self.climb_reward
            else:
                reward -= 100  # penalti jika kabur tanpa emas
            self.done = True
            return self._get_percept(), reward, True

        if self.pos == self.wumpus or self.pos in self.pits:
            reward += self.death_penalty
            self.done = True
            return self._get_percept(), reward, True

        return self._get_percept(), reward, self.done

    def render_path(self, path_positions):
        grid = [["." for _ in range(self.width)] for _ in range(self.height)]
        for (px, py) in self.pits: grid[self.height-py][px-1] = "P"
        wx, wy = self.wumpus; grid[self.height-wy][wx-1] = "W"
        gx, gy = self.gold;   grid[self.height-gy][gx-1] = "G"
        for i,(x,y) in enumerate(path_positions):
            if grid[self.height-y][x-1] == ".": grid[self.height-y][x-1] = str(i%10)
            else: grid[self.height-y][x-1] = grid[self.height-y][x-1].lower()
        sx, sy = self.start; grid[self.height-sy][sx-1] = "S"
        print("Grid legend: S=start, W=wumpus, G=gold, P=pit, numbers=path order")
        for row in grid: print(" ".join(row)); print()


def encode_state(p):
    at_start, breeze, stench, glitter, bump, has_gold, dir = p
    return ((((((at_start*2 + breeze)*2 + stench)*2 + glitter)*2 + bump)*2 + has_gold)*4) + dir


N_STATES, N_ACTIONS = 256, 5


class QTableAgent:
    def __init__(self, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.9995):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.Q = np.zeros((N_STATES, N_ACTIONS))

    def choose(self, s, greedy=False):
        if not greedy and random.random() < self.epsilon:
            return random.randrange(N_ACTIONS)
        q = self.Q[s]
        return random.choice(np.where(q == q.max())[0])

    def decay(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)




class QLearningAgent(QTableAgent):
    def train(self, env, episodes=10000, max_steps=100):
        hist = []
        for _ in range(episodes):
            s = encode_state(env.reset())
            tot, done = 0, False
            for _ in range(max_steps):
                a = self.choose(s)
                ns_p, r, done = env.step(a)
                ns = encode_state(ns_p)
                tot += r
                self.Q[s, a] += self.alpha * (r + self.gamma * self.Q[ns].max() - self.Q[s, a])
                s = ns
                if done:
                    break
            hist.append(tot)
            self.decay()
        return hist




class SARSAAgent(QTableAgent):
    def train(self, env, episodes=10000, max_steps=100):
        hist = []
        for _ in range(episodes):
            s = encode_state(env.reset())
            a = self.choose(s)
            tot, done = 0, False
            for _ in range(max_steps):
                ns_p, r, done = env.step(a)
                ns = encode_state(ns_p)
                tot += r
                a2 = self.choose(ns)
                self.Q[s, a] += self.alpha * (r + self.gamma * self.Q[ns, a2] - self.Q[s, a])
                s, a = ns, a2
                if done:
                    break
            hist.append(tot)
            self.decay()
        return hist



def greedy_path(env, Q, max_steps=100):
    percept = env.reset()
    s = encode_state(percept)
    path, acts, rews = [env.pos], [], []
    repeat, last_percept = 0, s
    for _ in range(max_steps):
        q = Q[s]
        a = int(random.choice(np.flatnonzero(q == q.max())))
        ns_p, r, done = env.step(a)
        ns = encode_state(ns_p)
        if ns == last_percept:
            repeat += 1
        else:
            repeat = 0
        last_percept = ns
        if repeat > 8:
            break
        path.append(env.pos)
        acts.append(WumpusEnv.ACTIONS[a])
        rews.append(r)
        s = ns
        if done:
            break
    return path, acts, rews, env.has_gold, env.pos, done