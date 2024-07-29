import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt


class Actions(Enum):
	Long=1
	Short=-1
	Neutral=0

class TradingEnv(gym.Env):
	def __init__(self, df, window_size,frame_bound):
		assert df.ndim == 2
		self.frame_bound=frame_bound
		self.seed()
		self.df = df
		self.window_size = window_size
		self.prices, self.signal_features,self.time_stamps = self._process_data()
		self.shape = (window_size, self.signal_features.shape[1])

		# spaces
		self.action_space = spaces.Discrete(len(Actions))
		self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float32)

		# episode
		self._start_tick = self.window_size
		self._end_tick = len(self.prices) - 1
		self._done = None
		self._current_tick = None
		self._last_trade_tick = None
		self._position = None
		self._position_history = None
		self._total_reward = None
		self._total_profit = None
		self._first_rendering = None

	def _process_data(self):
		prices = self.df.loc[:, 'close'].to_numpy()

		prices[self.frame_bound[0] - self.window_size]  # validate index (TODO: Improve validation)
		prices = prices[self.frame_bound[0]-self.window_size:self.frame_bound[1]]

		timeStamps = self.df.loc[:, 'timestamp']
		timeStamps = timeStamps[self.frame_bound[0]-self.window_size:self.frame_bound[1]]

		diff = np.insert(np.diff(prices), 0, 0)
		signal_features = np.column_stack((prices, diff))

		return prices, signal_features,timeStamps

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def reset(self):
		self._done = False
		self._current_tick = self._start_tick
		self._last_trade_tick = self._current_tick - 1
		self._position = Actions.Neutral
		self._position_history = (self.window_size * [None]) + [self._position]
		self._total_reward = 1.
		self._total_profit = 1.  # unit
		self._first_rendering = True
		return self._get_observation()

	def step(self, action_):
		if action_ in (0,1):
			action=Actions(action_)
		else:
			action=Actions.Short
		self._done = False
		self._current_tick += 1

		if self._current_tick == self._end_tick:
			self._done = True
		step_reward=0
		if action in (Actions.Long,Actions.Short):
			step_reward=Actions.Long.value*((self.prices[self._current_tick]-self.prices[self._current_tick-1])*100/self.prices[self._current_tick])-0.0015
		self._total_reward *= 1+step_reward/100

		# self._update_profit(action)
		self._position=action
		self._position_history.append(self._position)
		observation = self._get_observation()
		time=self.time_stamps[self._current_tick]
		info = dict(
			total_reward=self._total_reward,
			total_profit=self._total_profit,
			position=self._position.value,
			time=time
		)
		return observation, step_reward, self._done, info

	def _get_observation(self):
		return self.signal_features[(self._current_tick - self.window_size):self._current_tick]

	def render_all(self, mode='human'):
		plt.cla()
		window_ticks = np.arange(len(self._position_history))
		plt.plot(self.prices)

		short_ticks = []
		long_ticks = []
		for i, tick in enumerate(window_ticks):
			if self._position_history[i] == Actions.Short:
				short_ticks.append(tick)
			elif self._position_history[i] == Actions.Long:
				long_ticks.append(tick)

		plt.plot(short_ticks, self.prices[short_ticks], 'ro')
		plt.plot(long_ticks, self.prices[long_ticks], 'go')
		plt.show()