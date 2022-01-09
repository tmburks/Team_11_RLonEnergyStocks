import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import rl
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, LSTMCell, GRU
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent, DDPGAgent, CEMAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
import scipy.signal
from gym import Env
from gym.spaces import Discrete, Box
import random

MAX_ACCOUNT_BALANCE = 2147483647
MAX_OPEN_POSITIONS = 5
MAX_STEPS = 20000

INITIAL_ACCOUNT_BALANCE = 10000
cum_profits = []
networth = []
price_history = []
price_history_neg = []


class StockTradingEnv_KMI(Env):
    def __init__(self):
        super(StockTradingEnv_KMI, self).__init__()
        self.SetStartDate(2011, 12, 30)
        self.SetEndDate(2021, 12, 30)
        self.init_cash = self.SetCash(100000000/3)
        self.symbol = self.AddEquity("KMI", Resolution.Daily).Symbol
        df = self.History(self.symbol, self.SetStartDate, self.SetEndDate, Resolution.Daily)
        self.df = df.sort_values('Date')
        self.steps = (len(self.df.loc[:, 'Open'].values) - 6)
        self.reward_range = (0, MAX_ACCOUNT_BALANCE)
        # Actions of the format Buy x%, Sell x%, Hold, etc.
        self.action_space = Discrete(300)
        # Prices contains the OHCL values for the last five prices
        self.observation_space = Box(low=0, high=1, shape=(6, 6), dtype=np.float16)

    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)
        self.current_step += 1
        if self.current_step > len(self.df.loc[:, 'Open'].values) - 6:
            self.current_step = 0

        delay_modifier = (self.current_step / MAX_STEPS)

        reward = self.balance * delay_modifier
        done = self.net_worth <= 0
        obs = self._next_observation()

        return obs, reward, done, {}

    def _take_action(self, action):
        # Set the current price to a random price within the time step
        current_price = random.uniform(self.df.loc[self.current_step, "Open"], self.df.loc[self.current_step, "Close"])
        current_date = self.df.loc[self.current_step, "Date"]
        price_history.append(current_price)
        price_history_neg.append(-current_price)
        action_type = np.floor(action / 100)
        amount = (action - (action_type * 100)) / 100
        self.net_worth_prev = self.net_worth
        peaks = scipy.signal.find_peaks(price_history, distance=3)[0]
        peaks_vals = []
        for i in range(len(peaks)):
            peaks_vals.append(price_history[peaks[i]])

        lows = scipy.signal.find_peaks(price_history_neg, distance=3)[0]
        lows_vals = []
        for i in range(len(lows)):
            lows_vals.append(price_history[lows[i]])

        if (self.current_step - 1) in lows:
            total_possible = self.balance / current_price
            shares_bought = np.floor(total_possible * 1)
            if current_date == Today:
                self.Buy(self.symbol, shares_bought)
            prev_cost = self.cost_basis * self.shares_held
            additional_cost = shares_bought * current_price
            self.balance -= additional_cost
            self.cost_basis = (prev_cost + additional_cost) / (self.shares_held + shares_bought)
            self.shares_held += shares_bought
            print('Low-BUY')

        elif (self.current_step - 1) in peaks:
            # Sell amount % of shares held
            shares_sold = np.floor(self.shares_held * 1)
            if current_date == Today:
                self.Sell(self.symbol, shares_sold)
            self.balance += shares_sold * current_price
            self.shares_held -= shares_sold
            self.total_shares_sold += shares_sold
            self.total_sales_value += shares_sold * current_price
            print("Peak-SELL")

        else:
            if action_type == 2:
                # Buy amount % of balance in shares
                total_possible = self.balance / current_price
                shares_bought = np.floor(total_possible * amount)
                if current_date == Today:
                    self.Buy(self.symbol, shares_bought)
                prev_cost = self.cost_basis * self.shares_held
                additional_cost = shares_bought * current_price
                self.balance -= additional_cost
                self.cost_basis = (prev_cost + additional_cost) / (self.shares_held + shares_bought)
                self.shares_held += shares_bought
                print('BUY')
            elif action_type == 0:
                # Sell amount % of shares held
                shares_sold = np.floor(self.shares_held * amount)
                if current_date == Today:
                    self.Sell(self.symbol, shares_sold)
                self.balance += shares_sold * current_price
                self.shares_held -= shares_sold
                self.total_shares_sold += shares_sold
                self.total_sales_value += shares_sold * current_price
                print("SELL")
        self.net_worth = self.balance + self.shares_held * current_price
        networth.append(self.net_worth)
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth
        if self.shares_held == 0:
            self.cost_basis = 0

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        profit = self.net_worth - self.init_cash
        cum_profits.append(profit)
        print(f'Step: {self.current_step}')
        print(f'Profit: {profit}')

    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_step = 0
        current_price = random.uniform(self.df.loc[self.current_step, "Open"], self.df.loc[self.current_step, "Close"])
        total_possible = self.init_cash / current_price
        self.balance = self.init_cash - total_possible * current_price
        self.net_worth = self.init_cash
        self.max_net_worth = self.init_cash
        self.shares_held = total_possible
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = total_possible * current_price
        cum_profits.clear()
        networth.clear()
        price_history.clear()
        price_history_neg.clear()
        MAX_SHARE_PRICE = 0
        MAX_NUM_SHARES = 0

        return self._next_observation()

    def _next_observation(self):
        # Get the data points for the last 5 days and scale to between 0-1
        MAX_SHARE_PRICE = max(self.df.loc[self.current_step: self.current_step + 5, 'Open'].values)
        MAX_NUM_SHARES = max(self.df.loc[self.current_step: self.current_step + 5, 'Volume'].values)
        frame = np.array([
            self.df.loc[self.current_step: self.current_step + 5, 'Open'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step + 5, 'High'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step + 5, 'Low'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step + 5, 'Close'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step + 5, 'Volume'].values / MAX_NUM_SHARES, ])
        # Append additional data and scale each value to between 0-1
        obs = np.append(frame, [[
            self.balance / MAX_ACCOUNT_BALANCE,
            self.max_net_worth / MAX_ACCOUNT_BALANCE,
            self.shares_held / MAX_NUM_SHARES,
            self.cost_basis / MAX_SHARE_PRICE,
            self.total_shares_sold / MAX_NUM_SHARES,
            self.total_sales_value / (MAX_NUM_SHARES * MAX_SHARE_PRICE), ]], axis=0)
        return obs
