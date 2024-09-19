import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import math
from scipy.signal import argrelextrema

pd.set_option('mode.chained_assignment', None)

class StockAnalysis:
    def __init__(self, stock, initial_wealth=1000, long_MA=200, short_MA=17, period='3mo', interval='1d',
                 contraction_max=50, contraction_min=15, weeks_of_contraction=2, 
                 local_high_low_order=10, min_contractions=2, max_contractions=4, vol_contraction_days=(5, 30)):
        self.stock = stock
        self.initial_wealth = int(initial_wealth)
        self.long_MA = long_MA
        self.short_MA = short_MA
        self.period = period
        self.start_date = (datetime.today() - timedelta(days=1*365)).strftime('%Y-%m-%d')
        self.end_date = datetime.today().strftime('%Y-%m-%d')
        self.interval = interval
        self.df = None

        # Tunable parameters
        self.contraction_max = contraction_max  # Maximum acceptable contraction percentage
        self.contraction_min = contraction_min  # Minimum acceptable contraction percentage
        self.weeks_of_contraction = weeks_of_contraction  # Minimum number of weeks of contraction
        self.local_high_low_order = local_high_low_order  # Sensitivity for local high/low detection
        self.min_contractions = min_contractions  # Minimum number of contractions
        self.max_contractions = max_contractions  # Maximum number of contractions
        self.vol_contraction_days = vol_contraction_days  # (short-term, long-term) window for volume contraction
        
        self.vcp_patterns = []  # Store VCP patterns for graphing

    def get_stock_data(self):
        df = yf.download(tickers=self.stock, start=self.start_date, end=self.end_date, interval=self.interval, period=self.period)
        df.reset_index(inplace=True)
        df['date'] = df['Date'].dt.date
        self.df = df
        return df

    def ma_strategy(self):
        self.df['long_MA'] = self.df['Close'].rolling(int(self.long_MA)).mean()
        self.df['short_MA'] = self.df['Close'].rolling(int(self.short_MA)).mean()
        self.df['crosszero'] = np.where(self.df['short_MA'] < self.df['long_MA'], 1.0, 0.0)
        self.df['position'] = self.df['crosszero'].diff()
        self.df['position'].iloc[-1] = -1
        for i, row in self.df.iterrows():
            if self.df.loc[i, 'position'] == 1:
                self.df.loc[i, 'buy'] = round(self.df.loc[i, 'Close'], 2)
            if self.df.loc[i, 'position'] == -1:
                self.df.loc[i, 'sell'] = round(self.df.loc[i, 'Close'], 2)
        return self.df

    def ema_strategy(self):
        self.df['short_MA'] = self.df['Close'].ewm(span=self.short_MA).mean()
        self.df['long_MA'] = self.df['Close'].ewm(span=self.long_MA).mean()
        self.df['crosszero'] = np.where(self.df['short_MA'] < self.df['long_MA'], 1.0, 0.0)
        self.df['position'] = self.df['crosszero'].diff()
        self.df['position'].iloc[-1] = -1
        for i, row in self.df.iterrows():
            if self.df.loc[i, 'position'] == 1:
                self.df.loc[i, 'buy'] = round(self.df.loc[i, 'Close'], 2)
            if self.df.loc[i, 'position'] == -1:
                self.df.loc[i, 'sell'] = round(self.df.loc[i, 'Close'], 2)
        return self.df
    
    def _local_high_low(self):
        local_high = argrelextrema(self.df['High'].to_numpy(), np.greater, order=10)[0]
        local_low = argrelextrema(self.df['Low'].to_numpy(), np.less, order=10)[0]
        return local_high, local_low

    def _contractions(self, local_high, local_low):
        local_high = local_high[::-1]
        local_low = local_low[::-1]
        i, j = 0, 0
        contractions = []
        while i < len(local_low) and j < len(local_high):
            if local_low[i] > local_high[j]:
                contraction = round((self.df['High'][local_high[j]] - self.df['Low'][local_low[i]]) / self.df['High'][local_high[j]] * 100, 2)
                contractions.append(contraction)
                i += 1
                j += 1
            else:
                j += 1
        return contractions

    def _num_of_contractions(self, contractions):
        num_of_contraction = sum(1 for i in range(1, len(contractions)) if contractions[i] < contractions[i-1])
        return min(num_of_contraction, 4)  # Maximum of 4 contractions

    def vcp_strategy(self):
        """
        Identifies Volatility Contraction Pattern (VCP) based on price contraction.
        The pattern is a series of tightening price ranges, followed by a breakout.
        """
        print(f"Finding VCP for {self.stock}")

        local_high, local_low = self._local_high_low()
        contractions = self._contractions(local_high, local_low)
        num_of_contraction = self._num_of_contractions(contractions)
        
        # Conditions for VCP pattern
        if num_of_contraction < self.min_contractions:
            print(f"{self.stock}: Not enough contractions for a valid VCP")
            return None
        
        max_contraction = max(contractions[:num_of_contraction])
        min_contraction = min(contractions[:num_of_contraction])
        weeks_of_contraction = (len(self.df.index) - local_high[num_of_contraction - 1]) // 5
        
        # Volume contraction
        self.df['long_avg_volume'] = self.df['Volume'].rolling(window=self.vol_contraction_days[1]).mean()
        self.df['short_avg_volume'] = self.df['Volume'].rolling(window=self.vol_contraction_days[0]).mean()
        vol_contraction = self.df['short_avg_volume'].iloc[-1] < self.df['long_avg_volume'].iloc[-1]

        # Checking conditions
        if max_contraction > self.contraction_max or min_contraction > self.contraction_min:
            print(f"{self.stock}: Contraction depth invalid for VCP")
            return None
        if not vol_contraction:
            print(f"{self.stock}: No volume contraction detected")
            return None
        if weeks_of_contraction < self.weeks_of_contraction:
            print(f"{self.stock}: Not enough weeks of contraction")
            return None

        print(f"{self.stock} has a valid VCP pattern!")
        
        # Store VCP patterns for graphing
        self.vcp_patterns = [{"high": self.df['High'][local_high[i]], "low": self.df['Low'][local_low[i]], "date_high": self.df['Date'][local_high[i]], "date_low": self.df['Date'][local_low[i]]} for i in range(num_of_contraction)]
        
        return self.vcp_patterns
    

    def buy_sell_signals(self):
        total_profit = 0
        print(f'Stock: {self.stock}')
        print(f'Period: {self.start_date} - {self.end_date}')
        print('-' * 67)
        print('{:^7}{:^10}{:^15}{:^10}{:^15}'.format('S/N', 'Buy Date', 'Buy Price($)', 'Sell Date', 'Sell Price($)'))
        print('-' * 67)

        for i, row in self.df.iterrows():
            if self.df.loc[i, 'position'] == 1:
                buy_price = round(self.df.loc[i, 'buy'], 2)
                buydate = self.df.loc[i, 'Date']
            if self.df.loc[i, 'position'] == -1:
                sell_price = round(self.df.loc[i, 'sell'], 2)
                selldate = self.df.loc[i, 'Date']
                profit = sell_price - buy_price
                total_profit = round(total_profit + profit, 2)
                print('{:^7}{}{:^15}{}{:^15}'.format(i, buydate, buy_price, selldate, sell_price))
        print('')
        print('')
        return self.df

    def backtest(self):
        initial_wealth = self.initial_wealth
        position = 0
        total_profit = 0
        balance = initial_wealth
        MA_wealth = initial_wealth
        qty = 0
        self.df['position'].iloc[-1] = -1

        print(f'Stock: {self.stock}')
        print(f'Period: {self.start_date} - {self.end_date}')
        print(f'Initial Wealth: {initial_wealth}')
        print('-' * 100)
        print('{:^7}{:^15}{:^10}{:^15}{:^20}{:^20}{:^10}{:^20}{:^20}{:^20}{:^20}'.format('Sr. No', 'Buy Date', 'Buy Price($)', 'Sell Date', 'Sell Price($)', 'Investment($)', 'Qty', 'Total Buy($)', 'Total Sell($)', 'Profit/Loss($)', 'MA Wealth($)'))
        print('-' * 100)

        for i, row in self.df.iterrows():
            if position == 0 and self.df.loc[i, 'position'] == 1:
                buy_p = round(self.df.loc[i, 'Close'], 2)
                balance, qty, total_buy_p = self._buy_action(balance, buy_p)
                position = 1
            elif position == 1 and self.df.loc[i, 'position'] == -1:
                sell_p = round(self.df.loc[i, 'Close'], 2)
                total_sell_p, profitloss, total_profit, MA_wealth = self._sell_action(balance, qty, total_buy_p, sell_p, total_profit, MA_wealth)
                print(f'{i: ^7}{buy_p}{sell_p}{profitloss}{MA_wealth}')
                position = 0
            self.df.loc[i, 'MA_wealth'] = MA_wealth
        return self.df

    def _buy_action(self, balance, buy_p):
        qty = math.trunc(balance / buy_p)
        total_buy_p = round(buy_p * qty, 2)
        balance -= total_buy_p
        return balance, qty, total_buy_p

    def _sell_action(self, balance, qty, total_buy_p, sell_p, total_profit, MA_wealth):
        total_sell_p = round(sell_p * qty, 2)
        profitloss = round(total_sell_p - total_buy_p, 2)
        total_profit = round(total_profit + profitloss, 2)
        MA_wealth = round(balance + total_sell_p, 2)
        return total_sell_p, profitloss, total_profit, MA_wealth

    def graph(self):
        fig, ax = plt.subplots(figsize=[15, 6])
        # Plot long MA, short MA, and Close prices
        ax.plot(self.df['Date'], self.df['long_MA'], label='Long MA')
        ax.plot(self.df['Date'], self.df['short_MA'], color='orange', label='Short MA')
        ax.plot(self.df['Date'], self.df['Close'], color='black', label='Close')
        # Plot buy/sell signals
        ax.plot(self.df['Date'], self.df['buy'], color='green', label='Buy', marker='^', linestyle='None')
        ax.plot(self.df['Date'], self.df['sell'], color='red', label='Sell', marker='v', linestyle='None')

        # Plot VCP patterns
        if self.vcp_patterns:
            for pattern in self.vcp_patterns:
                ax.scatter(pattern['date_high'], pattern['high'], color='blue', marker='o', label='VCP High')
                ax.scatter(pattern['date_low'], pattern['low'], color='purple', marker='x', label='VCP Low')

        ax.legend(loc='upper right')
        ax.set_xlabel('Date')
        ax.set_title(self.stock)
        plt.show()

        fig, ax = plt.subplots(figsize=[15, 6])
        # Plot MA wealth and LT wealth
        ax.plot(self.df['Date'], self.df['MA_wealth'], color='black', label='MA strategy wealth')
        ax.plot(self.df['Date'], self.df['LT_wealth'], color='red', label='Buy and hold wealth')
        ax.legend(loc='upper left')
        ax.set_xlabel('Date')
        ax.set_title(self.stock)
        plt.show()


# Example usage
if __name__ == '__main__':
    analysis = StockAnalysis(stock='NVDA')
    df = analysis.get_stock_data()
    df = analysis.ema_strategy()
    df = analysis.buy_sell_signals()
    df = analysis.backtest()
    analysis.vcp_strategy()
    analysis.graph()