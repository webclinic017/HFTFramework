import datetime
import itertools
import json

from pandas._libs.tslibs.offsets import BDay

from configuration import DEFAULT_CANDLES_NUM_UNITS
from database.tick_db import TickDB, CandleType, CandleTimeResolution
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint
import os
from utils.pandas_utils.dataframe_utils import join_by_row, join_by_columns, join_two_timeseries_different_index_ffill, \
    join_by_columns_two_timeseries_different_index
from trading_algorithms.arbitrage.stat_arb import RegressionPriceType


class StatArbInstrument:
    '''
    Correlation analysis on returns : https://quantdare.com/correlation-prices-returns/
    Tutorial on : https://dm13450.github.io/2023/07/15/Stat-Arb-Walkthrough.html
    Cointegration test : https://www.quantstart.com/articles/Basics-of-Pairs-Trading/

    https://github.com/bradleyboyuyang/Statistical-Arbitrage/blob/master/statistical_arbitrage.ipynb
    https://nbviewer.jupyter.org/github/mapsa/seminario-doc-2014/blob/master/cointegration-example.ipynb

    '''

    @staticmethod
    def Hurst(series, kind='random_walk'):

        """Augmented Dickey Fuller test
        Has to be applied to residuals of a regression
        Parameters
        ----------
        series : array-like
                (Time-)series

         kind : str
        Kind of series
        possible values are 'random_walk', 'change' and 'price':
        - 'random_walk' means that a series is a random walk with random increments;
        - 'price' means that a series is a random walk with random multipliers;
        - 'change' means that a series consists of random increments
            (thus produced random walk is a cumulative sum of increments);

        Returns
        -------
        bool: boolean
            true if v pass the test -> has unit root serial correlation and not valid for regression
        """
        from hurst import compute_Hc
        H, c, data = compute_Hc(series, kind=kind, simplified=True)
        output = {"hust_exponent": H,
                  "hurst_mean_reversion": H < 0.5}

        return output

    @staticmethod
    def half_life(series):
        """
        Calculates the half life of a mean reversion
        """
        import statsmodels.api as sm
        ts = series.values[:, 0]
        ts = np.asarray(ts)

        # # make sure we are working with an array, convert if necessary

        #
        # #
        # # # delta = p(t) - p(t-1)
        # delta_ts = np.diff(ts)
        #
        # # calculate the vector of lagged values. lag = 1
        # lag_ts = np.vstack([ts[1:], np.ones(len(ts[1:]))]).T
        #
        # # calculate the slope of the deltas vs the lagged values
        # beta = np.linalg.lstsq(lag_ts, delta_ts)
        #
        # # compute and return half life
        # halflife= (np.log(2) / beta[0])[0]

        z_lag = np.roll(ts, 1)
        z_lag[0] = 0

        z_ret = ts - z_lag
        z_ret[0] = 0
        z_lag2 = sm.add_constant(z_lag)
        model = sm.OLS(z_ret, z_lag2)
        res = model.fit()
        halflife = -np.log(2) / res.params[1]
        return halflife

    @staticmethod
    def ADF(v, crit='5%', max_d=6, reg='c', autolag='AIC'):
        """Augmented Dickey Fuller test
        Has to be applied to residuals of a regression
        Parameters
        ----------
        v: ndarray matrix
            residuals matrix

        Returns
        -------
        dict: dictionary of results per confidence level
        """

        boolean = False

        from statsmodels.tsa.stattools import adfuller

        adf = adfuller(v, max_d, reg, autolag)
        value_adf = adf[0]
        output_dict = {}
        crit_dict = {
            '1%': 99,
            '5%': 95,
            '10%': 90
        }
        output_dict['pvalue'] = adf[1]
        for crit in crit_dict.keys():
            confidence_level = crit_dict[crit]
            output_dict[confidence_level] = adf[4][crit] > value_adf
        return output_dict

    @staticmethod
    def get_engle_granger_cointegration_test(X, y):
        """
        Engle-Granger cointegration test.
        Parameters
        ----------
        y: serie1
        x: serie 2
        Returns
        -------
        bool: boolean
            true if v pass the test -> has unit root serial correlation and not valid for regression
        """
        from statsmodels.tsa.stattools import coint
        coint_result = coint(X, y)
        tstat = coint_result[0]
        pvalue = coint_result[1]
        # null hypothesis is that there is no cointegration => reject null hypothesis if pvalue < 0.05
        cvt_90 = coint_result[2][2]  # 90%
        cvt_95 = coint_result[2][1]  # 95%
        cvt_99 = coint_result[2][0]  # 99%

        is_cointegrated = {
            90: False,
            95: False,
            99: False
        }
        is_cointegrated['pvalue'] = pvalue
        if tstat < cvt_99:
            is_cointegrated[99] = True
            # print("Reject null hypothesis at 99% confidence level - Cointegration exists.")
        if tstat < cvt_95:
            is_cointegrated[95] = True
            # print("Reject null hypothesis at 95% confidence level - Cointegration exists.")
        if tstat < cvt_90:
            is_cointegrated[90] = True
            # print("Reject null hypothesis at 90% confidence level - Cointegration exists.")
        return is_cointegrated

    @staticmethod
    def get_johansen(y, det_oder: int = -1, k_ar_diff: int = 1):
        """
        The result.lr1 array contains the test statistics for these hypothesis tests. The first element of result.lr1 is the test statistic for the null hypothesis of at most 0 cointegrating vectors,
        the second element is for the null hypothesis of at most 1 cointegrating vector, and so on.
        Get the cointegration vectors at 95% level of significance
        given by the trace statistic test.
        Parameters
        ----------
        y: ndarray matrix
            residuals matrix
        det_order : int (default: 0)
            * -1 - no deterministic terms
            * 0 - constant term
            * 1 - linear trend
        k_ar_diff : int, nonnegative (default: 1)
            Number of lagged differences in the model.
        confidence : float (default: 0.95) = 95%
        """

        N, l = y.shape

        from statsmodels.tsa.vector_ar.vecm import coint_johansen
        is_cointegrated = {
            90: False,
            95: False,
            99: False
        }
        try:
            result = coint_johansen(y, det_order=det_oder, k_ar_diff=k_ar_diff)
        except Exception as e:
            print(f"Error in coint_johansen {e}")
            return None, is_cointegrated

        trstat = result.lr1[0]  # trace statistic
        tsignf = result.cvt  # critical values (90%, 95%, 99%)
        trace_statistic = trstat
        cvt_90 = tsignf[:, 0][0]  # 90%
        cvt_95 = tsignf[:, 1][0]  # 95%
        cvt_99 = tsignf[:, 2][0]  # 99%

        # print("Johansen Test - Eigenvalues:", result.eig)
        # print("Johansen Test - Trace Statistic:", result.lr1)
        # print("Johansen Test - Critical Values (90%):", result.cvt[:, 0])
        # print("Johansen Test - Critical Values (95%):", result.cvt[:, 1])
        # print("Johansen Test - Critical Values (99%):", result.cvt[:, 2])
        # print('Trace statistic: %.2f' % trace_statistic)

        if trace_statistic > cvt_99:
            is_cointegrated[99] = True
            # print("Reject null hypothesis at 99% confidence level - Cointegration exists.")
        if trace_statistic > cvt_95:
            is_cointegrated[95] = True
            # print("Reject null hypothesis at 95% confidence level - Cointegration exists.")
        if trace_statistic > cvt_90:
            is_cointegrated[90] = True
            # print("Reject null hypothesis at 90% confidence level - Cointegration exists.")
        return result, is_cointegrated

    @staticmethod
    def get_backtest_df(returns_df: pd.DataFrame, price_df: pd.DataFrame, summaries_df: pd.DataFrame,
                        instrument_test: str, list_instruments: list, num_units: int, resolution: CandleTimeResolution,
                        period: int,
                        period_smooth_zscore: int = 1, beta_hedge=True, z_entry_buy: float = -2,
                        z_exit_buy: float = -0.5, z_entry_sell: float = 2,
                        z_exit_sell: float = 0.5, regression_price_type=RegressionPriceType.returns):
        import warnings;
        warnings.filterwarnings("ignore")
        summary_column = StatArbInstrument.get_pair_summary_column(num_units=num_units, resolution=resolution)

        backtest_df = returns_df.copy()
        for columns_prices in list(price_df.columns):
            backtest_df[f'{columns_prices}_price'] = price_df[columns_prices]
            backtest_df[f'{columns_prices}_log_price'] = np.log(price_df[columns_prices])

        intercept = summaries_df[f'intercept'][summary_column]
        backtest_df[f'synth_{instrument_test}'] = 0.0
        print(rf"intercept = {intercept}")
        print(rf"period = {round(summaries_df[f'residuals_half_life'][summary_column] * 2)}")
        for instrument in list_instruments:
            slope = summaries_df[f'beta_{instrument}'][summary_column]
            print(rf"beta {instrument} = {slope}")
            column_price = f'{instrument}_price'
            column_log_price = f'{instrument}_log_price'
            column_returns = f'{instrument}'

            if regression_price_type == RegressionPriceType.returns:
                backtest_df[f'synth_{instrument_test}'] += backtest_df[column_returns] * slope
            elif regression_price_type == RegressionPriceType.log_prices:
                backtest_df[f'synth_{instrument_test}'] += backtest_df[column_log_price] * slope
            else:
                backtest_df[f'synth_{instrument_test}'] += backtest_df[column_price] * slope

        backtest_df['intercept'] = summaries_df[f'intercept'][summary_column]
        backtest_df[f'synth_{instrument_test}'] += backtest_df['intercept']
        backtest_df[f'synth_{instrument_test}_price'] = backtest_df[f'synth_{instrument_test}'].copy()

        column_price = f'{instrument_test}_price'
        column_log_price = f'{instrument_test}_log_price'
        column_returns = f'{instrument_test}'

        if regression_price_type == RegressionPriceType.close_prices:
            spread = backtest_df[column_price] - backtest_df[f'synth_{instrument_test}_price']
            # and change df to returns for plotting later
            backtest_df[f'synth_{instrument_test}'] = backtest_df[f'synth_{instrument_test}'].pct_change().fillna(0)
        elif regression_price_type == RegressionPriceType.log_prices:
            spread = backtest_df[column_log_price] - backtest_df[f'synth_{instrument_test}']
            # and change df to returns for plotting later
            backtest_df[f'synth_{instrument_test}_price'] = backtest_df[f'synth_{instrument_test}'].copy()
            backtest_df[f'synth_{instrument_test}'] = backtest_df[f'synth_{instrument_test}_price'].pct_change().fillna(
                0)
        else:
            spread = backtest_df[column_returns] - backtest_df[f'synth_{instrument_test}']

        backtest_df['spread'] = spread
        # We believe the auxiliary process (cumulative sum of the residuals) can be modeled using a Ornstein-Uhlenbeck (OU) process.

        zscore = (spread - spread.rolling(period).mean()) / spread.rolling(period).std()
        # zscore= zscore.rolling(period_smooth_zscore).mean()

        # ewm(span=10, adjust=False).mean()
        # zscore = (spread - spread.ewm(span=period, adjust=False).mean()) / spread.ewm(span=period, adjust=False).std()
        if period_smooth_zscore > 1:
            print(rf"smooth zscore {period_smooth_zscore}")
            zscore = zscore.ewm(span=period_smooth_zscore, adjust=False).mean()

        backtest_df['zscore'] = zscore

        backtest_df.dropna(inplace=True)

        # BUY spread => Buy  main (btceur) + sell synthetic

        position_set = 1

        backtest_df['entry_buy'] = False
        backtest_df['exit_buy'] = False
        backtest_df['entry_sell'] = False
        backtest_df['exit_sell'] = False

        backtest_df['position_buy'] = None
        shift_entries = 0
        shift_exits = 0
        shift_positions = 1
        # operate as soon as zscore is starting to converge
        # operate after treshold passed
        mask_buy_entry = (backtest_df['zscore'] < z_entry_buy) & (backtest_df['zscore'].shift(1) > z_entry_buy)
        if shift_entries > 0:
            mask_buy_entry = mask_buy_entry.shift(shift_entries).fillna(False)

        # operate as soon as zscore is starting to converge
        # mask_buy_entry = (backtest_df['zscore'] > z_entry_buy) & (backtest_df['zscore'].shift(1) < z_entry_buy)

        backtest_df['position_buy'][mask_buy_entry] = position_set
        backtest_df['entry_buy'][mask_buy_entry] = True

        # mask_buy_exit = (backtest_df['position_buy'].ffill() == position_set) & (backtest_df['zscore'] > z_exit_buy)
        mask_buy_exit = (backtest_df['position_buy'].ffill().shift() == position_set) & (
                backtest_df['zscore'] > z_exit_buy)
        if shift_exits > 0:
            mask_buy_exit = mask_buy_exit.shift(shift_exits).fillna(False)

        backtest_df['position_buy'][mask_buy_exit] = 0
        backtest_df['exit_buy'][mask_buy_exit] = True

        backtest_df['position_buy'].ffill(inplace=True)
        backtest_df['position_buy'].fillna(0, inplace=True)
        backtest_df['position_buy'] = backtest_df['position_buy'].shift(shift_positions).fillna(0.0)

        backtest_df['entry_buy'] = (backtest_df['position_buy'].diff() > 0).shift(-1).fillna(False)
        backtest_df['exit_buy'] = (backtest_df['position_buy'].diff() < 0).shift(-1).fillna(False)

        # SELL spread  => Sell  main (btceur) + buy synthetic
        position_set = -1

        backtest_df['position_sell'] = None
        # operate after treshold passed
        mask_sell_entry = (backtest_df['zscore'] > z_entry_sell) & (backtest_df['zscore'].shift(1) < z_entry_sell)
        if shift_entries:
            mask_sell_entry = mask_sell_entry.shift(shift_entries).fillna(False)
        # operate as soon as zscore is starting to converge
        # mask_sell_entry = (backtest_df['zscore'] < z_entry_sell) & (backtest_df['zscore'].shift(1) > z_entry_sell)

        backtest_df['position_sell'][mask_sell_entry] = position_set
        backtest_df['entry_sell'][mask_sell_entry] = True
        mask_sell_exit = (backtest_df['position_sell'].ffill().shift() == position_set) & (
                backtest_df['zscore'] < z_exit_sell)
        if shift_exits > 0:
            mask_sell_exit = mask_sell_exit.shift(shift_exits).fillna(False)

        # mask_sell_exit = (backtest_df['position_sell'].ffill() == position_set) & (backtest_df['zscore'] < z_exit_sell)

        backtest_df['exit_sell'][mask_sell_exit] = True
        backtest_df['position_sell'][mask_sell_exit] = 0
        backtest_df['position_sell'].ffill(inplace=True)
        backtest_df['position_sell'].fillna(0, inplace=True)
        backtest_df['position_sell'] = backtest_df['position_sell'].shift(shift_positions).fillna(0.0)

        backtest_df['entry_sell'] = (backtest_df['position_sell'].diff() < 0).shift(-1).fillna(False)
        backtest_df['exit_sell'] = (backtest_df['position_sell'].diff() > 0).shift(-1).fillna(False)

        backtest_df['position'] = backtest_df['position_buy'] + backtest_df[
            'position_sell']
        del backtest_df['position_buy']
        del backtest_df['position_sell']

        for instrument in list_instruments:
            backtest_df[f'returns_{instrument}'] = 0
            slope = summaries_df[f'beta_{instrument}'][summary_column]
            backtest_df[f'returns_{instrument}_raw'] = backtest_df['position'] * backtest_df[instrument]
            backtest_df[f'beta_{instrument}'] = slope
            if beta_hedge:
                backtest_df[f'hedge_ratio_{instrument}'] = StatArbInstrument.get_hedge_ratio(intercept,
                                                                                             slope, backtest_df[
                                                                                                 instrument_test],
                                                                                             backtest_df[instrument])
                backtest_df[f'returns_{instrument}'] = backtest_df[
                                                           f'returns_{instrument}_raw'] * backtest_df[
                                                           f'hedge_ratio_{instrument}']
        # aggregate all hedges
        backtest_df['returns_synth'] = 0
        for instrument in list_instruments:
            backtest_df['returns_synth'] += backtest_df[f'returns_{instrument}']

        backtest_df[f'returns_{instrument_test}'] = backtest_df['position'] * backtest_df[instrument_test]

        backtest_df['returns'] = backtest_df[f'returns_{instrument_test}'] - backtest_df['returns_synth']
        backtest_df['returns_diff'] = backtest_df['returns'].diff()
        print(
            rf"betaHedge: {beta_hedge} period:{period} period_smooth_zscore:{period_smooth_zscore} diffReturns median {backtest_df['returns_diff'].median()}")
        return backtest_df

    @staticmethod
    def get_hedge_ratio(intercept, beta, price_main, price_underlying):
        # if main moves 1 is explained by hedger moves beta (0.5) => to hedge main with hedger we have to invest beta (2)
        # in java: com.lambda.investing.algorithmic_trading.hedging.LinearRegressionHedgeManager#getHedgeRatio
        return beta

    @staticmethod
    def get_backtest_with_returns_df(backtest_df: pd.DataFrame, returns_df: pd.DataFrame, instrument_test: str,
                                     list_instruments: list):
        distance = 0.00001
        buy_factor = -distance
        sell_factor = +distance

        backtest_df_with_returns = backtest_df.copy()
        for instrument in [instrument_test] + list_instruments:
            backtest_df_with_returns['returns_' + instrument] = returns_df[instrument].fillna(0)
            backtest_df_with_returns['cum_returns_' + instrument] = backtest_df_with_returns[
                'returns_' + instrument].cumsum()
            backtest_df_with_returns['entry_buy_' + instrument] = None
            backtest_df_with_returns['entry_sell_' + instrument] = None
            backtest_df_with_returns['exit_buy_' + instrument] = None
            backtest_df_with_returns['exit_sell_' + instrument] = None

            if instrument == instrument_test:
                backtest_df_with_returns['entry_buy_' + instrument][backtest_df_with_returns["entry_buy"]] = \
                    backtest_df_with_returns['cum_returns_' + instrument][
                        backtest_df_with_returns["entry_buy"]] + buy_factor
                backtest_df_with_returns['entry_sell_' + instrument][backtest_df_with_returns["entry_sell"]] = \
                    backtest_df_with_returns['cum_returns_' + instrument][
                        backtest_df_with_returns["entry_sell"]] + sell_factor
                backtest_df_with_returns['exit_buy_' + instrument][backtest_df_with_returns["exit_buy"]] = \
                    backtest_df_with_returns['cum_returns_' + instrument][
                        backtest_df_with_returns["exit_buy"]] + sell_factor
                backtest_df_with_returns['exit_sell_' + instrument][backtest_df_with_returns["exit_sell"]] = \
                    backtest_df_with_returns['cum_returns_' + instrument][
                        backtest_df_with_returns["exit_sell"]] + buy_factor
            else:
                backtest_df_with_returns['entry_sell_' + instrument][backtest_df_with_returns["entry_buy"]] = \
                    backtest_df_with_returns['cum_returns_' + instrument][
                        backtest_df_with_returns["entry_buy"]] + sell_factor
                backtest_df_with_returns['entry_buy_' + instrument][backtest_df_with_returns["entry_sell"]] = \
                    backtest_df_with_returns['cum_returns_' + instrument][
                        backtest_df_with_returns["entry_sell"]] + buy_factor
                backtest_df_with_returns['exit_sell_' + instrument][backtest_df_with_returns["exit_buy"]] = \
                    backtest_df_with_returns['cum_returns_' + instrument][
                        backtest_df_with_returns["exit_buy"]] + buy_factor
                backtest_df_with_returns['exit_buy_' + instrument][backtest_df_with_returns["exit_sell"]] = \
                    backtest_df_with_returns['cum_returns_' + instrument][
                        backtest_df_with_returns["exit_sell"]] + sell_factor
        return backtest_df_with_returns


    @staticmethod
    def plot_backtest_df(backtest_df: pd.DataFrame, returns_df: pd.DataFrame, instrument_test: str,
                         list_instruments: list, num_units: int, resolution: str, period_candles: int,
                         z_entry_buy: float = -2, z_entry_sell: float = 2, z_exit_buy: float = -0.2,
                         z_exit_sell: float = 0.2):
        backtest_df_with_returns = StatArbInstrument.get_backtest_with_returns_df(backtest_df, returns_df,
                                                                                  instrument_test, list_instruments)

        def rgb_to_hex(rgb):
            """Converts an RGB color tuple to hexadecimal."""
            return '#%02x%02x%02x' % tuple(int(255 * c) for c in rgb)

        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import matplotlib
        cmap = matplotlib.colormaps['tab10']
        # try:
        #     import matplotlib.cm as cm
        #     cmap = cm.get_cmap('tab10')  # Using 'tab10' colormap
        # except:
        #     import matplotlib.pyplot as plt
        #     cmap = plt.cm.get_cmap('tab10')

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            specs=[[{}], [{"secondary_y": True}], [{}]],
                            subplot_titles=(
                                f"Entry/Exit [{num_units} {resolution} candles ,{period_candles} period]", "Zscore",
                                "Equity curve"),
                            vertical_spacing=0.08)

        num_instruments = len(list_instruments) + 1

        colors = [rgb_to_hex(cmap(i)[:3]) for i in range(len(list_instruments) + 1)]

        # Plotting Instrument Entry points
        if len(list_instruments) > 1:
            synth_instrument = backtest_df_with_returns[f'synth_{instrument_test}']

            fig.add_trace(go.Scatter(x=backtest_df_with_returns.index,
                                     y=synth_instrument.cumsum(),
                                     mode='lines',
                                     line=dict(color="gray", width=0.7),
                                     showlegend=True,
                                     name="synthetic"),
                          row=1, col=1)

        for i, instrument in enumerate([instrument_test] + list_instruments):
            fig.add_trace(go.Scatter(x=backtest_df_with_returns.index,
                                     y=backtest_df_with_returns[instrument].cumsum(),
                                     mode='lines',
                                     line=dict(color=colors[i]),
                                     showlegend=True,
                                     name=instrument),
                          row=1, col=1)
            opacity = 0.4
            size = 8
            symbol_buy = 'triangle-up'  #
            color_buy = 'green'
            symbol_sell = 'triangle-down'  #
            color_sell = 'red'

            fig.add_trace(go.Scatter(x=backtest_df_with_returns['entry_buy_' + instrument].dropna().index,
                                     y=backtest_df_with_returns['entry_buy_' + instrument].dropna(),
                                     mode='markers',
                                     marker=dict(symbol=symbol_buy,
                                                 color=color_buy,
                                                 opacity=opacity,
                                                 size=size),
                                     showlegend=False,
                                     name='Long Entry ' + instrument),
                          row=1, col=1)

            fig.add_trace(go.Scatter(x=backtest_df_with_returns['exit_buy_' + instrument].dropna().index,
                                     y=backtest_df_with_returns['exit_buy_' + instrument].dropna(),
                                     mode='markers',
                                     marker=dict(symbol=symbol_sell,
                                                 color=color_sell,
                                                 opacity=opacity,
                                                 size=size),
                                     showlegend=False,
                                     name='Long Exit ' + instrument),
                          row=1, col=1)

            fig.add_trace(go.Scatter(x=backtest_df_with_returns['entry_sell_' + instrument].dropna().index,
                                     y=backtest_df_with_returns['entry_sell_' + instrument].dropna(),
                                     mode='markers',
                                     marker=dict(symbol=symbol_sell,
                                                 color=color_sell,
                                                 opacity=opacity,
                                                 size=size),
                                     showlegend=False,
                                     name='Short Entry ' + instrument),
                          row=1, col=1)

            fig.add_trace(go.Scatter(x=backtest_df_with_returns['exit_sell_' + instrument].dropna().index,
                                     y=backtest_df_with_returns['exit_sell_' + instrument].dropna(),
                                     mode='markers',
                                     marker=dict(symbol=symbol_buy,
                                                 color=color_buy,
                                                 opacity=opacity,
                                                 size=size),
                                     showlegend=False,
                                     name='Short Exit ' + instrument),
                          row=1, col=1)

            fig.update_yaxes(title_text="Returns", row=1, col=1)

        # Plotting zscore
        fig.add_trace(go.Scatter(x=backtest_df_with_returns.index,
                                 y=backtest_df_with_returns['zscore'],
                                 mode='lines', line_color="gray", showlegend=False,
                                 name='zscore'),
                      row=2, col=1)
        fig.add_hline(y=z_entry_buy, line_dash="solid", line_color="green", line_width=1, opacity=0.4,
                      name='z_entry_buy', row=2, col=1)
        fig.add_hline(y=z_entry_sell, line_dash="solid", line_color="red", line_width=1, opacity=0.4,
                      name='z_entry_sell', row=2, col=1)
        fig.add_hline(y=z_exit_buy, line_dash="dash", line_color="red", line_width=1, opacity=0.4, name='z_exit_buy',
                      row=2, col=1)
        fig.add_hline(y=z_exit_sell, line_dash="dash", line_color="green", line_width=1, opacity=0.4,
                      name='z_exit_sell', row=2, col=1)
        fig.update_yaxes(title_text="ZScore", row=2, col=1)

        spread = backtest_df_with_returns['spread']

        fig.add_trace(go.Scatter(x=backtest_df_with_returns.index,
                                 y=spread,
                                 line=dict(color="green", width=0.7, dash='dot'),
                                 mode='lines', showlegend=False,
                                 name='spread'),
                      row=2, col=1, secondary_y=True)
        fig.update_yaxes(title_text="Spread", row=2, col=1, secondary_y=True, color='green')

        # Plotting Equity curve
        fig.add_trace(go.Scatter(x=backtest_df_with_returns.index,
                                 y=backtest_df_with_returns['returns'].cumsum(),
                                 mode='lines', fill='tozeroy', line_color="gray", showlegend=False,
                                 name='Equity curve'),
                      row=3, col=1)
        fig.update_yaxes(title_text="PnL", row=3, col=1)

        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                          legend_title_text="Legend")
        fig.update_layout(height=1280, width=1024, showlegend=True)

        return fig

    def __init__(self, main_instrument: str, list_instruments: list,
                 regression_price_type: str = RegressionPriceType.close_prices):
        self.main_instrument = main_instrument
        self.synthetic_instruments = list_instruments
        if self.main_instrument in self.synthetic_instruments:
            self.synthetic_instruments.remove(self.main_instrument)

        self.all_instruments = [self.main_instrument] + self.synthetic_instruments
        self.tick = TickDB()
        # Returns are typically more stationary than prices, so we will use returns for beta calculation
        self.regression_price_type = regression_price_type

    def _download_trades(
            self, start_date: datetime.datetime, end_date: datetime.datetime
    ):
        assert end_date >= start_date, "end_date should be greater than start_date"
        trades_dict = {}
        for instrument in self.synthetic_instruments:
            print('downloading %s ....' % instrument)
            trades_df = self.tick.get_trades(
                instrument_pk=instrument, start_date=start_date, end_date=end_date
            )
            trades_dict[instrument] = trades_df
        return trades_dict

    def _download_depth(
            self, start_date: datetime.datetime, end_date: datetime.datetime
    ):
        assert end_date >= start_date, "end_date should be greater than start_date"
        dept_dict = {}

        drop_instruments = []
        for instrument in self.synthetic_instruments:
            try:
                print('downloading %s ....' % instrument)
                trades_df = self.tick.get_depth(
                    instrument_pk=instrument, start_date=start_date, end_date=end_date
                )
                dept_dict[instrument] = trades_df
            except Exception as e:
                print('Error downloading instrument %s -> skip it' % (instrument))
                drop_instruments.append(instrument)
        for drop_instrument in drop_instruments:
            self.synthetic_instruments.remove(drop_instrument)
        print(
            'final depth_dict of %d instruments  -> %s'
            % (len(dept_dict), ','.join(self.synthetic_instruments))
        )
        return dept_dict

    def _create_depth_from_reference(
            self, depth_reference_df: pd.DataFrame, raw_depth_dicts: dict
    ) -> pd.DataFrame:
        output_df = depth_reference_df.copy()
        instruments_list = [self.main_instrument]
        for instrument in raw_depth_dicts.keys():
            # index_to_search = raw_depth_dicts[instrument].index
            # index_selected = output_df.iloc[output_df.index.get_loc(index_to_search, method='ffill')]
            if len(raw_depth_dicts[instrument]) < len(depth_reference_df) / 100:
                self.synthetic_instruments.remove(instrument)
                continue
            instruments_list.append(instrument)

            new_df = raw_depth_dicts[instrument]['midprice'].to_frame(
                name='midprice_%s' % instrument
            )
            output_df = join_by_columns_two_timeseries_different_index(output_df, new_df)
            # output_df['midprice_%s' % instrument].iloc[index_selected]=raw_depth_dicts[instrument]
        output_df.sort_index(inplace=True)
        output_df.ffill(inplace=True, axis=0)
        output_df = output_df.loc[list(depth_reference_df.index)]
        output_df.drop_duplicates(inplace=True)
        output_df.sort_index(inplace=True)

        for instrument in instruments_list:
            output_df['return_%s' % instrument] = (
                    output_df['midprice_%s' % instrument].pct_change() + 1
            )
        output_df.dropna(axis=0, inplace=True)
        return output_df

    def _create_trades_from_reference(
            self, trade_reference_df: pd.DataFrame, raw_trade_dicts: dict
    ) -> pd.DataFrame:
        output_df = trade_reference_df.copy()
        instruments_list = [self.main_instrument]
        for instrument in raw_trade_dicts.keys():
            # index_to_search = raw_depth_dicts[instrument].index
            # index_selected = output_df.iloc[output_df.index.get_loc(index_to_search, method='ffill')]
            if len(raw_trade_dicts[instrument]) < len(trade_reference_df) / 100:
                self.synthetic_instruments.remove(instrument)
                continue
            instruments_list.append(instrument)

            new_df = raw_trade_dicts[instrument]['price'].to_frame(
                name='price_%s' % instrument
            )
            output_df = join_by_columns_two_timeseries_different_index(output_df, new_df)
            # output_df['midprice_%s' % instrument].iloc[index_selected]=raw_depth_dicts[instrument]
        output_df.sort_index(inplace=True)
        output_df.ffill(inplace=True, axis=0)
        output_df = output_df.loc[list(trade_reference_df.index)]
        output_df.drop_duplicates(inplace=True)
        output_df.sort_index(inplace=True)

        for instrument in instruments_list:
            output_df['return_%s' % instrument] = (
                    output_df['price_%s' % instrument].pct_change() + 1
            )
        output_df.dropna(axis=0, inplace=True)
        return output_df

    def _create_candles_from_reference(
            self, candle_reference_df: pd.DataFrame, raw_trades_dicts: dict
    ) -> pd.DataFrame:
        output_df = candle_reference_df.copy()
        trades_positioned = {}
        first_index_date = candle_reference_df.index[0]

        for instrument in self.synthetic_instruments:
            open_price = raw_trades_dicts[instrument].loc[:first_index_date]['price']
            if len(open_price) > 0:
                trade_price_after_index = open_price.iloc[-1]
            else:
                trade_price_after_index = None
            trades_positioned[instrument] = [trade_price_after_index]

        for index_date in candle_reference_df.index:
            for instrument in self.synthetic_instruments:
                close_price = raw_trades_dicts[instrument].loc[index_date:]
                if len(close_price) > 0:
                    trade_price_after_index = close_price['price'].iloc[0]
                else:
                    trade_price_after_index = None
                trades_positioned[instrument].append(trade_price_after_index)

        for instrument in self.synthetic_instruments:
            output_df['open_%s' % instrument] = trades_positioned[instrument][:-1]
            output_df['close_%s' % instrument] = trades_positioned[instrument][1:]
            output_df['return_%s' % instrument] = (
                    output_df['close_%s' % instrument] / output_df['open_%s' % instrument]
            )

        # %
        output_df = output_df.dropna(axis=0)
        return output_df

    def _create_candles_and_append(
            self,
            candle_reference_df: pd.DataFrame,
            start_date: datetime.datetime,
            end_date: datetime.datetime,
            type: str = 'volume',
            type_arg: float = 15.302083333333334 * 0.01295,
    ) -> pd.DataFrame:
        '''
        Will increase number of indexes ...not used
        :param candle_reference_df:
        :param start_date:
        :param end_date:
        :param type:
        :param type_arg:
        :return:
        '''
        assert end_date >= start_date, "end_date should be greater than start_date"
        if type != 'volume':
            raise NotImplementedError('only for volume candles!')

        combined_candle_df = None
        fillna_columns = []
        original_columns = []
        legend_list = []
        return_columns = []
        volume_candle_dict = {}
        volume_candle_dict[self.main_instrument] = candle_reference_df
        for instrument in self.synthetic_instruments:
            volume_candle_dict[instrument] = self.tick.get_candles_volume(
                instrument_pk=instrument,
                start_date=start_date,
                end_date=end_date,
                volume=type_arg,
            )

        for instrument in volume_candle_dict.keys():
            if instrument != self.main_instrument:
                data_to_append = volume_candle_dict[instrument][
                    ['open', 'high', 'low', 'close']
                ]

                data_to_append = data_to_append.rename(
                    columns={
                        'open': 'open_%s' % instrument,
                        'high': 'high_%s' % instrument,
                        'low': 'low_%s' % instrument,
                        'close': 'close_%s' % instrument,
                    }
                )

                data_to_append['return_%s' % instrument] = (
                        data_to_append['close_%s' % instrument]
                        / data_to_append['open_%s' % instrument]
                )
            else:
                data_to_append = volume_candle_dict[instrument]

            return_columns.append('return_%s' % instrument)
            fillna_columns += list(data_to_append.columns)
            data_to_append['original_%s' % instrument] = True
            original_columns.append('original_%s' % instrument)
            if combined_candle_df is None:
                combined_candle_df = data_to_append
            else:
                # add al indexes
                combined_candle_df = pd.concat([combined_candle_df, data_to_append])

        combined_candle_sorted = combined_candle_df.sort_index()
        combined_candle_sorted[fillna_columns] = combined_candle_sorted[
            fillna_columns
        ].ffill()
        combined_candle_sorted[original_columns] = combined_candle_sorted[
            original_columns
        ].fillna(False)
        # combined_candle_sorted = combined_candle_sorted.loc[volume_candle_dict[self.main_instrument].index]
        combined_candle_sorted.dropna(axis=0, inplace=True)
        return combined_candle_sorted

    def get_midprices_returns(
            self, start_date: datetime.datetime, end_date: datetime.datetime
    ) -> pd.DataFrame:
        assert end_date >= start_date, "end_date should be greater than start_date"
        main_ladder = self.tick.get_depth(
            instrument_pk=self.main_instrument, start_date=start_date, end_date=end_date
        )

        main_ladder = main_ladder[['midprice']]
        main_ladder = main_ladder.rename(
            columns={
                'midprice': 'midprice_%s' % self.main_instrument,
            }
        )

        depth_dict = self._download_depth(start_date=start_date, end_date=end_date)
        combined_depth_sorted = self._create_depth_from_reference(
            depth_reference_df=main_ladder, raw_depth_dicts=depth_dict
        )

        return_columns = []
        for column in combined_depth_sorted.columns:
            if column.startswith('return_'):
                return_columns.append(column)

        returns_df = combined_depth_sorted[return_columns]
        rename_dict = {}
        for column in list(returns_df.columns):
            # rename_dict[column] = column.lstrip('return_')#not working for etheur
            list_after_returns = column.split('_')[1:]
            rename_dict[column] = '_'.join(list_after_returns)

        returns_df = returns_df.rename(columns=rename_dict)
        return returns_df

    def get_trades_returns(
            self, start_date: datetime.datetime, end_date: datetime.datetime
    ) -> pd.DataFrame:
        assert end_date >= start_date, "end_date should be greater than start_date"

        main_trades = self.tick.get_trades(
            instrument_pk=self.main_instrument, start_date=start_date, end_date=end_date
        )

        main_trades = main_trades[['price']]
        main_trades = main_trades.rename(
            columns={
                'price': 'price_%s' % self.main_instrument,
            }
        )

        trades_dict = self._download_trades(start_date=start_date, end_date=end_date)
        combined_trades_sorted = self._create_trades_from_reference(
            trade_reference_df=main_trades, raw_trade_dicts=trades_dict
        )

        return_columns = []
        for column in combined_trades_sorted.columns:
            if column.startswith('return_'):
                return_columns.append(column)

        returns_df = combined_trades_sorted[return_columns]
        rename_dict = {}
        for column in list(returns_df.columns):
            # rename_dict[column] = column.lstrip('return_')#not working for etheur
            list_after_returns = column.split('_')[1:]
            rename_dict[column] = '_'.join(list_after_returns)

        returns_df = returns_df.rename(columns=rename_dict)
        return returns_df

    def get_candle_returns(
            self,
            start_date: datetime.datetime,
            end_date: datetime.datetime,
            type: CandleType = CandleType.CANDLE_MIDPRICE_TIME,
            resolution: CandleTimeResolution = CandleTimeResolution.MIN,
            num_units: int = 1,

    ) -> pd.DataFrame:
        assert end_date >= start_date, "end_date should be greater than start_date"
        instruments = [self.main_instrument] + self.synthetic_instruments
        main_candles = None
        return_columns_dict = {}
        for instrument in instruments:
            instrument_candles = self.tick.get_candles(
                candle_type=type,
                instrument_pk=instrument,
                start_date=start_date,
                end_date=end_date,
                resolution=resolution,
                num_units=num_units,
            )

            instrument_candles = instrument_candles[['open', 'high', 'low', 'close']]
            instrument_candles = instrument_candles.rename(
                columns={
                    'open': 'open_%s' % instrument,
                    'high': 'high_%s' % instrument,
                    'low': 'low_%s' % instrument,
                    'close': 'close_%s' % instrument,
                }
            )
            return_column = 'return_%s' % instrument
            return_columns_dict[return_column] = instrument

            instrument_candles[return_column] = (instrument_candles['close_%s' % instrument] / instrument_candles[
                'open_%s' % instrument]) - 1.0
            # instrument_candles[return_column] = (np.log(instrument_candles['close_%s' % instrument]) - np.log(
            #     instrument_candles['open_%s' % instrument]))

            if main_candles is None:
                main_candles = instrument_candles
            else:
                main_candles = join_by_columns_two_timeseries_different_index(main_candles, instrument_candles)

        returns_df = main_candles[return_columns_dict.keys()]
        returns_df = returns_df.rename(columns=return_columns_dict)
        returns_df.fillna(0.0, inplace=True)
        return returns_df

    def get_candle_close(
            self,
            start_date: datetime.datetime,
            end_date: datetime.datetime,
            type: CandleType = CandleType.CANDLE_MIDPRICE_TIME,
            resolution: CandleTimeResolution = CandleTimeResolution.MIN,
            num_units: int = 1,

    ) -> pd.DataFrame:
        '''

        Parameters
        ----------
        start_date: included
        end_date: excluded
        type
        resolution
        num_units

        Returns
        -------

        '''
        assert end_date >= start_date, "end_date should be greater than start_date"
        instruments = [self.main_instrument] + self.synthetic_instruments
        main_candles = None
        price_columns_dict = {}
        for instrument in instruments:
            try:
                instrument_candles = self.tick.get_candles(
                    candle_type=type,
                    instrument_pk=instrument,
                    start_date=start_date,
                    end_date=end_date,
                    resolution=resolution,
                    num_units=num_units,
                )

                instrument_candles = instrument_candles[['open', 'high', 'low', 'close']]
                instrument_candles = instrument_candles.rename(
                    columns={
                        'open': 'open_%s' % instrument,
                        'high': 'high_%s' % instrument,
                        'low': 'low_%s' % instrument,
                        'close': 'close_%s' % instrument,
                    }
                )
                price_column = 'price_%s' % instrument
                price_columns_dict[price_column] = instrument

                instrument_candles[price_column] = instrument_candles['close_%s' % instrument]
                if main_candles is None:
                    main_candles = instrument_candles
                else:
                    main_candles = join_by_columns_two_timeseries_different_index(main_candles, instrument_candles)
            except Exception as e:
                print(
                    f"ERROR: get_candle_close of {instrument} between {start_date} and {end_date} {type} {resolution} {num_units}: {e}")
                raise e

        price_df = main_candles[price_columns_dict.keys()]
        price_df = price_df.rename(columns=price_columns_dict)
        price_df.ffill(inplace=True)
        return price_df

    def correlation_analysis(self, returns_df: pd.DataFrame):
        corr = returns_df.corr().abs()
        import seaborn as sns

        # sns.set_theme(style="dark")

        # Generate a mask for the upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))

        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(11, 9))

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(230, 20, as_cmap=True)

        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(
            corr,
            mask=mask,
            vmax=1.0,
            vmin=0,
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.5},
        )

        plt.show()

    def linear_regression_inputs(
            self,
            returns_df: pd.DataFrame,
            list_instruments: list = None,
            test_size: float = 0.2,
            random_state=66,
    ):
        x_columns = []
        y_column = []
        if list_instruments is None:
            list_instruments = self.synthetic_instruments

        for column in list(returns_df.columns):
            if column == self.main_instrument:
                y_column.append(column)
            else:
                if column in list_instruments:
                    x_columns.append(column)

        X = returns_df[x_columns]
        y = returns_df[y_column]

        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=False
        )
        return X_train, X_test, y_train, y_test

    def get_linear_regression(self, returns_df: pd.DataFrame, list_instruments: list = None,
                              test_size: float = 0.2,
                              random_state=66):
        # import statsmodels.api as sm
        X_train, X_test, y_train, y_test = self.linear_regression_inputs(returns_df, test_size=test_size)
        from sklearn.linear_model import LinearRegression
        # Add a constant term for the regression
        # X = sm.add_constant(X_train)
        # model = sm.OLS(y_train, X).fit()
        # beta = model.params[1]

        regr = LinearRegression()
        regr.fit(X_train, y_train)
        # beta = regr.coef_[0][0]
        return regr, X_train, X_test, y_train, y_test

    def get_pair_summary(self, candles_df: pd.DataFrame, adf_kwarg: dict = {},
                         johansen_kwarg: dict = {"k_ar_diff": 1, "det_oder": 0}, test_size=0.2) -> dict:
        candles_df = candles_df.ffill()
        returns_df = (candles_df.pct_change()).fillna(0.0)
        candles_df = candles_df.fillna(0.0)
        output_dict = {}
        output_dict['mean_return'] = returns_df.abs().sum(axis=1).mean()
        # Johansen test

        johansen_result, johansen_is_cointegrated = StatArbInstrument.get_johansen(candles_df.values,
                                                                                   **johansen_kwarg)

        for johansen_result_key in johansen_is_cointegrated.keys():
            output_dict[rf"johansen_{johansen_result_key}_conf_cointegrated"] = johansen_is_cointegrated[
                johansen_result_key]

        if self.regression_price_type == RegressionPriceType.returns:
            regr, X_train, X_test, y_train, y_test = self.get_linear_regression(returns_df, test_size=test_size)
        elif self.regression_price_type == RegressionPriceType.log_returns:
            rregr, X_train, X_test, y_train, y_test = self.get_linear_regression(np.log(returns_df),
                                                                                 test_size=test_size)
        elif self.regression_price_type == RegressionPriceType.log_prices:
            regr, X_train, X_test, y_train, y_test = self.get_linear_regression(np.log(candles_df), test_size=test_size)
        else:
            regr, X_train, X_test, y_train, y_test = self.get_linear_regression(candles_df, test_size=test_size)

        # Regression
        for index, list_instrument in enumerate(self.synthetic_instruments):
            beta = regr.coef_[0][index]
            output_dict[rf"beta_{list_instrument}"] = beta
            # the time it takes for the spread to return 0.5 std deviations to the mean

        output_dict[rf"intercept"] = regr.intercept_[0]
        y_test_pred = regr.predict(X_test)
        y_train_pred = regr.predict(X_train)
        # get residuals of the linear regression
        residuals = y_train_pred - y_train
        residuals_test = y_test_pred - y_test
        # get adf test for the residuals
        output_dict[rf"residuals_mean_spread"] = residuals.mean().iloc[0]
        output_dict[rf"residuals_std_spread"] = residuals.std().iloc[0]
        output_dict[rf"residuals_half_life"] = StatArbInstrument.half_life(residuals)
        seconds_steps = candles_df.index.to_series().diff().dt.total_seconds().dropna().value_counts().idxmax()
        output_dict[rf"residuals_half_life_seconds"] = output_dict[rf"residuals_half_life"] * seconds_steps

        # Augmented Dickey Fuller test
        adf_train = StatArbInstrument.ADF(residuals, **adf_kwarg)
        try:
            adf_test = StatArbInstrument.ADF(residuals_test, **adf_kwarg)
        except Exception as e:
            print(rf"WARNING: ADF test not enough data -> set training ADF to test ADF")
            adf_test = adf_train

        for adf_result_key in adf_train.keys():
            output_dict[rf"adf_{adf_result_key}_conf_cointegrated"] = adf_train[
                adf_result_key]
            output_dict[rf"adf_{adf_result_key}_conf_cointegrated_test"] = adf_test[
                adf_result_key]

        # Engle-Granger test => takes so much time
        # if X_train.shape[1] == 1:
        #     # same as ADF on residuals?
        #     engle_granger_train = StatArbInstrument.get_engle_granger_cointegration_test(X_train, y_train)
        #     engle_granger_test = StatArbInstrument.get_engle_granger_cointegration_test(X_test, y_test)
        #     for engle_granger_key in engle_granger_train.keys():
        #         output_dict[rf"engle_granger_{engle_granger_key}_conf_cointegrated"] = engle_granger_train[
        #             engle_granger_key]
        #         output_dict[rf"engle_granger_{engle_granger_key}_conf_cointegrated_test"] = engle_granger_test[
        #             engle_granger_key]

        # Hurst residuals exponent
        hurst_dict = {}
        if len(residuals) > 100:
            hurst_dict = StatArbInstrument.Hurst(residuals)
        else:
            raise Exception(
                rf"ERROR: training Hurst not enough data {len(residuals)} for training between {candles_df.index[0]} and {candles_df.index[-1]}")

        if len(residuals_test) > 100:
            hurst_dict_test = StatArbInstrument.Hurst(residuals_test)
        else:
            print(
                rf"WARNING: Hurst test not enough data {len(residuals_test)} between {candles_df.index[0]} and {candles_df.index[-1]} and {test_size} test_pct -> set training hurst to test hurst")
            hurst_dict_test = hurst_dict

        hurst_dict_test = {f"{key}_test": value for key, value in hurst_dict_test.items()}
        # merge hust_dict in output_dict
        output_dict.update(hurst_dict)
        output_dict.update(hurst_dict_test)

        # Correlation
        # output_dict['correlation'] = returns_df.corr()

        return output_dict

    @staticmethod
    def get_pair_summary_column(num_units, resolution=CandleTimeResolution.SECOND):
        return rf"resolution_{resolution}-num_unit_{num_units}"

    def get_pair_summaries(self, start_date: datetime.datetime,
                           end_date: datetime.datetime, resolutions: list = [CandleTimeResolution.SECOND],
                           candle_type: CandleType = CandleType.CANDLE_MIDPRICE_TIME,
                           num_units: list = [6, 16, 26, 28, 31, 37, 42], spread_quantile: float = 0.5,
                           save_best_path: str = None, test_size=0.2) -> pd.DataFrame:
        assert end_date >= start_date, "end_date should be greater than start_date"
        # iterate over each combination of resolutions and num_units
        pair_result_df = None
        for resolution, num_unit in itertools.product(resolutions, num_units):
            try:
                if resolution == CandleTimeResolution.SECOND and num_unit % 60 == 0:
                    resolution = CandleTimeResolution.MIN
                    num_unit = num_unit // 60

                # we will change it later on get pair summary if required
                candles_df = self.get_candle_close(
                    start_date=start_date, end_date=end_date,
                    type=candle_type, resolution=resolution,
                    num_units=num_unit)

                if (len(candles_df)) * 1.2 < 100:
                    print(
                        f"WARNING: not enough data {len(candles_df)} for {resolution} {num_unit} between {start_date} and {end_date}")

                pair_result_dict = self.get_pair_summary(candles_df, test_size=test_size)
                pair_result_dict['resolution'] = resolution
                pair_result_dict['num_unit'] = num_unit

                # add depth spreads to the pair_result_dict
                pair_result_dict['depth_spread'] = 0.0
                for instrument in self.all_instruments:
                    depth_df = self.tick.get_depth(
                        instrument_pk=instrument,
                        start_date=start_date,
                        end_date=end_date,
                    )
                    spread = depth_df['spread'].quantile(spread_quantile)
                    pair_result_dict['depth_spread'] += spread
                    # add spreads quantiles statistics
                    # pair_result_dict[f"spread_{instrument}_std"] = depth_df['spread'].std()
                    # pair_result_dict[f"spread_{instrument}_mean"] = depth_df['spread'].mean()
                    # pair_result_dict[f"spread_{instrument}_median"] = depth_df['spread'].median()
                    # for quantile in [0.05, 0.25, 0.75, 0.1, 0.9, 0.95]:
                    #     pair_result_dict[f"spread_{instrument}_quantile_{quantile}"] = depth_df['spread'].quantile(quantile)
            except Exception as e:
                print(rf"ERROR: {e} for {resolution} {num_unit} between {start_date} and {end_date}  -> skip it")
                import traceback;
                traceback.print_exc()
                continue

            new_df = pd.DataFrame.from_dict(pair_result_dict,
                                            columns=[self.get_pair_summary_column(num_unit, resolution)],
                                            orient="index")
            if pair_result_df is None:
                pair_result_df = new_df
            else:
                pair_result_df = join_by_columns(pair_result_df, new_df)
        if pair_result_df is None:
            print(rf"ERROR:  for all pair summaries between {start_date} and {end_date}  -> skip it")
            return None
        pair_result_df = pair_result_df.T

        residual_std_spread = pair_result_df['residuals_std_spread'] + pair_result_df['residuals_mean_spread']
        residual_2std_spread = pair_result_df['residuals_std_spread'] * 2 + pair_result_df['residuals_mean_spread']
        pair_result_df["is_valid_1_std"] = pair_result_df["depth_spread"] < residual_std_spread
        pair_result_df["is_valid_2_std"] = pair_result_df["depth_spread"] < residual_2std_spread
        # pair_result_df = pair_result_df.sort_values(by="hust_exponent", ascending=True)
        pair_result_df = pair_result_df.sort_values(by="hust_exponent_test", ascending=True)
        if save_best_path is not None:
            StatArbInstrument.get_best_pair_result(pair_result_df, save_best_path, self.all_instruments,
                                                   regression_price_type=self.regression_price_type)

        return pair_result_df

    @staticmethod
    def get_best_pair_result(pair_result_df: pd.DataFrame, save_best_path, all_instruments,
                             regression_price_type=RegressionPriceType.returns):
        # get all columns that contain beta in name
        beta_columns = [col for col in pair_result_df.columns if 'beta' in col]
        summaries_best_df = pair_result_df[['resolution', 'num_unit', 'intercept',
                                            'adf_pvalue_conf_cointegrated',
                                            'adf_pvalue_conf_cointegrated_test',
                                            'hust_exponent', 'hust_exponent_test', 'residuals_half_life',
                                            'residuals_half_life_seconds',
                                            'residuals_mean_spread', 'residuals_std_spread', 'depth_spread',
                                            'is_valid_1_std', 'is_valid_2_std'] + beta_columns].sort_values(
            by="hust_exponent")
        best_candles = summaries_best_df['num_unit'].iloc[0]  # transform to seconds always

        best_candles_resolution = summaries_best_df['resolution'].iloc[0]
        if best_candles_resolution == CandleTimeResolution.MIN:
            best_candles = best_candles * 60
        if best_candles_resolution == CandleTimeResolution.HOUR:
            best_candles = best_candles * 3600

        best_mean = summaries_best_df['residuals_mean_spread'].iloc[0]
        best_std = summaries_best_df['residuals_std_spread'].iloc[0]
        half_life_seconds = summaries_best_df['residuals_half_life_seconds'].iloc[0]
        half_life_candles = summaries_best_df['residuals_half_life'].iloc[0]
        intercept = summaries_best_df['intercept'].iloc[0]
        betas = {}
        for instrument in all_instruments:
            if f'beta_{instrument}' in summaries_best_df.columns:
                betas[instrument] = summaries_best_df[f'beta_{instrument}'].iloc[0]

        dict_to_save = StatArbInstrument.create_dict_configuration(betas_dict=betas,
                                                                   half_life=half_life_candles,
                                                                   half_life_seconds=half_life_seconds,
                                                                   mean=best_mean, std=best_std,
                                                                   intercept=intercept,
                                                                   seconds_candles=best_candles,
                                                                   regression_price_type=regression_price_type
                                                                   )

        # print dict as json pretty
        str_dict = json.dumps(dict_to_save, indent=4)
        print(
            rf"{str_dict} -> best candles {best_candles} seconds half_life {half_life_candles} candles resolution {best_candles_resolution} into {save_best_path}")
        dirname = os.path.dirname(save_best_path)
        if len(dirname) > 0:
            os.makedirs(dirname, exist_ok=True)
        with open(save_best_path, 'w') as f:
            json.dump(dict_to_save, f)

    @staticmethod
    def create_dict_configuration(betas_dict: dict, half_life: float = 0.0, half_life_seconds: float = -1.0,
                                  mean: float = 0.0, std: float = 0.0, intercept: float = 0.0,
                                  seconds_candles: int = 60,
                                  regression_price_type: str = RegressionPriceType.returns) -> dict:
        '''
        {
            "etheur_kraken":11.48873636090238,
            "half_life_seconds": 21967.694437,
            "mean": 0.00085,
            "std": 0.0,
            "intercept": -0.000002
        }

        Returns dictionary
        -------

        '''
        output_dict = {}
        for key, value in betas_dict.items():
            output_dict[key] = value.astype(float)
        output_dict["half_life"] = half_life.astype(float)
        output_dict["period"] = (int)(round(half_life * 2))
        output_dict["half_life_seconds"] = half_life_seconds.astype(float)
        output_dict["mean"] = mean.astype(float)
        output_dict["std"] = std.astype(float)
        output_dict["intercept"] = intercept.astype(float)
        output_dict['seconds_candles'] = seconds_candles
        output_dict['regression_price_type'] = regression_price_type
        return output_dict

    def backtest(self,
                 training_start_date: datetime.datetime,
                 training_end_date: datetime.datetime,
                 start_date: datetime.datetime,
                 end_date: datetime.datetime,
                 type=CandleType.CANDLE_MIDPRICE_TIME,
                 resolution: CandleTimeResolution = CandleTimeResolution.MIN,
                 num_units: int = 100,
                 period: int = 100, period_smooth_zscore: int = 100,
                 z_entry_buy: float = -2, z_entry_sell: float = 2,
                 z_exit_buy: float = -0.2, z_exit_sell: float = 0.2,
                 beta_hedge: bool = False, plot_it: bool = True,
                 return_all_df: bool = False,
                 test_size: float = 0.2,
                 save_best_path: str = None,
                 ):
        '''
        Backtest the strategy on the given dates
        Parameters
        ----------
        training_start_date
        training_end_date
        start_date
        end_date
        type
        resolution
        num_units
        period
        period_smooth_zscore
        z_entry_buy
        z_entry_sell
        z_exit_buy
        z_exit_sell
        beta_hedge
        plot_it
        return_all_df
        test_size
        Returns backtest_df and fig if return_all_df is True , returns backtest_df, fig, returns_df, price_df, summaries_df
        -------

        '''

        summaries_df = self.get_pair_summaries(start_date=training_start_date,
                                               end_date=training_end_date,
                                               resolutions=[resolution],
                                               num_units=[num_units],
                                               test_size=test_size,
                                               save_best_path=save_best_path
                                               )
        price_df = self.get_candle_close(
            start_date=start_date, end_date=end_date,
            type=type, resolution=resolution,
            num_units=num_units)

        returns_df = self.get_candle_returns(
            start_date=start_date, end_date=end_date,
            type=type, resolution=resolution,
            num_units=num_units)
        fig = None
        backtest_df = StatArbInstrument.get_backtest_df(returns_df=returns_df, price_df=price_df,
                                                        summaries_df=summaries_df, instrument_test=self.main_instrument,
                                                        list_instruments=self.synthetic_instruments,
                                                        num_units=num_units, resolution=resolution, period=period,
                                                        period_smooth_zscore=period_smooth_zscore,
                                                        beta_hedge=beta_hedge,
                                                        z_entry_buy=z_entry_buy, z_exit_buy=z_exit_buy,
                                                        z_entry_sell=z_entry_sell, z_exit_sell=z_exit_sell,
                                                        regression_price_type=self.regression_price_type
                                                        )
        if plot_it:
            fig = StatArbInstrument.plot_backtest_df(backtest_df, returns_df, self.main_instrument,
                                                     self.synthetic_instruments, num_units, resolution, period,
                                                     z_entry_buy=z_entry_buy, z_exit_buy=z_exit_buy,
                                                     z_entry_sell=z_entry_sell,
                                                     z_exit_sell=z_exit_sell)

        if return_all_df:
            return backtest_df, fig, returns_df, price_df, summaries_df

        return backtest_df, fig


# create main
if __name__ == '__main__':
    instrument_test = 'eurusd_darwinex'
    list_instruments = ['gbpusd_darwinex']
    training_days = 10
    backtesting_days = 1

    statarb_instrument = StatArbInstrument(
        main_instrument=instrument_test, list_instruments=list_instruments,
        regression_price_type=RegressionPriceType.close_prices
    )

    resolution = CandleTimeResolution.SECOND
    reference_date = datetime.datetime(year=2024, month=8, day=1)

    start_date = reference_date - BDay(training_days + backtesting_days)
    end_date = reference_date - BDay(backtesting_days)
    #
    # summaries_df = statarb_instrument.get_pair_summaries(
    #     start_date=start_date, end_date=start_date + BDay(1),
    #     candle_type=CandleType.CANDLE_MIDPRICE_TIME,
    #     resolutions=[resolution],
    #     num_units=DEFAULT_CANDLES_NUM_UNITS,
    #     spread_quantile=0.3,
    #     save_best_path="test.json"
    # )

    z_entry_buy = -2.0
    z_exit_buy = 0.5
    z_entry_sell = 2.0
    z_exit_sell = -0.5

    period = 40
    num_units = 5
    period_smooth_zscore = 0
    beta_hedge = True

    backstests_dict = {}
    backtest_day = reference_date - BDay(backtesting_days)
    backtest_start_date = backtest_day.replace(hour=0)
    backtest_end_date = backtest_day.replace(hour=23)
    days_add = 0
    start_date_temp = backtest_start_date + BDay(days_add)
    end_date_temp = backtest_end_date + BDay(days_add)

    training_start_date = start_date + BDay(days_add)
    training_end_date = end_date + BDay(days_add)
    print(rf"{start_date_temp} training from {training_start_date} to {training_end_date}")

    backtest2_df, fig = statarb_instrument.backtest(
        training_start_date=training_start_date, training_end_date=training_end_date,
        start_date=start_date_temp, end_date=end_date_temp,
        type=CandleType.CANDLE_MIDPRICE_TIME, resolution=CandleTimeResolution.MIN,
        num_units=num_units, period=period,
        period_smooth_zscore=period_smooth_zscore, beta_hedge=beta_hedge,
        z_entry_buy=z_entry_buy, z_exit_buy=z_exit_buy, z_entry_sell=z_entry_sell, z_exit_sell=z_exit_sell,
        plot_it=True
    )
