import signal
import subprocess
from enum import Enum
import glob
import datetime


from trading_algorithms.algorithm_enum import AlgorithmEnum
from backtest.input_configuration import (
    InputConfiguration,
    BacktestConfiguration,
    AlgorithmConfiguration, MultiThreadConfiguration,
)
import os
import threading
import time
from pathlib import Path
import pandas as pd
from configuration import LAMBDA_OUTPUT_PATH, operative_system, is_jupyter_notebook

from utils.pandas_utils.dataframe_utils import join_by_row


class BacktestState(Enum):
    created = 0
    running = 1
    finished = 2


OUTPUT_PATH = LAMBDA_OUTPUT_PATH
REMOVE_FINAL_CSV = True
REMOVE_INPUT_JSON = True

DEFAULT_JVM_WIN = '-Xmx6G'
DEFAULT_JVM_UNIX = '-Xmx6G'
TIMEOUT_SECONDS = 60 * 20  # 20 minutes


class ListenerKill:
    def notify_kill(self, code):
        pass


class TimeoutException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class BacktestLauncher:
    VERBOSE_OUTPUT = False
    if operative_system == 'windows':
        DEFAULT_JVM = DEFAULT_JVM_WIN
    else:
        DEFAULT_JVM = DEFAULT_JVM_UNIX

    def __init__(
            self,
            input_configuration: InputConfiguration,
            id: str,
            jar_path='Backtest.jar',
            jvm_options: str = DEFAULT_JVM,
    ):
        self.pid = None
        self.proc = None
        self._thread = None

        self.input_configuration = input_configuration
        self.jar_path = jar_path
        self.output_path = OUTPUT_PATH
        self.class_path_folder = Path(self.jar_path).parent

        self.algorithm_name = (
            self.input_configuration.algorithm_configuration.algorithm_name
        )
        self.jvm_options = jvm_options + f' -Dlog.appName={self.algorithm_name}'  # change log name
        self.jvm_options += (
            f' -Duser.timezone=GMT'  # GMT for printing logs and backtest configuration
        )

        self.task = 'java %s -jar %s' % (self.jvm_options, self.jar_path)
        self.state = BacktestState.created
        self.id = id
        self.listeners_kill = []

        if not os.path.isdir(self.output_path):
            print("mkdir %s" % self.output_path)
            os.mkdir(self.output_path)

    def __getstate__(self):
        """Return state values to be pickled."""
        state = self.__dict__.copy()
        # Don't pickle non-serializable objects
        state['proc'] = None
        state['pid'] = None
        state['_thread'] = None
        return state

    def __setstate__(self, state):
        """Restore state from the unpickled state values."""
        self.__dict__.update(state)
        # Reset non-serializable objects
        self.proc = None
        self.pid = None
        self._thread = None

    def register_kill_listener(self, listener: ListenerKill):
        self.listeners_kill.append(listener)

    def notify_kill_listeners(self, code: int):
        for listener in self.listeners_kill:
            listener.notify_kill(code)

    def _launch_process_os(self, command_to_run: str):
        '''
        java process is not printing in notebook
        Parameters
        ----------
        command_to_run

        Returns
        -------

        '''
        self.pid = None
        ret = os.system(command_to_run)
        return ret

    def _wait_subproccess(self):
        ret = self.proc.wait(timeout=TIMEOUT_SECONDS)
        return ret
        # if not is_jupyter_notebook():
        #     ret = self.proc.wait(timeout=TIMEOUT_SECONDS)
        #     return ret
        # else:
        #     start_time = time.time()
        #     while self.proc.poll() is None:
        #         elapsed_seconds = time.time() - start_time
        #         if elapsed_seconds > TIMEOUT_SECONDS:
        #             raise TimeoutException(
        #                 "timeout %d seconds expired subprocess pid:%d in jupyter"
        #                 % (TIMEOUT_SECONDS, self.proc.pid)
        #             )
        #         text = self.proc.stdout.read1().decode("utf-8")
        #
        #         print(text, end='', flush=True)
        #
        #     ret = self.proc.returncode
        #     return ret

    def _launch_process_subprocess(self, command_to_run: str):
        import subprocess

        # https://stackoverflow.com/questions/56138384/capture-jupyter-notebook-stdout-with-subprocess
        stderr_option = None
        stdout_option = None
        #
        # if not is_jupyter_notebook():
        #     stderr_option = None
        #     stdout_option = None
        # else:
        #     stderr_option = subprocess.STDOUT
        #     stdout_option = subprocess.STDOUT

        self.proc = subprocess.Popen(
            command_to_run, stderr=stderr_option, stdout=stdout_option
        )  # <-- redirect stderr to stdout

        self.pid = (
            self.proc.pid
        )  # <--- access `pid` attribute to get the pid of the child process.
        try:
            ret = self._wait_subproccess()
        except (subprocess.TimeoutExpired, TimeoutException):
            print(
                rf"{datetime.datetime.now()} timeout java {self.id} {self.algorithm_name} {command_to_run} {TIMEOUT_SECONDS} seconds expired -> kill the process pid:{self.proc.pid}"
            )
            self.proc.kill()
            ret = -1

        return ret

    def run(self):
        self.state = BacktestState.running
        file_content = self.input_configuration.get_json()

        # save it into file
        filename = os.getcwd() + os.sep + self.input_configuration.get_filename()
        textfile = open(filename, 'w')
        textfile.write(file_content)
        textfile.close()

        command_to_run = self.task + ' %s 1' % filename
        print('pwd=%s' % os.getcwd())
        if self.VERBOSE_OUTPUT:
            command_to_run += '>%sout.log' % (os.getcwd() + os.sep)

        ret = self._launch_process_subprocess(command_to_run)
        # ret = self._launch_process_os(command_to_run)
        if ret != 0:
            print(f"{datetime.datetime.now()} error  launching %s" % (command_to_run))

        print(f'{datetime.datetime.now()} %s %s finished with code %d' % (self.id, self.algorithm_name, ret))
        self.notify_kill_listeners(ret)
        self.state = BacktestState.finished
        # remove input file
        if REMOVE_INPUT_JSON and os.path.exists(filename):
            os.remove(filename)

        self.proc = None
        self.pid = None

    def start(self):
        """Start the backtest in a separate thread."""
        self._thread = threading.Thread(target=self.run)
        self._thread.start()

    def is_alive(self):
        """Check if the thread is still running."""
        if self._thread:
            return self._thread.is_alive()
        return False

    def join(self, timeout=None):
        """Wait for the thread to complete."""
        if self._thread:
            self._thread.join(timeout)

    def kill(self):
        import traceback

        try:
            if self.proc is not None:
                print(rf"WARNING: kill the process pid:{self.proc.pid}")
                # traceback.print_stack()
                self.proc.kill()
                return

            if self.pid is not None:
                print(rf"WARNING: kill the os.pid:{self.pid}")
                # traceback.print_stack()
                os.kill(self.pid, signal.SIGTERM)
        except Exception as e:
            print(f"kill error:{e}")

    def __del__(self):
        self.kill()


class BacktestLauncherController:
    def __init__(self, backtest_launchers: list, max_simultaneous: int = 4):
        self.backtest_launchers = backtest_launchers
        self.max_simultaneous = max_simultaneous
        self.last_output = {}

    def _initial_clean(self, backtest_launcher):
        input_configuration = backtest_launcher.input_configuration
        algo_name = input_configuration.algorithm_configuration.algorithm_name

        csv_filenames = glob.glob(
            backtest_launcher.output_path + os.sep + 'trades_table_%s_*.csv' % algo_name
        )
        for csv_filename in csv_filenames:
            os.remove(csv_filename)

    def execute_lambda(self):
        sent = []
        start_time = time.time()
        while 1:
            running = 0
            for backtest_launcher in self.backtest_launchers:
                if backtest_launcher.state == BacktestState.running:
                    running += 1
            if (self.max_simultaneous - running) > 0:
                backtest_waiting = [
                    backtest
                    for backtest in self.backtest_launchers
                    if backtest not in sent
                ]

                for idx in range(
                        min(self.max_simultaneous - running, len(backtest_waiting))
                ):
                    backtest_launcher = backtest_waiting[idx]
                    print("launching %s" % backtest_launcher.id)
                    self._initial_clean(backtest_launcher=backtest_launcher)
                    backtest_launcher.start()
                    sent.append(backtest_launcher)

            processed = [t for t in sent if t.state == BacktestState.finished]
            if len(processed) == len(self.backtest_launchers):
                seconds_elapsed = time.time() - start_time
                print(
                    'finished %d backtests in %d minutes'
                    % (len(self.backtest_launchers), seconds_elapsed / 60)
                )
                break
            time.sleep(0.01)

    def execute_joblib(self):
        from utils.paralellization_util import process_jobs
        from pathos.multiprocessing import ProcessPool

        jobs = []
        for backtest_launcher in self.backtest_launchers:
            job = {"func": backtest_launcher.run}
            jobs.append(job)
        process_jobs(jobs=jobs, num_threads=self.max_simultaneous, pool_class=ProcessPool)

    def _get_trades_raw_df(self, backtest_launcher, algo_name: str, path: list) -> dict:
        csv_filenames = glob.glob(
            backtest_launcher.output_path + os.sep + f'trades_table_{algo_name}_*.csv'
        )
        output = {}
        for csv_filename in csv_filenames:
            path.append(csv_filename)
            try:
                df_temp = pd.read_csv(csv_filename)
                instrument_pk_list = csv_filename.split(os.sep)[-1].split('_')[-2:]
                instrument_pk = '_'.join(instrument_pk_list).split('.')[0]
                output[instrument_pk] = df_temp
            except Exception as e:
                print(f'something goes wrong reading output csv {csv_filename} : {e}')
                continue
        return output, path

    def _get_trades_df(self, backtest_launcher, algo_name: str, path: list, start_arb: bool,
                       resample_period: str) -> pd.DataFrame:
        csv_filenames = glob.glob(
            backtest_launcher.output_path + os.sep + f'trades_table_{algo_name}_*.csv'
        )
        df = None
        instrument_pks = []
        for csv_filename in csv_filenames:
            try:
                df_temp = pd.read_csv(csv_filename)
                instrument_pk_list = csv_filename.split(os.sep)[-1].split('_')[-2:]
                instrument_pk = '_'.join(instrument_pk_list).split('.')[0]
                instrument_pks.append(instrument_pk)
                df_temp['main_instrument'] = False
                if instrument_pk == backtest_launcher.input_configuration.backtest_configuration.instrument_pk:
                    df_temp['main_instrument'] = True

                df_temp['historicalUnrealizedPnl'] = df_temp['historicalUnrealizedPnl'].diff().fillna(0.0)
                df_temp['historicalTotalPnl'] = df_temp['historicalTotalPnl'].diff().fillna(0.0)
                df_temp['historicalRealizedPnl'] = df_temp['historicalRealizedPnl'].diff().fillna(0.0)
                df_temp['numberTrades'] = df_temp['numberTrades'].diff().fillna(0.0)
                df_temp[f'netPosition_{instrument_pk}'] = df_temp['netPosition']
                df_temp[f'price_{instrument_pk}'] = df_temp['price']
                df_temp[f'bidPrice_{instrument_pk}'] = df_temp['bidPrice']
                df_temp[f'askPrice_{instrument_pk}'] = df_temp['askPrice']
                df_temp[f'midPrice_{instrument_pk}'] = df_temp['midPrice']
                df_temp[f'quantity_{instrument_pk}'] = df_temp['quantity']
                if df is None:
                    df = df_temp
                else:
                    df = join_by_row(df, df_temp)

                path.append(csv_filename)
            except Exception as e:
                print(f'something goes wrong reading output csv {csv_filename} : {e}')
                continue

        if df is not None and len(df) > 0:
            if isinstance(df['timestamp'], str):
                df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
            # df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.sort_values(by=['date'], ascending=[True])
            if start_arb:
                columns_copy_from_main = ['zscore_buy', 'zscore_sell', 'zscore_mid', 'action']
                df[columns_copy_from_main] = df.groupby('numberTrades')[columns_copy_from_main].ffill().bfill().fillna(
                    0.0)
            df['numberTrades'] = df['numberTrades'].cumsum()
            df['historicalUnrealizedPnl'] = df['historicalUnrealizedPnl'].cumsum()
            df['historicalRealizedPnl'] = df['historicalRealizedPnl'].cumsum()
            df['historicalTotalPnl'] = df['historicalUnrealizedPnl'] + df['historicalRealizedPnl']
            df['netInvestment'] = 0
            df['netPosition'] = df.loc[df['main_instrument'] == True, 'netPosition'].ffill()

            df['bidPrice'] = df.loc[df['main_instrument'] == True, 'bidPrice'].ffill()
            df['askPrice'] = df.loc[df['main_instrument'] == True, 'askPrice'].ffill()
            df['midPrice'] = df.loc[df['main_instrument'] == True, 'midPrice'].ffill()
            df['price'] = df.loc[df['main_instrument'] == True, 'price'].ffill()
            df['quantity'] = df.loc[df['main_instrument'] == True, 'quantity'].ffill()

            for instrument_pk in instrument_pks:
                df.loc[:, f'netPosition_{instrument_pk}'] = df[f'netPosition_{instrument_pk}'].ffill()
                df.loc[:, f'price_{instrument_pk}'] = df[f'price_{instrument_pk}'].ffill()
                df.loc[:, f'quantity_{instrument_pk}'] = df[f'quantity_{instrument_pk}'].ffill()
                df.loc[:, f'bidPrice_{instrument_pk}'] = df[f'bidPrice_{instrument_pk}'].ffill()
                df.loc[:, f'askPrice_{instrument_pk}'] = df[f'askPrice_{instrument_pk}'].ffill()
                df.loc[:, f'midPrice_{instrument_pk}'] = df[f'midPrice_{instrument_pk}'].ffill()

                df['netInvestment'] += df[f'price_{instrument_pk}'] * df[f'quantity_{instrument_pk}']

            # drop individual strategies columns
            drop_columns = ["clientOrderId", "verb", "fee", "avgOpenPrice",
                            "instrument", "main_instrument", "algorithmInfo"]
            df.drop(columns=drop_columns, inplace=True, errors='ignore')
            df.ffill(inplace=True)
            # df groupby column date in 10 seconds and last
            resampled = df.set_index('date').resample(resample_period).last().ffill()
            df = resampled.reset_index()

        return df, path

    def run(self, raw_results: bool = False) -> dict:
        from configuration import SHARPE_BACKTEST_FREQ
        # self.execute_lambda()
        self.execute_joblib()
        # get output dataframes
        output = {}

        for idx, backtest_launcher in enumerate(self.backtest_launchers):
            path = []
            df = None
            input_configuration = backtest_launcher.input_configuration
            algo_name = input_configuration.algorithm_configuration.algorithm_name
            output[backtest_launcher.id] = None
            stat_arb = False
            if algo_name.startswith(AlgorithmEnum.stat_arb) or algo_name.startswith(AlgorithmEnum.stat_arb_quoting):
                stat_arb = True
            if raw_results:
                df_dicts, path = self._get_trades_raw_df(backtest_launcher, algo_name, path)
                output[backtest_launcher.id] = df_dicts
            else:
                df, path = self._get_trades_df(backtest_launcher, algo_name, path, start_arb=stat_arb,
                                               resample_period=SHARPE_BACKTEST_FREQ)
                if df is None:
                    continue

                output[backtest_launcher.id] = df
                if df is None or len(df) == 0:
                    print('%s with None trades' % (algo_name))
                else:
                    print(
                        f'{algo_name} with {df["numberTrades"].max()} trades [{idx}/{len(self.backtest_launchers) - 1}]'
                    )

            if REMOVE_FINAL_CSV:
                try:
                    if isinstance(path, str) and os.path.exists(path):
                        os.remove(path)
                    if isinstance(path, list):
                        for path_i in path:
                            if os.path.exists(path_i):
                                os.remove(path_i)
                except Exception as e:
                    print('error removing csv %s : %s' % (path, str(e)))

            # remove position json
            position_files = glob.glob(
                backtest_launcher.output_path
                + os.sep
                + '%s_paperTradingEngine_position.json' % (algo_name)
            )
            try:
                for position_file in position_files:
                    if os.path.exists(position_file):
                        os.remove(position_file)
            except Exception as e:
                print(
                    'error removing position json %s : %s'
                    % (backtest_launcher.output_path, str(e))
                )

        self.last_output = output
        print(rf"Finished backtest with {len(self.backtest_launchers)} launchers")
        return output


if __name__ == '__main__':
    import datetime
    from trading_algorithms.algorithm import Algorithm, AlgorithmParameters
    from trading_algorithms.market_making.avellaneda_stoikov import AvellanedaStoikovParameters

    Algorithm.MULTITHREAD_CONFIGURATION = MultiThreadConfiguration.singlethread
    Algorithm.DELAY_MS = 0


    backtest_configuration = BacktestConfiguration(
        start_date=datetime.datetime(year=2020, day=8, month=12),
        end_date=datetime.datetime(year=2020, day=8, month=12),
        instrument_pk='btcusdt_binance',
    )

    parameters = {
        AvellanedaStoikovParameters.risk_aversion: "0.9",
        AvellanedaStoikovParameters.position_multiplier: "100",
        AvellanedaStoikovParameters.midprice_period_window: "100",
        AvellanedaStoikovParameters.seconds_change_k: "600",
        AlgorithmParameters.quantity: "0.0001",
        AvellanedaStoikovParameters.k_default: "0.00769",
        AvellanedaStoikovParameters.spread_multiplier: "5.0",
        AlgorithmParameters.first_hour: "7",
        AlgorithmParameters.last_hour: "19",
    }

    algorith_configuration = AlgorithmConfiguration(
        algorithm_name='AvellanedaStoikov', parameters=parameters
    )

    parameters_2 = {
        AvellanedaStoikovParameters.risk_aversion: "0.2",
        AvellanedaStoikovParameters.position_multiplier: "100",
        AvellanedaStoikovParameters.midprice_period_window: "100",
        AvellanedaStoikovParameters.seconds_change_k: "600",
        AlgorithmParameters.quantity: "0.0001",
        AvellanedaStoikovParameters.k_default: "0.00769",
        AvellanedaStoikovParameters.spread_multiplier: "5.0",
        AlgorithmParameters.first_hour: "7",
        AlgorithmParameters.last_hour: "19",
    }

    algorith_configuration_2 = AlgorithmConfiguration(
        algorithm_name='AvellanedaStoikov', parameters=parameters_2
    )

    input_configuration = InputConfiguration(
        backtest_configuration=backtest_configuration,
        algorithm_configuration=algorith_configuration,
    )
    backtest_launcher_1 = BacktestLauncher(
        input_configuration=input_configuration,
        id='main_test_as',
        # jar_path=rf'D:\javif\Coding\cryptotradingdesk\java\executables\Backtest\target\Backtest.jar',
    )

    input_configuration_2 = InputConfiguration(
        backtest_configuration=backtest_configuration,
        algorithm_configuration=algorith_configuration_2,
    )

    backtest_launcher_2 = BacktestLauncher(
        input_configuration=input_configuration_2,
        id='main_test_as_2',
        # jar_path=rf'D:\javif\Coding\cryptotradingdesk\java\executables\Backtest\target\Backtest.jar',
    )

    backtest_launchers = [backtest_launcher_1, backtest_launcher_2]
    # train_launchers = [backtest_launcher_1]
    backtest_controller = BacktestLauncherController(
        backtest_launchers=backtest_launchers
    )

    backtest_controller.run()
