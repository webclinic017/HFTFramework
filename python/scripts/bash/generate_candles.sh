cd ~/market_making_fw
git pull
cd ~

PATH=$PATH:~/.local/bin:$JAVA_HOME/bin:$M2_HOME/bin:$HADOOP_HOME/sbin:$HADOOP_HOME/bin:/usr/bin
PYTHONPATH=$PYTHONPATH:~/market_making_fw/python_lambda

export LAMBDA_OUTPUT_PATH="/home/tradeasystems/lambda_data/output_models"
export LAMBDA_INPUT_PATH="/home/tradeasystems/lambda_data/input_models"
export LAMBDA_TEMP_PATH="/home/tradeasystems/lambda_temp"
export LAMBDA_LOGS_PATH="/home/tradeasystems/lambda_data/logs"
export LAMBDA_PYTHON_PATH="/home/tradeasystems/market_making_fw/python_lambda"
export LAMBDA_DATA_PATH="/home/tradeasystems/lambda_data"
export LAMBDA_JAR_PATH="/home/tradeasystems/market_making_fw/java/executables/Backtest/bin"
export CANDLES_BATCH_SIZE="10000"
export CACHE_DISABLED="1"
SPLITS=8

for i in $(seq 1 $SPLITS)
do
    python3 ~/market_making_fw/python_lambda/scripts/generate_candles.py --splits $SPLITS --index $i >$LAMBDA_LOGS_PATH/scripts/generate_candles$i.log
done