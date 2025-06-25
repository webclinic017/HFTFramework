$env:PYTHONPATH =$env:LAMBDA_PYTHON_PATH
conda activate lambda
python minute_candle_publisher_launcher.py localhost 666