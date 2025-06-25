$env:PYTHONPATH =$env:LAMBDA_PYTHON_PATH
$env:LAMBDA_LOGS_PATH="X:\logs"
conda activate lambda2
python algotrading_zeromq_launcher.py -Xmx2048M