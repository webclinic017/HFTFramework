SET PYTHONPATH=%LAMBDA_PYTHON_PATH%
conda run -n lambda python algotrading_zeromq_launcher.py -Xmx2048M
