#$env:TERM="ansi"
#conda init
git pull --force
.\package_jar.ps1
#better to add to environment settings
#$env:PYTHONPATH ="C:\Users\javif\Coding\market_making_fw\python_lambda"
.\only_notebook_lambda.ps1

#conda activate lambda
#cd notebooks
#jupyter notebook --ip=* --port=8890 --NotebookApp.iopub_data_rate_limit=100000 --NotebookApp.iopub_msg_rate_limit=8000 --NotebookApp.rate_limit_window=8000 --no-browser