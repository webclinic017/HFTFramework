#$env:TERM="ansi"
#conda init
git pull --force
#better to add to environment settings
#$env:PYTHONPATH ="C:\Users\javif\Coding\market_making_fw\python_lambda"
conda activate lambda2
cd notebooks
jupyter notebook --ip=* --port=8890 --no-browser