$DATE_FOLDER =Get-Date -Format "yyMMdd"
$LOG_PATH="$env:LAMBDA_LOGS_PATH\$DATE_FOLDER\minute_candle_publisher.log"
Write-Host "Logging into "$LOG_PATH
.\minute_candle_publisher.ps1 >> $LOG_PATH
