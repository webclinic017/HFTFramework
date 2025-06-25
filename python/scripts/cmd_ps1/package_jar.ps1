cd ../../../java
git pull

cd parent_pom
mvn install -DskipTests=true
cd ../common
mvn install -DskipTests=true
cd ../algorithmic_trading_framework
mvn install -DskipTests=true
cd ../trading_algorithms
mvn install -DskipTests=true
cd ../backtest_engine
mvn install -DskipTests=true
cd ../executables/Backtest
mvn install package -DskipTests=true
cd ../AlgoTradingZeroMq
mvn install package -DskipTests=true
cd ../../../python_lambda