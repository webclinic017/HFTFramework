package com.lambda.investing.algorithmic_trading.market_making.constant_spread;


import com.google.common.primitives.Ints;
import com.lambda.investing.algorithmic_trading.AlgorithmConnectorConfiguration;
import com.lambda.investing.algorithmic_trading.LogLevels;
import com.lambda.investing.algorithmic_trading.market_making.reinforcement_learning.RLAbstractMarketMaking;
import com.lambda.investing.algorithmic_trading.reinforcement_learning.action.ConstantSpreadAction;
import org.apache.logging.log4j.LogManager;

import java.util.HashMap;
import java.util.Map;

public class AlphaConstantSpread extends RLAbstractMarketMaking {


    protected ConstantSpreadAlgorithm algorithm;

    public AlphaConstantSpread(AlgorithmConnectorConfiguration algorithmConnectorConfiguration, String algorithmInfo,
                               Map<String, Object> parameters) {
        super(algorithmConnectorConfiguration, algorithmInfo, parameters);
        logger = LogManager.getLogger(AlphaConstantSpread.class);
    }


    @Override
    public void setParameters(Map<String, Object> parameters) {
        super.setParameters(parameters);
        //ACTION configuration
        int[] levels = getParameterArrayInt(parameters, "levelAction");
        int[] skewLevels = getParameterArrayInt(parameters, "skewLevelAction");
        this.action = new ConstantSpreadAction(levels, skewLevels);
        //initial values to underlying
        //creation of the algorithm
        parameters.put("level", Ints.max(levels));
        parameters.put("skewLevel", 0);

        algorithm = new ConstantSpreadAlgorithm(algorithmConnectorConfiguration, algorithmInfo, parameters);
        setMarketMakerAlgorithm(algorithm, parameters);


        logger.info("[{}] initial values   {}\n level:{} skewLevel:{}", getCurrentTime(), algorithmInfo,
                algorithm.level, algorithm.skewLevel);

        //STATE configuration
        logger.info("[{}]set parameters  {}", getCurrentTime(), algorithmInfo);
        algorithmNotifier.notifyObserversOnUpdateParams(this.parameters);

    }

    @Override
    public String printAlgo() {
        return String
                .format("%s quantityBuy=%.5f quantitySell=%.5f   ConstantDQNSpreadAlgorithm level=%d  skew_level=%d",
                        algorithmInfo, algorithm.quantityBuy, algorithm.quantitySell,
                        algorithm.level, algorithm.skewLevel);
    }

    @Override
    protected void updateCurrentCustomColumn() {
        addCurrentCustomColumn(this.instrument.getPrimaryKey(), "level", (double) algorithm.level);
        addCurrentCustomColumn(this.instrument.getPrimaryKey(), "skewLevel", (double) algorithm.skewLevel);
        addCurrentCustomColumn(this.instrument.getPrimaryKey(), "iterations", (double) iterations.get());
        try {
            addCurrentCustomColumn(this.instrument.getPrimaryKey(), "bid", getLastDepth(instrument).getBestBid());
            addCurrentCustomColumn(this.instrument.getPrimaryKey(), "ask", getLastDepth(instrument).getBestAsk());
            addCurrentCustomColumn(this.instrument.getPrimaryKey(), "bid_qty", getLastDepth(instrument).getBestBidQty());
            addCurrentCustomColumn(this.instrument.getPrimaryKey(), "ask_qty", getLastDepth(instrument).getBestAskQty());
            addCurrentCustomColumn(this.instrument.getPrimaryKey(), "imbalance", getLastDepth(instrument).getImbalance());
            addCurrentCustomColumn(this.instrument.getPrimaryKey(), "reward", (double) this.rewardStartStep);
        } catch (Exception e) {
        }

    }

    @Override
    public void setAction(double[] actionValues) {
        super.setAction(actionValues);

        if (actionValues == null || actionValues.length == 0) {
            logger.warn("[{}] setAction called with empty actionValues before -> skip it", this.getCurrentTime());
            return;
        }
        actionValues = GetActionValues(actionValues);
        if (actionValues == null || actionValues.length < ConstantSpreadAction.SIZE_ARRAY_ACTION) {
            if (actionValues == null) {
                logger.error("[{}] setAction called with empty actionValues -> skip it", this.getCurrentTime());
            } else {
                logger.error("[{}] setAction called with empty actionValues or {} less columns than {} -> skip it", this.getCurrentTime(), actionValues.length, ConstantSpreadAction.SIZE_ARRAY_ACTION);
            }
            return;
        }

        if (!Double.isNaN(actionValues[ConstantSpreadAction.LEVEL_INDEX]))
            algorithm.level = (int) Math.round(actionValues[ConstantSpreadAction.LEVEL_INDEX]);

        if (!Double.isNaN(actionValues[ConstantSpreadAction.SKEW_LEVEL_INDEX]))
            algorithm.skewLevel = (int) Math.round(actionValues[ConstantSpreadAction.SKEW_LEVEL_INDEX]);

        if (LOG_LEVEL > LogLevels.SOME_ITERATION_LOG.ordinal()) {
            //// Here the action is not finished
            logger.info("[{}][iteration {} start] actionReceived: {} reward:{}  ->level={}  skewLevel={}", this.getCurrentTime(),
                    this.iterations.get() - 1, getLastActionString(), getCurrentReward(), algorithm.level, algorithm.skewLevel);
        }
        notifyParameters();
    }

    @Override
    protected void notifyParameters() {
        Map<String, Object> parameters = new HashMap<>();
        parameters.put("level", algorithm.level);
        parameters.put("skewLevel", algorithm.skewLevel);

        algorithmNotifier.notifyObserversOnUpdateParams(parameters);
    }

    @Override
    protected boolean onFinishedIteration(long msElapsed, double reward, double[] state) {
        if (super.onFinishedIteration(msElapsed, reward, state)) {
            if (LOG_LEVEL > LogLevels.SOME_ITERATION_LOG.ordinal()) {
                logger.info(
                        "[{}][iteration {} end] onFinishedIteration  {} ms later  reward={}  -> level={}  skewLevel={}",
                        this.getCurrentTime(), iterations.get() - 1, msElapsed, getLastReward(), algorithm.level,
                        algorithm.skewLevel);
            }
        }
        return true;
    }

}
