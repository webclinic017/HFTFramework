package com.lambda.investing.algorithmic_trading.market_making.constant_spread;


import com.lambda.investing.algorithmic_trading.AlgorithmConnectorConfiguration;
import com.lambda.investing.algorithmic_trading.market_making.MarketMakingAlgorithm;
import com.lambda.investing.model.exception.LambdaTradingException;
import com.lambda.investing.model.market_data.Depth;
import com.lambda.investing.model.trading.ExecutionReport;
import com.lambda.investing.model.trading.OrderRequest;
import com.lambda.investing.model.trading.QuoteRequest;
import com.lambda.investing.model.trading.QuoteRequestAction;
import org.apache.commons.math3.util.Precision;

import java.util.Map;

public class ConstantSpreadAlgorithm extends MarketMakingAlgorithm {

    private static double MAX_TICKS_MIDPRICE_PRICE_DEV = 100;
    private double minQuantityFollow;
    public int level;//0-4
    public int skewLevel;//-4 - 4
    private double quantity;
    private double quantityLimit;
    private double lastValidSpread, lastValidMid = 0.01;

    public ConstantSpreadAlgorithm(AlgorithmConnectorConfiguration algorithmConnectorConfiguration,
                                   String algorithmInfo, Map<String, Object> parameters) {
        super(algorithmConnectorConfiguration, algorithmInfo, parameters);
        setParameters(parameters);
    }

    public ConstantSpreadAlgorithm(String algorithmInfo, Map<String, Object> parameters) {
        super(algorithmInfo, parameters);
        setParameters(parameters);
    }


    @Override
    public void setParameters(Map<String, Object> parameters) {
        super.setParameters(parameters);
        this.level = getParameterIntOrDefault(parameters, "level", 0);
        this.skewLevel = getParameterIntOrDefault(parameters, "skewLevel", 0);
        this.quantity = getParameterDouble(parameters, "quantity");
        this.quantityBuy = quantity;
        this.quantitySell = quantity;
        this.quantityLimit = getParameterDoubleOrDefault(parameters, "quantityLimit", "quantity_limit", -1);
        this.minQuantityFollow = getParameterDoubleOrDefault(parameters, "minQuantityFollow", 0.0);
    }


    @Override
    public String printAlgo() {
        if (quantityLimit != -1) {
            return String
                    .format("%s  level=%d  quantity=%.5f minQuantityFollow=%.5f skewLevel=%d quantityLimit=%.5f ", algorithmInfo, level,
                            quantity, minQuantityFollow, skewLevel, quantityLimit);
        } else {
            return String
                    .format("%s  level=%d    quantity=%.5f minQuantityFollow=%.5f skewLevel=%d", algorithmInfo, level, quantity, minQuantityFollow, skewLevel);
        }
    }

    //	@Override public AlgorithmState getAlgorithmState() {
    //		return AlgorithmState.STARTED;
    //	}


    @Override
    public boolean onDepthUpdate(Depth depth) {
        boolean output = super.onDepthUpdate(depth);
        if (!output || !depth.getInstrument().equals(instrument.getPrimaryKey())) {
            return false;
        }

        if (!depth.isDepthFilled()) {
            logger.info("stopping algorithm because depth is incomplete!");
            stop();
            return false;
        } else {
            start();
        }

        try {

            double currentSpread = 0;
            double midPrice = 0;
            double askPrice = 0.0;
            double bidPrice = 0.0;
            boolean askPriceValid = true;
            boolean bidPriceValid = true;
            try {

                currentSpread = depth.getSpread();
                midPrice = depth.getMidPrice();
                //ASK = level+skew
                double candidatePriceAsk = depth.getAskPriceFromLevel(level, minQuantityFollow);
                if (candidatePriceAsk == Double.MAX_VALUE) {
                    candidatePriceAsk = depth.getWorstAsk();
                    askPriceValid = false;
                }
                int askPriceLevel = Math.max(depth.getLevelAskFromPrice(candidatePriceAsk) + skewLevel, 0);
                askPrice = depth.getAskPriceFromLevel(askPriceLevel);
                if (askPrice == Double.MAX_VALUE) {
                    askPrice = depth.getWorstAsk();
//                    askPriceValid = false;
                }
                //BID = level-skew
                double candidatePriceBid = depth.getBidPriceFromLevel(level, minQuantityFollow);
                if (candidatePriceBid == Double.MIN_VALUE) {
                    candidatePriceBid = depth.getWorstBid();
                    bidPriceValid = false;
                }
                int bidPriceLevel = Math.max(depth.getLevelBidFromPrice(candidatePriceBid) - skewLevel, 0);
                bidPrice = depth.getBidPriceFromLevel(bidPriceLevel);
                if (bidPrice == Double.MIN_VALUE) {
                    bidPrice = depth.getWorstBid();
//                    bidPriceValid = false;
                }
            } catch (Exception e) {
                return false;
            }

            if (currentSpread == 0) {
                currentSpread = lastValidSpread;
            } else {
                lastValidSpread = currentSpread;
            }

            if (midPrice == 0) {
                midPrice = lastValidMid;
            } else {
                lastValidMid = midPrice;
            }

            askPrice = Precision.round(askPrice, instrument.getNumberDecimalsPrice());

            double askQty = this.quantity;
            if (this.quantityLimit > 0 && getPosition(this.instrument) < -quantityLimit) {
                askQty = 0.0;
            }


            bidPrice = Precision.round(bidPrice, instrument.getNumberDecimalsPrice());
            double bidQty = this.quantity;

            if (this.quantityLimit > 0 && getPosition(this.instrument) > quantityLimit) {
                bidQty = 0.0;
            }

            if (!bidPriceValid) {
                logger.info("bidPrice out of limits for quantityLimit {} unquote bid", quantityLimit);
                bidQty = 0.0;
            }
            if (!askPriceValid) {
                logger.info("askPrice out of limits for quantityLimit {} unquote ask", quantityLimit);
                askQty = 0.0;
            }
            //Check not crossing the mid price!
            askPrice = Math.max(askPrice, depth.getMidPrice() + instrument.getPriceTick());
            bidPrice = Math.min(bidPrice, depth.getMidPrice() - instrument.getPriceTick());

            //			Check worst price
            //			double maxAskPrice = depth.getMidPrice() + MAX_TICKS_MIDPRICE_PRICE_DEV * instrument.getPriceTick();
            //			askPrice = Math.min(askPrice, maxAskPrice);
            //			double minBidPrice = depth.getMidPrice() - MAX_TICKS_MIDPRICE_PRICE_DEV * instrument.getPriceTick();
            //			bidPrice = Math.max(bidPrice, minBidPrice);


            //create quote request
            QuoteRequest quoteRequest = createQuoteRequest(this.instrument);
            quoteRequest.setQuoteRequestAction(QuoteRequestAction.On);
            quoteRequest.setBidPrice(bidPrice);
            quoteRequest.setAskPrice(askPrice);
            quoteRequest.setBidQuantity(bidQty);
            quoteRequest.setAskQuantity(askQty);

            try {
                sendQuoteRequest(quoteRequest);

                //				logger.info("quoting  {} bid {}@{}   ask {}@{}", instrument.getPrimaryKey(), quantity, bidPrice,
                //						quantity, askPrice);

            } catch (LambdaTradingException e) {
                logger.error("can't quote {} bid {}@{}   ask {}@{}", instrument.getPrimaryKey(), quantity, bidPrice,
                        quantity, askPrice, e);
            }
        } catch (Exception e) {
            logger.error("error onDepth  : ", e);
        }

        return true;
    }

    @Override
    public void sendOrderRequest(OrderRequest orderRequest) throws LambdaTradingException {
        //		logger.info("sendOrderRequest {} {}", orderRequest.getOrderRequestAction(), orderRequest.getClientOrderId());
        super.sendOrderRequest(orderRequest);

    }

    @Override
    public boolean onExecutionReportUpdate(ExecutionReport executionReport) {
        super.onExecutionReportUpdate(executionReport);

        //		logger.info("onExecutionReportUpdate  {}  {}:  {}", executionReport.getExecutionReportStatus(),
        //				executionReport.getClientOrderId(), executionReport.getRejectReason());

        //		boolean isTrade = executionReport.getExecutionReportStatus().equals(ExecutionReportStatus.CompletellyFilled)
        //				|| executionReport.getExecutionReportStatus().equals(ExecutionReportStatus.PartialFilled);

        //		if (isTrade) {
        //			try {
        //				//				logger.info("{} received {}  {}@{}",executionReport.getExecutionReportStatus(),executionReport.getVerb(),executionReport.getLastQuantity(),executionReport.getPrice());
        //				QuoteRequest quoteRequest = createQuoteRequest(executionReport.getInstrument());
        //				quoteRequest.setQuoteRequestAction(QuoteRequestAction.Off);
        //				sendQuoteRequest(quoteRequest);
        //				//				logger.info("unquoting because of trade in {} {}", executionReport.getVerb(),
        //				//						executionReport.getClientOrderId());
        //			} catch (LambdaTradingException e) {
        //				logger.error("cant unquote {}", instrument.getPrimaryKey(), e);
        //			}
        //		}
        return true;
    }
}
