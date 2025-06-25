package com.lambda.investing.algorithmic_trading.hedging;

import com.lambda.investing.algorithmic_trading.Algorithm;
import com.lambda.investing.algorithmic_trading.hedging.synthetic_portfolio.SyntheticInstrument;
import com.lambda.investing.model.asset.Instrument;
import com.lambda.investing.model.exception.LambdaTradingException;
import com.lambda.investing.model.market_data.Depth;
import com.lambda.investing.model.market_data.Trade;
import com.lambda.investing.model.trading.ExecutionReport;
import com.lambda.investing.model.trading.OrderRequest;
import com.lambda.investing.model.trading.Verb;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.FileNotFoundException;
import java.util.Date;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

public class LinearRegressionHedgeManager implements HedgeManager {

    protected Logger logger = LogManager.getLogger(LinearRegressionHedgeManager.class);
    protected Instrument instrument;
    protected SyntheticInstrument syntheticInstrument;
    protected Set<Instrument> interestedInstruments = new HashSet<>();
    protected Set<String> interestedInstrumentPks = new HashSet<>();
    protected String syntheticInstrumentFile;
    protected Map<String, Double> askMap = new ConcurrentHashMap<>();
    protected Map<String, Double> bidMap = new ConcurrentHashMap<>();

    public LinearRegressionHedgeManager(Instrument instrument, String syntheticInstrumentFile) throws FileNotFoundException {
        this.instrument = instrument;
        this.syntheticInstrumentFile = syntheticInstrumentFile;
        setSyntheticInstrument(new SyntheticInstrument(this.syntheticInstrumentFile));
    }

    public void setSyntheticInstrument(SyntheticInstrument instrument) {
        this.syntheticInstrument = instrument;
        interestedInstruments.addAll(this.syntheticInstrument.getInstruments());
        interestedInstrumentPks.addAll(this.syntheticInstrument.getInstrumentPks());
    }

    @Override
    public boolean onExecutionReportUpdate(ExecutionReport executionReport) {
        //Hedger instrument

        if (interestedInstrumentPks.contains(executionReport.getInstrument())) {
            logger.info("[{}] {}-{}  {}", new Date(executionReport.getTimestampCreation()),
                    executionReport.getClientOrderId(), executionReport.getExecutionReportStatus(),
                    executionReport);
        }

        return false;
    }

    @Override
    public boolean onDepthUpdate(Depth depth) {
        askMap.put(depth.getInstrument(), depth.getBestAsk());
        bidMap.put(depth.getInstrument(), depth.getBestBid());
        return true;
    }

    @Override
    public boolean onTradeUpdate(Trade trade) {
        return true;
    }

    protected void tradeSpread(Algorithm algorithm, Verb verbMain, double quantity) {
        hedgeSyntheticInstrumentsMarket(algorithm, verbMain, quantity);

    }

    private double getHedgeRatio(Verb verbMain, Instrument underlyingInstrument, double beta) {
        // if main moves 1 is explained by hedger moves beta (0.5) => to hedge main with hedger we have to invest beta (2)
        // in python stat_arb_instrument.stat_arb_instrument.StatArbInstrument.get_hedge_ratio
        return beta;
    }


    protected void hedgeSyntheticInstrumentsMarket(Algorithm algorithm, Verb verbMain, double quantityOnMain) {
        Verb verbHedge = Verb.OtherSideVerb(verbMain);

        for (Instrument underlyingInstrument : this.syntheticInstrument.getInstruments()) {
            double beta = this.syntheticInstrument.getBeta(underlyingInstrument);
            OrderRequest orderRequestSynth = null;
            double hedgeRatio = getHedgeRatio(verbMain, underlyingInstrument, beta);
            double quantityTrade = Math.abs(quantityOnMain * hedgeRatio);
            quantityTrade = underlyingInstrument.roundQty(quantityTrade);

            if (beta == 0) {
                continue;
            }
            if (quantityTrade == 0) {
                logger.warn(
                        "something is wrong trying to trade zeroQuantity on synthetic {}  beta:{}   main qty:{}  => more qty on main or remove instrument",
                        underlyingInstrument, beta, quantityOnMain);
                continue;
            }
            double minQty = underlyingInstrument.getQuantityTick();
            if (quantityTrade < minQty) {
                logger.warn("no hedging on {} with beta:{} because quantityToTrade {} < {}minQty instrument",
                        underlyingInstrument, beta, quantityTrade, minQty);
            }

            Verb verbHedgeUnderlyingInstrument = verbHedge;
            if (beta < 0) {
                //reversed
                verbHedgeUnderlyingInstrument = Verb.OtherSideVerb(verbHedge);
            }

            orderRequestSynth = algorithm
                    .createMarketOrderRequest(underlyingInstrument, verbHedgeUnderlyingInstrument, quantityTrade);

            logger.info("[{}] {}-{}  {}", algorithm.getCurrentTime(),
                    orderRequestSynth.getClientOrderId(), orderRequestSynth.getOrderRequestAction(),
                    orderRequestSynth);

            try {
                algorithm.sendOrderRequest(orderRequestSynth);
            } catch (LambdaTradingException e) {
                logger.error("error sending {} order on {}", verbHedge, underlyingInstrument, e);
            }

        }

    }

    @Override
    public boolean hedge(Algorithm algorithm, Instrument instrument, double quantityMain, Verb verbMain) {
        try {
            tradeSpread(algorithm, verbMain, quantityMain);
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    @Override
    public Set<Instrument> getInstrumentsHedgeList() {
        return interestedInstruments;
    }
}
