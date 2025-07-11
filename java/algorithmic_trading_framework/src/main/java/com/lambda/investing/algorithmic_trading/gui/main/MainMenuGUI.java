package com.lambda.investing.algorithmic_trading.gui.main;

import com.intellij.uiDesigner.core.GridConstraints;
import com.intellij.uiDesigner.core.GridLayoutManager;
import com.lambda.investing.algorithmic_trading.*;
import com.lambda.investing.algorithmic_trading.gui.algorithm.AlgorithmGui;
import com.lambda.investing.algorithmic_trading.gui.algorithm.arbitrage.statistical_arbitrage.StatisticalArbitrageAlgorithmGui;
import com.lambda.investing.algorithmic_trading.gui.algorithm.market_making.MarketMakingAlgorithmGui;
import com.lambda.investing.model.market_data.Depth;
import com.lambda.investing.model.market_data.Trade;
import com.lambda.investing.model.trading.ExecutionReport;
import com.lambda.investing.model.trading.OrderRequest;
import org.jfree.chart.ChartTheme;

import javax.swing.*;
import java.awt.*;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * To compile from maven
 * "Generate GUI into:" from "Binary class files" --> "Java source code" in the settings (found in Project|Settings|Editor|GUI Designer).
 */
public class MainMenuGUI extends JFrame implements AlgorithmObserver {
    private JTabbedPane depthTabs;
    private JPanel panel1;

    public static boolean IS_BACKTEST = true;

    private static String TITLE = "[%s]Lambda Algotrading  %d algorithms";
    private List<Algorithm> algorithmsList;
    private Map<String, AlgorithmGui> algorithmsMap;

    protected ChartTheme theme;

    public MainMenuGUI(ChartTheme theme, List<Algorithm> algorithmsList) {
        //		https://www.fdi.ucm.es/profesor/jpavon/poo/tema6resumido.pdf
        super(String.format(TITLE, new Date(), algorithmsList.size()));
        this.theme = theme;
        this.algorithmsList = algorithmsList;

        algorithmsMap = new ConcurrentHashMap<>();
    }

    public MainMenuGUI(ChartTheme theme, Algorithm algorithm) {
        super(String.format(TITLE, new Date(), 1));
        this.theme = theme;
        this.algorithmsList = new ArrayList<>();
        this.algorithmsList.add(algorithm);
        algorithmsMap = new ConcurrentHashMap<>();
    }

    private void startGUI() {
        this.add(depthTabs);
        setDefaultCloseOperation(EXIT_ON_CLOSE);
    }

    public void start() {
        try {
            startGUI();
            depthTabs.remove(0);//remove na tab

            for (Algorithm algorithm : algorithmsList) {

                switch (algorithm.getAlgorithmType()) {
                    case MarketMaking:

                        MarketMakingAlgorithmGui marketMakingAlgorithmGui = new MarketMakingAlgorithmGui(theme, ((SingleInstrumentAlgorithm) algorithm).getInstrument());
                        depthTabs.add(algorithm.getAlgorithmInfo(), marketMakingAlgorithmGui.getPanel());
                        algorithmsMap.put(algorithm.getAlgorithmInfo(), marketMakingAlgorithmGui);
                        break;
                    case Arbitrage:
                        StatisticalArbitrageAlgorithmGui statisticalArbitrageAlgorithmGui = new StatisticalArbitrageAlgorithmGui(theme);
                        depthTabs.add(algorithm.getAlgorithmInfo(), statisticalArbitrageAlgorithmGui.getPanel());
                        algorithmsMap.put(algorithm.getAlgorithmInfo(), statisticalArbitrageAlgorithmGui);
                        break;
                    default:
                        System.err.println("Algorithm type not supported: " + algorithm.getAlgorithmType());
                        break;
                }
                algorithm.register(this);
            }
            this.pack();
            setSize(1024, 768);
            setVisible(true);
        } catch (Exception e) {
            System.err.println("Error starting GUI: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private void updateTitle(Date date) {
        setTitle(String.format(TITLE, date, algorithmsList.size()));
    }

    @Override
    public void onUpdateDepth(String algorithmInfo, Depth depth) {
        updateTitle(new Date(depth.getTimestamp()));
        algorithmsMap.get(algorithmInfo).updateDepth(depth);
    }

    @Override
    public void onUpdatePnlSnapshot(String algorithmInfo, PnlSnapshot pnlSnapshot) {
        algorithmsMap.get(algorithmInfo).updatePnlSnapshot(pnlSnapshot);
    }

    @Override
    public void onUpdatePortfolioSnapshot(String algorithmInfo, PortfolioSnapshot portfolioSnapshot) {
        algorithmsMap.get(algorithmInfo).updatePortfolioSnapshot(portfolioSnapshot);
    }

    @Override
    public void onUpdateTrade(String algorithmInfo, Trade trade) {
        updateTitle(new Date(trade.getTimestamp()));
        algorithmsMap.get(algorithmInfo).updateTrade(trade);
    }

    @Override
    public void onUpdateParams(String algorithmInfo, Map<String, Object> newParams) {
        algorithmsMap.get(algorithmInfo).updateParams(newParams);
    }

    @Override
    public void onUpdateMessage(String algorithmInfo, String name, String body) {
        algorithmsMap.get(algorithmInfo).updateMessage(name, body);
    }

    @Override
    public void onOrderRequest(String algorithmInfo, OrderRequest orderRequest) {
        algorithmsMap.get(algorithmInfo).updateOrderRequest(orderRequest);
    }

    @Override
    public void onExecutionReportUpdate(String algorithmInfo, ExecutionReport executionReport) {
        algorithmsMap.get(algorithmInfo).updateExecutionReport(executionReport);
    }

    @Override
    public void onCustomColumns(long timestamp, String algorithmInfo, String instrumentPk, String key, Double value) {
        algorithmsMap.get(algorithmInfo).updateCustomColumn(timestamp, instrumentPk, key, value);
    }

    {
// GUI initializer generated by IntelliJ IDEA GUI Designer
// >>> IMPORTANT!! <<<
// DO NOT EDIT OR ADD ANY CODE HERE!
        $$$setupUI$$$();
    }

    /**
     * Method generated by IntelliJ IDEA GUI Designer
     * >>> IMPORTANT!! <<<
     * DO NOT edit this method OR call it in your code!
     *
     * @noinspection ALL
     */
    private void $$$setupUI$$$() {
        panel1 = new JPanel();
        panel1.setLayout(new GridLayoutManager(1, 1, new Insets(0, 0, 0, 0), -1, -1));
        panel1.setEnabled(false);
        depthTabs = new JTabbedPane();
        depthTabs.setEnabled(true);
        panel1.add(depthTabs, new GridConstraints(0, 0, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_BOTH, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, null, new Dimension(1280, 1024), null, 1, false));
        final JPanel panel2 = new JPanel();
        panel2.setLayout(new GridLayoutManager(1, 1, new Insets(0, 0, 0, 0), -1, -1));
        panel2.setEnabled(false);
        depthTabs.addTab("na", panel2);
        depthTabs.setEnabledAt(0, false);
    }

    /**
     * @noinspection ALL
     */
    public JComponent $$$getRootComponent$$$() {
        return panel1;
    }

}
