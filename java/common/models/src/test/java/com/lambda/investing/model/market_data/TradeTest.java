package com.lambda.investing.model.market_data;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class TradeTest {

    private Trade trade;
    private static final String INSTRUMENT_PK = "btcusdt";

    @BeforeEach
    void setUp() {
        trade = Trade.getInstance();
        trade.setInstrument(INSTRUMENT_PK);
        trade.setPrice(49050.0);
        trade.setQuantity(1.0);
        trade.setTimestamp(System.currentTimeMillis());
    }

    @Test
    void testGetInstrument() {
        assertEquals(INSTRUMENT_PK, trade.getInstrument());
    }

    @Test
    void testSetInstrument() {
        String newInstrument = "ethusdt";
        trade.setInstrument(newInstrument);
        assertEquals(newInstrument, trade.getInstrument());
    }

    @Test
    void testGetPrice() {
        assertEquals(49050.0, trade.getPrice());
    }

    @Test
    void testSetPrice() {
        double newPrice = 50000.0;
        trade.setPrice(newPrice);
        assertEquals(newPrice, trade.getPrice());
    }

    @Test
    void testGetQuantity() {
        assertEquals(1.0, trade.getQuantity());
    }

    @Test
    void testSetQuantity() {
        double newQuantity = 2.0;
        trade.setQuantity(newQuantity);
        assertEquals(newQuantity, trade.getQuantity());
    }

    @Test
    void testGetTimestamp() {
        long timestamp = System.currentTimeMillis();
        trade.setTimestamp(timestamp);
        assertEquals(timestamp, trade.getTimestamp());
    }

    @Test
    void testSetTimestamp() {
        long newTimestamp = System.currentTimeMillis() + 1000;
        trade.setTimestamp(newTimestamp);
        assertEquals(newTimestamp, trade.getTimestamp());
    }


    @Test
    void testClone() throws CloneNotSupportedException {
        Trade clonedTrade = (Trade) trade.clone();
        assertNotSame(trade, clonedTrade);
        assertEquals(trade.getInstrument(), clonedTrade.getInstrument());
        assertEquals(trade.getPrice(), clonedTrade.getPrice());
        assertEquals(trade.getQuantity(), clonedTrade.getQuantity());
        assertEquals(trade.getTimestamp(), clonedTrade.getTimestamp());
    }
}