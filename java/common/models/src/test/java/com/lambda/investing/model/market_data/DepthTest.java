package com.lambda.investing.model.market_data;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class DepthTest {

    private Depth depth;
    private static final String INSTRUMENT_PK = "btcusdt";

    @BeforeEach
    void setUp() {
        depth = Depth.getInstance();
        depth.setInstrument(INSTRUMENT_PK);
        depth.setBids(new double[]{49000.0, 48900.0});
        depth.setAsks(new double[]{49100.0, 49200.0});
        depth.setBidsQuantities(new double[]{1.0, 2.0});
        depth.setAsksQuantities(new double[]{1.0, 2.0});
        depth.setLevelsFromData();
    }

    @Test
    void testGetLevelBidFromPrice() {
        assertEquals(0, depth.getLevelBidFromPrice(49000.0));
        assertEquals(1, depth.getLevelBidFromPrice(48900.0));
        assertEquals(2, depth.getLevelBidFromPrice(48800.0));
    }

    @Test
    void testGetBidPriceFromLevel() {
        assertEquals(49000.0, depth.getBidPriceFromLevel(0));
        assertEquals(48900.0, depth.getBidPriceFromLevel(1));
        assertEquals(Double.MIN_VALUE, depth.getBidPriceFromLevel(2));
    }

    @Test
    void testGetAskPriceFromLevel() {
        assertEquals(49100.0, depth.getAskPriceFromLevel(0));
        assertEquals(49200.0, depth.getAskPriceFromLevel(1));
        assertEquals(Double.MAX_VALUE, depth.getAskPriceFromLevel(2));
    }

    @Test
    void testGetLevelAskFromPrice() {
        assertEquals(0, depth.getLevelAskFromPrice(49100.0));
        assertEquals(1, depth.getLevelAskFromPrice(49200.0));
        assertEquals(2, depth.getLevelAskFromPrice(49300.0));
    }

    @Test
    void testGetMidPriceFromLevel() {
        assertEquals(49050.0, depth.getMidPriceFromLevel(0, 0));
        assertEquals(49050.0, depth.getMidPriceFromLevel(0, 1.0));
        assertEquals(Double.NaN, depth.getMidPriceFromLevel(2, 1.0));
    }

    @Test
    void testGetSpreadFromLevel() {
        assertEquals(100.0, depth.getSpreadFromLevel(0, 0));
        assertEquals(100.0, depth.getSpreadFromLevel(0, 1.0));
        assertEquals(Double.NaN, depth.getSpreadFromLevel(2, 1.0));
    }

    @Test
    void testGetBidVolume() {
        assertEquals(3.0, depth.getBidVolume());
    }

    @Test
    void testGetAskVolume() {
        assertEquals(3.0, depth.getAskVolume());
    }

    @Test
    void testGetBidVolumeByPrice() {
        assertEquals(1.0, depth.getBidVolume(49000.0));
        assertEquals(2.0, depth.getBidVolume(48900.0));
        assertEquals(0.0, depth.getBidVolume(48800.0));
    }

    @Test
    void testGetAskVolumeByPrice() {
        assertEquals(1.0, depth.getAskVolume(49100.0));
        assertEquals(2.0, depth.getAskVolume(49200.0));
        assertEquals(0.0, depth.getAskVolume(49300.0));
    }

    @Test
    void testClone() throws CloneNotSupportedException {
        Depth clonedDepth = (Depth) depth.clone();
        assertNotSame(depth, clonedDepth);
        assertEquals(depth.getBids(), clonedDepth.getBids());
        assertEquals(depth.getAsks(), clonedDepth.getAsks());
    }

    @Test
    void testIsDefaultValue() {
        assertFalse(depth.isDefaultValue(Double.MIN_VALUE));
        assertFalse(depth.isDefaultValue(0.0));
        assertFalse(depth.isDefaultValue(1.0));
        assertFalse(depth.isDefaultValue(-1.0));
        assertFalse(depth.isDefaultValue(Double.MAX_VALUE));
        assertTrue(depth.isDefaultValue(Depth.DEFAULT_VALUE));
        assertTrue(depth.isDefaultValue(Double.NaN));
    }
}