package com.lambda.investing.model.market_data;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;

class ObjectPoolTest {

    private ObjectPool<String> objectPool;
    private static AtomicInteger counter = new AtomicInteger(0);
    @BeforeEach
    void setUp() {
        objectPool = new ObjectPool<>() {
            @Override
            protected String create() {
                return "newObject" + counter.incrementAndGet();
            }
        };
    }

    @Test
    void testCheckOut() {
        String obj = objectPool.checkOut();
        assertNotNull(obj);
        assertEquals("newObject" + counter.get(), obj);
        assertEquals(0, objectPool.available.size());
        assertEquals(1, objectPool.inUse.size());

        String obj1 = objectPool.checkOut();
        assertNotNull(obj1);
        assertEquals("newObject" + counter.get(), obj1);
        assertEquals(0, objectPool.available.size());
        assertEquals(2, objectPool.inUse.size());
    }

    @Test
    void testCheckIn() {
        String obj = objectPool.checkOut();
        objectPool.checkIn(obj);
        assertEquals(1, objectPool.available.size());
        assertEquals(0, objectPool.inUse.size());
    }

    @Test
    void testCheckOutAndIn() {
        String obj = objectPool.checkOut();
        assertEquals(0, objectPool.available.size());
        assertEquals(1, objectPool.inUse.size());

        String obj1 = objectPool.checkOut();
        assertEquals(0, objectPool.available.size());
        assertEquals(2, objectPool.inUse.size());

        objectPool.checkIn(obj);
        assertEquals(1, objectPool.available.size());
        assertEquals(1, objectPool.inUse.size());

        objectPool.checkIn(obj1);
        assertEquals(2, objectPool.available.size());
        assertEquals(0, objectPool.inUse.size());

        String obj3 = objectPool.checkOut();
        assertEquals(1, objectPool.available.size());
        assertEquals(1, objectPool.inUse.size());

        objectPool.checkIn(obj3);
        assertEquals(2, objectPool.available.size());
        assertEquals(0, objectPool.inUse.size());

        objectPool.checkOut();
        objectPool.checkOut();
        objectPool.checkOut();
        assertEquals(0, objectPool.available.size());
        assertEquals(3, objectPool.inUse.size());


    }

//    @Test
//    void testLazyCheckIn() throws InterruptedException {
//        String obj = objectPool.checkOut();
//        objectPool.lazyCheckIn(obj, 100);
//        assertEquals(0, objectPool.available.size());
//        assertEquals(1, objectPool.inUse.size());
//
//        Thread.sleep(200);
//        assertEquals(1, objectPool.available.size());
//        assertEquals(0, objectPool.inUse.size());
//    }

    @Test
    void testToString() {
        String obj = objectPool.checkOut();
        String expected = "Pool available=0 inUse=1";
        assertEquals(expected, objectPool.toString());
    }

//    @Test
//    void testRun() throws InterruptedException {
//        String obj = objectPool.checkOut();
//        objectPool.lazyCheckIn(obj, 100);
//        Thread.sleep(200);
//        assertEquals(1, objectPool.available.size());
//        assertEquals(0, objectPool.inUse.size());
//    }

//    @Test
//    void testStartCheckInThread() throws InterruptedException {
//        String obj = objectPool.checkOut();
//        objectPool.lazyCheckIn(obj, 100);
//        Thread.sleep(200);
//        assertTrue(objectPool.lazyCheckinThread.isAlive());
//    }
//
//    @Test
//    void testStopCheckInThread() throws InterruptedException {
//        String obj = objectPool.checkOut();
//        objectPool.lazyCheckIn(obj, 100);
//        Thread.sleep(200);
//        objectPool.stopCheckInThread();
//        Thread.sleep(200);
//        assertFalse(objectPool.lazyCheckinThread.isAlive());
//    }
}