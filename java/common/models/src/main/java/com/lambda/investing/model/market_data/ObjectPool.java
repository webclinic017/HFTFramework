package com.lambda.investing.model.market_data;

import com.lambda.investing.model.exception.LambdaException;
import net.openhft.affinity.AffinityLock;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * https://java-design-patterns.com/patterns/object-pool/#programmatic-example-of-object-pool-pattern-in-java
 *
 * @param <T>
 */
public abstract class ObjectPool<T> implements Runnable {
    private Logger logger = LogManager.getLogger(ObjectPool.class);
    public Thread lazyCheckinThread;
    private boolean lazyCheckinThreadRunning;

    public ObjectPool() {
//        lazyCheckinThreadRunning = true;
//        lazyCheckinThread = new Thread(this::runAffinity, "ObjectPool_Thread");
//        lazyCheckinThread.start();
    }

    private final ConcurrentHashMap<Long, T> timeToCheckIn = new ConcurrentHashMap<>();
    public final Set<T> available = new HashSet<>();
    public final Set<T> inUse = new HashSet<>();
    private final Object lock = new Object();

    protected abstract T create();

    public T checkOut() {
        synchronized (lock) {
            if (available.isEmpty()) {
                available.add(create());
            }
            var instance = available.iterator().next();
            available.remove(instance);
            inUse.add(instance);
            return instance;
        }
    }

    public void lazyCheckIn(T instance, int milliseconds) {
        if (lazyCheckinThread == null) {
            throw new RuntimeException("ObjectPool thread is not running");
        }

        long timeToChecking = System.currentTimeMillis() + milliseconds;
        timeToCheckIn.put(timeToChecking, instance);
    }

    public void checkIn(T instance) {
        synchronized (lock) {
            if (inUse.remove(instance)) {
                available.add(instance);
            }
        }
    }

    @Override
    public synchronized String toString() {
        return String.format("Pool available=%d inUse=%d", available.size(), inUse.size());
    }

    public static int NUMBER_OF_CORES = Runtime.getRuntime().availableProcessors();
    public static boolean IS_LINUX = System.getProperty("os.name").toLowerCase().contains("linux");

    public static int[] GET_AFFINITY_CPUS() throws LambdaException {
        if (IS_LINUX || NUMBER_OF_CORES < 32) {
            // Linux or less than 32 cores, use all available cores
            int[] cpus = new int[NUMBER_OF_CORES];
            for (int i = 0; i < NUMBER_OF_CORES; i++) {
                cpus[i] = i;
            }
            return cpus;
        }
        throw new LambdaException("Unsupported OS or number of cores: " + System.getProperty("os.name") + " with " + NUMBER_OF_CORES + " cores. Only Linux or less than 32 cores are supported.");

//        // Windows or more than 32 cores, use only the first 32 cores
//        int[] cpus = new int[32];
//        for (int i = 0; i < 32; i++) {
//            cpus[i] = i;
//        }
//
//        return cpus;


    }

    public void runAffinity() {
        try (AffinityLock al = AffinityLock.acquireLock(GET_AFFINITY_CPUS())) {
            run();
        } catch (LambdaException e) {
            run();
        } catch (Exception e) {
            logger.warn("error AffinityLock ", e);
            if (IS_LINUX) {
                System.err.println("error AffinityLock  -> " + e.toString());
            }
            run();
        }
    }

    @Override
    public void run() {
        //Iterate over the timeToCheckIn and check if the time is up
        //If it is, check in the object
        while (true) {
            try {
                if (!timeToCheckIn.isEmpty()) {
                    long currentTime = System.currentTimeMillis();
                    for (var entry : timeToCheckIn.entrySet()) {
                        if (entry.getKey() < currentTime) {
                            checkIn(entry.getValue());
                            timeToCheckIn.remove(entry.getKey());
                        }
                    }
                }
//                Thread.onSpinWait();
                Thread.sleep(5000);
            } catch (Exception e) {
                logger.error("Error in ObjectPool thread", e);
                e.printStackTrace();
            }
        }

    }


    public void stopCheckInThread() {
        lazyCheckinThreadRunning = false;
    }
}

