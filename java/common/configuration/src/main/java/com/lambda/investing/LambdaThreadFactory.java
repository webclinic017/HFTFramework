package com.lambda.investing;

import com.google.common.util.concurrent.ThreadFactoryBuilder;
import net.openhft.affinity.AffinityThreadFactory;

import java.util.concurrent.ThreadFactory;

import static net.openhft.affinity.AffinityStrategies.DIFFERENT_CORE;

public class LambdaThreadFactory {

    public static ThreadFactory createThreadFactory(String name, int priority) {
        if (Configuration.USE_THREAD_AFFINITY) {
            return new AffinityThreadFactory(name, DIFFERENT_CORE);
        } else {
            ThreadFactoryBuilder threadFactoryBuilder = new ThreadFactoryBuilder();
            threadFactoryBuilder.setNameFormat(name + "-%d");
            threadFactoryBuilder.setPriority(priority);
            return threadFactoryBuilder.build();
        }
    }

    public static ThreadFactory createThreadFactory(String name) {
        return createThreadFactory(name, Thread.NORM_PRIORITY);
    }
}
