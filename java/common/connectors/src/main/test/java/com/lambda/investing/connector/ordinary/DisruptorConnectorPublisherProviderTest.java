package com.lambda.investing.connector.ordinary;

import com.google.common.base.Stopwatch;
import com.lambda.investing.Configuration;
import com.lambda.investing.connector.ConnectorConfiguration;
import com.lambda.investing.connector.ConnectorListener;
import com.lambda.investing.model.messaging.TypeMessage;
import lombok.AllArgsConstructor;
import lombok.Getter;

import org.junit.jupiter.api.*;
import org.springframework.util.StringUtils;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;

import static org.junit.jupiter.api.Assertions.assertEquals;


public class DisruptorConnectorPublisherProviderTest implements ConnectorListener {
    DisruptorConnectorPublisherProvider disruptorConnectorPublisherProvider = null;
    ConnectorConfiguration connectorConfiguration = new OrdinaryConnectorConfiguration();
    List<ReceivedItem> lastItemsUpdate = new ArrayList();
    CountDownLatch waiter;

    @AllArgsConstructor
    @Getter
    private class ReceivedItem {
        ConnectorConfiguration configuration;
        long timestampReceived;
        TypeMessage typeMessage;
        String content;

        @Override
        public String toString() {
            return "ReceivedItem{" +
                    "timestampReceived=" + timestampReceived +
                    ", typeMessage=" + typeMessage +
                    ", content='" + content + '\'' +
                    '}';
        }
    }

    @Override
    public void onUpdate(ConnectorConfiguration configuration, long timestampReceived, TypeMessage typeMessage, String content) {
        lastItemsUpdate.add(new ReceivedItem(configuration, timestampReceived, typeMessage, content));
        if (waiter != null) {
            waiter.countDown();
        }

    }


    //sometimes we received two messages!
//    @Test
//    @RepeatedTest (25)
    public void testSendReceiveSimple() throws InterruptedException {
        Stopwatch timer = Stopwatch.createStarted();
        disruptorConnectorPublisherProvider = new DisruptorConnectorPublisherProvider("junit_test", 0, 4096);
        disruptorConnectorPublisherProvider.register(connectorConfiguration, this);

        String topic = "topic1";
        TypeMessage typeMessage = TypeMessage.info;
        String message = Configuration.formatLog("message_{}", System.currentTimeMillis());
        System.out.println(message);
        waiter = new CountDownLatch(1);
        lastItemsUpdate.clear();
        disruptorConnectorPublisherProvider.publish(connectorConfiguration, typeMessage, topic, message);
        waiter.await();

        if (lastItemsUpdate.size() > 1) {
            System.out.println(StringUtils.arrayToDelimitedString(lastItemsUpdate.toArray(), ","));
        }

        assertEquals(1, lastItemsUpdate.size());
        ReceivedItem itemReceived = lastItemsUpdate.get(0);
        assertEquals(typeMessage, itemReceived.getTypeMessage());
        assertEquals(message, itemReceived.getContent());
        System.out.println("Method took: " + timer.stop());

    }
}
