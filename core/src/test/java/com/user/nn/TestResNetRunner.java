package com.user.nn;

import org.junit.jupiter.api.Test;
import com.user.nn.examples.TrainResNetCifar10;

public class TestResNetRunner {
    @Test
    public void runResNet() throws Exception {
        TrainResNetCifar10.main(new String[0]);
    }
}
