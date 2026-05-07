package com.user.nn;

import jcuda.jcudnn.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Tag;
import static jcuda.jcudnn.JCudnn.*;
import static org.junit.jupiter.api.Assertions.*;

public class TestCudnn {

    @Test
    @Tag("gpu")
    void testCudnnInit() {
        cudnnHandle handle = new cudnnHandle();
        int result = cudnnCreate(handle);
        assertEquals(cudnnStatus.CUDNN_STATUS_SUCCESS, result, 
                "cuDNN should initialize successfully on a GPU system");
        
        System.out.println("cuDNN Version: " + cudnnGetVersion());
        cudnnDestroy(handle);
    }
}
