import com.user.nn.core.*;
import java.util.*;

public class TestDims {
    public static void main(String[] args) {
        Tensor t1 = Torch.tensor(new float[3072], 3, 32, 32);
        System.out.println("t1 shape: " + Arrays.toString(t1.shape));
        
        List<Tensor> list = new ArrayList<>();
        list.add(t1);
        list.add(t1);
        
        Tensor stacked = Torch.stack(list, 0);
        System.out.println("stacked shape: " + Arrays.toString(stacked.shape));
        
        NN lib = new NN();
        NN.Conv2d conv = new NN.Conv2d(lib, 3, 64, 3, 3, 32, 32, 1, 1, false);
        Tensor out = conv.forward(stacked);
        System.out.println("conv out shape: " + Arrays.toString(out.shape));
        
        NN.BatchNorm2d bn = new NN.BatchNorm2d(lib, 64);
        Tensor bnOut = bn.forward(out);
        System.out.println("bn out shape: " + Arrays.toString(bnOut.shape));
    }
}
