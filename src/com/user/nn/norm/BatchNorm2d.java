package com.user.nn.norm;

import com.user.nn.core.*;
import com.user.nn.core.Module;

public class BatchNorm2d extends Module {
    public int numFeatures;
    public float momentum;
    public float epsilon;
    public Parameter gamma;
    public Parameter beta;
    public Tensor runningMean;
    public Tensor runningVar;

    public BatchNorm2d(int numFeatures) {
        this(numFeatures, 0.1f, 1e-5f);
    }

    public BatchNorm2d(int numFeatures, float momentum, float epsilon) {
        this.numFeatures = numFeatures;
        this.momentum = momentum;
        this.epsilon = epsilon;

        Tensor g = new Tensor(numFeatures);
        Torch.nn.init.ones_(g);
        this.gamma = new Parameter(g);
        addParameter("gamma", this.gamma);

        Tensor b = new Tensor(numFeatures);
        Torch.nn.init.zeros_(b);
        this.beta = new Parameter(b);
        addParameter("beta", this.beta);

        this.runningMean = new Tensor(numFeatures);
        Torch.nn.init.zeros_(this.runningMean);
        
        this.runningVar = new Tensor(numFeatures);
        Torch.nn.init.ones_(this.runningVar);
    }

    @Override
    public void toGPU() {
        super.toGPU();
        runningMean.toGPU();
        runningVar.toGPU();
    }

    @Override
    public void toCPU() {
        super.toCPU();
        runningMean.toCPU();
        runningVar.toCPU();
    }

    @Override
    public Tensor forward(Tensor x) {
        if (!x.isGPU()) x.toGPU();
        
        Tensor out = new Tensor(x.shape);
        out.toGPU();

        Tensor g = this.gamma.getTensor();
        Tensor b = this.beta.getTensor();

        if (this.is_training()) {
            CUDAOps.batchNorm2dForwardTraining(x, out, g, b, runningMean, runningVar, momentum, epsilon);
        } else {
            CUDAOps.batchNorm2dForwardInference(x, out, g, b, runningMean, runningVar, epsilon);
        }

        if (Torch.is_grad_enabled() && (x.requires_grad || g.requires_grad || b.requires_grad)) {
            out.requires_grad = true;
            out.grad_fn = new Tensor.GradFn(x, g, b) {
                @Override
                public void apply(Tensor gradOutput) {
                    if (!gradOutput.isGPU()) gradOutput.toGPU();

                    Tensor dx = new Tensor(x.shape);
                    dx.toGPU();

                    Tensor dg = new Tensor(g.shape);
                    dg.toGPU();

                    Tensor db = new Tensor(b.shape);
                    db.toGPU();

                    CUDAOps.batchNorm2dBackward(x, gradOutput, dx, g, dg, db, epsilon);

                    x.backwardStep(dx);
                    g.backwardStep(dg);
                    b.backwardStep(db);
                }
            };
        }

        return out;
    }
}
