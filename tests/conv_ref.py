import sys
import numpy as np

# usage: conv_ref.py input.csv weight.csv bias.csv batch inC inH inW kernelH kernelW stride pad out_file
if len(sys.argv) < 12:
    print("usage: conv_ref.py input.csv weight.csv bias.csv batch inC inH inW kernelH kernelW stride pad out_file")
    sys.exit(1)

in_file = sys.argv[1]
w_file = sys.argv[2]
b_file = sys.argv[3]
batch = int(sys.argv[4])
inC = int(sys.argv[5])
inH = int(sys.argv[6])
inW = int(sys.argv[7])
kh = int(sys.argv[8])
kw = int(sys.argv[9])
stride = int(sys.argv[10])
pad = int(sys.argv[11])
out_file = sys.argv[12]

X = np.loadtxt(in_file, delimiter=',').astype(np.float32)
W = np.loadtxt(w_file, delimiter=',').astype(np.float32)
B = np.loadtxt(b_file, delimiter=',').astype(np.float32)

# reshape
X = X.reshape(batch, inC, inH, inW)
# W is (inC*kh*kw) x outC -> reshape
ksz = inC * kh * kw
B = B.reshape(-1)
if W.ndim == 1:
    # single column
    outC = 1
    W = W.reshape(ksz, outC)
else:
    outC = W.shape[1]
    W = W.reshape(ksz, outC)
if B.ndim == 0:
    B = np.array([float(B)])
else:
    B = B.reshape(-1)
B = B.reshape(-1)

outH = (inH + 2*pad - kh) // stride + 1
outW = (inW + 2*pad - kw) // stride + 1
Y = np.zeros((batch, outC, outH, outW), dtype=np.float32)

for b in range(batch):
    for oc in range(outC):
        for oh in range(outH):
            for ow in range(outW):
                s = 0.0
                for ic in range(inC):
                    for kh_i in range(kh):
                        for kw_i in range(kw):
                            ih = oh*stride - pad + kh_i
                            iw = ow*stride - pad + kw_i
                            if 0 <= ih < inH and 0 <= iw < inW:
                                s += X[b, ic, ih, iw] * W[(ic*kh*kw + kh_i*kw + kw_i), oc]
                Y[b, oc, oh, ow] = s + (B[oc] if B.size>0 else 0.0)

np.savetxt(out_file, Y.reshape(batch, -1), delimiter=',')
print('wrote', out_file)
