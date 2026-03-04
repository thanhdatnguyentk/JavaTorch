import sys
import numpy as np

# Usage: python linear_relu_ref.py input.csv weight.csv bias.csv out.csv
if len(sys.argv) < 5:
    print("usage: linear_relu_ref.py input.csv weight.csv bias.csv out.csv")
    sys.exit(1)

in_file = sys.argv[1]
w_file = sys.argv[2]
b_file = sys.argv[3]
out_file = sys.argv[4]

X = np.loadtxt(in_file, delimiter=',').astype(np.float32)
W = np.loadtxt(w_file, delimiter=',').astype(np.float32)
B = np.loadtxt(b_file, delimiter=',').astype(np.float32)

# Ensure B is 1-D
if B.ndim == 2 and B.shape[0] == 1:
    B = B.reshape(-1)

Y = X.dot(W) + B
Y = np.maximum(Y, 0.0)

np.savetxt(out_file, Y, delimiter=',')
print('wrote', out_file)
