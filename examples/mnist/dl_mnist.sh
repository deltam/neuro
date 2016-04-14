#!/bin/sh
# download from http://yann.lecun.com/exdb/mnist/

curl http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz | gunzip > train-images-idx3-ubyte
curl http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz | gunzip > train-labels-idx1-ubyte
curl http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz | gunzip > t10k-images-idx3-ubyte
curl http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz | gunzip > t10k-labels-idx1-ubyte
