# MNIST example

Classify [MNIST](http://yann.lecun.com/exdb/mnist/) handwritten digits by [neuro](https://github.com/deltam/neuro).

## Usage

```sh
$ ./dl_mnist.sh
$ lein gorilla :port 8888 :nrepl-port 8889
Gorilla-REPL: 0.4.0
Started nREPL server on port 8889
Running at http://127.0.0.1:8888/worksheet.html .
Ctrl+C to exit.
```

connect repl (`M-x cider-connct`)

```clojure
(require '[mnist.core :as mnist])

(def net2 (mnist/train))
; epoch 0: 950 / 10000
; epoch 1: 8911 / 10000
; epoch 2: 9153 / 10000
; epoch 3: 9259 / 10000
; epoch 4: 9335 / 10000
; ...
```

open monitoring page

[http://127.0.0.1:8888/worksheet.html?filename=mnist-train-monitor.gorilla](http://127.0.0.1:8888/worksheet.html?filename=mnist-train-monitor.gorilla)

![mnist monitor](https://github.com/deltam/neuro/blob/master/examples/mnist/mnist_monitor.png?raw=true)

## License

```
MIT License

Copyright (c) 2018 MISUMI Masaru

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
