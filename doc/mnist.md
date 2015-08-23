```clojure

(require '[neuro.mnist :as mn])
(require '[neuro.train :as tr])
(require '[neuro.network :as nw])

(def traindata-8 (traindata-2class 8))
(def testdata-8 (take 100 (shuffle traindata-8)))

(def nn (nw/gen-nn :rand 784 1000 1))

(time (def nn-mnist
        (tr/train-sgd nn tr/err-fn-2class traindata-8 testdata-8 (fn [_ d] (< d 0.8)))
        ))


```
