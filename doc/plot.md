```clojure

(ns jubilant-dawn
  (:require [gorilla-plot.core :as plot])
  (:require [neuro.train :as tr]))

(plot/compose
  (plot/list-plot (take-last 100 @tr/+train-err-vec+) :joined true :plot-size 800 :color :blue)
  (plot/list-plot (take-last 100 @tr/+test-err-vec+) :joined true :plot-size 800 :color :red)
)
```
