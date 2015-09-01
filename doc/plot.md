```clojure

(ns jubilant-dawn
  (:require [gorilla-plot.core :as plot])
  (:require [neuro.train :as tr]))

(last @tr/+train-err-vec+)
(let [range 500]
  (plot/compose
    (plot/list-plot (take-last range @tr/+train-err-vec+) :joined true :plot-size 800 :color :blue)
    (plot/list-plot (take-last range @tr/+test-err-vec+) :joined true :plot-size 800 :color :red)
))
```

学習データのプロット

```clojure
(let [data0 (filter (fn [{[a] :ans}] (= a 0)) user/traindata-2class)
      series0 (map (fn [{[x y _] :x}] [x y]) data0)
        data1 (filter (fn [{[a] :ans}] (= a 1)) user/traindata-2class)
      series1 (map (fn [{[x y _] :x}] [x y]) data1)]
  (plot/compose
    (plot/list-plot series0 :aspect-ratio 1.0 :plot-range [[0 9] [0 9]] :color :blue)
    (plot/list-plot series1 :aspect-ratio 1.0 :plot-range [[0 9] [0 9]] :color :red)))
```

学習結果のプロット

```clojure
(let [series0 (for [x (range 0 10 0.3) y (range 0 10 0.3)
                    :let [[a] (neuro.core/nn-calc tr/+now-nn+ [x y])]
                    :when (< a 0.5)]
                [x y])
      series1 (for [x (range 0 10 0.3) y (range 0 10 0.3)
                    :let [[a] (neuro.core/nn-calc tr/+now-nn+ [x y])]
                    :when (>= a 0.5)]
                [x y])]
  (plot/compose
    (plot/list-plot series0 :aspect-ratio 1.0 :plot-range [[0 9] [0 9]] :symbol-size 30 :color :blue)
    (plot/list-plot series1 :aspect-ratio 1.0 :plot-range [[0 9] [0 9]] :symbol-size 30 :color :red)))
```