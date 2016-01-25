```clojure

(ns jubilant-dawn
  (:require [gorilla-plot.core :as plot])
  (:require [neuro.train :as tr]))
(count @tr/+train-err-vec+)
(last @tr/+train-err-vec+)
(let [range 500]
  (plot/compose
    (plot/list-plot (take-last range @tr/+train-err-vec+) :joined true :plot-size 500 :color :blue)
    (plot/list-plot (take-last range @tr/+test-err-vec+) :joined true :plot-size 500 :color :red)
))
```

学習データのプロット

```clojure
(let [range [[0 10] [0 10]]
      data0 (filter (fn [[_ {[a _] :w}]] (= a 0)) user/traindata-2class)
      series0 (map (fn [[{[x y] :w} _]] [x y]) data0)
      data1 (filter (fn [[_ {[a _] :w}]] (= a 1)) user/traindata-2class)
      series1 (map (fn [[{[x y] :w} _]] [x y]) data1)]
  (plot/compose
    (plot/list-plot series0 :aspect-ratio 1.0 :plot-range range :color :blue)
    (plot/list-plot series1 :aspect-ratio 1.0 :plot-range range :color :red)))
```

学習結果のプロット

```clojure
(let [calc-range (range 0 10 0.5)
      plot-range [[0 10] [0 10]]
      series0 (for [x calc-range y calc-range
                    :let [{[a _] :w} (neuro.network/calc @tr/+now-net+ (neuro.vol/vol [x y]))]
                    :when (< a 0.5)]
                [x y])
      series1 (for [x calc-range y calc-range
                    :let [{[a _] :w} (neuro.network/calc @tr/+now-net+ (neuro.vol/vol [x y]))]
                    :when (>= a 0.5)]
                [x y])]
  (plot/compose
    (plot/list-plot series0 :aspect-ratio 1.0 :plot-range plot-range :symbol-size 30 :color :blue)
    (plot/list-plot series1 :aspect-ratio 1.0 :plot-range plot-range :symbol-size 30 :color :red)))
```
