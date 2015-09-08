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
      data0 (filter (fn [{[a] :ans}] (= a 0)) user/traindata-2class)
      series0 (map (fn [{[x y _] :x}] [x y]) data0)
      data1 (filter (fn [{[a] :ans}] (= a 1)) user/traindata-2class)
      series1 (map (fn [{[x y _] :x}] [x y]) data1)]
  (plot/compose
    (plot/list-plot series0 :aspect-ratio 1.0 :plot-range range :color :blue)
    (plot/list-plot series1 :aspect-ratio 1.0 :plot-range range :color :red)))
```

学習結果のプロット

```clojure
(let [calc-range (range 0 10 0.5)
      plot-range [[0 10] [0 10]]
      series0 (for [x calc-range y calc-range
                    :let [[a] (neuro.core/nn-calc @tr/+now-nn+ [x y])]
                    :when (< a 0.5)]
                [x y])
      series1 (for [x calc-range y calc-range
                    :let [[a] (neuro.core/nn-calc @tr/+now-nn+ [x y])]
                    :when (>= a 0.5)]
                [x y])]
  (plot/compose
    (plot/list-plot series0 :aspect-ratio 1.0 :plot-range plot-range :symbol-size 30 :color :blue)
    (plot/list-plot series1 :aspect-ratio 1.0 :plot-range plot-range :symbol-size 30 :color :red)))
```

```clojure:座標の歪み具合を見る
; (def nn (nw/map-nn (fn [w _ i _] (if (zero? i) 0 w)) (nw/gen-nn :rand 2 2 4 1)))
; (require '[neuro.func :as fc])
(let [nn (take 2 @tr/+now-nn+)
      data0 (filter (fn [{[a] :ans}] (= a 0)) user/traindata-2class-2)
      data1 (filter (fn [{[a] :ans}] (= a 1)) user/traindata-2class-2)
      series0 (map (fn [{[x y] :x}] (let [[x2 y2] (neuro.core/nn-calc nn [x y])] [(fc/logit x2) (fc/logit y2)])) data0)
      series1 (map (fn [{[x y] :x}] (let [[x2 y2] (neuro.core/nn-calc nn [x y])] [(fc/logit x2) (fc/logit y2)])) data1)]
  (plot/compose
    (plot/list-plot series0 :aspect-ratio 1.0  :color :blue)
    (plot/list-plot series1 :aspect-ratio 1.0  :color :red)))

(let [nn (take 2 @tr/+now-nn+)
      v-range (range 1.1 10 0.5)
      series (for [x v-range, y v-range :let [[x2 y2] (neuro.core/nn-calc nn [x y])]]
               [(fc/logit x2) (fc/logit y2)])]
  (plot/list-plot series :aspect-ratio 1.0  :symbol-size 30 :color :black))
```