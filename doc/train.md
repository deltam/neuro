

```clj

(comment

(def nn-2class (nw/gen-nn :rand 3 6 1))


(def traindata-2class [{:x [2 5] :ans [0]}
                       {:x [3 2] :ans [0]}
                       {:x [4 1] :ans [0]}
                       {:x [8 3] :ans [0]}
                       {:x [1 8] :ans [0]}
                       {:x [9 5] :ans [0]}
                       {:x [5 2] :ans [0]}
                       {:x [4 2] :ans [0]}
                       {:x [3 3] :ans [0]}
                       {:x [2 6] :ans [0]}
                       {:x [1 8] :ans [0]}
                       {:x [9 5] :ans [0]}
                       {:x [1 4] :ans [0]}
                       {:x [2 4] :ans [0]}
                       {:x [1 6] :ans [0]}
                       {:x [2 3] :ans [0]}
                       {:x [6 1] :ans [0]}
                       {:x [9 4] :ans [0]}
                       {:x [7 2] :ans [0]}
                       {:x [6 2] :ans [0]}
                       {:x [8 1] :ans [0]}
                       {:x [9 2] :ans [0]}
                       {:x [2 1] :ans [0]}
                       {:x [2 7] :ans [0]}
                       {:x [1 3] :ans [0]}
                       {:x [1 7] :ans [0]}


                       {:x [3 7] :ans [1]}
                       {:x [4 4] :ans [1]}
                       {:x [7 6] :ans [1]}
                       {:x [3 7] :ans [1]}
                       {:x [4 8] :ans [1]}
                       {:x [7 8] :ans [1]}
                       {:x [7 6] :ans [1]}
                       {:x [4 5] :ans [1]}
                       {:x [4 7] :ans [1]}
                       {:x [8 5] :ans [1]}
                       {:x [6 3] :ans [1]}
                       {:x [4 7] :ans [1]}
                       {:x [9 6] :ans [1]}
                       {:x [7 4] :ans [1]}
                       {:x [3 8] :ans [1]}
                       {:x [3 4] :ans [1]}
                       {:x [5 6] :ans [1]}
                       {:x [6 5] :ans [1]}
                       {:x [5 4] :ans [1]}
                       {:x [5 3] :ans [1]}
                       {:x [4 6] :ans [1]}
                       {:x [8 7] :ans [1]}
                       {:x [3 6] :ans [1]}
                       {:x [6 7] :ans [1]}

                       ])

(def traindata-2class-2
  [{:x [3 2] :ans [0]}
   {:x [4 1] :ans [0]}
   {:x [8 3] :ans [0]}
   {:x [1 8] :ans [0]}
   {:x [9 5] :ans [0]}
   {:x [5 2] :ans [0]}
   {:x [4 2] :ans [0]}
   {:x [3 3] :ans [0]}
   {:x [2 6] :ans [0]}
   {:x [1 8] :ans [0]}
   {:x [9 5] :ans [0]}
   {:x [1 4] :ans [0]}
   {:x [2 4] :ans [0]}
   {:x [1 6] :ans [0]}
   {:x [2 3] :ans [0]}
   {:x [6 1] :ans [0]}
   {:x [9 4] :ans [0]}
   {:x [7 2] :ans [0]}
   {:x [6 2] :ans [0]}
   {:x [8 1] :ans [0]}
   {:x [9 2] :ans [0]}
   {:x [2 1] :ans [0]}
   {:x [2 7] :ans [0]}
   {:x [1 3] :ans [0]}
   {:x [1 7] :ans [0]}
   {:x [7 5] :ans [0]}
   {:x [6 7] :ans [0]}
   {:x [8 6] :ans [0]}
   {:x [7 8] :ans [0]}
   {:x [3 9] :ans [0]}
   {:x [4 9] :ans [0]}
   {:x [5 9] :ans [0]}
   {:x [5 8] :ans [0]}
   {:x [2 8] :ans [0]}
   {:x [2 9] :ans [0]}
   {:x [6 6] :ans [0]}
   {:x [2 5] :ans [0]}
   {:x [3 5] :ans [0]}
   {:x [5 7] :ans [0]}
   {:x [7 6] :ans [0]}
   {:x [9 6] :ans [0]}
   {:x [8 8] :ans [0]}
   {:x [6 8] :ans [0]}
   {:x [9 8] :ans [0]}


   {:x [3 7] :ans [1]}
   {:x [4 4] :ans [1]}
   {:x [6 4] :ans [1]}
   {:x [3 7] :ans [1]}
   {:x [4 8] :ans [1]}
   {:x [7 3] :ans [1]}
   {:x [8 4] :ans [1]}
   {:x [4 5] :ans [1]}
   {:x [4 7] :ans [1]}
   {:x [8 5] :ans [1]}
   {:x [6 3] :ans [1]}
   {:x [4 7] :ans [1]}
   {:x [5 5] :ans [1]}
   {:x [7 4] :ans [1]}
   {:x [3 8] :ans [1]}
   {:x [3 4] :ans [1]}
   {:x [5 6] :ans [1]}
   {:x [6 5] :ans [1]}
   {:x [5 4] :ans [1]}
   {:x [5 3] :ans [1]}
   {:x [4 6] :ans [1]}
   {:x [4 3] :ans [1]}
   {:x [3 6] :ans [1]}
   {:x [3 7] :ans [1]}
   {:x [4 4] :ans [1]}
   {:x [6 4] :ans [1]}
   {:x [3 7] :ans [1]}
   {:x [4 8] :ans [1]}
   {:x [7 3] :ans [1]}
   {:x [8 4] :ans [1]}
   {:x [4 5] :ans [1]}
   {:x [4 7] :ans [1]}
   {:x [8 5] :ans [1]}
   {:x [6 3] :ans [1]}
   {:x [4 7] :ans [1]}
   {:x [5 5] :ans [1]}
   {:x [7 4] :ans [1]}
   {:x [3 8] :ans [1]}
   {:x [3 4] :ans [1]}
   {:x [5 6] :ans [1]}
   {:x [6 5] :ans [1]}
   {:x [5 4] :ans [1]}
   {:x [5 3] :ans [1]}
   {:x [4 6] :ans [1]}
   {:x [4 3] :ans [1]}
   {:x [3 6] :ans [1]}
])


(time
 (def nn-g
   (train nn-2class err-fn-2class weight-gradient traindata-2class
          (fn [_ d] (< d 0.1)))
   ))

(time
 (def nn-r
   (train nn-2class err-fn-2class weight-randomize traindata-2class
          (fn [_ d] (< d 0.1)))
   ))

(defn nn-test [nn dataset ans ans-test]
  (let [t (map (fn [x] (core/nn-calc nn x))
               (for [t traindata-2class :when (= (:ans t) ans)]
                 (:x t)))
        cnt (count t)]
    (/ (reduce (fn [r [w]] (if (ans-test w) (inc r) r)) 0.0 t)
       cnt)))

(nn-test nn-g traindata-2class [0] #(< % 0.5))
(nn-test nn-g traindata-2class [1] #(> % 0.5))
(nn-test nn-r traindata-2class [0] #(< % 0.5))
(nn-test nn-r traindata-2class [1] #(> % 0.5))


(defn plot-classify
  "ランダムな数値を分類させて結果をCSVで出力する"
  [nn count]
  (binding [gr/*rnd* (java.util.Random. (System/currentTimeMillis))]
    (let [samples (for [i (range count)
                        :let [x1 (int (* 10 (gr/double)))
                              x2 (int (* 10 (gr/double)))
                              [v] (core/nn-calc nn [x1 x2 1])
                              ok (if (< 0.5 v) 1 0)]]
                    [x1 x2 1 ok v])]
      (doseq [[x1 x2 _ ok _] (sort-by #(nth % 4) samples)]
        (printf "%d,%d,%d\n" x1 x2 ok)))))



)
```
