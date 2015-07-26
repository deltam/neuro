(ns neuro.train
  (:require [neuro.core :as core]
            [neuro.network :as nw]))


(defn train [dataset init-nn limit dfn]
  (loop [cur-nn init-nn]
    (if (< (dfn cur-nn dataset) limit)
      cur-nn
      (recur (train-next cur-nn dataset dfn)))))

(defn train-next [nn dataset dfn]
  (let [nn1 (next-nn nn)
        diff1 (dfn nn1 dataset)
        nn2 (next-nn nn)
        diff2 (dfn nn2 dataset)]
    (cond (< diff1 diff2) nn1
          (> diff1 diff2) nn2
          :else nn)))

(defn next-nn [nn]
  (let [weights (:weights nn)]
    (mapv (fn [w-mat] (weight-randomize w-mat)) weights)))


(defn diff-fn [nn dataset]
  (* 0.5
     (square-sum
      (for [{x :x, ans :ans} dataset
            :let [v-seq (core/nn-calc nn x)]]
        (apply +
               (map (fn [v d] (- v d)) v-seq ans))))))


(defn diff-fn-2class
  "2値分類の誤差関数"
  [nn dataset]
  (* -1.0
     (apply +
            (for [{x :x, a :ans} dataset
                  :let [v (core/nn-calc nn x)
                        [y d] (first (map vector v a))]]
              (+ (* d (Math/log y))
                 (* (- 1.0 d) (Math/log (- 1.0 y))))))))




(defn- weight-randomize [w-mat]
  (apply vector
         (for [w-seq w-mat]
           (apply vector (map #(rand-add %) w-seq)))))

(defn- rand-add [x]
  (+ x
     (rand-nth [0.01 0 -0.01])))

(defn- square [x]
  (* x x))

(defn- square-sum [x-seq]
  (apply +
         (map square x-seq)))



(comment

(def nn (nw/gen-nn 0.1 3 2 1))

(def traindata [{:x [1 2 1] :ans [1 0]}
                {:x [2 3 1] :ans [0 1]}
                {:x [1 4 1] :ans [1 0]}
                {:x [2 8 1] :ans [0 1]}
                {:x [1 9 1] :ans [1 0]}
               ])

)
