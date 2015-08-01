(ns neuro.train
  (:require [neuro.core :as core]
            [neuro.network :as nw]))


(def ^:dynamic *weight-inc-val* 0.00001)
(def ^:dynamic *learning-param* 0.00001)


(defn train [init-nn dfn dataset]
  (loop [cur-nn init-nn, diff (dfn init-nn dataset) , cnt 0]
    (let [next (train-next cur-nn dataset dfn)
          next-diff (dfn next dataset)]
      (if (zero? (mod cnt 100))
        (println cnt " now diff: " next-diff))
      (if (< (Math/abs (- diff next-diff)) 0.00000001)
        cur-nn
        (recur next, next-diff, (inc cnt))))))

(defn train-next [nn dataset dfn]
  (let [diff1 (dfn nn dataset)
        nn2 (next-nn nn dfn dataset)
        diff2 (dfn nn2 dataset)]
    (cond (<= diff1 diff2) nn
          (>  diff1 diff2) nn2)))

(defn next-nn [nn dfn dataset]
  (weight-gradient nn dfn dataset))



(defn diff-fn-2class
  "2値分類の誤差関数 nn の出力層は1ニューロン"
  [nn dataset]
  (* -1.0
     (apply +
            (for [{x :x, [d] :ans} dataset
                  :let [[y] (core/nn-calc nn x)]]
              (+ (* d (Math/log y))
                 (* (- 1.0 d) (Math/log (- 1.0 y))))))))




(defn- weight-gradient
  "勾配降下法で重みを更新する"
  [nn dfn dataset]
  (nw/map-weights (fn [w l i o]
                    (update-by-gradient w nn dfn dataset l i o))
                  nn))

(defn- update-by-gradient
  "重みを勾配に従って更新した値を返す"
  [w nn dfn dataset level in out]
  (let [grd (gradient nn dfn dataset level in out)
        diff (dfn nn dataset)]
    (- w (* *learning-param* diff))))

(defn- gradient
  "nnの微小増分の傾きを返す"
  [nn dfn dataset level in out]
  (let [w (nw/weight nn level in out)
        nn-inc (nw/update-weight nn (+ w *weight-inc-val*) level in out)
        y (dfn nn dataset)
        y-inc (dfn nn-inc dataset)]
    (/ (- y-inc y) *weight-inc-val*)))



(defn diff-fn-regression
  "回帰解析用の誤差関数"
  [nn dataset]
  (* 0.5
     (square-sum
      (for [{x :x, ans :ans} dataset
            :let [v-seq (core/nn-calc nn x)]]
        (apply +
               (map (fn [v d] (- v d)) v-seq ans))))))

(defn- weight-randomize [w-mat]
  (apply vector
         (for [w-seq w-mat]
           (apply vector (map #(rand-add %) w-seq)))))

(defn- rand-add [x]
  (+ x
     (rand-nth [0.000001 0 -0.000001])))

(defn- square [x]
  (* x x))

(defn- square-sum [x-seq]
  (apply +
         (map square x-seq)))



(comment

(def nn-2class (nw/gen-nn 0.1 3 2 1))

(def traindata-2class [{:x [1 1 1] :ans [1]}
                       {:x [2 3 1] :ans [1]}
                       {:x [1 4 1] :ans [1]}
                       {:x [2 8 1] :ans [0]}
                       {:x [1 9 1] :ans [0]}
                       {:x [9 2 1] :ans [0]}
                       ])

(def train-nn-2class (train nn-2class traindata-2class diff-fn-2class))

)
