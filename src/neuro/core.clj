(ns neuro.core
  (:require [neuro.func :as fnc]))

(declare nn-calc-level nn-calc-node transposed-matrix)

(defn nn-calc
  "多層ニューラルネットの計算をする"
  [nn x-seq]
  (let [weights (:weights nn)
        level-nodes (:nodes nn)]
    (loop [w-mats weights, mid-ans x-seq]
      (let [w-mat (first w-mats)]
        (if (nil? w-mat)
          mid-ans
          (let [ans (nn-calc-level mid-ans w-mat)]
            (recur (rest w-mats) ans)))))))


(defn nn-calc-level [x-seq w-mat]
  (let [weight-by-out (transposed-matrix w-mat)]
    (mapv (fn [w-seq] (nn-calc-node x-seq w-seq))
          weight-by-out)))

;; TODO 関数を可変にする
(defn nn-calc-node [x-seq w-seq]
  (fnc/logistic-func
   (apply +
          (map (fn [x w] (* x w)) x-seq w-seq))))

(defn- transposed-matrix [mat]
  (let [outs (count (first mat))]
    (map (fn [out] (map (fn [row] (nth row out)) mat)) (range outs))))
