(ns neuro.core
  (:require [neuro.network :as nw])
  (:require [neuro.func :as fnc]))

;; TODO 関数を可変にする
(defn nn-calc-node [x-seq w-seq]
  (fnc/logistic-func
   (apply +
          (map (fn [x w] (* x w)) x-seq w-seq))))

(defn nn-calc
  "多層ニューラルネットの計算をする"
  [nn x-seq]
  (nw/reduce-nn nn-calc-node x-seq nn))
