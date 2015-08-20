(ns neuro.core
  (:require [neuro.network :as nw])
  (:require [neuro.func :as fnc]))

(defn nn-calc-node [f x-seq w-seq bias]
  (let [activation-f (fnc/dict f)]
    (activation-f
     (+
      (apply +
             (map (fn [x w] (* x w)) x-seq w-seq))
      bias))))

(defn nn-calc
  "多層ニューラルネットの計算をする"
  [nn x-seq]
  (let [node-f (fn [xs ws b] (nn-calc-node (:func nn) xs ws b))]
    (nw/reduce-nn node-f x-seq nn)))
