(ns neuro.core
  (:require [neuro.network :as nw])
  (:require [neuro.func :as fnc]))

(defn nn-calc-node [f x-seq w-seq]
  (let [activation-f (fnc/dict f)]
    (activation-f
      (apply +
             (map (fn [x w] (* x w)) x-seq w-seq)))))

(defn forward [in-seq layer]
  (let [activation-f (fnc/dict (:func layer))
        w-mat (:weights layer)
        [in out] (:nodes layer)
        w-seq (map (fn [o]
                     (map (fn [i] (nth (nth w-mat i) o))
                          (range (inc in))))
                   (range out))
        wx-b (map (fn [ws] (apply + (map #(* %1 %2) ws in-seq)))
                w-seq)]
    (mapv activation-f wx-b)))

(defn nn-calc
  "多層ニューラルネットの計算をする"
  [nn-layers in-seq]
  (reduce (fn [v layer] (forward (into [1] v) layer))
          in-seq
          nn-layers))
