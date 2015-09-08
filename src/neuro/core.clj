(ns neuro.core
  (:require [neuro.network :as nw])
  (:require [neuro.func :as fnc]))


(defn node-input
  "ノードの入力値、wx+bを返す"
  [layer x-seq]
  (let[w-mat (:weights layer)
       [in out] (:nodes layer)
       w-by-out (map (fn [o]  ; transposed
                       (map (fn [i] (nth (nth w-mat i) o))
                            (range (inc in))))
                     (range out))]
    (mapv (fn [ws] (apply + (map * ws
                                 (into [1] x-seq)))) ; with bias
         w-by-out)))


(defn forward-one
  "1層の順伝播計算を進める"
  [layer in-seq]
  (let [activation-f (fnc/dict (:func layer))
        wx-b (node-input layer in-seq)
        output (mapv activation-f wx-b)]
    (-> layer
        (assoc :z-val in-seq)
        (assoc :output output))))

(defn forward
  "多層ニューラルネットの順伝播計算を最後まですすめる"
  [nn-layers in-seq]
  (vec (rest
        (reduce (fn [r layer]
                  (let [x-seq (:output (last r))]
                    (conj r (forward-one layer x-seq))))
                [{:output in-seq}]
                nn-layers))))

(defn backward-one
  "1層の逆伝播の計算を進める"
  [layer delta-seq]
  (let [activation-df (fnc/d-dict (:func layer))
        w-mat (rest (:weights layer)) ; without bias
        z-val (:z-val layer)
        cur-delta (map-indexed (fn [idx z]
                                 (let [w-seq (nth w-mat idx)
                                       wd (apply + (map * w-seq delta-seq))]
                                   (* (activation-df z) wd)))
                               z-val)]
    (-> layer
        (assoc :forward-delta (vec cur-delta))
        (assoc :delta delta-seq))))

(defn backward
  "逆伝播を最後まで実行する"
  [nn-layers delta-seq]
  (let [max-idx (dec (count nn-layers))]
    (loop [ret-nn [], idx max-idx, delta delta-seq]
      (if (< idx 0)
        ret-nn
        (let [cur-layer (nth nn-layers idx)
              backed (backward-one cur-layer delta)]
          (recur (into [backed] ret-nn)
                 (dec idx)
                 (:forward-delta backed)))))))


(defn backprop
  "誤差逆伝播法で重みを勾配を算出する"
  [nn-layers in-seq train-seq]
  (let [forwarded (forward nn-layers in-seq)
        output (:output (last forwarded))
        delta-seq (map - output train-seq)
        backed (backward forwarded delta-seq)]
    (mapv (fn [{w-mat :weights, delta-seq :delta, z-val :z-val}]
            (nw/map-matrix-indexed (fn [i o _]
                                     (* (nth delta-seq o)
                                        (nth (into [1] z-val) i))) ; bias
                                  w-mat))
          backed)))


(defn nn-calc
  "多層ニューラルネットの計算をする"
  [nn-layers in-seq]
  (let [forwarded (forward nn-layers in-seq)]
    (:output (last forwarded))))
