(ns neuro.network
  (:require [clojure.data.generators :as gr]))

(declare gen-num-vec gen-num-matrix seq-by-2-items update-at update-matrix-at)

(defn gen-layer
  "NNの1層を作る"
  [init in out]
  {:nodes [in out]
   :weights (gen-num-matrix init (inc in) out)
   :func :logistic})

(defn gen-nn
  "多層ニューラルネットを定義する"
  [init & layer-nodes]
  (let [nodes (seq-by-2-items layer-nodes)]
    (mapv (fn [[in out]] (gen-layer init in out))
          nodes)))

(defn stack-layer
  "NNに層を追加する"
  [nn layer]
  (conj nn layer))

(defn wget
  "重みを取得する"
  [nn layer in out]
  (let [w-mat (:weights (nth nn layer))]
    ((w-mat in) out)))

(defn wput
  "重みを更新する"
  [nn layer in-node out-node w]
  (let [cur-layer (nth nn layer)
        w-mat (:weights cur-layer)
        updated (update-matrix-at w-mat in-node out-node w)
        new-layer (assoc cur-layer :weights updated)]
    (update-at nn layer new-layer)))


(defn- mapv-indexed [f coll]
  (vec (map-indexed f coll)))

(defn- map-matrix-indexed
  [f mat cols rows]
  (mapv-indexed (fn [c w-vec]
                  (mapv-indexed (fn [r w] (f c r w))
                                w-vec))
                mat))

(defn map-nn
  "重みの更新を一括して行なう
  (f l i o w)"
  [f nn]
  (mapv-indexed (fn [idx layer]
                  (let [[in out] (:nodes layer)
                        w-mat (:weights layer)]
                    (assoc layer :weights
                           (map-matrix-indexed (fn [i o w] (f idx i o w)) w-mat (inc in) out))))
                nn))



(defn- weight-init-f [init]
  (if (= :rand init)
    (binding [gr/*rnd* (java.util.Random. (System/currentTimeMillis))]
      (fn [] (gr/double)))
    (if (fn? init)
      init
      (fn [] init))))

(defn- gen-num-vec [init n]
  (let [init-f (weight-init-f init)]
    (apply vector
           (repeatedly n init-f))))

(defn- gen-num-matrix [init x y]
  (gen-num-vec #(gen-num-vec init y) x))

(defn- seq-by-2-items [s]
  (map vector s (rest s)))

(defn- update-at [v idx val]
  (apply vector
         (concat (subvec v 0 idx)
                 [val]
                 (subvec v (inc idx)))))

(defn- update-matrix-at [mat x y val]
  (update-at mat x
             (update-at (mat x) y val)))




(comment

(def nn [{:node [3 2]
          :weights [[0.0 0.0] ; bias
                    [0.0 0.0]
                    [0.0 0.0]
                    [0.0 0.0]]
          :func :logistic}
         {:node [2 1]
          :weights [[0.0]     ; bias
                    [0.0]
                    [0.0]]
          :func :logistic}
         ])

(forward (first nn) [1 2 3])
;=> [1 1]

)
