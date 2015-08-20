(ns neuro.network
  (:require [clojure.data.generators :as gr]))

(declare gen-num-vec gen-num-matrix seq-by-2-items update-at update-matrix-at)

(defn gen-nn
  "多層ニューラルネットを定義する"
  [init & layer-nodes]
  {:nodes (apply vector layer-nodes)
   :weights (apply vector
                   (for [[in out] (seq-by-2-items layer-nodes)]
                     (gen-num-matrix init in out)))
   :bias (apply vector
                (mapv #(gen-num-vec init %) (rest layer-nodes)))
   :func :logistic})

(defn weight [nn layer in out]
  (let [w-mat ((:weights nn) layer)]
    ((w-mat in) out)))

(defn update-weight
  "重みを更新する"
  [nn w layer in-node out-node]
  (let [w-mat (:weights nn)
        updated (update-matrix-at (w-mat layer) in-node out-node w)]
    (assoc nn :weights (update-at w-mat layer updated))))

(defn map-nn
  "重みの更新を一括して行なう"
  [f nn]
  (let [ws (:weights nn)
        layers (range (count ws))
        idx (for [l layers
                  in (range (count (ws l)))
                  out (range (count ((ws l) in)))]
              [l in out])]
    (reduce (fn [ret [l i o]]
              (let [w (weight ret l i o)]
                (update-weight ret (f w l i o) l i o)))
            nn
            idx)))

(defn- in-weight-vec
  "あるノードへ入力されるパスの重みを返す"
  [nn layer node]
  (let [in (nth (:nodes nn) layer)]
    (mapv (fn [i] (weight nn layer i node))
          (range in))))

(defn- get-bias [nn layer node]
  (let [bias-seq (:bias nn)]
    (nth (nth bias-seq layer) node)))

(defn- reduce-1-nn
  "ひとつの層について関数を適用する"
  [f xs nn layer out]
  (mapv (fn [node] (f xs (in-weight-vec nn layer node) (get-bias nn layer node)))
        (range out)))

(defn reduce-nn
  "各層ごとに関数を適用して値を集計する
  (f x-seq weight-seq)"
  [f xv nn]
  (let [ns (seq-by-2-items (:nodes nn))]
    (first
     (reduce (fn [[xs layer] [in out]]
               [(reduce-1-nn f xs nn layer out) (inc layer)])
             [xv 0]
             ns))))



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

(def nn {:nodes [3 2 1]
         :weights [
                   [[0.0 0.0]
                    [0.0 0.0]
                    [0.0 0.0]]
                   [[0.0]
                    [0.0]]
                   ]
         :bias [[0.1 0.2]
                [0.3]]
         :func :logistic})

)
