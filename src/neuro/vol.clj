(ns neuro.vol
  "NNの最小構成要素"
  (:require [clojure.data.generators :as gr]))


;; util
(defn fill-vec [len fill] (vec (repeat len fill)))
(defn zero-vec [len] (fill-vec len 0.0))
(defn rand-vec [len]
  (binding [gr/*rnd* (java.util.Random. (System/currentTimeMillis))]
    (vec (repeatedly len (fn [] (- (gr/double) 0.5))))))

(defn vol
  "as matrix, col -> :sy, row -> :sx"
  ([ix iy w-vec]
   {:sx ix, :sy iy
    :w w-vec})
  ([ix iy]
   (vol ix iy (rand-vec (* ix iy))))
  ([w-vec] ; 1 dim
   (vol 1 (count w-vec) w-vec)))

(defn- xy->i
  "2次元から1次元への座標変換"
  [v x y]
  (+ x (* y (:sx v))))

(defn wget [v x y] (nth (:w v) (xy->i v x y)))

(defn wset [v x y w]
  (assoc v :w
         (assoc (:w v) (xy->i v x y) w)))


(defn transposed
  "転置行列"
  [v]
  (vol (:sy v) (:sx v)
       (vec
        (for [x (range (:sx v)), y (range (:sy v))]
          (wget v x y)))))
(def T transposed) ; alias

(defn w-elm-op
  "行列の要素ごとの演算"
  [f v1 v2]
  (vol (:sx v1) (:sy v1)
       (mapv f (:w v1) (:w v2))))

(defn w+
  "w行列の同じ要素同士を足し合わせる, v1,v2は同じサイズとする"
  [v1 v2]
  (w-elm-op + v1 v2))

(defn w-
  [v1 v2]
  (w-elm-op - v1 v2))

(defn dot
  "w行列の掛け算"
  [v1 v2]
  (vol (:sx v2) (:sy v1)
       (vec
        (for [y (range (:sy v1)), x (range (:sx v2))
              :let [v1-vec (map #(wget v1 % y) (range (:sx v1)))
                    v2-vec (map #(wget v2 x %) (range (:sy v2)))]]
          (apply + (map * v1-vec v2-vec))))))


(defn w*
  "w行列のアダマール積"
  [v1 v2]
  (w-elm-op * v1 v2))

(defn w-sum-row
  "行を足し合わせて1xNの行列にする"
  [v]
  (dot v (vol 1 (:sx v) (fill-vec (:sx v) 1))))

(defn w-max
  [v]
  (apply max (:w v)))



(defn reduce-elm
  "要素を集計する"
  ([f init v]
   (reduce f init (:w v)))
  ([f v]
   (reduce-elm f 0 v)))

(defn map-w
  ([f v]
   (vol (:sx v) (:sy v)
        (mapv f (:w v))))
  ([f v1 v2]
   (vol (:sx v1) (:sy v1)
        (mapv f (:w v1) (:w v2)))))
