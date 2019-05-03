(ns neuro.vol
  "Matrix for Neural Network"
;  (:require [taoensso.tufte :refer [p]])
  )


;; util
(defn fill-vec [len fill] (vec (repeat len fill)))
(defn zero-vec [len] (fill-vec len 0.0))

(defn gauss-vec
  "Random numbers in accordance with Gaussian distribution"
  [len c]
  (let [rnd (java.util.Random. (System/currentTimeMillis))]
    (vec (repeatedly len (fn [] (* c (.nextGaussian rnd)))))))

(defn xavier-vec
  "Random numbers for Xavier initilization"
  [len num-nodes]
  (gauss-vec len (/ 1.0 (Math/sqrt num-nodes))))

(defn he-vec
  "Random numbers for He initilization"
  [len num-nodes]
  (gauss-vec len (Math/sqrt (/ 2.0 num-nodes))))

(defn- xy->i
  "2次元から1次元への座標変換"
  [v x y]
  (+ x (* y (:sx v))))


(defrecord VecVol [sx sy w])

(defn vol
  "as matrix, col -> :sy, row -> :sx"
  ([ix iy wv]
   (->VecVol ix iy wv))
  ([ix iy]
   (vol ix iy (xavier-vec (* ix iy) ix)))
  ([wv] ; 1 dim
   (vol 1 (count wv) wv)))

(defn shape [v]
  [(:sx v) (:sy v)])

(defn wget [this x y] (nth (:w this) (xy->i this x y)))

(defn wset [this x y w]
  (assoc this :w
         (assoc (:w this) (xy->i this x y) w)))

(defn w-elm-op [f this other]
  (vol (:sx this) (:sy this) (mapv f (:w this) (:w other))))

(defn w+
  "w行列の同じ要素同士を足し合わせる, v1,v2は同じサイズとする"
  [v1 v2]
  (w-elm-op + v1 v2))

(defn w-
  [v1 v2]
  (w-elm-op - v1 v2))

(defn w*
  "w行列のアダマール積"
  [v1 v2]
  (w-elm-op * v1 v2))

(defn dot
  "w行列の掛け算"
  [v1 v2]
  (let [[sx1 sy1] (shape v1)
        [sx2 sy2] (shape v2)]
    (vol sx2 sy1
         (let [v1-y-range (range sy1)
               v1-rows (mapv (fn [y] (map #(wget v1 % y) (range sx1)))
                             v1-y-range)
               v2-x-range (range sx2)
               v2-cols (mapv (fn [x] (map #(wget v2 x %) (range sy2)))
                             v2-x-range)
               xy (for [y (range sy1), x (range sx2)]
                    [x y])]
           (mapv (fn [[x y]] (apply + (map * (nth v1-rows y) (nth v2-cols x))))
                 xy)))))

(defn dot-Tv-v
  "(dot (T v1) v2)"
  [v1 v2]
  (vol (:sx v2) (:sx v1)
       (let [v1-y-range (range (:sx v1))
             v1-rows (mapv (fn [y] (map #(wget v1 y %) (range (:sy v1))))
                           v1-y-range)
             v2-x-range (range (:sx v2))
             v2-cols (mapv (fn [x] (map #(wget v2 x %) (range (:sy v2))))
                           v2-x-range)
             xy (for [y (range (:sx v1)), x (range (:sx v2))]
                  [x y])]
         (mapv (fn [[x y]] (apply + (map * (nth v1-rows y) (nth v2-cols x))))
               xy))))

(defn dot-v-Tv
  "(dot v1 (T v2))"
  [v1 v2]
  (vol (:sy v2) (:sy v1)
       (let [v1-y-range (range (:sy v1))
             v1-rows (mapv (fn [y] (map #(wget v1 % y) (range (:sx v1))))
                           v1-y-range)
             v2-x-range (range (:sy v2))
             v2-cols (mapv (fn [x] (map #(wget v2 % x) (range (:sx v2))))
                           v2-x-range)
             xy (for [y (range (:sy v1)), x (range (:sy v2))]
                  [x y])]
         (mapv (fn [[x y]] (apply + (map * (nth v1-rows y) (nth v2-cols x))))
               xy))))


(defn T
  "転置行列"
  [v]
  (vol (:sy v) (:sx v)
       (let [xy (for [x (range (:sx v)), y (range (:sy v))]
                  [x y])]
         (mapv (fn [[x y]] (wget v x y)) xy))))

(defn repeat-vol
  [v n]
  (dot v (vol n 1 (fill-vec n 1))))

(defn sum-vol
  "行を足し合わせて1xNの行列にする"
  [v]
  (dot v (vol 1 (:sx v) (fill-vec (:sx v) 1))))

(defn row [v row]
  (vol 1 (:sy v) (mapv #(wget v row %) (range (:sy v)))))

(defn rows [v]
  (map #(row v %) (range (:sx v))))

(defn append-row [v row]
  (T
   (vol (:sy v)
        (inc (:sx v))
        (vec (concat (:w (T v)) (:w row))))))

(defn stack-rows [& rows]
  (reduce append-row (first rows) (rest rows)))

(defn w-max
  [v]
  (apply max (:w v)))

(defn reduce-elm
  "要素を集計する"
  ([f init v]
   (reduce f init (:w v)))
  ([f v]
   (reduce f 0 (:w v))))

(defn map-w
  ([f v]
   (vol (:sx v) (:sy v)
        (mapv f (:w v))))
  ([f v1 v2]
   (vol (:sx v1) (:sy v1)
        (mapv f (:w v1) (:w v2)))))

(defn map-row
  ([f v]
   (let [done (map f (rows v))]
     (vol (:sx v) (:sy v)
          (vec (for [y (range (:sy v)), x (range (:sx v))]
                 (wget (nth done x) 0 y))))))
  ([f v1 v2]
   (let [done (map f (rows v1) (rows v2))]
     (reduce append-row (first done) (rest done)))))

(defn argmax [v]
  (let [m (apply max (:w v))]
    (reduce-kv (fn [acc i w] (if (= w m) i acc))
               0
               (:w v))))
