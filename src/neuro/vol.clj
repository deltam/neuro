(ns neuro.vol
  "Matrix for Neural Network")


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
  (gauss-vec len (/ 2.0 (Math/sqrt num-nodes))))

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
  (vol (:sx v2) (:sy v1)
       (let [v1-y-range (range (:sy v1))
             v1-rows (mapv (fn [y] (map #(wget v1 % y) (range (:sx v1))))
                           v1-y-range)
             v2-x-range (range (:sx v2))
             v2-cols (mapv (fn [x] (map #(wget v2 x %) (range (:sy v2))))
                           v2-x-range)
             xy (for [y (range (:sy v1)), x (range (:sx v2))]
                  [x y])]
         (mapv (fn [[x y]] (apply + (map * (nth v1-rows y) (nth v2-cols x))))
               xy))))

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
   (reduce f 0 (:w v))))

(defn map-w
  ([f v]
   (vol (:sx v) (:sy v)
        (mapv f (:w v))))
  ([f v1 v2]
   (vol (:sx v1) (:sy v1)
        (mapv f (:w v1) (:w v2)))))
