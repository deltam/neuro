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


(defrecord VecVol [sx sy w T])

(defn vol
  "as matrix, col -> :sy, row -> :sx"
  ([ix iy wv]
   (->VecVol ix iy wv false))
  ([ix iy]
   (vol ix iy (xavier-vec (* ix iy) ix)))
  ([wv] ; 1 dim
   (vol 1 (count wv) wv)))

(defn shape [v]
  (if (:T v)
    [(:sy v) (:sx v)]
    [(:sx v) (:sy v)]))

(defn- xy->i
  "2次元から1次元への座標変換"
  [v x y]
  (if (:T v)
    (+ y (* x (:sx v)))
    (+ x (* y (:sx v)))))

(defn wget [this x y]
  (nth (:w this) (xy->i this x y)))

(defn wset [this x y w]
  (assoc this :w
         (assoc (:w this) (xy->i this x y) w)))

;; todo test
(defn w-vec [v]
  (let [[sx sy] (shape v)]
    (vec (for [y (range sy), x (range sx)]
           (wget v x y)))))

(defn w-elm-op [f this other]
  (let [[sx sy] (shape this)]
    (vol sx sy (mapv f (w-vec this) (w-vec other)))))

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

(defn T
  "転置行列"
  [v]
  (update v :T not))

(defn repeat-vol
  [v n]
  (dot v (vol n 1 (fill-vec n 1))))

(defn sum-vol
  "行を足し合わせて1xNの行列にする"
  [v]
  (let [[n _] (shape v)]
    (dot v (vol 1 n (fill-vec n 1)))))

(defn row [v row]
  (let [[_ row-len] (shape v)]
    (vol 1 row-len (mapv #(wget v row %) (range row-len)))))

(defn rows [v]
  (let [[row-len _] (shape v)]
    (map #(row v %) (range row-len))))

(defn append-row [v row]
  (let [[sx sy] (shape v)
        rows (conj (vec (rows v)) row)]
    (T
     (vol sy
          (inc sx)
          (apply concat (map :w rows))))))

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
   (let [[sx sy] (shape v)]
     (vol sx sy (mapv f (w-vec v)))))
  ([f v1 v2]
   (let [[sx sy] (shape v1)]
     (vol sx sy (mapv f (w-vec v1) (w-vec v2))))))

;; todo test
(defn map-row
  [f v]
  (let [done (map f (rows v))
        [sx sy] (shape v)]
    (T
     (vol sy sx
          (vec (apply concat (map w-vec done))))))
  )

(defn argmax [v]
  (let [m (apply max (:w v))]
    (reduce-kv (fn [acc i w] (if (= w m) i acc))
               0
               (:w v))))
