(ns neuro.vol
  "Matrix for Neural Network"
  (:require [taoensso.tufte :refer [p]])
  (:refer-clojure :exclude [repeat shuffle])
  )


;; util
(defn fill-vec [len fill] (vec (clojure.core/repeat len fill)))

(def ^:private zero-vec-inf (iterate #(conj % 0.0) []))
(defn zero-vec [len] (nth zero-vec-inf len))

(defn one-hot-vec
  "Generate one-hot vector"
  [n max]
  (assoc (zero-vec max) n 1.0))

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


(defrecord VecVol [shape posf w])

(defn gen-posf
  [[sx sy] f]
  (fn [x y]
    (if (and (<= 0 x) (< x sx) (<= 0 y) (< y sy))
      (f x y)
      (throw (IndexOutOfBoundsException. (str "Index=" [x y] ": shape=" [sx sy]))))))

(defn vol
  "as matrix, col -> :sy, row -> :sx"
  ([ix iy wv]
   (->VecVol [ix iy]
             (gen-posf [ix iy] (fn [x y] (+ x (* y ix))))
             (vec wv)))
  ([ix iy]
   (vol ix iy (xavier-vec (* ix iy) ix)))
  ([wv] ; 1 dim
   (vol 1 (count wv) wv)))

(def raw :w)

(def shape :shape)

(defn pos [v x y]
  ((:posf v) x y))

(defn wget [this x y]
  (nth (raw this) (pos this x y)))

(defn pos-seq [v]
  (let [[sx sy] (shape v)]
    (for [y (range sy), x (range sx)]
      (pos v x y))))

(defn raw-vec [v]
  (vec (map (raw v) (pos-seq v))))

(defn copy [v]
  (let [[sx sy] (shape v)]
    (vol sx sy (raw-vec v))))

(defn slice [v start end]
  (let [[_ sy] (shape v)
        len (- end start)]
    (assoc v
           :shape [len sy]
           :posf (gen-posf [len sy] (fn [x y] (pos v (+ x start) y))))))

(defn rows
  ([v]
   (let [[len _] (shape v)]
     (mapv (fn [x] (slice v x (inc x)))
           (range len))))
  ([v i]
   (nth (rows v) i)))

;; test
(defn partition-row [v n]
  (let[[len _] (shape v)
       step (range 0 (inc len) n)]
    (map (fn [s e] (slice v s e))
         step
         (rest step))))

(defn T
  "転置行列"
  [v]
  (let [[sx sy] (shape v)]
    (assoc v
           :shape [sy sx]
           :posf (gen-posf [sy sx] (fn [x y] (pos v y x))))))

(defn dot
  "w行列の掛け算"
  [v1 v2]
  (p :dot
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
                    xy))))))

(defn w-elm-op [f this other]
  (p :w-elm-op
     (let [[sx sy] (shape this)]
       (vol sx sy (mapv f (raw-vec this) (raw-vec other))))))

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


(defn repeat
  [v n]
  (p :repeat
     (let [[_ sy] (shape v)]
       (assoc v
              :shape [n sy]
              :posf (gen-posf [n sy] (fn [x y] (pos v 0 y)))))))

(defn sum-row
  "行を足し合わせて1xNの行列にする"
  [v]
  (p :sum-vol
     (let [[n _] (shape v)]
       (dot v (vol 1 n (fill-vec n 1))))))


(defn w-max
  [v]
  (apply max (raw-vec v)))

(defn reduce-elm
  "要素を集計する"
  ([f init v]
   (reduce f init (raw-vec v)))
  ([f v]
   (reduce f 0 (raw-vec v))))

(defn map-w
  ([f v]
   (let [[sx sy] (shape v)]
     (vol sx sy (mapv f (raw-vec v)))))
  ([f v1 v2]
   (let [[sx sy] (shape v1)]
     (vol sx sy (mapv f (raw-vec v1) (raw-vec v2))))))

;; todo test
(defn map-row
  [f v]
  (let [done (map f (rows v))
        [sx sy] (shape v)]
    (T
     (vol sy sx
          (vec (apply concat (map raw-vec done)))))))

(defn argmax [v]
  (let [[_ size] (shape v)]
    (reduce (fn [max-i i] (if (< (wget v 0 max-i) (wget v 0 i))
                            i
                            max-i))
            0
            (range size))))

(defn gen-perm
  "Generate Permutation index vector of sx"
  [v]
  (let [[len _] (shape v)]
    (vec (clojure.core/shuffle (range len)))))

(defn shuffle
  "Shuffle index of sx"
  ([v]
   (shuffle v (gen-perm v)))
  ([v perm]
   (assoc v :posf (fn [x y] (pos v (perm x) y)))))
