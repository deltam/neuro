(ns neuro.vol
  "Matrix for Neural Network"
  (:require [taoensso.tufte :refer [p]])
  (:refer-clojure :exclude [repeat shuffle partition print rand])
  )


;; util
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

(def raw :w)

(def shape :shape)

(defn pos [v c r]
  ((:posf v) c r))

(defn gen-posf
  [[col row] f]
  (fn [c r]
    (if (and (<= 0 c) (< c col) (<= 0 r) (< r row))
      (f c r)
      (throw (IndexOutOfBoundsException. (str "index=" [c r] ": shape=" [col row]))))))


;; generate

(defn vol
  "as matrix col, row"
  ([col row wv]
   (->VecVol [col row]
             (gen-posf [col row] (fn [c r] (+ (* c row) r)))
             (vec wv)))
  ([wv] ; 1 dim
   (vol 1 (count wv) wv)))

(defn fill
  ([col row fillv]
   (let [v (vol col row [fillv])]
     (assoc v :posf (gen-posf (shape v)
                              (fn [_ _] 0)))))
  ([len fillv] (fill 1 len fillv)))

(defn zeros
  ([col row] (fill col row 0.0))
  ([len] (zeros 1 len)))

(defn ones
  ([col row] (fill col row 1.0))
  ([len] (ones 1 len)))

(defn rand
  ([col row]
   (vol col row (xavier-vec (* col row) col)))
  ([len] (rand 1 len)))

(defn one-hot [max idx-seq]
  (let [v (vol (count idx-seq) max [0.0 1.0])]
    (assoc v :posf
           (gen-posf (shape v)
                     (fn [c r]
                       (let [i (nth idx-seq c)]
                         (if (= r i) 1 0)))))))




(defn wget [this c r]
  (nth (raw this) (pos this c r)))

(defn pos-seq [v]
  (let [[col row] (shape v)]
    (for [c (range col), r (range row)]
      (pos v c r))))

(defn raw-vec [v]
  (vec (map (raw v) (pos-seq v))))

(defn clone [v]
  (let [[col row] (shape v)]
    (vol col row (raw-vec v))))

(defn print [v]
  (let [[col row] (shape v)]
    (doseq [c (range col), r (range row)]
      (clojure.core/print (wget v c r))
      (if (= (inc r) row)
        (println)
        (clojure.core/print "\t")))))


(defn slice [v start-col end-col]
  (let [[_ row] (shape v)
        len (- end-col start-col)]
    (assoc v
           :shape [len row]
           :posf (gen-posf [len row] (fn [c r] (pos v (+ c start-col) r))))))

(defn rows
  ([v]
   (let [[len _] (shape v)]
     (mapv (fn [c] (slice v c (inc c)))
           (range len))))
  ([v i]
   (nth (rows v) i)))

(defn partition [v n]
  (let[[len _] (shape v)
       step (range 0 (inc len) n)]
    (map (fn [s e] (slice v s e))
         step
         (rest step))))



(defn T
  "転置行列"
  [v]
  (let [[col row] (shape v)]
    (assoc v
           :shape [row col]
           :posf (gen-posf [row col] (fn [c r] (pos v r c))))))

(defn dot
  "w行列の掛け算 (dot [N M] [M K]) = [N K]"
  [v1 v2]
  (p :dot
     (let [[col1 row1] (shape v1)
           [col2 row2] (shape v2)
           raw-vec1 (clojure.core/partition row1 (raw-vec v1))
           raw-vec2 (clojure.core/partition col2 (raw-vec (T v2)))]
       (vol col1 row2
            (for [rv1 raw-vec1, rv2 raw-vec2]
              (apply + (map * rv1 rv2)))))))


(defn w-elm-op [f this other]
  (let [[sx sy] (shape this)]
    (vol sx sy (map f (raw-vec this) (raw-vec other)))))

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
  (let [[_ row] (shape v)]
    (assoc v
           :shape [n row]
           :posf (gen-posf [n row] (fn [_ r] (pos v 0 r))))))

(defn sum-row
  "行を足し合わせて1xNの行列にする"
  [v]
  (p :sum-row
     (let [[n _] (shape v)]
       (dot (ones n) v))))


(defn w-max
  [v]
  (apply max (raw-vec v)))

(defn argmax [v]
  (let [[col row] (shape v)]
    (reduce (fn [[mc mr] [c r]] (if (< (wget v mc mr) (wget v c r))
                                  [c r]
                                  [mc mr]))
            [0 0]
            (for [c (range col), r (range row)]
              [c r]))))


(defn reduce-elm
  "要素を集計する"
  ([f init v]
   (reduce f init (raw-vec v)))
  ([f v]
   (reduce f 0 (raw-vec v))))

(defn map-w
  ([f v]
   (let [[col row] (shape v)]
     (vol col row (map f (raw-vec v)))))
  ([f v1 v2]
   (let [[col row] (shape v1)]
     (vol col row (map f (raw-vec v1) (raw-vec v2))))))


(defn gen-perm
  "Generate Permutation index vector of cols"
  [v]
  (let [[len _] (shape v)]
    (vec (clojure.core/shuffle (range len)))))

(defn shuffle
  "Shuffle index of sx"
  ([v]
   (shuffle v (gen-perm v)))
  ([v perm]
   (assoc v :posf (fn [c r] (pos v (perm c) r)))))
