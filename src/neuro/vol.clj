(ns neuro.vol
  (:require [clojure.data.generators :as gr]))


;; util
(defn zero-vec [len] (vec (repeat len 0.0)))
(defn rand-vec [len]
  (binding [gr/*rnd* (java.util.Random. (System/currentTimeMillis))]
    (vec (repeatedly len (fn [] (- (gr/double) 0.5))))))

;; 最小構成要素
(defn vol
  ([ix iy w-vec dw-vec]
   {:sx ix, :sy iy
    :w w-vec
    :dw dw-vec})
  ([ix iy w-vec]
   (vol ix iy w-vec (zero-vec (* ix iy))))
  ([ix iy]
   (vol ix iy
        (rand-vec (* ix iy))
        (zero-vec (* ix iy))))
  ([len] ; 1 dim
   (vol 1 len (zero-vec len) (zero-vec len))))

(defn- xy->i
  "2次元から1次元への座標変換"
  [v x y]
  (+ x (* y (:sx v))))

(defn wget [v x y] (nth (:w v) (xy->i v x y)))

(defn dwget [v x y] (nth (:dw v) (xy->i v x y)))

(defn wset [v x y w]
  (assoc v :w
         (assoc (:w v) (xy->i v x y) w)))

(defn dwset [v x y dw]
  (assoc v :dw
         (assoc (:dw v) (xy->i v x y) dw)))

(defn w-add
  "w行列の同じ要素同士を足し合わせる, v1,v2は同じサイズとする"
  [v1 v2]
  (vol (:sx v1) (:sy v1)
       (mapv + (:w v1) (:w v2))))

(defn w-mul
  "w行列の掛け算"
  [v1 v2]
  (vol (:sx v2) (:sy v1)
       (vec
        (for [x (range (:sx v2)), y (range (:sy v1))
              :let [v1-vec (map #(wget v1 % y) (range (:sx v1)))
                    v2-vec (map #(wget v2 x %) (range (:sy v2)))]]
          (apply + (map * v1-vec v2-vec))))))


(defn map-w [f v]
  (vol (:sx v) (:sy v)
       (mapv f (:w v))))

(comment

(def vol {:node 5
          :w [1 2 3 4 5]
          :dw [0 0 0 0 0]})

)
