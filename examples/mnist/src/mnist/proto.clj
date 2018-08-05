(ns mnist.proto
  "ためしにいちから実装してみる
  http://nnadl-ja.github.io/nnadl_site_ja/chap1.html"
  (:require [neuro.vol :as vl]
            [mnist.data :as md]))

(defn rand-vol [x y] (vl/vol x y (vl/rand-vec (* x y))))
(defn zero-vol [x y] (vl/vol x y (vl/zero-vec (* x y))))


(defn vectorize [f] (fn [v] (vl/map-w f v)))

(defn sigmoid [x] (/ 1.0 (+ 1.0 (Math/exp (- x)))))
(def sigmoid-vec (vectorize sigmoid))

(defn sigmoid-prime [x] (* (sigmoid x) (- 1.0 (sigmoid x))))
(def sigmoid-prime-vec (vectorize sigmoid-prime))


(defn network
  "Generate Neural Network"
  [& more]
  {:num_layers (count more)
   :sizez more
   :biases (mapv (fn [y] (rand-vol 1 y))
                 (rest more))
   :weights (mapv (fn [x y] (rand-vol x y))
                  more
                  (rest more))})

(defn feedforward [nn in-vol]
  (reduce (fn [r [wv bv]]
            (sigmoid-vec
             (vl/w-add (vl/w-prod wv r) bv)))
          in-vol
          (map vector (:weights nn) (:biases nn))))


(def load-data (md/load-train))

(def my-data (shuffle load-data))
(def test-data (take 1000 my-data))
(def train-data (drop 1000 my-data))

(declare update-mini-batch backprop evaluate)

(defn sgd [nn train-data epoch mini-batch-size eta test-data]
  (let [mini-batchs (partition mini-batch-size train-data)]
    (loop [ep 0, cur-nn nn]
      (printf "Epoch %d: %d / %d\n" ep (evaluate cur-nn test-data) (count test-data))
      (flush)
      (if (< (inc ep) epoch)
        (recur (inc ep)
               (reduce (fn [net b] (update-mini-batch net b eta))
                       cur-nn
                       mini-batchs))
        cur-nn))))


(defn size-zero-vol [{x :sx, y :sy}] (zero-vol x y))
(defn update-eta [rate] (fn [p dp] (- p (* dp rate))))

(defn update-mini-batch [nn mini-batch eta]
  (let [[acm-b acm-w] (reduce (fn [[rb rw] [in out]]
                                (let [[db dw] (backprop nn in out)]
                                  [(map vl/w-add rb db)
                                   (map vl/w-add rw dw)]))
                              [(map size-zero-vol (:biases nn))
                               (map size-zero-vol (:weights nn))]
                              mini-batch)
        n (count mini-batch)]
    (assoc nn
           :biases (map #(vl/map-w (update-eta (/ eta n)) %1 %2) (:biases nn) acm-b)
           :weights (map #(vl/map-w (update-eta (/ eta n)) %1 %2) (:weights nn) acm-w))))


(defn cost-derivative [nn act out]
  (vl/w-sub act out))

(defn backprop [nn in-vol out-vol]
  ;; feedforwad
  (let [[acts zs] (reduce (fn [[as zs] [b w]]
                            (let [x (last as)
                                  z (vl/w-add (vl/w-prod w x) b)
                                  a (sigmoid-vec z)]
                              [(conj as a) (conj zs z)]))
                          [[in-vol] []]
                          (map vector (:biases nn) (:weights nn)))
        delta (vl/w-prod-h (cost-derivative nn (last acts) out-vol)
                           (sigmoid-prime-vec (last zs)))
        ;; backward
        [db dw] (reduce (fn [[dbs dws delta] [act z w]]
                          (let [d (vl/w-prod-h (vl/w-prod (vl/T w) delta)
                                               (sigmoid-prime-vec z))
                                dw (vl/w-prod d (vl/T act))]
                            [(concat [d] dbs) (concat [dw] dws) d]))
                        [[delta] [(vl/w-prod delta (vl/T (first (rest (reverse acts)))))] delta]
                        (map vector
                             (drop 2 (reverse acts))
                             (drop 1 (reverse zs))
                             (reverse (:weights nn))))]
    [db dw]))

(defn argmax [x-vol]
  (last
   (reduce (fn [[rx ri] [x i]] (if (< rx x)
                                 [x i]
                                 [rx ri]))
           (map vector (:w x-vol) (range)))))

(defn evaluate [nn test-data]
  (count (filter (fn [[in out]]
                   (= (argmax out) (argmax (feedforward nn in))))
                 test-data)))
