(ns neuro.train
  (:require [neuro.core :as core]
            [neuro.vol :as vl]
            [neuro.layer :as ly]
            [neuro.network :as nw]))


(def ^:dynamic *weight-decay-param* 0.001)
(def ^:dynamic *mini-batch-size* 10)
(def ^:dynamic *momentum-param* 0.6)

(def +learning-rate+ (atom 0.01))

(def +now-epoch+ (atom 0))
(def +now-net+ (atom nil))
(def +train-err-vec+ (atom []))
(def +test-err-vec+ (atom []))


(defn gen-train-pairs
  "vector -> input, answer pair"
  [in-test-vec]
  (map vec
       (partition 2
                  (map vl/vol in-test-vec))))


(defn init []
  (reset! +now-epoch+ 0)
  (reset! +now-net+ nil)
  (reset! +train-err-vec+ [])
  (reset! +test-err-vec+ []))

(defn- monitoring
  "学習過程をレポートする"
  ([net train-err]
   (reset! +now-net+ net)
   (swap! +train-err-vec+ conj train-err))
  ([net train-err test-err]
   (swap! +test-err-vec+ conj test-err)
   (monitoring net train-err)))


;; 重み更新関数

(defn w-updater
  []
  (fn [w dw] (- w (* @+learning-rate+ dw))))
;  (fn [w dw] (pl/p :update (- w (* @+learning-rate+ dw)))))

(def default-updater (w-updater))

;(defn- update-by-gradient
;  "重みを勾配に従って更新した値を返す"
;  [w grad bias?]
;  (let [de (if bias? grad
;               (+ grad (* *weight-decay-param* w)))]
;    (- w (* @+learning-rate+ de))))

;(defn- momentum
;  "モメンタム項の計算"
;  [pre-nn cur-nn nn]
;  (nw/map-nn (fn [l i o w]
;               (let [dw ( - (nw/wget cur-nn l i o)
;                            (nw/wget pre-nn l i o))]
;                 (+ w (* *momentum-param* dw))))
;             nn))



;; train funcs

(defn train-seq
  [net train-pairs]
  (nw/backprop-n-seq net train-pairs default-updater))

(defn train-batch
  [t-seq terminate-f]
  (loop [cur (second t-seq), s (rest (rest t-seq))]
    (monitoring cur (nw/loss cur))
    (if (terminate-f (nw/loss cur))
      cur
      (recur (first s) (rest s)))))

(defn train
  [net train-pairs test-pairs terminate-f]
  (let [batchs (partition *mini-batch-size* (shuffle train-pairs))]
    (loop [epoch 1, cur net, b (first batchs), r (rest batchs)]
      (reset! +now-epoch+ epoch)
      (if (nil? b)
        cur
        (recur (inc epoch)
               (train-batch (train-seq cur b) terminate-f)
               (first r) (rest r))))))
