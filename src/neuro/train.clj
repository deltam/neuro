(ns neuro.train
  (:require [neuro.core :as core]
            [neuro.vol :as vl]
            [neuro.layer :as ly]
            [neuro.network :as nw]))


(def ^:dynamic *weight-decay-param* 0.001)
(def ^:dynamic *mini-batch-size* 10)
(def ^:dynamic *momentum-param* 0.6)

(def +learning-rate+ (atom 0.01))

(def +train-err-vec+ (atom []))
(def +test-err-vec+ (atom []))
(def +now-net+ (atom nil))


(defn gen-train-pairs
  "vector -> input, answer pair"
  [in-test-vec]
  (map vec
       (partition 2
                  (map vl/vol in-test-vec))))


(defn init []
  (reset! +train-err-vec+ [])
  (reset! +test-err-vec+ [])
  (reset! +now-net+ nil))

(defn- monitoring
  "学習過程をレポートする"
  [epoc train-err test-err net]
  (do (swap! +train-err-vec+ conj train-err)
      (swap! +test-err-vec+ conj test-err)
      (reset! +now-net+ net)))


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

(defn train-one
  [t-seq terminate-f]
  (loop [cur (second t-seq), s (rest (rest t-seq)), loss 0.0, epoc 0]
    (monitoring epoc loss 0.0 cur)
    (if (terminate-f (nw/loss cur))
      cur
      (recur (first s) (rest s) (nw/loss cur) (inc epoc)))))

(defn train
  [net train-pairs test-pairs terminate-f]
  (let [batchs (partition *mini-batch-size* (shuffle train-pairs))]
    (loop [cur net, b (first batchs), r (rest batchs)]
      (if (nil? b)
        cur
        (recur (train-one (train-seq cur b) terminate-f)
               (first r) (rest r))))))
