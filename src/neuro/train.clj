(ns neuro.train
  (:require [neuro.core :as core]
            [neuro.vol :as vl]
            [neuro.layer :as ly]
            [neuro.network :as nw]))


(def ^:dynamic +train-config+
  {:learning-rate 0.01
   :mini-batch-size 10
   :epoch-limit 10
   :weight-decay 0.001
   :momentum 0.6
   :updater nil
   :epoch-reporter (fn [epoch net] nil)})


;; train current status
(def +now-epoch+ (atom 0))
(def +now-net+ (atom nil))
(def +train-loss-history+ (atom []))
(def +test-loss-history+ (atom []))

(defn init []
  (reset! +now-epoch+ 0)
  (reset! +now-net+ nil)
  (reset! +train-loss-history+ [])
  (reset! +test-loss-history+ []))


;; 重み更新関数生成
(defn gen-w-updater
  []
  (if-let [f (:updater +train-config+)]
    f
    (fn [w dw]
      (- w (* (:learning-rate +train-config+) dw)))))

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
  (nw/backprop-n-seq net train-pairs (gen-w-updater)))

(defn train-batch
  [net batch]
  (nw/backprop-n net batch (gen-w-updater)))

(defn train-batch-all
  [init-net batchs]
  (reduce (fn [net b]
            (swap! +train-loss-history+ conj (nw/loss net))
            (train-batch net b))
          init-net
          batchs))

(defn train
  [net train-pairs & confs]
  (binding [+train-config+ (merge +train-config+ (apply hash-map confs))]
    (let [batchs (partition (:mini-batch-size +train-config+) train-pairs)]
      (reset! +now-net+ net)
      (reset! +now-epoch+ 0)
      (loop [epoch 0, cur net]
        (reset! +now-epoch+ epoch)
        (reset! +now-net+ cur)
        ((:epoch-reporter +train-config+) epoch cur)
        (if (< epoch (:epoch-limit +train-config+))
          (recur (inc epoch) (train-batch-all cur batchs))
          cur)))))
