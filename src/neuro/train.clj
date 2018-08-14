(ns neuro.train
;  (:require [taoensso.tufte :as tufte :refer [p]])
  (:require [neuro.layer :as ly]
            [neuro.network :as nw]))


(def ^:dynamic *train-params*
  {:learning-rate 0.01
   :mini-batch-size 10
   :epoch-limit 10
   :weight-decay 0.001
   :momentum 0.6
   :updater nil
   :epoch-reporter (fn [epoch net] nil)})

(defmacro with-params
  "specified train parameters"
  [params-vec train-expr]
  (let [conf (merge *train-params* (apply hash-map params-vec))]
    `(binding [*train-params* ~conf]
       ~train-expr)))


;; train current status
(def ^:dynamic *now-epoch* (atom 0))
(def ^:dynamic *now-net* (atom nil))
(def ^:dynamic *train-loss-history* (atom []))
(def ^:dynamic *test-loss-history* (atom []))
(def ^:dynamic *num-batchs* (atom 0))

(defn init []
  (reset! *now-epoch* 0)
  (reset! *now-net* nil)
  (reset! *train-loss-history* [])
  (reset! *test-loss-history* []))


;; 重み更新関数生成
(defn gen-w-updater
  [mini-batch-size]
  (if-let [f (:updater *train-params*)]
    f
    (let [rate (/ (:learning-rate *train-params*) mini-batch-size)]
      (fn [w dw]
        (- w (* rate dw))))))

;(defn- update-by-gradient
;  "重みを勾配に従って更新した値を返す"
;  [w grad bias?]
;  (let [de (if bias? grad
;               (* grad (* *weight-decay-param* w)))]
;    (- w (* @+learning-rate* de))))

;(defn- momentum
;  "モメンタム項の計算"
;  [pre-nn cur-nn nn]
;  (nw/map-nn (fn [l i o w]
;               (let [dw ( - (nw/wget cur-nn l i o)
;                            (nw/wget pre-nn l i o))]
;                 (* w (* *momentum-param* dw))))
;             nn))



;;; backpropagation

(defn backprop
  "誤差逆伝播法でネットを更新する"
  [net in-vol answer-vol]
  (let [net-f (ly/forward net in-vol)]
    (ly/backward net-f answer-vol)))

(defn backprop-n
  "複数の入力ー回答データに対して誤差逆伝播法を適用する"
  [net train-pairs]
  (let [merged (reduce ly/merge-w
                       (pmap (fn [[in-vol answer-vol]]
                              (backprop net in-vol answer-vol))
                            train-pairs))
        loss (/ (nw/loss merged) (count train-pairs))]
    (nw/update-loss merged loss)))


;;; train funcs

(defn update-mini-batch
  [net batch]
  (ly/update-w (backprop-n net batch)
               (gen-w-updater (count batch))))

(defn reduce-mini-batchs
  [init-net batchs]
  (reduce (fn [net b]
            (swap! *train-loss-history* conj (nw/loss net))
            (update-mini-batch net b))
          init-net
          batchs))

(defn sgd
  "Stochastic gradient descent"
  [net train-pairs]
  (let [batchs (partition (:mini-batch-size *train-params*) train-pairs)]
    (reset! *num-batchs* (count batchs))
    (loop [epoch 0, cur net]
      (reset! *now-epoch* epoch)
      (reset! *now-net* cur)
      (future ((:epoch-reporter *train-params*) epoch cur))
      (if (< epoch (:epoch-limit *train-params*))
        (recur (inc epoch) (reduce-mini-batchs cur batchs))
        cur))))
