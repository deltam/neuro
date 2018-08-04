(ns neuro.train
  (:require [taoensso.tufte :as tufte :refer (p)])
  (:require [neuro.layer :as ly]
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
(def +num-batchs+ (atom 0))

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



;;; backpropagation

(defn backprop
  "誤差逆伝播法でネットを更新する"
  [net in-vol train-vol updater]
  (p ::backprop
     (let [net-b (p ::backward
                    (ly/backward
                     (p ::forward
                        (ly/forward net in-vol))
                     train-vol))]
       (ly/update-w net-b updater))))

(defn backprop-n
  "複数の入力ー回答データに対して誤差逆伝播法を適用する"
  [net train-pairs updater]
  (p ::backprop-n
     (let [n (count train-pairs)
           merged (reduce ly/merge-w
                          (map (fn [[in-vol train-vol]]
                                 (backprop net in-vol train-vol updater))
                               train-pairs))
           trained (ly/map-w merged #(/ % n))
           loss (/ (nw/loss merged) n)]
       (nw/update-loss trained loss))))


;;; train funcs

(defn update-mini-batch
  [net batch]
  (p ::update-mini-batch
     (backprop-n net batch (gen-w-updater))))

(defn reduce-mini-batchs
  [init-net batchs]
  (p ::reduce-mini-batch
     (reduce (fn [net b]
               (swap! +train-loss-history+ conj (nw/loss net))
               (update-mini-batch net b))
             init-net
             batchs)))

(defn sgd
  "Stochastic gradient descent"
  [net train-pairs & confs]
  (binding [+train-config+ (merge +train-config+ (apply hash-map confs))]
    (let [batchs (partition (:mini-batch-size +train-config+) train-pairs)]
      (reset! +now-net+ net)
      (reset! +now-epoch+ 0)
      (reset! +num-batchs+ (count batchs))
      (loop [epoch 0, cur net]
        (reset! +now-epoch+ epoch)
        (reset! +now-net+ cur)
        ((:epoch-reporter +train-config+) epoch cur)
        (if (< epoch (:epoch-limit +train-config+))
          (recur (inc epoch) (reduce-mini-batchs cur batchs))
          cur)))))
