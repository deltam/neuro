(ns neuro.train
  (:require [neuro.core :as core]
            [neuro.vol :as vl]
            [neuro.layer :as ly]
            [neuro.network :as nw]))


(declare w-updater)

(def ^:dynamic +train-config+
  {:learning-rate 0.01
   :mini-batch-size 10
   :weight-decay 0.001
   :momentum 0.6
   :updater w-updater
   :terminater #(< % 0.1)})


;; train congress
(def +now-epoch+ (atom 0))
(def +now-net+ (atom nil))
(def +train-loss-history+ (atom []))
(def +test-loss-history+ (atom []))

(defn init []
  (reset! +now-epoch+ 0)
  (reset! +now-net+ nil)
  (reset! +train-loss-history+ [])
  (reset! +test-loss-history+ []))

(defn- monitoring
  "学習過程をレポートする"
  ([net train-loss]
   (reset! +now-net+ net)
   (swap! +train-loss-history+ conj train-loss))
  ([net train-loss test-loss]
   (swap! +test-loss-history+ conj test-loss)
   (monitoring net train-loss)))


;; 重み更新関数

(defn w-updater
  [w dw]
  (- w (* (:learning-rate +train-config+) dw)))
;  (fn [w dw] (pl/p :update (- w (* @+learning-rate+ dw)))))


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
  (nw/backprop-n-seq net train-pairs (:updater +train-config+)))

(defn train-batch
  [t-seq]
  (loop [cur (second t-seq), s (rest (rest t-seq))]
    (monitoring cur (nw/loss cur))
    (if ((:terminater +train-config+) (nw/loss cur))
      cur
      (recur (first s) (rest s)))))

(defn train
  [net train-pairs test-pairs & confs]
  (binding [+train-config+ (merge +train-config+ (apply hash-map confs))]
    (let [batchs (partition (:mini-batch-size +train-config+) train-pairs)]
      (loop [epoch 1, cur net, b (first batchs), br (rest batchs)]
        (reset! +now-epoch+ epoch)
        (if (nil? b)
          cur
          (recur (inc epoch)
                 (train-batch (train-seq cur b))
                 (first br)
                 (rest br)))))))
