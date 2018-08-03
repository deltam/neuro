(ns neuro.train
  (:require [taoensso.tufte :as tufte :refer (p profiled profile)])
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
           loss (/ (loss merged) n)]
       (update-loss trained loss))))



(defn backprop-seq
  "誤差逆伝播法で更新したネットのシーケンスを返す"
  [net in-vol train-vol updater]
  (iterate (fn [cur-net]
             (backprop cur-net in-vol train-vol updater))
           net))

(defn backprop-n-seq
  [net train-pairs updater]
  (iterate (fn [cur-net]
             (backprop-n cur-net train-pairs updater))
           net))



;;; train funcs

(defn train-seq
  [net train-pairs]
  (backprop-n-seq net train-pairs (gen-w-updater)))

(defn train-batch
  [net batch]
  (p ::train-batch
     (backprop-n net batch (gen-w-updater))))

(defn train-batch-all
  [init-net batchs]
  (p ::tb-all
     (reduce (fn [net b]
               (swap! +train-loss-history+ conj (nw/loss net))
               (train-batch net b))
             init-net
             batchs)))

(defn train ;-sgd
  [net train-pairs & confs]
  (binding [+train-config+ (merge +train-config+ (apply hash-map confs))]
    (let [batchs (partition (:mini-batch-size +train-config+) train-pairs)]
      (reset! +now-net+ net)
      (reset! +now-epoch+ 0)
      (reset! +num-batchs+ (count batchs))
      (loop [epoch 0, cur net]
        (reset! +now-epoch+ epoch)
        (reset! +now-net+ cur)
        (p ::report
           ((:epoch-reporter +train-config+) epoch cur))
        (if (< epoch (:epoch-limit +train-config+))
          (recur (inc epoch) (train-batch-all cur batchs))
          cur)))))


;;; gradient checking (for debug)

(defn add-w-eps
  [net l i o eps]
  (let [layer (nth (:layer net) l)
        v (:w layer)
        we (+ (vl/wget v i o) eps)]
    (assoc net :layer
           (assoc (vec (:layer net)) l
                  (assoc layer :w
                         (vl/wset v i o we))))))

(defn get-dw
  [net l i o]
  (let [layer (nth (:layer net) l)
        v (:dw layer)]
    (vl/wget v i o)))

(defn calc-loss
  [net in-vol train-vol]
  (:loss
   (ly/backward (loss-layer (ly/forward net in-vol))
                train-vol)))

(defn gradient-checking
  "To use when debug of backpropagation.
  See http://ufldl.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization

  (def net (nw/network
            (ly/input 2)
            (ly/fc 2 5)
            (ly/relu 5)
            (ly/fc 5 4)
            (ly/sigmoid 4)
            (ly/fc 4 3)
            (ly/tanh 3)
            (ly/fc 3 2)
            (ly/softmax 2)))

  (gradient-checking net (vl/vol [2 3]) (vl/vol [1 0]) 5 1 0)
  ;> {:result true, :dw -0.11955934238580034, :grad -0.11955934228480292}

  (gradient-checking net (vl/vol [2 3]) (vl/vol [0 1]) 5 1 0)
  ;> {:result true, :dw 0.07476710855760384, :grad 0.07476710852882817}

  (gradient-checking net (vl/vol [2 3]) (vl/vol [0 1]) 7 1 0)
  ;> {:result true, :dw 0.061648897173240895, :grad 0.06164889717413802}"
  [net in-vol train-vol l i o]
  (let [net-bp (backprop net in-vol train-vol (fn [w dw] (- w (* 0.01 dw))))
        dw (get-dw net-bp l i o)
        eps 0.0001
        net1 (add-w-eps net l i o eps)
        net2 (add-w-eps net l i o (- eps))
        loss1 (calc-loss net1 in-vol train-vol)
        loss2 (calc-loss net2 in-vol train-vol)
        grad (/ (- loss1 loss2)
                (* 2 eps))]
    {:result (< (Math/abs (- dw grad))
                eps)
     :dw dw
     :grad grad}))
