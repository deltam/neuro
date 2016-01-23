(ns neuro.network
  (:require [neuro.vol :as vl])
  (:require [neuro.layer :as ly]))


; util
(defn- map-with-args
  "ひとつ前の関数適用の結果より引数を抜き出して受け渡しながらmapする"
  [f coll init-val arg-key]
  (loop [cur (first coll), r (rest coll), done [], v init-val]
    (if (nil? cur)
      done
      (let [next (f cur v)]
        (recur (first r) (rest r) (conj done next) (arg-key next))))))


; neural network
(defn network [& layers]
  {:type :network
   :layer layers})

(defmethod ly/forward :network
  [this in-vol]
  (assoc this :layer
         (map-with-args ly/forward (:layer this) in-vol :out-vol)))

(defmethod ly/backward :network
  [this delta-vol]
  (let [back-layer (reverse (:layer this))]
    (assoc this :layer
           (reverse
            (map-with-args ly/backward back-layer delta-vol :delta-vol)))))

(defmethod ly/update :network
  [this f]
  (let [layers (:layer this)
        updated (map #(ly/update % f) layers)]
    (assoc this :layer updated)))

(defmethod ly/merge-w :network
  [this net]
  (assoc this :layer
         (map (fn [l1 l2] (ly/merge-w l1 l2))
              (:layer this)
              (:layer net))))

(defmethod ly/map-w :network
  [this f]
  (assoc this :layer
         (map #(ly/map-w % f)
              (:layer this))))



;; util

(defn output
  [net]
  (let [out-layer (last (:layer net))]
    (:out-vol out-layer)))

(defn layer
  [net idx]
  (nth (:layer net) idx))

(defn loss-layer
  [net]
  (last (:layer net)))

(defn loss
  [net]
  (:loss (loss-layer net)))

(defn update-loss
  [net loss]
  (assoc net :layer
         (map #(if (nil? (:loss %))
                 %
                 (assoc % :loss loss))
              (:layer net))))



;; backpropagation

(defn backprop
  "誤差逆伝播法でネットを更新する"
  [net in-vol train-vol updater]
  (let [net-f (ly/forward net in-vol)
        net-b (ly/backward net-f train-vol)]
    (ly/update net-b updater)))

(defn backprop-n
  "複数の入力ー回答データに対して誤差逆伝播法を適用する"
  [net train-pairs updater]
  (let [merged (reduce (fn [r v] (ly/merge-w r v))
                       (map (fn [[in-vol train-vol]]
                              (backprop net in-vol train-vol updater))
                            train-pairs))
        n (count train-pairs)
        trained (ly/map-w merged (fn [w] (/ w n)))
        loss (/ (loss merged) n)]
    (update-loss trained loss)))



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



;; gradient checking

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






(comment

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

(nw/gradient-checking net (vl/vol [2 3]) (vl/vol [1 0]) 5 1 0)
;> {:result true, :dw -0.11955934238580034, :grad -0.11955934228480292}

(nw/gradient-checking net (vl/vol [2 3]) (vl/vol [0 1]) 5 1 0)
;> {:result true, :dw 0.07476710855760384, :grad 0.07476710852882817}

(nw/gradient-checking net (vl/vol [2 3]) (vl/vol [0 1]) 7 1 0)
;> {:result true, :dw 0.061648897173240895, :grad 0.06164889717413802}

)
