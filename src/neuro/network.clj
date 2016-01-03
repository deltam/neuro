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



(comment

(def net (nw/network
          (ly/input-layer 2)
          (ly/fc-layer 2 3)
          (ly/sigmoid-layer 3)
          (ly/fc-layer 3 3)
          (ly/softmax-layer 3)))

)
