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


(defn output
  [net]
  (let [out-layer (last (:layer net))]
    (:out-vol out-layer)))

(defn loss
  [net]
  (let [loss-layer (last (:layer net))]
    (:loss loss-layer)))


(defn backprop
  "誤差逆伝播法でネットを更新する"
  [net in-vol train-vol updater]
  (let [net-f (ly/forward net in-vol)
        net-b (ly/backward net-f train-vol)]
    (ly/update net-b updater)))

(defn backprop-seq
  "誤差逆伝播法で更新したネットのシーケンスを返す"
  [net in-vol train-vol updater]
  (iterate (fn [cur-net]
             (backprop cur-net in-vol train-vol updater))
           net))




(comment

(def net (nw/network
          (ly/input-layer 2)
          (ly/fc-layer 2 3)
          (ly/sigmoid-layer 3)
          (ly/fc-layer 3 3)
          (ly/softmax-layer 3)))

)
