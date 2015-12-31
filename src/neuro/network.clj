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
  [this back-vol]
  (let [back-layer (reverse (:layer this))]
    (assoc this :layer
           (reverse
            (map-with-args ly/backward back-layer back-vol :back-vol)))))

(defmethod ly/update :network
  [this f]
  (let [layers (:layer this)
        updated (map (fn [l] (ly/update l f)) layers)]
    (assoc this :layer updated)))


(defn output
  [net]
  (let [out-layer (last (:layer net))]
    (:out-vol out-layer)))


(defn backprop
  "誤差逆伝播法でネットを更新する"
  [net in-vol train-vol updater]
  (let [forwarded (ly/forward net in-vol)
        out-vol (output forwarded)
        delta-vol (vl/w-sub train-vol out-vol)
        backwarded (ly/backward forwarded delta-vol)]
    (ly/update backwarded updater)))




(comment

(def net (nw/network
          (ly/input-layer 2)
          (ly/fc-layer 2 3)
          (ly/sigmoid-layer 3)
          (ly/fc-layer 3 3)
          (ly/softmax-layer 3)))

)
