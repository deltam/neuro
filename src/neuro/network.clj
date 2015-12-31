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




(defn map-nn
  [f nn]
  (let [layers (:layer nn)]
    (apply ly/network
           (map-indexed (fn [l layer]
                          (let [w-vol (:w layer)]
                            (if (nil? w-vol)
                              layer
                              (assoc layer :w
                                     (reduce (fn [v [i o]]
                                               (let [w (vl/wget v i o)]
                                                 (vl/wset v i o (f l i o w))))
                                             w-vol
                                             (for [x (range (:sx w-vol)) y (range (:sy w-vol))]
                                               [x y]))))))
                        layers))))

(defn map-nn-bias
  [f nn]
  (let [layers (:layer nn)]
    (apply ly/network
           (map-indexed (fn [l layer]
                          (let [bias-vol (:bias layer)]
                            (if (nil? bias-vol)
                              layer
                              (assoc layer :bias
                                     (reduce (fn [v o]
                                               (let [b (vl/wget v 0 o)]
                                                 (vl/wset v 0 o (f l o b))))
                                             bias-vol
                                             (range (:sy bias-vol)))))))
                        layers))))


(defn nn-put
  [nn l i o w]
  (let [layer (nth (:layer nn) l)
        w-vol (:w layer)
        w2-vol (vl/wset w-vol i o w)
        layer2 (assoc layer :w w2-vol)]
    (assoc nn :layer
           (assoc (vec (:layer nn)) l layer2))))

(defn nn-put-bias
  [nn l o b]
  (let [layer (nth (:layer nn) l)
        b-vol (:bias layer)
        b2-vol (vl/wset b-vol 0 o b)
        layer2 (assoc layer :bias b2-vol)]
    (assoc nn :layer
           (assoc (vec (:layer nn)) l layer2))))






(comment

(def net (nw/network
          (ly/input-layer 2)
          (ly/fc-layer 2 3)
          (ly/sigmoid-layer 3)
          (ly/fc-layer 3 3)
          (ly/softmax-layer 3)))

)
