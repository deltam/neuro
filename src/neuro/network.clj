(ns neuro.network
  (:require [neuro.layer :as ly]))


(declare map-with-args)

(defrecord Network [layer]
  ly/Layer
  (forward [this in-vol]
    (assoc this :layer
           (map-with-args ly/forward (:layer this) in-vol :out-vol)))
  (backward [this delta-vol]
    (let [back-layer (reverse (:layer this))]
      (assoc this :layer
             (reverse
              (map-with-args ly/backward back-layer delta-vol :delta-vol)))))
  (update-w [this f]
    (let [layers (:layer this)
          updated (map #(ly/update-w % f) layers)]
      (assoc this :layer updated)))
  (merge-w [this other]
    (assoc this :layer
           (map ly/merge-w (:layer this) (:layer other))))
  (map-w [this f]
    (assoc this :layer
           (map #(ly/map-w % f)
                (:layer this)))))


(defn network [& layers]
  (->Network layers))

(defn parse-net [& defs]
  (let [grp (partition 3 (concat defs [nil nil]))
        lds (map (fn [l [_ p _]] (vec (concat l [p]))) grp (concat (rest grp) [nil nil]))]
    (mapcat (fn [[lt p1 con p2]]
              (let [l (ly/gen lt p1)]
                (if (nil? con)
                  [l]
                  [l (ly/gen con p1 p2)])))
            lds)))

(defn gen-net
  "generate neural net
  ;; sample
  (gen-net
    :input 10 :fc
    :sigmoid 80 :fc
    :relu 20 :fc
    :softmax 2)"
  [& defs]
  (apply network (apply parse-net defs)))

(declare output)

(defn feedforward
  [net in-vol]
  (output
   (ly/forward net in-vol)))



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

(defn- map-with-args
  "ひとつ前の関数適用の結果より引数を抜き出して受け渡しながらmapする"
  [f coll init-val arg-key]
  (loop [cur (first coll), r (rest coll), done [], v init-val]
    (if (nil? cur)
      done
      (let [next (f cur v)]
        (recur (first r) (rest r) (conj done next) (arg-key next))))))
