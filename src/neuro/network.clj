(ns neuro.network
  (:require [taoensso.tufte :refer [p]])
  (:require [neuro.layer :as ly]))


(defn- reduce-layers [layers f init out-fn]
  (first
   (reduce (fn [[acc out] cur]
             (let [next (f cur out)]
               [(conj acc next) (out-fn next)]))
           [[] init]
           layers)))

(defrecord Network [layer]
  ly/Executable
  (forward [this in-vol]
    (p :net-for
       (assoc this :layer (reduce-layers (:layer this) ly/forward in-vol ly/output))))
  (backward [this answer-vol]
    (p :net-back
       (assoc this :layer (vec (reverse
                                (reduce-layers (reverse (:layer this)) ly/backward answer-vol ly/grad))))))
  (output [this]
    (let [out-layer (last (:layer this))]
      (ly/output out-layer)))
  (grad [this] (map ly/grad (:layer this)))
  ly/Optimizable
  (update-p [this f]
    (p :net-up
       (assoc this :layer (map #(if (satisfies? ly/Optimizable %)
                                  (ly/update-p % f)
                                  %)
                               (:layer this))))))


(defn network [& layers]
  (->Network (vec layers)))

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

(defn feedforward
  [net in-vol]
  (ly/output (ly/forward net in-vol)))



;; util

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
  (p :update-loss
     (assoc net :layer
            (map #(if (nil? (:loss %))
                    %
                    (assoc % :loss loss))
                 (:layer net)))))
