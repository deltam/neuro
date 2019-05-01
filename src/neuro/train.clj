(ns neuro.train
;  (:require [taoensso.tufte :refer [p]])
  (:require [neuro.layer :as ly]
            [neuro.network :as nw]
            [neuro.vol :as vl]))


(def ^:dynamic *train-params*
  "train hyper parameters"
  {:learning-rate 0.01
   :mini-batch-size 10
   :epoch-limit 10
   :updater nil
   :epoch-reporter (fn [epoch net] nil)})

(defn new-status []
  "Generate train progress status"
  {:now-epoch 0
   :now-net nil
   :num-batchs 0
   :train-loss-history []})

(def ^:dynamic *train-status*
  "train progress status"
  (atom (new-status)))

(defn init! []
  (reset! *train-status* (new-status)))

(defn- add-train-loss! [loss]
  (swap! *train-status* assoc :train-loss-history (conj (:train-loss-history @*train-status*) loss)))


(defmacro with-params
  "specified train parameters"
  [params-vec train-expr]
  (let [pm (apply hash-map params-vec)
        conf (merge *train-params* (dissoc pm :train-status-var))
        train-status-bind (if (:train-status-var pm) `(*train-status* ~(:train-status-var pm)))]
    `(binding [*train-params* ~conf
               ~@train-status-bind]
       ~train-expr)))



(defn gen-w-updater
  "generate updater for weights and biases"
  [mini-batch-size]
  (if-let [f (:updater *train-params*)]
    f
    (let [rate (/ (:learning-rate *train-params*) mini-batch-size)]
      (fn [w dw]
        (- w (* rate dw))))))



;;; backpropagation

(defn backprop
  "do backpropagation of 1 train-pair"
  [net in-vol answer-vol]
  (let [net-f (ly/forward net in-vol)]
    (ly/backward net-f answer-vol)))

(defn backprop-n
  "do backpropagation of multiple train-pairs"
  [net train-pairs]
  (let [merged (reduce ly/merge-p
                       (pmap (fn [[in-vol answer-vol]]
                              (backprop net in-vol answer-vol))
                            train-pairs))
        loss (/ (nw/loss merged) (count train-pairs))]
    (nw/update-loss merged loss)))


;;; train funcs

(defn update-mini-batch
  [net vols]
  (let [in-vol (apply vl/stack-rows (map first vols))
        answer-vol (apply vl/stack-rows (map second vols))]
    (ly/update-p (backprop net in-vol answer-vol)
                 (gen-w-updater (count vols)))))

(defn reduce-mini-batchs
  [init-net batchs]
  (reduce (fn [net b]
            (update-mini-batch net b))
          init-net
          batchs))

(defn sgd
  "Stochastic gradient descent"
  [net train-pairs]
  (let [batchs (partition (:mini-batch-size *train-params*) train-pairs)]
    (swap! *train-status* assoc :num-batchs (count batchs))
    (loop [epoch 0, cur net]
      (swap! *train-status* assoc :now-epoch epoch)
      (swap! *train-status* assoc :now-net cur)
      (add-train-loss! (nw/loss cur))
      (future ((:epoch-reporter *train-params*) epoch cur))
      (if (< epoch (:epoch-limit *train-params*))
        (recur (inc epoch) (reduce-mini-batchs cur batchs))
        cur))))
