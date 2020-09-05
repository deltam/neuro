(ns neuro.train
;  (:require [taoensso.tufte :refer [p]])
  (:require [neuro.layer :as ly]
            [neuro.network :as nw]
            [neuro.vol :as vl]))


;;; backpropagation

(defn backprop
  "do backpropagation"
  [net in-vol answer-vol]
  (let [net-f (ly/forward net in-vol)]
    (ly/backward net-f answer-vol)))

(defn split-mini-batch
  ([in-vol ans-vol size]
   (map vector
        (vl/partition in-vol size)
        (vl/partition ans-vol size)))
  ([[iv av] size] (split-mini-batch iv av size)))

(defn mini-batch-updater [optimizer net [in-vol answer-vol]]
  (let [backed (backprop net in-vol answer-vol)]
    (ly/update-p backed optimizer)))

(defn gen-sgd-optimizer [lr]
  (fn [w dw] (- w (* lr dw))))



;;; train

(defn iterate-train-fn [model train-data-seq]
  (fn [trainer]
    (->> (iterate (fn [{m :model, [td & r] :rest, i :index, tl :total-loss}]
                    (let [m2 (trainer m td)]
                      {:model m2
                       :index (inc i)
                       :loss (nw/loss m2)
                       :total-loss (cons (nw/loss m2) tl)
                       :rest r}))
                  {:model model
                   :index -1
                   :total-loss '()
                   :rest train-data-seq})
         (map #(dissoc % :rest))
         (rest))))

(defn iterate-mini-batch-train-fn [model mini-batchs]
  (let [size (count mini-batchs)]
    (fn [optimizer]
      (->> ((iterate-train-fn model (cycle mini-batchs)) (partial mini-batch-updater optimizer))
           (map #(assoc %
                        :epoch (int (/ (:index %) size))
                        :mini-batch (mod (:index %) size)))))))



;;; reporting

(defn wrap-prepare [f s]
  (map #(do (f %)
            %)
       s))

(defn with-report
  ([step rf s]
   (wrap-prepare #(when (zero? (mod (:index %) step))
                    (rf %))
                 s))
  ([rf s] (with-report 1 rf s)))

(defn with-epoch-report [f s]
  (wrap-prepare #(when (and (< 0 (:index %))
                            (zero? (:mini-batch %)))
                   (f %))
                s))

(defn accuracy [b mini-batch-size]
  (float
   (/ (apply + (take mini-batch-size (:total-loss b)))
      mini-batch-size)))




(comment

  (def train-seq-fn
    (let [batchs (nt/split-mini-batch train-data 20)]
      (nt/iterate-mini-batch-train-fn net batchs)))

  (def train-result
    (->> (train-seq-fn (nt/gen-sgd-optimizer 1.0))
         (nt/with-report 10 #(printf "%d - %02d: loss %f\n" (:epoch %) (:mini-batch %) (:loss %)))
         (drop-while #(< (:epoch %) 10))
         (first)))

)
