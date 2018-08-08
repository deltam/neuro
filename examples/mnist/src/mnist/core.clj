(ns mnist.core
  (:require [mnist.data :as md]
            [neuro.layer :as ly]
            [neuro.network :as nw]
            [neuro.train :as nt]
            [clojure.java.shell :refer (sh)]))

(def net
  (nw/gen-net
   :input 784 :fc
   :sigmoid 30 :fc
   :softmax 10))

(def my-data (shuffle (md/load-train)))
(def test-data (take 10000 my-data))
(def train-data (drop 10000 my-data))

(defn evaluate [net test-pairs]
  (count
   (filter (fn [[img-vol label-vol]]
             (= (md/vol->digit label-vol)
                (md/vol->digit (nw/feedforward net img-vol))))
           test-pairs)))

(def start-time-now-epoch (atom 0))

(defn report [ep net]
  (let [ok (evaluate net test-data)
        n (count test-data)
        elapsed (- (System/currentTimeMillis) @start-time-now-epoch)]
    (printf "epoch %d: %d / %d  (%4.2f min)\n" ep ok n (float (/ elapsed 60000.0)))
    (flush)
    (reset! start-time-now-epoch (System/currentTimeMillis))
    (sh "say" (str "epoc " ep " " (float (* 100 (/ ok n))))) ; for mac user
    ))


(defn train
  ([] (train net))
  ([net]
   (reset! start-time-now-epoch (System/currentTimeMillis))
   (nt/init)
   (nt/with-params [:mini-batch-size 20
                    :epoch-limit 30
                    :learning-rate 1.0
                    :epoch-reporter report]
     (nt/sgd net train-data))))


(defn print-time-to-next-epoch []
  (let [done-batchs (mod (count @nt/+train-loss-history+) @nt/+num-batchs+)
        now-elapsed (- (System/currentTimeMillis) @start-time-now-epoch)
        msec-per-batch (if (zero? done-batchs)
                         now-elapsed
                         (/ now-elapsed done-batchs))
        until-msec (* (- @nt/+num-batchs+ done-batchs) msec-per-batch)]
    (printf "start next epoch at %4.2f min later\n"
            (float (/ until-msec 60000)))))
