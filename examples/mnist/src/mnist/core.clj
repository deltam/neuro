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

(defn report [ep net]
  (let [ok (evaluate net test-data)
        n (count test-data)]
    (printf "epoch %d: %d / %d\n" ep ok n)
    (flush)
;    (sh "say" (str "epoc " ep " " (float (* 100 (/ ok n))))) ; for mac user
    ))

(defn train
  ([] (train net))
  ([net]
   (nt/init)
   (nt/sgd net
           train-data
           :mini-batch-size 10
           :epoch-limit 30
           :learning-rate 3.0
           :epoch-reporter report)))
