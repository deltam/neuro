(ns mnist.core
;  (:require [taoensso.tufte :refer [p]])
  (:require [mnist.data :as md]
            [neuro.core :as nc]
            [neuro.train :as nt]
            [neuro.vol :as vl]
            [clojure.java.shell :refer [sh]]))

(def net
  (nc/gen-net
   :input 784 :fc
   :sigmoid 30 :fc
   :softmax 10))

(def train-size 60000)

(def my-data (let [[img lbl] (md/load-train)
                   perm (vl/gen-perm img)
                   imgr (vl/shuffle img perm)
                   lblr (vl/shuffle lbl perm)]
               [imgr lblr]))
(def test-data (map #(vl/slice % 0 1000) my-data))
(def train-data (map #(vl/slice % 1000 train-size) my-data))

(defn evaluate [net [img-vol label-vol]]
  (let [done (nc/feedforward net img-vol)
        check (pmap (fn [done-vol lbl-vol]
                      (= (md/vol->digit done-vol)
                         (md/vol->digit lbl-vol)))
                    (vl/rows done)
                    (vl/rows label-vol))]
    (count (filter true? check))))

(def ^:private start-time-now-epoch (atom 0))

(def test-error-rates (atom []))

(defn report [ep net]
  (let [ok (evaluate net test-data)
        [n _] (vl/shape (first test-data))
        elapsed (- (System/currentTimeMillis) @start-time-now-epoch)]
    (printf "\nepoch %d: %d / %d  (%4.2f min)\n" ep ok n (float (/ elapsed 60000.0)))
    (flush)
    (swap! test-error-rates conj (- 1.0 (/ (float ok) (float n))))
    (reset! start-time-now-epoch (System/currentTimeMillis))
;    (sh "say" (str "epoc " ep " " (float (* 100 (/ ok n))))) ; for mac user
    ))

(defn mini-batch-report [net loss]
  (print "."))


(def mnist-train-status (atom nil))

(defn train
  ([] (train net))
  ([net]
   (reset! start-time-now-epoch (System/currentTimeMillis))
   (reset! test-error-rates [])
   (reset! mnist-train-status (nt/new-status))
   (nc/with-params [:train-status-var mnist-train-status
                    :mini-batch-size 20
                    :epoch-limit 30
                    :learning-rate 1.0
                    :epoch-reporter report
                    :mini-batch-reporter mini-batch-report]
     (let [[img-vol lbl-vol] train-data]
       (nc/sgd net img-vol lbl-vol)))))


(defn print-time-to-next-epoch []
  (let [{nb :num-batchs, tlh :train-loss-history} @mnist-train-status
        done-batchs (mod (count tlh) nb)
        now-elapsed (- (System/currentTimeMillis) @start-time-now-epoch)
        msec-per-batch (if (zero? done-batchs)
                         now-elapsed
                         (/ now-elapsed done-batchs))
        until-msec (* (- nb done-batchs) msec-per-batch)]
    (printf "start next epoch at %4.2f min later\n"
            (float (/ until-msec 60000)))))
