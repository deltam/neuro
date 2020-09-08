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

(def all-train-data (let [[img lbl] (md/load-train)
                          perm (vl/gen-perm img)
                          imgr (vl/shuffle img perm)
                          lblr (vl/shuffle lbl perm)]
                      [imgr lblr]))
(def all-train-data-size (let [img (first all-train-data)]
                           (first (vl/shape img))))

(defn slice-all-train-data [s e]
  (map #(vl/slice % s e) all-train-data))


(def test-size 1000)
(def train-size (- all-train-data-size test-size))

(def test-data (slice-all-train-data 0 test-size))
(def train-data (slice-all-train-data test-size (+ test-size train-size)))

(defn evaluate [net [img-vol label-vol]]
  (let [done (nc/feedforward net img-vol)
        check (pmap (fn [done-vol lbl-vol]
                      (= (md/vol->digit done-vol)
                         (md/vol->digit lbl-vol)))
                    (vl/rows done)
                    (vl/rows label-vol))]
    (count (filter true? check))))

(def ^:private start-time-now-epoch (atom 0))
(defn init-start-time-now-epoch! [] (reset! start-time-now-epoch (System/currentTimeMillis)))

(def test-error-rates (atom []))

(defn report [test-data {net :model, ep :epoch}]
  (let [ok (evaluate net test-data)
        [n _] (vl/shape (first test-data))
        elapsed (- (System/currentTimeMillis) @start-time-now-epoch)]
    (printf "epoch %d: %d / %d  (%4.2f min)\n" ep ok n (float (/ elapsed 60000.0)))
    (flush)
    (swap! test-error-rates conj (- 1.0 (/ (float ok) (float n))))
    (init-start-time-now-epoch!)
;    (sh "say" (str "epoc " ep " " (float (* 100 (/ ok n))))) ; for mac user
    ))

(defn mini-batch-report [_]
  (print ".")
  (flush))


(defn train
  ([] (train net))
  ([net] (train net train-data test-data))
  ([net train-data test-data]
   (init-start-time-now-epoch!)
   (reset! test-error-rates [])
   (let [learning-rate 1.0
         limit         10
         size          20
         batchs (nt/split-mini-batch train-data size)
         f (nt/iterate-mini-batch-train-fn net batchs)]
     (->> (f (nt/gen-sgd-optimizer learning-rate))
          (nt/with-epoch-report (partial report test-data))
          (drop-while #(< (:epoch %) limit))
          (first)))))



(comment
  (def train-seq-fn
    (let [batchs (neuro.train/split-mini-batch train-data 20)]
      (neuro.train/iterate-mini-batch-train-fn net batchs)))

  (init-start-time-now-epoch!)
  (def result
    (->> (train-seq-fn (neuro.train/gen-sgd-optimizer 1.0))
         (neuro.train/with-epoch-report (partial report test-data))
         (drop-while #(< (:epoch %) 10))
         (first)
         (:loss)))
)
