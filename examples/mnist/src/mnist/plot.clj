(ns mnist.plot
  "plot lein-grilla train status"
  (:require [gorilla-plot.core :as plot]
            [gorilla-repl.image :as image]
            [neuro.train :as nt]
            [neuro.network :as nw]
            [mnist.data :as md]))


(defn plot-train [& opts]
  (printf "epoch %d, last loss %f\n" @nt/+now-epoch+ (last @nt/+train-loss-history+))
  (apply plot/list-plot
         (take-last 100 @nt/+train-loss-history+)
         (concat [:joined true] opts)))

(def ^:private test-data (take 1000 (md/load-train)))

(defn rand-check-view [& opts]
  (let [[img-vol digit-vol] (rand-nth test-data)
        digit (md/vol->digit digit-vol)
        result (md/vol->digit (nw/feedforward @nt/+now-net+ img-vol))]
    (if (= digit result)
      (println "Succeed!" digit)
      (println "Failed!  label" digit "!=" "result" result))
    (apply image/image-view
           (md/vol->image img-vol)
           (concat [:width 150] opts))))
