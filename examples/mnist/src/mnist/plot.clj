(ns mnist.plot
  "plot lein-grilla train status"
  (:require [gorilla-plot.core :as plot]
            [gorilla-repl.image :as image]
            [neuro.train :as nt]
            [neuro.network :as nw]
            [mnist.data :as md]
            [mnist.core :as core]))


(defn plot-train [& opts]
  (let [{ep :now-epoch, tlh :train-loss-history} @core/*mnist-train-status*]
    (printf "epoch %d, last loss %f\n" ep (last tlh))
    (apply plot/list-plot
           (take-last 100 tlh)
           (concat [:joined true] opts))))

(def ^:private test-data (take 1000 (md/load-train)))

(defn rand-check-view [& opts]
  (let [[img-vol digit-vol] (rand-nth test-data)
        digit (md/vol->digit digit-vol)
        result (md/vol->digit (nw/feedforward (:now-net @core/*mnist-train-status*) img-vol))]
    (if (= digit result)
      (println "Succeed!" digit)
      (println "Failed!  label" digit "!=" "result" result))
    (apply image/image-view
           (md/vol->image img-vol)
           (concat [:width 150] opts))))
