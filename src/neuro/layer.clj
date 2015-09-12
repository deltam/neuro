(ns neuro.layer
  (:require [neuro.vol :as vl]
            [neuro.func :as fnc]))

(defn forward
  [layer in-vol]
  ((:forward layer) layer in-vol))

(defn backward
  [layer back-vol]
  ((:backward layer) layer back-vol))

;; input layer
(defn input-layer
  [in]
  {:type :input
   :out in
   :forward (fn [this in-vol] in-vol)
   :backward (fn [this back-vol] back-vol)})


;; full connection layer
(defn forward-fc [layer in-vol]
  (let [w-vol (:w layer)
        bias-vol (:bias layer)]
    (vl/w-add (vl/w-mul w-vol in-vol)
              bias-vol)))

(defn backward-fc [layer in-vol]
  in-vol)

(defn fc-layer
  [in out]
  {:type :fc
   :in in, :out out
   :w (vl/vol in out)
   :bias (vl/vol 1 out)
   :forward forward-fc
   :backward backward-fc})



;; activation layer
(defn sigmoid-layer
  [in]
  {:type :sigmoid
   :out in
   :forward (fn [this in-vol] (vl/map-w fnc/sigmoid in-vol))
   :backward nil})




;; loss layer
(defn loss-layer
  [in]
  {:type :loss
   :out in
   :forward (fn [this in-vol] in-vol)
   :backward nil ; 誤差関数
})



;; network
(defn network [& layers]
  {:layer layers
   :forward (fn [this in-vol]
              (reduce (fn [r l] (forward l r))
                      in-vol (:layer this)))
   :backward (fn [this back-vol] nil)})


(comment

(def nn
  [{:type :input
    :out 2}
   {:type :fc
    :out 6
    :w [[0 0 0 0 0 0] ; bias
        [1 1 1 1 1 1]
        [1 2 3 4 5 6]]}
   {:type :sigmoid
    :in 6 :out 6}
   {:type :fc
    :in 6 :out 2
    :w [[0 0] ;bias
        [1 2]
        [1 2]
        [1 2]
        [1 2]
        [1 2]
        [1 2]]}
   {:type :sigmoid
    :in 2 :out 2}
   {:type :fc
    :in 2 :out 1
    :w [[0]
        [1]]}])
)
