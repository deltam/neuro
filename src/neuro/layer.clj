(ns neuro.layer
  (:require [neuro.vol :as vl]
            [neuro.func :as fnc]))

(defmulti forward :type)
(defmulti backward :type)


;; input layer
(defn input-layer
  [in]
  {:type :input
   :out in})

(defmethod forward :input
  [this in-vol] in-vol)

(defmethod backward :input
  [this back-vol]
  (assoc this :back-vol back-vol))


;; connection layer
(defn fc-layer
  [in out]
  {:type :fc
   :in in, :out out
   :w (vl/vol in out)
   :bias (vl/vol 1 out)})

(defmethod forward :fc
  [this in-vol]
  (let [w-vol (:w this)
        bias-vol (:bias this)]
    (vl/w-add (vl/w-mul w-vol in-vol)
              bias-vol)))

(defmethod backward :fc
  [this back-vol]
  (let [w-vol (:w this)
        prod-vol (vl/vol (:sx w-vol) (:sy w-vol)
                         (vec (flatten (repeat (:sx w-vol) (:w back-vol)))))
        dw-vol (vl/w-mul-h w-vol prod-vol)
        dbias-vol (vl/w-mul-h (:bias this) back-vol)]
    (-> this
        (assoc :dw dw-vol)
        (assoc :dbias dbias-vol)
        (assoc :back-vol (vl/w-sum-row (vl/transposed dw-vol))))))




;; activation layer
(defn sigmoid-layer
  [in]
  {:type :sigmoid
   :out in})

(defmethod forward :sigmoid
  [this in-vol]
  (vl/map-w fnc/sigmoid in-vol))

(defmethod backward :sigmoid
  [this back-vol]
  (assoc this :back-vol
         (vl/map-w fnc/d-sigmoid back-vol)))




;; loss layer
(defn loss-layer
  [in]
  {:type :loss
   :out in})

(defmethod forward :loss
  [this in-vol]
  in-vol)

(defmethod backward :loss
  [this back-vol]
  (assoc this :back-vol back-vol)) ; 誤差関数


;; output layer
(defn output-layer
  [out]
  {:type :output
   :out out})

(defmethod forward :output [this in-vol] in-vol)
(defmethod backward :output [this back-vol] (assoc this :back-vol back-vol))



;; network
(defn network [& layers]
  {:type :network
   :layer layers})

(defmethod forward :network
  [this in-vol]
  (apply network
         (let [max (count (:layer this))]
           (loop [idx 0, done [], v in-vol]
             (if (< idx max)
               (let [cur (nth (:layer this) idx)
                     v2 (forward cur v)]
                 (recur (inc idx) (conj done (assoc cur :in-vol v)) v2))
               done)))))

(defmethod backward :network
  [this back-vol]
  (apply network
         (loop [idx (dec (count (:layer this))), done [], v back-vol]
           (if (< idx 0)
             (reverse done)
             (let [cur (nth (:layer this) idx)
                   next (backward cur v)]
               (recur (dec idx) (conj done next) (:back-vol next)))))))


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
