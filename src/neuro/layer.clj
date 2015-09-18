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
   :bias (vl/vol 1 out (vl/zero-vec out))})

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
           (loop [i 0, done [], v in-vol]
             (if (< i max)
               (let [cur (nth (:layer this) i)
                     v2 (forward cur v)]
                 (recur (inc i) (conj done (assoc cur :in-vol v)) v2))
               done)))))

(defmethod backward :network
  [this back-vol]
  (apply network
         (loop [i (dec (count (:layer this))), done [], v back-vol]
           (if (< i 0)
             (reverse done)
             (let [cur (nth (:layer this) i)
                   next (backward cur v)]
               (recur (dec i) (conj done next) (:back-vol next)))))))
