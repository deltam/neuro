(ns neuro.layer
  (:require [neuro.vol :as vl]
            [neuro.func :as fnc]))

(defmulti forward :type)
(defmulti backward :type)
(defmulti update :type)


(defmethod update :default
  [this f] this)


;; input layer
(defn input-layer
  [in]
  {:type :input
   :out in})

(defmethod forward :input
  [this in-vol]
  (assoc this :in-vol in-vol))

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
    (assoc this :in-vol
           (vl/w-add (vl/w-mul w-vol in-vol)
                     bias-vol))))

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

(defmethod update :fc
  [this f]
  (let [w (:w this)
        dw (:dw this)
        bias (:bias this)
        dbias (:dbias this)]
    (-> this
        (assoc :w (vl/map-w f w dw))
        (assoc :bias (vl/map-w f bias dbias)))))






;; activation layer
(defn sigmoid-layer
  [in]
  {:type :sigmoid
   :out in})

(defmethod forward :sigmoid
  [this in-vol]
  (assoc this :in-vol
         (vl/map-w fnc/sigmoid in-vol)))

(defmethod backward :sigmoid
  [this back-vol]
  (assoc this :back-vol
         (vl/map-w fnc/d-sigmoid back-vol)))

;; ReLU
(defn relu-layer
  [in]
  {:type :relu
   :out in})

(defmethod forward :relu
  [this in-vol]
  (assoc this :in-vol
         (vl/map-w fnc/relu in-vol)))

(defmethod backward :relu
  [this back-vol]
  (assoc this :back-vol
         (vl/map-w fnc/d-relu back-vol)))


;; tanh
(defn tanh-layer
  [in]
  {:type :tanh
   :out in})

(defmethod forward :tanh
  [this in-vol]
  (assoc this :in-vol
         (vl/map-w fnc/tanh in-vol)))

(defmethod backward :tanh
  [this back-vol]
  (assoc this :back-vol
         (vl/map-w fnc/d-tanh back-vol)))




;; loss layer
(defn softmax-layer
  [in]
  {:type :softmax
   :out in})

(defmethod forward :softmax
  [this in-vol]
  (let [es (vl/map-w #(Math/exp %) in-vol)
        sum (vl/reduce-elm + es)]
    (assoc this :in-vol
           (vl/map-w #(/ % sum) es))))

(defmethod backward :softmax
  [this back-vol]
  (assoc this :back-vol back-vol)) ; 誤差関数
