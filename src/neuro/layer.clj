(ns neuro.layer
  (:require [neuro.vol :as vl]
            [neuro.func :as fnc]))

(defmulti forward :type)
(defmulti backward :type)

(defmulti update :type)
(defmulti merge-w :type)
(defmulti map-w :type)


(defmethod update :default
  [this f] this)

(defmethod merge-w :default
  [this layer] this)

(defmethod map-w :default
  [this f] this)


;; input layer
(defn input-layer
  [in]
  {:type :input
   :out in})

(defmethod forward :input
  [this in-vol]
  (assoc this :out-vol in-vol))

(defmethod backward :input
  [this delta-vol]
  this)




;; connection layer
(defn fc-layer
  [in out]
  {:type :fc
   :in in, :out out
   :w (vl/vol in out)
   :bias (vl/vol 1 out (vl/zero-vec out))})

(defmethod forward :fc
  [this in-vol]
  (let [{w :w, bias :bias} this]
    (assoc this :out-vol
           (vl/w-add (vl/w-prod w in-vol) bias))))

(defmethod backward :fc
  [this delta-vol]
  (let [w-vol (:w this)
        prod-vol (vl/vol (:sx w-vol) (:sy w-vol)
                         (vec (flatten (repeat (:sx w-vol) (:w delta-vol)))))
        dw-vol (vl/w-prod-h w-vol prod-vol)]
    (assoc this
           :dw dw-vol
           :dbias delta-vol
           :delta-vol (vl/w-sum-row (vl/transposed dw-vol)))))

(defmethod update :fc
  [this f]
  (let [{w :w, dw :dw} this
        {b :bias, db :dbias} this]
    (assoc this
           :w (vl/map-w f w dw)
           :bias (vl/map-w f b db))))

(defmethod merge-w :fc
  [this layer]
  (let [w1 (:w this)
        bias1 (:bias this)
        w2 (:w layer)
        bias2 (:bias layer)]
    (assoc this
           :w (vl/map-w + w1 w2)
           :bias (vl/map-w + bias1 bias2))))

(defmethod map-w :fc
  [this f]
  (let [w (:w this)
        bias (:bias this)]
    (assoc this
           :w (vl/map-w f w)
           :bias (vl/map-w f bias))))





;; activation layer
(defn sigmoid-layer
  [in]
  {:type :sigmoid
   :out in})

(defmethod forward :sigmoid
  [this in-vol]
  (assoc this :out-vol
         (vl/map-w fnc/sigmoid in-vol)))

(defmethod backward :sigmoid
  [this delta-vol]
  (let [y (:out-vol this)]
    (assoc this :delta-vol
           (vl/w-prod-h
            (vl/map-w fnc/d-sigmoid y)
            delta-vol))))


(defn relu-layer
  [in]
  {:type :relu
   :out in})

(defmethod forward :relu
  [this in-vol]
  (assoc this :out-vol
         (vl/map-w fnc/relu in-vol)))

(defmethod backward :relu
  [this delta-vol]
  (let [y (:out-vol this)]
    (assoc this :delta-vol
           (vl/w-prod-h
            (vl/map-w fnc/d-relu y)
            delta-vol))))


(defn tanh-layer
  [in]
  {:type :tanh
   :out in})

(defmethod forward :tanh
  [this in-vol]
  (assoc this :out-vol
         (vl/map-w fnc/tanh in-vol)))

(defmethod backward :tanh
  [this delta-vol]
  (let [y (:out-vol this)]
    (assoc this :delta-vol
           (vl/w-prod-h
            (vl/map-w fnc/d-tanh y)
            delta-vol))))




;; loss layer
(defn softmax-layer
  [in]
  {:type :softmax
   :out in})

(defmethod forward :softmax
  [this in-vol]
  (let [wm (vl/w-max in-vol)
        es (vl/map-w #(Math/exp (- % wm)) in-vol)
        sum (vl/reduce-elm + es)]
    (assoc this :out-vol
           (vl/map-w #(/ % sum) es))))

(defn- cross-entropy
  "cross-entropy 誤差関数"
  [train-vol out-vol]
  (- (vl/reduce-elm + (vl/map-w (fn [d y] (* d (Math/log y)))
                                train-vol
                                out-vol))))

(defmethod backward :softmax
  [this train-vol]
  (let [loss (cross-entropy train-vol (:out-vol this))
        delta-vol (vl/map-w - (:out-vol this) train-vol)]
    (assoc this
           :delta-vol delta-vol
           :loss loss)))

(defmethod merge-w :softmax
  [this layer]
  (assoc this :loss
         (+ (:loss this) (:loss layer))))
