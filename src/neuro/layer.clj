(ns neuro.layer
;  (:require [taoensso.tufte :as tufte :refer (p)])
  (:require [neuro.vol :as vl]
            [neuro.func :as fnc]))

(defmulti forward :type)
(defmulti backward :type)

(defmulti update-w :type)
(defmulti merge-w :type)
(defmulti map-w :type)


(defmethod update-w :default
  [this f] this)

(defmethod merge-w :default
  [this layer] this)

(defmethod map-w :default
  [this f] this)


;; input layer
(defn input
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
(defn fc
  [in out]
  {:type :fc
   :in in, :out out
   :w (vl/vol in out)
   :bias (vl/vol 1 out (vl/zero-vec out))})

(defmethod forward :fc
  [this in-vol]
  (let [{w :w, bias :bias} this]
    (assoc this
           :in-vol in-vol
           :out-vol (vl/w+ (vl/dot w in-vol) bias))))

(defmethod backward :fc
  [this grad-vol]
  (assoc this
         :dw (vl/dot grad-vol (vl/T (:in-vol this)))
         :dbias grad-vol
         :delta-vol (vl/dot (vl/T (:w this)) grad-vol)))

(defmethod update-w :fc
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
(defn sigmoid
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
           (vl/w* (vl/map-w fnc/d-sigmoid y) delta-vol))))


(defn relu
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
           (vl/w* (vl/map-w fnc/d-relu y) delta-vol))))


(defn tanh
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
           (vl/w* (vl/map-w fnc/d-tanh y) delta-vol))))




;; loss layer
(defn softmax
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

(defn- normalize
  "1e-10 - 1.0 の間に重みを正規化"
  [v]
  (let [wmax (vl/w-max v)
        wmin (apply min (:w v))]
    (vl/map-w #(/ (+ (- % wmin) 1e-10) wmax) v)))

(defn- cross-entropy
  "cross-entropy 誤差関数"
  [train-vol out-vol]
  (- (vl/reduce-elm + (vl/map-w (fn [d y] (* d (Math/log y)))
                                train-vol
                                (normalize out-vol)))))

(defmethod backward :softmax
  [this answer-vol]
  (assoc this
         :delta-vol (vl/w- (:out-vol this) answer-vol)
         :loss (cross-entropy answer-vol (:out-vol this))))

(defmethod merge-w :softmax
  [this layer]
  (assoc this :loss
         (+ (:loss this) (:loss layer))))


;; generator
(defn gen [l & params]
  (if-let [genf (l
                 {:input input
                  :fc fc
                  :sigmoid sigmoid
                  :relu relu
                  :tanh tanh
                  :softmax softmax})]
    (apply genf params)))
