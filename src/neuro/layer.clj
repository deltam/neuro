(ns neuro.layer
  (:require [taoensso.tufte :as tufte :refer (p)])
  (:require [neuro.vol :as vl]
            [neuro.func :as fnc]))

(defprotocol Layer
  "Neural Network Layer"
  (forward [this in-vol] "feedfoward")
  (backward [this delta-vol] "use backprop")
  (update-w [this f] "update weights")
  (merge-w [this other] "merge-w 2 layer")
  (map-w [this f] "map f weights"))



;; input layer

(defrecord Input [out out-vol]
  Layer
  (forward [this in-vol] (assoc this :out-vol in-vol))
  (backward [this delta-vol] this)
  (update-w [this f] this)
  (merge-w [this other] this)
  (map-w [this f] this))

(defn input [in]
  (->Input in nil))



;; connection layer
(defrecord FullConn [in out w bias in-vol out-vol dw dbias delta-vol]
  Layer
  (forward [this in-vol]
    (p ::forward-fc
       (let [{w :w, bias :bias} this]
         (assoc this
                :in-vol in-vol
                :out-vol (vl/w+ (vl/dot w in-vol) bias)))))
  (backward [this grad-vol]
    (p ::backward-fc
       (assoc this
              :dw (vl/dot grad-vol (vl/T (:in-vol this)))
              :dbias grad-vol
              :delta-vol (vl/dot (vl/T (:w this)) grad-vol))))
  (update-w [this f]
    (p ::update-w-fc
       (let [{w :w, dw :dw} this
             {b :bias, db :dbias} this]
         (assoc this
                :w (vl/map-w f w dw)
                :bias (vl/map-w f b db)))))
  (merge-w [this other]
    (p ::merge-w-fc
       (let [w1 (:w this)
             bias1 (:bias this)
             w2 (:w other)
             bias2 (:bias other)]
         (assoc this
                :w (vl/w+ w1 w2)
                :bias (vl/w+ bias1 bias2)))))
  (map-w [this f]
    (p ::map-w-fc
       (let [w (:w this)
             bias (:bias this)]
         (assoc this
                :w (vl/map-w f w)
                :bias (vl/map-w f bias))))))

(defn fc
  [in out]
  (->FullConn in out
              (vl/vol in out)
              (vl/vol 1 out (vl/zero-vec out))
              nil
              nil
              nil
              nil
              nil))




;; activation layer

(defrecord Sigmoid [out out-vol delta-vol]
  Layer
  (forward [this in-vol]
    (p ::forward-sigmoid
       (assoc this :out-vol
              (vl/map-w fnc/sigmoid in-vol))))
  (backward [this delta-vol]
    (p ::backward-sigmoid
       (let [y (:out-vol this)]
         (assoc this :delta-vol
                (vl/w* (vl/map-w fnc/d-sigmoid y) delta-vol)))))
  (update-w [this f] this)
  (merge-w [this other] this)
  (map-w [this f] this))

(defn sigmoid
  [in]
  (->Sigmoid in nil nil))



(defrecord ReLU [out out-vol delta-vol]
  Layer
  (forward [this in-vol]
    (assoc this :out-vol
           (vl/map-w fnc/relu in-vol)))
  (backward [this delta-vol]
    (let [y (:out-vol this)]
      (assoc this :delta-vol
             (vl/w* (vl/map-w fnc/d-relu y) delta-vol))))
  (update-w [this f] this)
  (merge-w [this other] this)
  (map-w [this f] this))

(defn relu
  [in]
  (->ReLU in nil nil))



(defrecord Tanh [out out-vol delta-vol]
  Layer
  (forward [this in-vol]
    (assoc this :out-vol
           (vl/map-w fnc/tanh in-vol)))
  (backward [this delta-vol]
    (let [y (:out-vol this)]
      (assoc this :delta-vol
             (vl/w* (vl/map-w fnc/d-tanh y) delta-vol))))
  (update-w [this f] this)
  (merge-w [this other] this)
  (map-w [this f] this))

(defn tanh
  [in]
  (->Tanh in nil nil))





;; loss layer

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

(defrecord Softmax [out out-vol delta-vol loss]
  Layer
  (forward [this in-vol]
    (p ::forward-softmax
       (let [wm (vl/w-max in-vol)
             es (vl/map-w #(Math/exp (- % wm)) in-vol)
             sum (vl/reduce-elm + es)]
         (assoc this :out-vol
                (vl/map-w #(/ % sum) es)))))
  (backward [this answer-vol]
    (p ::backward-softmax
       (assoc this
              :delta-vol (vl/w- (:out-vol this) answer-vol)
              :loss (cross-entropy answer-vol (:out-vol this)))))
  (update-w [this f] this)
  (merge-w [this other]
    (p ::merge-w-softmax
       (assoc this :loss
              (+ (:loss this) (:loss other)))))
  (map-w [this f] this))

(defn softmax
  [in]
  (->Softmax in nil nil nil))




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
