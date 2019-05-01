(ns neuro.layer
  "Neural Network Layer"
  (:require [neuro.vol :as vl]))

(defprotocol Executable
  "Neural Network Layer"
  (forward [this in-vol] "feedfoward")
  (backward [this grad-vol] "use backprop")
  (output [this] "output by feedforward")
  (grad [this] "grad by backward"))

(defprotocol Optimizable
  "Neural Network Layer that has parameters to be aggregated"
  (update-p [this f] "update params"))



;; input layer

(defrecord Input [out out-vol]
  Executable
  (forward [this in-vol] (assoc this :out-vol in-vol))
  (backward [this grad-vol] this)
  (output [this] (:out-vol this))
  (grad [this] nil))

(defn input [in]
  (->Input in nil))



;; connection layer
(defrecord FullConn [in out w bias in-vol out-vol dw dbias delta-vol]
  Executable
  (forward [this in-vol]
    (let [{w :w, bias :bias} this]
      (assoc this
             :in-vol in-vol
             :out-vol (vl/w+ (vl/dot w in-vol) (vl/repeat-vol bias (:sx in-vol))))))
  (backward [this grad-vol]
    (assoc this
           :dw (vl/dot grad-vol (vl/T (:in-vol this)))
           :dbias (vl/sum-vol grad-vol)
           :delta-vol (vl/dot-Tv-v (:w this) grad-vol)))
  (output [this] (:out-vol this))
  (grad [this] (:delta-vol this))
  Optimizable
  (update-p [this f]
    (let [{w :w, dw :dw} this
          {b :bias, db :dbias} this]
      (assoc this
             :w (vl/map-w f w dw)
             :bias (vl/map-w f b db)))))

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

(defn- sigmoid-f [x] (/ 1.0 (+ 1.0 (Math/exp (- x)))))
(defn- sigmoid-df [y] (* y (- 1 y)))

(defrecord Sigmoid [out out-vol delta-vol]
  Executable
  (forward [this in-vol]
    (assoc this :out-vol (vl/map-w sigmoid-f in-vol)))
  (backward [this grad-vol]
    (let [y (:out-vol this)]
      (assoc this :delta-vol (vl/w* (vl/map-w sigmoid-df y) grad-vol))))
  (output [this] (:out-vol this))
  (grad [this] (:delta-vol this)))

(defn sigmoid
  [in]
  (->Sigmoid in nil nil))



(defn- relu-f [x] (max x 0))
(defn- relu-df [x] (if (< 0 x) 1 0))

(defrecord ReLU [out out-vol delta-vol]
  Executable
  (forward [this in-vol]
    (assoc this
           :out-vol (vl/map-w relu-f in-vol)))
  (backward [this grad-vol]
    (let [y (:out-vol this)]
      (assoc this :delta-vol (vl/w* (vl/map-w relu-df y) grad-vol))))
  (output [this] (:out-vol this))
  (grad [this] (:delta-vol this)))

(defn relu
  [in]
  (->ReLU in nil nil))



(defn- tanh-f [x] (Math/tanh x))
(defn- tanh-df [y] (- 1 (* y y)))

(defrecord Tanh [out out-vol delta-vol]
  Executable
  (forward [this in-vol]
    (assoc this :out-vol (vl/map-w tanh-f in-vol)))
  (backward [this grad-vol]
    (let [y (:out-vol this)]
      (assoc this :delta-vol (vl/w* (vl/map-w tanh-df y) grad-vol))))
  (output [this] (:out-vol this))
  (grad [this] (:delta-vol this)))

(defn tanh
  [in]
  (->Tanh in nil nil))





;; loss layer

(defn softmax-f [in-vol]
  (let [wm (vl/w-max in-vol)
        es (vl/map-w #(Math/exp (- % wm)) in-vol)
        sum (vl/reduce-elm + es)]
    (vl/map-w #(/ % sum) es)))

(defn softmax-f-n [in-vol]
  (vl/map-row softmax-f in-vol))

(defn- clip
  "1e-10 - 1.0 の間に重みを正規化"
  [v]
  (let [wmax (vl/w-max v)
        wmin (apply min (:w v))]
    (vl/map-w #(/ (+ (- % wmin) 1e-10) wmax) v)))

(defn cross-entropy
  "cross-entropy 誤差関数"
  [answer-vol out-vol]
  (- (vl/reduce-elm + (vl/map-w (fn [d y] (* d (Math/log y)))
                                answer-vol
                                (clip out-vol)))))

(defn cross-entropy-n
  [answer-vol out-vol]
  (/
   (apply + (map cross-entropy (vl/rows answer-vol) (vl/rows out-vol)))
   (:sx answer-vol)))

(defrecord Softmax [out out-vol delta-vol loss]
  Executable
  (forward [this in-vol]
    (assoc this
           :out-vol (softmax-f-n in-vol)))
  (backward [this answer-vol]
    (assoc this
           :delta-vol (vl/w- (:out-vol this) answer-vol)
           :loss (cross-entropy-n answer-vol (:out-vol this))))
  (output [this] (:out-vol this))
  (grad [this] (:delta-vol this))
  Optimizable
  (update-p [this f] this))

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
