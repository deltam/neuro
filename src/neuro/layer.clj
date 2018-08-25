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
  (update-p [this f] "update params")
  (merge-p [this other] "merge 2 layer"))

(defprotocol Compilable
  "Compile neural net to clojure code"
  (compile [this] "output source code"))


;; input layer

(defrecord Input [out out-vol]
  Executable
  (forward [this in-vol] (assoc this :out-vol in-vol))
  (backward [this grad-vol] this)
  (output [this] (:out-vol this))
  (grad [this] nil)
  Compilable
  (compile [this]
    `(fn [vs#] vs#)))

(defn input [in]
  (->Input in nil))



;; connection layer

(defn sym-args [len]
  (mapv #(gensym (str "x" %)) (range len)))

(defrecord FullConn [in out w bias in-vol out-vol dw dbias delta-vol]
  Executable
  (forward [this in-vol]
    (let [{w :w, bias :bias} this]
      (assoc this
             :in-vol in-vol
             :out-vol (vl/w+ (vl/dot w in-vol) bias))))
  (backward [this grad-vol]
    (assoc this
           :dw (vl/dot-v-Tv grad-vol (:in-vol this))
           :dbias grad-vol
           :delta-vol (vl/dot-Tv-v (:w this) grad-vol)))
  (output [this] (:out-vol this))
  (grad [this] (:delta-vol this))
  Optimizable
  (update-p [this f]
    (let [{w :w, dw :dw} this
          {b :bias, db :dbias} this]
      (assoc this
             :w (vl/map-w f w dw)
             :bias (vl/map-w f b db))))
  (merge-p [this other]
    (let [{dw1 :dw, dbias1 :dbias} this
          {dw2 :dw, dbias2 :dbias} other]
      (assoc this
             :dw (vl/w+ dw1 dw2)
             :dbias (vl/w+ dbias1 dbias2))))
  Compilable
  (compile [this]
    (let [arg (gensym "arg")
          wsym (map #(gensym (str "w" %)) (range (:out this)))
          in-v (gensym "in-v")]
      `(let [~@(mapcat (fn [sym out-idx]
                         `(~sym ~(conj (mapv #(vl/wget (:w this) % out-idx)
                                             (range (:in this)))
                                       (vl/wget (:bias this) 0 out-idx))))
                    wsym
                    (range (:out this)))]
         (fn [~arg]
           (let [~in-v (conj ~arg 1.0)]
             ~(mapv (fn [out-idx]
                      `(apply + (map * ~in-v ~(nth wsym out-idx))))
                    (range (:out this)))))))))

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
  (grad [this] (:delta-vol this))
  Compilable
  (compile [this]
    (let [arg (gensym "arg")]
      `(let [sig# (fn [x#] (/ 1.0 (+ 1.0 (Math/exp (- x#)))))]
         (fn [~arg]
           (mapv sig# ~arg))))))

(defn sigmoid
  [in]
  (->Sigmoid in nil nil))



(defn relu-f [x] (max x 0))
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



(defn tanh-f [x] (Math/tanh x))
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

(defn- clip
  "1e-10 - 1.0 の間に重みを正規化"
  [v]
  (let [wmax (vl/w-max v)
        wmin (apply min (:w v))]
    (vl/map-w #(/ (+ (- % wmin) 1e-10) wmax) v)))

(defn- cross-entropy
  "cross-entropy 誤差関数"
  [answer-vol out-vol]
  (- (vl/reduce-elm + (vl/map-w (fn [d y] (* d (Math/log y)))
                                answer-vol
                                (clip out-vol)))))

(defrecord Softmax [out out-vol delta-vol loss]
  Executable
  (forward [this in-vol]
    (let [wm (vl/w-max in-vol)
          es (vl/map-w #(Math/exp (- % wm)) in-vol)
          sum (vl/reduce-elm + es)]
      (assoc this
             :out-vol (vl/map-w #(/ % sum) es))))
  (backward [this answer-vol]
    (assoc this
           :delta-vol (vl/w- (:out-vol this) answer-vol)
           :loss (cross-entropy answer-vol (:out-vol this))))
  (output [this] (:out-vol this))
  (grad [this] (:delta-vol this))
  Optimizable
  (update-p [this f] this)
  (merge-p [this other]
    (assoc this
           :loss (+ (:loss this) (:loss other))))
  Compilable
  (compile [this]
    (let [arg (gensym "arg")]
      `(fn [~arg]
         (let [mn# (apply max ~arg)
               es# (map #(Math/exp (- % mn#)) ~arg)
               sm# (apply + es#)]
           (mapv #(/ % sm#) es#))))))

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
