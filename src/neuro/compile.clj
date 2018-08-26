(ns neuro.compile
  "compile nn"
  (:require [neuro.vol :as vl]))


(defprotocol Compilable
  "Compile neural net to clojure code"
  (compile [this] "output source code"))

(defn sym-args [len]
  (mapv #(gensym (str "x" %)) (range len)))

(extend-protocol Compilable
  neuro.layer.Input
  (compile [this]
    `(fn [vs#] vs#))

  neuro.layer.FullConn
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
                    (range (:out this))))))))

  neuro.layer.Sigmoid
  (compile [this]
    (let [arg (gensym "arg")]
      `(let [sig# (fn [x#] (/ 1.0 (+ 1.0 (Math/exp (- x#)))))]
         (fn [~arg]
           (mapv sig# ~arg)))))

  neuro.layer.Softmax
  (compile [this]
    (let [arg (gensym "arg")]
      `(fn [~arg]
         (let [mn# (apply max ~arg)
               es# (map #(Math/exp (- % mn#)) ~arg)
               sm# (apply + es#)]
           (mapv #(/ % sm#) es#)))))

  neuro.network.Network
  (compile [this]
    (if (every? #(satisfies? Compilable %) (:layer this))
      (let [arg (gensym "arg")]
        `(let [fs# ~(mapv compile (:layer this))]
           (fn [~arg]
             (reduce (fn [inv# f#] (f# inv#))
                     ~arg
                     fs#)))))))
