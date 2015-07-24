(ns neuro.train
  (:require [neuro.func :as fnc])


(defn rand-add [x]
  (+ x
     (rand-nth [0.01 0 -0.01])))

(defn weight-randomize [w-matrix]
  (apply vector
         (for [w-seq w-matrix]
           (apply vector (map #(rand-add %) w-seq)))))

(defn bias-randomize [bias-vec]
  (apply vector (map #(rand-add %) bias-vec)))

(defn next-params [params]
  (let [p (assoc params :weight (weight-randomize (:weight params)))]
    (assoc p :bias (bias-randomize (:bias params)))))

(defn train-next [dataset first-params dfn]
  (let [param1 (next-params first-params)
        diff1 (dfn dataset param1)
        param2 (next-params first-params)
        diff2 (dfn dataset param2)]
    (cond (< diff1 diff2) param1
          (> diff1 diff2) param2
          :else first-params)))

(defn train [dataset init-params limit dfn]
  (loop [params init-params]
    (if (< (dfn dataset params) limit)
      params
      (recur (train-next dataset params dfn)))))





(comment

(def testdata [{:x [1 2] :ans [1]}
               {:x [2 3] :ans [0]}
               {:x [1 4] :ans [1]}
               {:x [2 8] :ans [0]}
               {:x [1 9] :ans [1]}
               ])
(def params {:weight [[0.2 0.7]]
;                      [0.1 0.9]]
             :bias [0.1 0.1]
             :fn fnc/logistic-func})

)
