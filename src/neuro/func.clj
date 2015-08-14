(ns neuro.func)

(defn dict [name]
  (condp = name
        :relu retified-linear-unit
        :logistic logistic-func))

(defn retified-linear-unit
  "正規化線形関数"
  [x]
  (max x 0))

(defn logistic-func
  "ロジスティック関数"
  [x]
  (/ 1.0 (+ 1.0 (Math/exp (- x)))))
