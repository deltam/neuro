(ns neuro.func)

(defn retified-linear-unit
  "正規化線形関数"
  [x]
  (max x 0))

(defn i-relu
  "正規化線形関数の逆関数"
  [y]
  (if (< y 0)
    0
    y))

(defn logistic-func
  "ロジスティック関数"
  [x]
  (/ 1.0 (+ 1.0 (Math/exp (- x)))))

(defn logit
  "ロジット関数、ロジスティック関数の逆関数"
  [y]
  (Math/log (/ y (- 1 y))))

(defn tanh
  "双曲線正接関数"
  [x]
  (Math/tanh x))

(defn i-tanh
  "逆双曲線正接関数"
  [y]
  (* 0.5 (Math/log (/ (+ 1 y)
                      (- 1 y)))))


(defn dict
  "関数本体を返す"
  [name]
  (condp = name
        :relu retified-linear-unit
        :logistic logistic-func
        :tanh tanh))

(defn idict
  "逆関数を返す"
  [name]
  (condp = name
    :relu i-relu
    :logistic logit
    :tanh i-tanh))
