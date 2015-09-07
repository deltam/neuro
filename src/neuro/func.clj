(ns neuro.func)

(defn relu
  "正規化線形関数"
  [x]
  (max x 0))

(defn i-relu
  "正規化線形関数の逆関数"
  [y]
  (if (< y 0)
    0
    y))

(defn d-relu
  [x]
  (if (< 0 x) 1 0))

(defn sigmoid
  "シグモイド関数"
  [x]
  (/ 1.0 (+ 1.0 (Math/exp (- x)))))

(defn d-sigmoid
  "シグモイド関数の微分"
  [x]
  (* (sigmoid x)
     (- 1 (sigmoid x))))

(defn logit
  "ロジット関数、シグモイド関数の逆関数"
  [y]
  (Math/log (/ y (- 1 y))))

(defn tanh
  "双曲線正接関数"
  [x]
  (Math/tanh x))

(defn d-tanh
  "tanhの微分"
  [x]
  (- 1 (* (tanh x) (tanh x))))

(defn i-tanh
  "逆双曲線正接関数"
  [y]
  (* 0.5 (Math/log (/ (+ 1 y)
                      (- 1 y)))))


(defn dict
  "関数本体を返す"
  [name]
  (condp = name
        :relu relu
        :sigmoid sigmoid
        :tanh tanh))

(defn d-dict
  "関数の微分を返す"
  [name]
  (condp = name
        :relu d-relu
        :sigmoid d-sigmoid
        :tanh d-tanh))

(defn i-dict
  "逆関数を返す"
  [name]
  (condp = name
    :relu i-relu
    :sigmoid logit
    :tanh i-tanh))
