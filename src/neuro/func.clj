(ns neuro.func)

(defn retified-linear-func
  "正規化線形関数"
  [x]
  (max x 0))

(defn logistic-func
  "ロジスティック関数"
  [x]
  (/ 1.0 (+ 1.0 (Math/exp (- x)))))

(defn calc [x-seq weight-seq bias activation-func]
  (let [pair (map vector x-seq weight-seq)
        u (+ (apply + (map (fn [[x w]] (* x w)) pair)) bias)]
    (activation-func u)))

(defn multi-calc [x-seq {weight-matrix :weight, bias-seq :bias, activation-func :fn}]
  (for [[w-seq b] (map vector weight-matrix bias-seq)]
    (calc x-seq w-seq b activation-func)))


(defn square [x]
  (* x x))

(defn square-sum [x-seq]
  (apply +
         (map square x-seq)))

(defn diff-fn [dataset params]
  (* 0.5
     (square-sum
      (for [{x :x, ans :ans} dataset
            :let [multi (multi-calc x params)
                  [x d] (first (map vector multi ans))]]
        (- x d)))))

(defn diff-fn-2class
  "2値分類の誤差関数"
  [dataset params]
  (* -1.0
     (apply +
            (for [{x :x, a :ans} dataset
                  :let [m (multi-calc x params)
                        [y d] (first (map vector m a))]]
              (+ (* d (Math/log y))
                 (* (- 1.0 d) (Math/log (- 1.0 y))))))))
