(ns neuro.network)

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






(def testdata [{:x [1 2] :ans [1]}
               {:x [2 3] :ans [0]}
               {:x [1 4] :ans [1]}
               {:x [2 8] :ans [0]}
               {:x [1 9] :ans [1]}
               ])
(def params {:weight [[0.2 0.7]]
;                      [0.1 0.9]]
             :bias [0.1 0.1]
             :fn logistic-func})
