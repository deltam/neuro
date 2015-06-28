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
      (for [data dataset
            :let [multi (multi-calc (:x data) params)
                  [d x] (first (map vector multi (:test data)))]]
        (- x d)))))

(defn rand-add [x]
  (+ x
     (rand-nth [0.01 0 -0.01])))

(defn weight-randomize [w-matrix]
  (apply vector
         (for [w-seq w-matrix]
           (apply vector (map #(rand-add %) w-seq)))))

(defn next-params [params]
  (assoc params :weight (weight-randomize (:weight params))))

(defn train-next [dataset first-params]
  (let [param1 (next-params first-params)
        diff1 (diff-fn dataset param1)
        param2 (next-params first-params)
        diff2 (diff-fn dataset param2)]
    (cond (< diff1 diff2) param1
          (> diff1 diff2) param2
          :else first-params)))

(defn train [dataset init-params limit]
  (loop [params init-params]
    (if (< (diff-fn dataset params) limit)
      params
      (recur (train-next dataset params)))))






(def testdata [{:x [1 2] :test [1]}
               {:x [2 3] :test [0]}
               {:x [1 4] :test [1]}
               {:x [2 8] :test [0]}
               {:x [1 9] :test [1]}
               ])
(def params {:weight [[0.2 0.7]]
;                      [0.1 0.9]]
             :bias [0.1 0.1]
             :fn logistic-func})
