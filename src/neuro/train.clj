(ns neuro.train
  (:require [neuro.core :as core]
            [neuro.network :as nw]
            [clojure.data.generators :as gr]))

(declare train-next)

(def ^:dynamic *weight-inc-val* 0.00001)
(def ^:dynamic *learning-rate* 0.00001)
(def ^:dynamic *weight-random-diff* 0.001)

(def ^:dynamic *report-period* 100)


(defn train [init-nn dfn w-updater dataset terminate-f]
  (loop [cur-nn init-nn, diff (dfn init-nn dataset) , cnt 0]
    (let [next (train-next cur-nn dfn w-updater dataset)
          next-diff (dfn next dataset)]
      (if (zero? (mod cnt *report-period*))
        (println cnt " now diff: " next-diff))
      (if (terminate-f diff next-diff)
        cur-nn
        (recur next, next-diff, (inc cnt))))))

(defn train-next [nn dfn w-updater dataset]
  (let [nn1 (w-updater nn dfn dataset)
        diff1 (dfn nn1 dataset)
        nn2 (w-updater nn dfn dataset)
        diff2 (dfn nn2 dataset)]
    (cond (<= diff1 diff2) nn1
          (>  diff1 diff2) nn2)))



(defn diff-fn-2class
  "2値分類の誤差関数 nn の出力層は1ニューロン"
  [nn dataset]
  (let [samples (count dataset)
        diff-sum (apply +
                        (for [{x :x, [d] :ans} dataset
                              :let [[y] (core/nn-calc nn x)]]
                          (+ (* d (Math/log y))
                             (* (- 1.0 d) (Math/log (- 1.0 y))))))]
    (/ (* -1 diff-sum) samples)))



;; 勾配降下法

(defn- gradient
  "nnの微小増分の傾きを返す"
  [nn dfn dataset level in out]
  (let [w (nw/weight nn level in out)
        nn-inc (nw/update-weight nn (+ w *weight-inc-val*) level in out)
        y (dfn nn dataset)
        y-inc (dfn nn-inc dataset)]
    (/ (- y-inc y) *weight-inc-val*)))

(defn- update-by-gradient
  "重みを勾配に従って更新した値を返す"
  [w nn dfn dataset level in out]
  (let [grd (gradient nn dfn dataset level in out)
        diff (dfn nn dataset)]
    (- w (* *learning-rate* diff))))

(defn weight-gradient
  "勾配降下法で重みを更新する"
  [nn dfn dataset]
  (nw/map-weights (fn [w l i o]
                    (update-by-gradient w nn dfn dataset l i o))
                  nn))



;; ランダム更新

(defn- rand-add [x]
  (+ x
     (gr/rand-nth [*weight-random-diff*
                   0
                   (* -1 *weight-random-diff*)])))

(defn weight-randomize
  "重みをランダムに更新する"
  [nn dfn dataset]
  (binding [gr/*rnd* (java.util.Random. (System/currentTimeMillis))]
    (nw/map-weights (fn [w l i o] (rand-add w))
                    nn)))


;(defn diff-fn-regression
;  "回帰解析用の誤差関数"
;  [nn dataset]
;  (* 0.5
;     (square-sum
;      (for [{x :x, ans :ans} dataset
;            :let [v-seq (core/nn-calc nn x)]]
;        (apply +
;               (map (fn [v d] (- v d)) v-seq ans))))))
;
;(defn- square [x]
;  (* x x))
;
;(defn- square-sum [x-seq]
;  (apply +
;         (map square x-seq)))
;


(comment

(def nn-2class (let [nn (nw/gen-nn :rand 3 4  1)]
                 ;; biasノードの初期値は0
                 (nw/map-weights (fn [w l i o]
                                   (if (or (and (= l 0) (= i 2))
                                           (and (= l 1) (= i 3)))
                                      0.0
                                     w))
                                 nn)))

(def traindata-2class [{:x [2 5 1] :ans [0]}
                       {:x [3 2 1] :ans [0]}
                       {:x [4 1 1] :ans [0]}
                       {:x [8 3 1] :ans [0]}
                       {:x [1 8 1] :ans [0]}
                       {:x [9 5 1] :ans [0]}
                       {:x [5 2 1] :ans [0]}
                       {:x [4 2 1] :ans [0]}
                       {:x [3 3 1] :ans [0]}
                       {:x [2 6 1] :ans [0]}
                       {:x [1 8 1] :ans [0]}
                       {:x [9 5 1] :ans [0]}


                       {:x [3 7 1] :ans [1]}
                       {:x [4 4 1] :ans [1]}
                       {:x [7 6 1] :ans [1]}
                       {:x [3 7 1] :ans [1]}
                       {:x [4 8 1] :ans [1]}
                       {:x [7 8 1] :ans [1]}
                       {:x [7 6 1] :ans [1]}
                       {:x [4 5 1] :ans [1]}
                       {:x [4 7 1] :ans [1]}
                       {:x [8 5 1] :ans [1]}
                       {:x [6 3 1] :ans [1]}
                       {:x [4 7 1] :ans [1]}
                       {:x [9 6 1] :ans [1]}
                       {:x [7 4 1] :ans [1]}
                       ])
(time
 (def train-nn-2class
   (train nn-2class diff-fn-2class weight-gradient traindata-2class
          #(< (Math/abs (- %1 %2)) 0.00001)))
 )

(time
 (def nn2
   (train train-nn-2class diff-fn-2class weight-randomize traindata-2class
          (fn [_ d] (< d 0.5333))))
)

(defn nn-test [nn dataset ans ans-test]
  (let [t (map (fn [x] (core/nn-calc nn2 x))
               (for [t traindata-2class :when (= (:ans t) ans)]
                 (:x t)))
        cnt (count t)]
    (/ (reduce (fn [r [w]] (if (ans-test w) (inc r) r)) 0.0 t)
       cnt)))

(nn-test nn2 traindata-2class [0] #(< % 0.5))
(nn-test nn2 traindata-2class [1] #(> % 0.5))

)
