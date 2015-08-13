(ns neuro.train
  (:require [neuro.core :as core]
            [neuro.network :as nw]
            [clojure.data.generators :as gr]))


(def ^:dynamic *weight-inc-val* 0.00001)
(def ^:dynamic *learning-rate* 0.01)
(def ^:dynamic *weight-random-diff* 0.001)

(def ^:dynamic *report-period* 100)
(def ^:dynamic *mini-batch-size* 10)


(defn- train-next [nn efn w-updater dataset]
  (let [nn1 (w-updater nn efn dataset)
        d1 (efn nn1 dataset)
        nn2 (w-updater nn efn dataset)
        d2 (efn nn2 dataset)]
    (cond (<= d1 d2) nn1
          (>  d1 d2) nn2)))

(defn train
  "NNの学習を行なう"
  [init-nn efn w-updater dataset terminate-f]
  (loop [cur-nn init-nn, diff (efn init-nn dataset) , cnt 0]
    (let [next (train-next cur-nn efn w-updater dataset)
          next-diff (efn next dataset)]
      (if (zero? (mod cnt *report-period*))
        (println cnt " now diff: " next-diff))
      (if (terminate-f diff next-diff)
        cur-nn
        (recur next, next-diff, (inc cnt))))))

(defn train-sgd
  "訓練データをシャッフルしてミニバッチ方式で学習する"
  [init-nn efn dataset terminate-f]
  (let [batch-data (partition *mini-batch-size* (shuffle dataset))]
    (loop [idx (dec (count batch-data)), nn init-nn]
      (if (< idx 0)
          (do (println "train finish!!!")
              nn)
          (do  (printf "batch start %d\n" idx)
               (println nn)
               (recur (dec idx) (train nn efn weight-gradient (nth batch-data idx) terminate-f)))))))



(defn err-fn-2class
  "2値分類の誤差関数 nn の出力層は1ニューロン"
  [nn dataset]
  (let [samples (count dataset)
        er (apply +
                  (for [{x :x, [d] :ans} dataset
                        :let [[y] (core/nn-calc nn x)]]
                    (+ (* d (Math/log y))
                       (* (- 1.0 d) (Math/log (- 1.0 y))))))]
    (/ (* -1 er) samples)))



;; 勾配降下法

(defn- gradient
  "nnの微小増分の傾きを返す"
  [nn efn dataset level in out]
  (let [w (nw/weight nn level in out)
        nn2 (nw/update-weight nn (+ w *weight-inc-val*) level in out)
        y (efn nn dataset)
        y2 (efn nn2 dataset)]
    (/ (- y2 y) *weight-inc-val*)))

(defn- update-by-gradient
  "重みを勾配に従って更新した値を返す"
  [w nn efn dataset level in out]
  (let [grd (gradient nn efn dataset level in out)]
    (- w (* *learning-rate* grd))))

(defn weight-gradient
  "勾配降下法で重みを更新する"
  [nn efn dataset]
  (nw/map-nn (fn [w l i o]
               (update-by-gradient w nn efn dataset l i o))
             nn))



;; ランダム更新

(defn- rand-add [x]
  (+ x
     (gr/rand-nth [*weight-random-diff*
                   0
                   (* -1 *weight-random-diff*)])))

(defn weight-randomize
  "重みをランダムに更新する"
  [nn efn dataset]
  (binding [gr/*rnd* (java.util.Random. (System/currentTimeMillis))]
    (nw/map-nn (fn [w l i o] (rand-add w))
               nn)))




(comment

(def nn-2class (nw/gen-nn :rand 3 6 1))


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
                       {:x [1 4 1] :ans [0]}
                       {:x [2 4 1] :ans [0]}
                       {:x [1 6 1] :ans [0]}
                       {:x [2 3 1] :ans [0]}
                       {:x [6 1 1] :ans [0]}
                       {:x [9 4 1] :ans [0]}
                       {:x [7 2 1] :ans [0]}
                       {:x [6 2 1] :ans [0]}
                       {:x [8 1 1] :ans [0]}
                       {:x [9 2 1] :ans [0]}
                       {:x [2 1 1] :ans [0]}
                       {:x [2 7 1] :ans [0]}
                       {:x [1 3 1] :ans [0]}
                       {:x [1 7 1] :ans [0]}


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
                       {:x [3 8 1] :ans [1]}
                       {:x [3 4 1] :ans [1]}
                       {:x [5 6 1] :ans [1]}
                       {:x [6 5 1] :ans [1]}
                       {:x [5 4 1] :ans [1]}
                       {:x [5 3 1] :ans [1]}
                       {:x [4 6 1] :ans [1]}
                       {:x [8 7 1] :ans [1]}
                       {:x [3 6 1] :ans [1]}
                       {:x [6 7 1] :ans [1]}

                       ])

(def traindata-2class-2
  [{:x [3 2 1] :ans [0]}
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
   {:x [1 4 1] :ans [0]}
   {:x [2 4 1] :ans [0]}
   {:x [1 6 1] :ans [0]}
   {:x [2 3 1] :ans [0]}
   {:x [6 1 1] :ans [0]}
   {:x [9 4 1] :ans [0]}
   {:x [7 2 1] :ans [0]}
   {:x [6 2 1] :ans [0]}
   {:x [8 1 1] :ans [0]}
   {:x [9 2 1] :ans [0]}
   {:x [2 1 1] :ans [0]}
   {:x [2 7 1] :ans [0]}
   {:x [1 3 1] :ans [0]}
   {:x [1 7 1] :ans [0]}
   {:x [7 5 1] :ans [0]}
   {:x [6 7 1] :ans [0]}
   {:x [8 6 1] :ans [0]}
   {:x [7 8 1] :ans [0]}
   {:x [3 9 1] :ans [0]}
   {:x [4 9 1] :ans [0]}
   {:x [5 9 1] :ans [0]}
   {:x [5 8 1] :ans [0]}
   {:x [2 8 1] :ans [0]}
   {:x [2 9 1] :ans [0]}
   {:x [6 6 1] :ans [0]}
   {:x [2 5 1] :ans [0]}
   {:x [3 5 1] :ans [0]}
   {:x [5 7 1] :ans [0]}
   {:x [7 6 1] :ans [0]}
   {:x [9 6 1] :ans [0]}
   {:x [8 8 1] :ans [0]}
   {:x [6 8 1] :ans [0]}
   {:x [9 8 1] :ans [0]}


   {:x [3 7 1] :ans [1]}
   {:x [4 4 1] :ans [1]}
   {:x [6 4 1] :ans [1]}
   {:x [3 7 1] :ans [1]}
   {:x [4 8 1] :ans [1]}
   {:x [7 3 1] :ans [1]}
   {:x [8 4 1] :ans [1]}
   {:x [4 5 1] :ans [1]}
   {:x [4 7 1] :ans [1]}
   {:x [8 5 1] :ans [1]}
   {:x [6 3 1] :ans [1]}
   {:x [4 7 1] :ans [1]}
   {:x [5 5 1] :ans [1]}
   {:x [7 4 1] :ans [1]}
   {:x [3 8 1] :ans [1]}
   {:x [3 4 1] :ans [1]}
   {:x [5 6 1] :ans [1]}
   {:x [6 5 1] :ans [1]}
   {:x [5 4 1] :ans [1]}
   {:x [5 3 1] :ans [1]}
   {:x [4 6 1] :ans [1]}
   {:x [4 3 1] :ans [1]}
   {:x [3 6 1] :ans [1]}
   {:x [3 7 1] :ans [1]}
   {:x [4 4 1] :ans [1]}
   {:x [6 4 1] :ans [1]}
   {:x [3 7 1] :ans [1]}
   {:x [4 8 1] :ans [1]}
   {:x [7 3 1] :ans [1]}
   {:x [8 4 1] :ans [1]}
   {:x [4 5 1] :ans [1]}
   {:x [4 7 1] :ans [1]}
   {:x [8 5 1] :ans [1]}
   {:x [6 3 1] :ans [1]}
   {:x [4 7 1] :ans [1]}
   {:x [5 5 1] :ans [1]}
   {:x [7 4 1] :ans [1]}
   {:x [3 8 1] :ans [1]}
   {:x [3 4 1] :ans [1]}
   {:x [5 6 1] :ans [1]}
   {:x [6 5 1] :ans [1]}
   {:x [5 4 1] :ans [1]}
   {:x [5 3 1] :ans [1]}
   {:x [4 6 1] :ans [1]}
   {:x [4 3 1] :ans [1]}
   {:x [3 6 1] :ans [1]}
])


(time
 (def nn-g
   (train nn-2class err-fn-2class weight-gradient traindata-2class
          (fn [_ d] (< d 0.1)))
   ))

(time
 (def nn-r
   (train nn-2class err-fn-2class weight-randomize traindata-2class
          (fn [_ d] (< d 0.1)))
   ))

(defn nn-test [nn dataset ans ans-test]
  (let [t (map (fn [x] (core/nn-calc nn x))
               (for [t traindata-2class :when (= (:ans t) ans)]
                 (:x t)))
        cnt (count t)]
    (/ (reduce (fn [r [w]] (if (ans-test w) (inc r) r)) 0.0 t)
       cnt)))

(nn-test nn-g traindata-2class [0] #(< % 0.5))
(nn-test nn-g traindata-2class [1] #(> % 0.5))
(nn-test nn-r traindata-2class [0] #(< % 0.5))
(nn-test nn-r traindata-2class [1] #(> % 0.5))


(defn plot-classify
  "ランダムな数値を分類させて結果をCSVで出力する"
  [nn count]
  (binding [gr/*rnd* (java.util.Random. (System/currentTimeMillis))]
    (let [samples (for [i (range count)
                        :let [x1 (int (* 10 (gr/double)))
                              x2 (int (* 10 (gr/double)))
                              [v] (core/nn-calc nn [x1 x2 1])
                              ok (if (< 0.5 v) 1 0)]]
                    [x1 x2 1 ok v])]
      (doseq [[x1 x2 _ ok _] (sort-by #(nth % 4) samples)]
        (printf "%d,%d,%d\n" x1 x2 ok)))))



)
