(ns neuro.train
  (:require [neuro.core :as core]
            [neuro.network :as nw]
            [clojure.data.generators :as gr]))


(def ^:dynamic *weight-inc-val* 0.00001)
(def ^:dynamic *learning-rate* 10.0)
(def ^:dynamic *weight-random-diff* 0.001)

(def ^:dynamic *report-period* 1)
(def ^:dynamic *mini-batch-size* 10)

(def +train-err-vec+ (atom []))
(def +test-err-vec+ (atom []))
(def +learning-rate+ (atom 8.0))

(defn train-init []
  (reset! +train-err-vec+ [])
  (reset! +test-err-vec+ []))

(defn- train-next [nn efn w-updater dataset]
  (let [nn1 (w-updater nn efn dataset)
        d1 (efn nn1 dataset)
        nn2 (w-updater nn efn dataset)
        d2 (efn nn2 dataset)]
    (cond (<= d1 d2) nn1
          (>  d1 d2) nn2)))

(defn train
  "NNの学習を行なう"
  [init-nn efn w-updater dataset testset terminate-f]
  (loop [cur-nn init-nn, err (efn init-nn dataset) , cnt 0]
    (let [next (train-next cur-nn efn w-updater dataset)
          train-err (efn next dataset)
          test-err (efn next testset)]
      (if (zero? (mod cnt *report-period*))
        (do (swap! +train-err-vec+ conj train-err)
            (swap! +test-err-vec+ conj test-err)))
      (if (terminate-f err train-err)
        cur-nn
        (recur next, train-err, (inc cnt))))))

(declare weight-gradient)
(defn train-sgd
  "訓練データをシャッフルしてミニバッチ方式で学習する"
  [init-nn efn dataset testset terminate-f]
  (let [batch-data (partition *mini-batch-size* (shuffle dataset))]
    (loop [idx (dec (count batch-data)), nn init-nn]
      (if (< idx 0)
          (do (println "train finish!!!")
              nn)
          (do (println "batch start " idx)
              (recur (dec idx) (train nn efn weight-gradient (nth batch-data idx) testset terminate-f)))))))



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
  [nn1 nn2 efn dataset]
  (let [y1 (efn nn1 dataset)
        y2 (efn nn2 dataset)]
    (/ (- y2 y1) *weight-inc-val*)))

(defn- update-by-gradient
  "重みを勾配に従って更新した値を返す"
  [w nn1 nn2 efn dataset]
  (let [grd (gradient nn1 nn2 efn dataset)]
    (- w (* @+learning-rate+ grd))))

(defn weight-gradient
  "勾配降下法で重みを更新する"
  [nn efn dataset]
  (nw/map-nn (fn [w l i o]
               (let [nn2 (nw/update-weight nn (+ w *weight-inc-val*) l i o)]
                 (update-by-gradient w nn nn2 efn dataset)))
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
