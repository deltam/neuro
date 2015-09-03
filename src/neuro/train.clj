(ns neuro.train
  (:require [neuro.core :as core]
            [neuro.network :as nw]
            [clojure.data.generators :as gr]))


(def ^:dynamic *weight-inc-val* 0.00001)
(def ^:dynamic *learning-rate* 10.0)
(def ^:dynamic *weight-random-diff* 0.001)

(def ^:dynamic *report-period* 1)
(def ^:dynamic *mini-batch-size* 10)
(def ^:dynamic *momentum-param* 0.6)

(def +train-err-vec+ (atom []))
(def +test-err-vec+ (atom []))
(def +learning-rate+ (atom 3.0))
(def +go-next-batch+ (atom false))
(def +now-nn+ (atom nil))


(defn init []
  (reset! +train-err-vec+ [])
  (reset! +test-err-vec+ [])
  (reset! +go-next-batch+ false)
  (reset! +now-nn+ nil))

(defn- monitoring
  "学習過程をレポートする"
  [epoc train-err test-err nn]
  (if (zero? (mod epoc *report-period*))
    (do (swap! +train-err-vec+ conj train-err)
        (swap! +test-err-vec+ conj test-err)
        (reset! +now-nn+ nn))))

(defn- momentum
  "モメンタム項の計算"
  [pre-nn cur-nn nn]
  (nw/map-nn (fn [l i o w]
               (let [dw ( - (nw/wget cur-nn l i o)
                            (nw/wget pre-nn l i o))]
                 (+ w (* *momentum-param* dw))))
             nn))

(defn train
  "NNの学習を行なう"
  [init-nn efn w-updater dataset testset terminate-f]
  (loop [pre-nn init-nn, cur-nn init-nn, err (efn init-nn dataset) , epoc 0]
    (let [next-nn (momentum pre-nn cur-nn (w-updater cur-nn efn dataset))
          train-err (efn next-nn dataset)
          test-err (efn next-nn testset)]
      (monitoring epoc train-err test-err next-nn)
      (if (or @+go-next-batch+ (terminate-f err train-err))
        (do (reset! +go-next-batch+ false)
            cur-nn)
        (recur cur-nn, next-nn, train-err, (inc epoc))))))

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
  (nw/map-nn (fn [l i o w]
               (let [nn2 (nw/wput nn l i o (+ w *weight-inc-val*))]
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
    (nw/map-nn (fn [l i o w] (rand-add w))
               nn)))
