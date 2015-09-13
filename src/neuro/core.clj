(ns neuro.core
  (:require [neuro.layer :as ly])
  (:require [neuro.network :as nw])
  (:require [neuro.func :as fnc]))


(defn backprop
  "誤差逆伝播法で重みを勾配を算出する"
  [net in-seq train-seq]
  (let [forwarded (ly/forward net in-seq)
        delta-seq (map - forwarded train-seq)]
    (ly/backward net delta-seq)))


(defn nn-calc
  "多層ニューラルネットの計算をする"
  [net in-seq]
  (let [l (last (:layer (ly/forward net in-seq)))]
    (:in-vol l)))
