(ns neuro.core
  (:require [neuro.network :as nw]
            [neuro.train :as tr]))

;; copy from funcool/octet.util
(defmacro ^:private defalias
  [sym sym2]
  `(do
     (def ~sym (var ~sym2))
     (alter-meta! (var ~sym) merge (dissoc (meta (var ~sym2)) :name))))


(defalias gen-net nw/gen-net)
(defalias feedforward nw/feedforward)

(defalias iterate-train-fn tr/iterate-train-fn)
(defalias iterate-mini-batch-train-fn tr/iterate-mini-batch-train-fn)
(defalias split-mini-batch tr/split-mini-batch)
(defalias gen-sgd-optimizer tr/gen-sgd-optimizer)

(defalias with-epoch-report tr/with-epoch-report)
