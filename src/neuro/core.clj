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
(defalias with-params tr/with-params)
(defalias init tr/init)
(defalias sgd tr/sgd)
