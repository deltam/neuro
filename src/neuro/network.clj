(ns neuro.network)

(defn gen-nn
  "多層ニューラルネットを定義する"
  [init & level-nodes]
  {:nodes (apply vector level-nodes)
   :weights (apply vector
                   (for [[in out] (seq-by-2-items level-nodes)]
                     (gen-num-matrix init in out)))
   :func :logistic})

(defn weight [nn level in out]
  (let [w-mat ((:weights nn) level)]
    ((w-mat in) out)))

(defn update-weight
  "重みを更新する"
  [nn w level in-node out-node]
  (let [w-mat (:weights nn)
        updated (update-matrix-at (w-mat level) in-node out-node w)]
    (assoc nn :weights (update-at w-mat level updated))))

(defn map-weights
  "重みの更新を一括して行なう"
  [f nn]
  (let [ws (:weights nn)
        levels (range (count ws))
        idx (for [l levels, in (range (count (ws l))), out (range (count ((ws l) in)))]
              [l in out])]
    (reduce (fn [ret [l i o]]
              (let [w (weight ret l i o)]
                (update-weight ret (f w l i o) l i o)))
            nn
            idx)))





(defn- gen-num-vec [init n]
  (let [init-f #(if (number? init) init (init))]
    (apply vector
           (repeat n (init-f)))))

(defn- gen-num-matrix [init x y]
  (gen-num-vec #(gen-num-vec init y) x))

(defn- seq-by-2-items [s]
  (loop [ret [], cur s, next (rest s)]
    (if (and (not-empty cur) (not-empty next))
      (recur (conj ret [(first cur) (first next)]) (rest cur) (rest next))
      ret)))

(defn- update-at [v idx val]
  (apply vector
         (concat (subvec v 0 idx)
                 [val]
                 (subvec v (inc idx)))))

(defn- update-matrix-at [mat x y val]
  (update-at mat x
             (update-at (mat x) y val)))




(comment

(def nn {:nodes [3 2 1]
         :weights [
                   [[0.0 0.0]
                    [0.0 0.0]
                    [0.0 0.0]]
                   [[0.0]
                    [0.0]]
                   ]
         :func :logistic})

)
