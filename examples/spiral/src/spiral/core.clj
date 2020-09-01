(ns spiral.core
  (:require [neuro.core :as nc]
            [neuro.train :as nt]
            [neuro.vol :as vl]))

(def net
  (nc/gen-net
   :input 2 :fc
   :sigmoid 10 :fc
   :softmax 3))

(defn spiral-raw [n cat-num]
  (for [cat (range cat-num), i (range n)
        :let [rate (/ i n)
              radius (* 1.0 rate)
              theta (+ (* cat 4.0) (* 4.0 rate) (rand 0.2))]]
    {:cat cat
     :x (* radius (Math/sin theta))
     :y (* radius (Math/cos theta))}))

(defn spiral->vol [raw cat-num]
  (let [xy (mapcat #(vector (:x %) (:y %))
                   raw)
        cat (map :cat raw)]
    (vector (vl/vol (count raw) 2 xy)
            (vl/one-hot cat-num cat))))


(def raw (shuffle (spiral-raw 300 3)))
(def test-data (spiral->vol (take 50 raw) 3))
(def train-data (spiral->vol (drop 50 raw) 3))


(defn evaluate [net [xy-vol cat-vol]]
  (let [done (nc/feedforward net xy-vol)
        check (map (fn [done-vol cat-vol]
                      (= (vl/argmax done-vol)
                         (vl/argmax cat-vol)))
                    (vl/rows done)
                    (vl/rows cat-vol))]
    (count (filter true? check))))

(def ^:private start-time-now-epoch (atom (System/currentTimeMillis)))

(def test-error-rates (atom []))

(defn report [{ep :epoch, net :model, loss :loss}]
  (let [elapsed (- (System/currentTimeMillis) @start-time-now-epoch)
        ok (evaluate net test-data)
        [n _] (vl/shape (first test-data))]
    (printf "epoch %d:  loss = %4.4f, test = %d / %d (%4.2f min)\n" ep loss ok n (float (/ elapsed 60000.0)))
    (flush)
    (swap! test-error-rates conj (- 1.0 (/ (float ok) (float n))))
    (reset! start-time-now-epoch (System/currentTimeMillis))))


(defn train [net]
  (let [batchs (nt/split-mini-batch train-data 30)
        f (nc/iterate-mini-batch-train-fn net batchs)]
    (->> (f (nt/gen-sgd-optimizer 1.0))
         (nt/with-epoch-report report)
         (drop-while #(< (:epoch %) 50))
         (first))))

(comment

  (def train-seq-fn
    (let [batchs (nt/split-mini-batch score/train-data 30)]
      (nc/iterate-mini-batch-train-fn net batchs)))

  (def result
    (->> (train-seq-fn (nt/gen-sgd-optimizer 1.0))
         (nt/with-epoch-report score/report)
         (drop-while #(< (:epoch %) 100))
         (first)))

)


(def decision-boundary-plot
  {
   :width 600
   :height 600
   :autosize "pad",

;   :signals [{:name "classes" :value [0 1 2]}
;             {:name "meshGrid"
;              :value {
;                      :width (count (distinct (map :x mesh)))
;                      :height (count (distinct (map :y mesh)))
;                      :values (map :val mesh)
;                      }}
;             ]

   :data [
          {
           :name "contours",
           :transform [
                       {
                        :type "contour"
                        :size [{:signal "meshGrid.width"} {:signal "meshGrid.height"}]
                        :values {:signal "meshGrid.values"}
                        :thresholds {:signal "classes"}
                        }
                       ]
           }
          {:name "spiral"
           :values raw}
          ]

   :projections [
                 {
                  :name "projection",
                  :type "identity",
                  :scale {:signal "width / meshGrid.width"}
                  }
                 ],

   :scales [
            {
             :name "x",
             :type "linear",
             :domain {:data "spiral", :field "x"},
             :domainMax 1.0
             :domainMin -1.0
             :zero true
             :range "width"
             },
            {
             :name "y",
             :type "linear",
             :domain {:data "spiral", :field "y"},
             :domainMax 1.0
             :domainMin -1.0
             :zero true
             :range "height"
             },
            {
             :name "color",
             :type "sequential",
             :zero true,
             :domain {:data "contours", :field "value"},
             :range "heatmap"
             }
            {
             :name "class"
             :type "threshold"
             :zero true
             :domain {:signal "classes"}
             :range {:scheme "dark2"}
             }
            ],

   :axes [
          {
           :scale "x",
           :grid true,
           :domain false,
           :orient "bottom",
           :title "x"
           },
          {
           :scale "y",
           :grid true,
           :domain false,
           :orient "left",
           :title "y"
           }
          ],

   :legends [
             {:fill "class", :type "category"}
             {:fill "color", :type "gradient"}
             ],

   :marks [
           {
            :type "path",
            :from {:data "contours"},
            :encode {
                     :enter {
                             :stroke {:value "transparent"},
                             :strokeWidth {:value 0},
                             :fill {:scale "color", :field "value"},
                             :fillOpacity {:value 0.35}
                             }
                     },
            :transform [
                        {:type "geopath"
                         :field "datum"
                         :projection "projection"
                         }
                        ]
            }
           {
            :type "symbol"
            :from {:data "spiral"}
            :encode {
                     :enter {
                             :x {:scale "x" :field "x"}
                             :y {:scale "y" :field "y"}
                             :shape {:value "circle"}
                             :opacity {:value 0.5}
                             :fill {:scale "class", :field "cat"}
                             :size {:value 10}
                             }
                     }
            }
           ],

   :config {
            :range {
                    :heatmap {:scheme "greenblue"}
                    }
            }
   })
