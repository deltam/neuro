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


(def raw (shuffle (spiral-raw 100 3)))
(def dataset (spiral->vol raw 3))


(def train-status (atom nil))

(defn report [ep net]
  (let [loss (last (:train-loss-history @train-status))]
    (when (zero? (mod ep 10))
      (printf "epoch %d: %f" ep loss)
;      (let [[xy-vol cat-vol] dataset
;            [len _] (vl/shape xy-vol)
;            pred (nc/feedforward net xy-vol)]
;        (print "\ttest: " (count
;                           (filter (fn [[p cat]]
;                                     (= (vl/argmax p) (vl/argmax cat)))
;                                   (map vector
;                                        (vl/rows pred)
;                                        (vl/rows cat-vol))))
;               "/" len))
      (println)
      (flush))))

(defn train
  ([] (train net))
  ([net]
   (reset! train-status (nt/new-status))
   (let [start (System/currentTimeMillis)
         [xy-vol cat-vol] dataset
         trained (nc/sgd net xy-vol cat-vol)]
     (println "elapsed" (/ (- (System/currentTimeMillis) start) 1000.0) "sec")
     trained
   )))

(comment
(def net2
  (neuro.core/with-params [:epoch-limit 300
                           :mini-batch-size 30
                           :learning-rate 1.0
                           :train-status-var train-status
                           :epoch-reporter report]
    (train net)))
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
