

```clojure

(comment

(def net (nw/network
          (ly/input-layer 2)
          (ly/fc-layer 2 3)
          (ly/sigmoid-layer 3)
          (ly/fc-layer 3 3)
          (ly/softmax-layer 2)))



(def traindata-2class
	 (tr/gen-train-dataset
	  [[2 5] [0 1]
       [3 2] [0 1]
       [4 1] [0 1]
       [8 3] [0 1]
       [1 8] [0 1]
       [9 5] [0 1]
       [5 2] [0 1]
       [4 2] [0 1]
       [3 3] [0 1]
       [2 6] [0 1]
       [1 8] [0 1]
       [9 5] [0 1]
       [1 4] [0 1]
       [2 4] [0 1]
       [1 6] [0 1]
       [2 3] [0 1]
       [6 1] [0 1]
       [9 4] [0 1]
       [7 2] [0 1]
       [6 2] [0 1]
       [8 1] [0 1]
       [9 2] [0 1]
       [2 1] [0 1]
       [2 7] [0 1]
       [1 3] [0 1]
       [1 7] [0 1]


       [3 7] [1 0]
       [4 4] [1 0]
       [7 6] [1 0]
       [3 7] [1 0]
       [4 8] [1 0]
       [7 8] [1 0]
       [7 6] [1 0]
       [4 5] [1 0]
       [4 7] [1 0]
       [8 5] [1 0]
       [6 3] [1 0]
       [4 7] [1 0]
       [9 6] [1 0]
       [7 4] [1 0]
       [3 8] [1 0]
       [3 4] [1 0]
       [5 6] [1 0]
       [6 5] [1 0]
       [5 4] [1 0]
       [5 3] [1 0]
       [4 6] [1 0]
       [8 7] [1 0]
       [3 6] [1 0]
       [6 7] [1 0]
]))

(def shuffled (shuffle traindata-2class))
(def testdata (take 5 shuffled))
(def traindata (drop 0 shuffled))

(time
 (def nn2
   (train-sgd nn err-fn-2class traindata testdata
          (fn [_ _ e] (< e 0.2)))
   ))


(def traindata-2class-2
  [{:x [3 2] :ans [0]}
   {:x [4 1] :ans [0]}
   {:x [8 3] :ans [0]}
   {:x [1 8] :ans [0]}
   {:x [9 5] :ans [0]}
   {:x [5 2] :ans [0]}
   {:x [4 2] :ans [0]}
   {:x [3 3] :ans [0]}
   {:x [2 6] :ans [0]}
   {:x [1 8] :ans [0]}
   {:x [9 5] :ans [0]}
   {:x [1 4] :ans [0]}
   {:x [2 4] :ans [0]}
   {:x [1 6] :ans [0]}
   {:x [2 3] :ans [0]}
   {:x [6 1] :ans [0]}
   {:x [9 4] :ans [0]}
   {:x [7 2] :ans [0]}
   {:x [6 2] :ans [0]}
   {:x [8 1] :ans [0]}
   {:x [9 2] :ans [0]}
   {:x [2 1] :ans [0]}
   {:x [2 7] :ans [0]}
   {:x [1 3] :ans [0]}
   {:x [1 7] :ans [0]}
   {:x [7 5] :ans [0]}
   {:x [6 7] :ans [0]}
   {:x [8 6] :ans [0]}
   {:x [7 8] :ans [0]}
   {:x [3 9] :ans [0]}
   {:x [4 9] :ans [0]}
   {:x [5 9] :ans [0]}
   {:x [5 8] :ans [0]}
   {:x [2 8] :ans [0]}
   {:x [2 9] :ans [0]}
   {:x [6 6] :ans [0]}
   {:x [2 5] :ans [0]}
   {:x [3 5] :ans [0]}
   {:x [5 7] :ans [0]}
   {:x [7 6] :ans [0]}
   {:x [9 6] :ans [0]}
   {:x [8 8] :ans [0]}
   {:x [6 8] :ans [0]}
   {:x [9 8] :ans [0]}

   {:x [1 1] :ans [0]}
   {:x [3 1] :ans [0]}
   {:x [7 7] :ans [0]}
   {:x [8 7] :ans [0]}
   {:x [9 7] :ans [0]}
   {:x [1 5] :ans [0]}
   {:x [1 9] :ans [0]}
   {:x [4 8.5] :ans [0]}
   {:x [3 8.5] :ans [0]}
   {:x [2 8.5] :ans [0]}
   {:x [1 8.5] :ans [0]}
   {:x [6 9] :ans [0]}
   {:x [7 9] :ans [0]}
   {:x [8 9] :ans [0]}
   {:x [9 9] :ans [0]}
   {:x [5 1] :ans [0]}
   {:x [5 0.5] :ans [0]}


   {:x [3 7] :ans [1]}
   {:x [4 4] :ans [1]}
   {:x [6 4] :ans [1]}
   {:x [3 7] :ans [1]}
   {:x [4 8] :ans [1]}
   {:x [7 3] :ans [1]}
   {:x [8 4] :ans [1]}
   {:x [4 5] :ans [1]}
   {:x [4 7] :ans [1]}
   {:x [8 5] :ans [1]}
   {:x [6 3] :ans [1]}
   {:x [4 7] :ans [1]}
   {:x [5 5] :ans [1]}
   {:x [7 4] :ans [1]}
   {:x [3 8] :ans [1]}
   {:x [3 4] :ans [1]}
   {:x [5 6] :ans [1]}
   {:x [6 5] :ans [1]}
   {:x [5 4] :ans [1]}
   {:x [5 3] :ans [1]}
   {:x [4 6] :ans [1]}
   {:x [4 3] :ans [1]}
   {:x [3 6] :ans [1]}
   {:x [3 7] :ans [1]}
   {:x [4 4] :ans [1]}
   {:x [6 4] :ans [1]}
   {:x [3 7] :ans [1]}
   {:x [4 8] :ans [1]}
   {:x [7 3] :ans [1]}
   {:x [8 4] :ans [1]}
   {:x [4 5] :ans [1]}
   {:x [4 7] :ans [1]}
   {:x [8 5] :ans [1]}
   {:x [6 3] :ans [1]}
   {:x [4 7] :ans [1]}
   {:x [5 5] :ans [1]}
   {:x [7 4] :ans [1]}
   {:x [3 8] :ans [1]}
   {:x [3 4] :ans [1]}
   {:x [5 6] :ans [1]}
   {:x [6 5] :ans [1]}
   {:x [5 4] :ans [1]}
   {:x [5 3] :ans [1]}
   {:x [4 6] :ans [1]}
   {:x [4 3] :ans [1]}
   {:x [3 6] :ans [1]}

   {:x [5 4.5] :ans [1]}
   {:x [3 6.5] :ans [1]}
   {:x [3 3.5] :ans [1]}
   {:x [5 4.5] :ans [1]}
   {:x [4.5 4.5] :ans [1]}
   {:x [3 4.5] :ans [1]}
   {:x [4 7.5] :ans [1]}
   {:x [4.5 8] :ans [1]}
   {:x [5.5 4.5] :ans [1]}
   {:x [8 4.5] :ans [1]}
   {:x [6.3 5] :ans [1]}
   {:x [5.5 4.5] :ans [1]}

])

(def shuffled (shuffle traindata-2class-2))
(def testdata (take 5 shuffled))
(def traindata (drop 5 shuffled))

(def nn (nw/gen-nn :rand 2 6 2 1))

(time
 (def nn2
   (train-sgd nn err-fn-2class traindata testdata
          (fn [_ _ e] (< e 0.2)))
   ))

```
