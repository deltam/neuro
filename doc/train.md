```clojure

(comment

(require '[neuro.train :as tr]
         '[neuro.network :as nw]
         '[neuro.layer :as ly]
		 '[neuro.vol :as vl])

(def traindata-2class
     (tr/gen-train-pairs
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

(def net (nw/network
          (ly/input 2)
          (ly/fc 2 5)
          (ly/sigmoid 5)
          (ly/fc 5 3)
          (ly/sigmoid 3)
          (ly/fc 3 2)
          (ly/softmax 2)))

(time
 (def net2
   (tr/train net traindata testdata
          (fn [e] (< e 0.2)))
   ))








(def traindata-2class-2
     (tr/gen-train-pairs
       [[3 2] [0 1]
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
       [7 5] [0 1]
       [6 7] [0 1]
       [8 6] [0 1]
       [7 8] [0 1]
       [3 9] [0 1]
       [4 9] [0 1]
       [5 9] [0 1]
       [5 8] [0 1]
       [2 8] [0 1]
       [2 9] [0 1]
       [6 6] [0 1]
       [2 5] [0 1]
       [3 5] [0 1]
       [5 7] [0 1]
       [7 6] [0 1]
       [9 6] [0 1]
       [8 8] [0 1]
       [6 8] [0 1]
       [9 8] [0 1]

       [1 1] [0 1]
       [3 1] [0 1]
       [7 7] [0 1]
       [8 7] [0 1]
       [9 7] [0 1]
       [1 5] [0 1]
       [1 9] [0 1]
       [4 8.5] [0 1]
       [3 8.5] [0 1]
       [2 8.5] [0 1]
       [1 8.5] [0 1]
       [6 9] [0 1]
       [7 9] [0 1]
       [8 9] [0 1]
       [9 9] [0 1]
       [5 1] [0 1]
       [5 0.5] [0 1]


       [3 7] [1 0]
       [4 4] [1 0]
       [6 4] [1 0]
       [3 7] [1 0]
       [4 8] [1 0]
       [7 3] [1 0]
       [8 4] [1 0]
       [4 5] [1 0]
       [4 7] [1 0]
       [8 5] [1 0]
       [6 3] [1 0]
       [4 7] [1 0]
       [5 5] [1 0]
       [7 4] [1 0]
       [3 8] [1 0]
       [3 4] [1 0]
       [5 6] [1 0]
       [6 5] [1 0]
       [5 4] [1 0]
       [5 3] [1 0]
       [4 6] [1 0]
       [4 3] [1 0]
       [3 6] [1 0]
       [3 7] [1 0]
       [4 4] [1 0]
       [6 4] [1 0]
       [3 7] [1 0]
       [4 8] [1 0]
       [7 3] [1 0]
       [8 4] [1 0]
       [4 5] [1 0]
       [4 7] [1 0]
       [8 5] [1 0]
       [6 3] [1 0]
       [4 7] [1 0]
       [5 5] [1 0]
       [7 4] [1 0]
       [3 8] [1 0]
       [3 4] [1 0]
       [5 6] [1 0]
       [6 5] [1 0]
       [5 4] [1 0]
       [5 3] [1 0]
       [4 6] [1 0]
       [4 3] [1 0]
       [3 6] [1 0]

       [5 4.5] [1 0]
       [3 6.5] [1 0]
       [3 3.5] [1 0]
       [5 4.5] [1 0]
       [4.5 4.5] [1 0]
       [3 4.5] [1 0]
       [4 7.5] [1 0]
       [4.5 8] [1 0]
       [5.5 4.5] [1 0]
       [8 4.5] [1 0]
       [6.3 5] [1 0]
       [5.5 4.5] [1 0]
]))

(def shuffled (shuffle traindata-2class-2))
(def testdata (take 5 shuffled))
(def traindata (drop 5 shuffled))

(def net (nw/network
          (ly/input 2)
          (ly/fc 2 3)
          (ly/sigmoid 3)
          (ly/fc 3 3)
          (ly/softmax 2)))

(time
 (def net2
   (tr/train net traindata testdata
          (fn [e] (< e 0.2)))
   ))

```
