(ns neuro.vol-test
  (:require [clojure.test :refer :all]
            [neuro.vol :refer :all]))

(deftest vol-test
  (testing "create vol on default"
    (is (= #neuro.vol.VecVol{:sx 2 :sy 2
            :w [1 2
                3 4]
            :T false}
           (vol 2 2 [1 2
                     3 4]))))
  (testing "create vol by x,y"
    (let [{x :sx, y :sy, w :w, T :T} (vol 2 3)]
      (is (= 2 x))
      (is (= 3 y))
      (is (= (* 2 3) (count w)))
      (is (false? T))))
  (testing "create vol by vector"
    (let [{x :sx, y :sy, w :w, T :T} (vol [1 2 3 4])]
      (is (= 1 x))
      (is (= 4 y))
      (is (= [1
              2
              3
              4]
             w))
      (is (false? T)))))

(deftest wget-test
  (let [v1 (vol [1
                 2
                 3
                 4
                 5])
        v2 (vol 2 3 [1 2
                     3 4
                     5 6])]
    (testing "wget for vector"
      (is (= 2 (wget v1 0 1))))
    (testing "wget for matrix"
      (is (= 4 (wget v2 1 1))))))

(deftest wset-test
  (let [v1 (vol [1
                 2
                 3
                 4
                 5])
        v2 (vol 2 3 [1 2
                     3 4
                     5 6])]
    (testing "wset for vector"
      (is (= (vol [1
                   2
                   100
                   4
                   5])
             (wset v1 0 2 100))))
    (testing "wget for matrix"
      (is (= (vol 2 3 [1 2
                       3 4
                       5 100])
             (wset v2 1 2 100))))))

(deftest transposed-test
  (let [v1 (vol [1 2 3 4 5])
        v2 (vol 2 3 [1 2
                     3 4
                     5 6])]
    (testing "transposed for vector"
      (let [tv1 (T v1)
            [x y] (shape tv1)
            {T :T} tv1]
        (is (= 5 x))
        (is (= 1 y))
        (is (true? T))
        (is (= [1
                2
                3
                4
                5]
               [(wget tv1 0 0)
                (wget tv1 1 0)
                (wget tv1 2 0)
                (wget tv1 3 0)
                (wget tv1 4 0)]))
        ))
    (testing "transposed for matrix"
      (let [tv2 (T v2)
            [x y] (shape tv2)
            {T :T} tv2]
        (is (= 3 x))
        (is (= 2 y))
        (is (= [1 2
                3 4
                5 6]
               [(wget tv2 0 0) (wget tv2 0 1)
                (wget tv2 1 0) (wget tv2 1 1)
                (wget tv2 2 0) (wget tv2 2 1)]))
        (is (true? T))))))

(deftest w-elm-op-test
  (let [v1 (vol 2 3 [1 2
                     3 4
                     5 6])
        v2 (vol 2 3 [7   8
                     9  10
                     11 12])]
    (testing "w-elm-op vector"
      (let [{x :sx, y :sy, w :w} (w-elm-op (fn [a b] [a b]) v1 v2)]
        (is (= 2 x))
        (is (= 3 y))
        (is (= [[1  7] [2  8]
                [3  9] [4 10]
                [5 11] [6 12]]
               w))))))

(deftest dot-test
  (testing "prod 2x2"
    (let [v1 (vol 2 2 [1 2
                       3 4])
          v2 (vol 2 2 [10 20
                       30 40])
          {x :sx, y :sy, w :w} (dot v1 v2)]
      (is (= 2 x))
      (is (= 2 y))
      (is (= [(+ (* 1 10) (* 2 30))  (+ (* 1 20) (* 2 40))
              (+ (* 3 10) (* 4 30))  (+ (* 3 20) (* 4 40))]
             w))))
  (testing "prod 1x3 2x1 = 2x3"
    (let [v1 (vol 1 3 [1
                       2
                       3])
          v2 (vol 2 1 [10 20])
          {x :sx, y :sy, w :w} (dot v1 v2)]
      (is (= 2 x))
      (is (= 3 y))
      (is (= [(* 1 10) (* 1 20)
              (* 2 10) (* 2 20)
              (* 3 10) (* 3 20)]
             w)))))
