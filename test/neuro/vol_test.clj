(ns neuro.vol-test
  (:refer-clojure :exclude [repeat shuffle partition rand print])
  (:require [clojure.test :refer :all]
            [neuro.vol :refer :all]))

(deftest vol-test
  (testing "create vol on default"
    (let [v (vol 2 2 [1 2
                      3 4])]
      (is (= [2 2] (shape v)))
      (is (= [1 2 3 4] (raw v)))
      (is (function? (:posf v)))))
  (testing "create vol by vector"
    (let [v (vol [1 2 3 4])]
      (is (= [1 4] (shape v)))
      (is (= [1 2 3 4]
             (raw v)))
      (is (function? (:posf v))))))

(deftest wget-test
  (let [v1 (vol [1 2 3 4 5])
        v2 (vol 2 3 [1 2 3
                     4 5 6])]
    (testing "wget for vector"
      (is (= 1 (wget v1 0 0)))
      (is (= 2 (wget v1 0 1)))
      (is (= 3 (wget v1 0 2)))
      (is (= 4 (wget v1 0 3)))
      (is (= 5 (wget v1 0 4)))
      (is (thrown? IndexOutOfBoundsException (wget v1 0 5)))
      (is (thrown? IndexOutOfBoundsException (wget v1 1 0)))
      (is (thrown? IndexOutOfBoundsException (wget v1 -1 0)))
      (is (thrown? IndexOutOfBoundsException (wget v1 0 -1))))
    (testing "wget for matrix"
      (is (= 1 (wget v2 0 0)))
      (is (= 2 (wget v2 0 1)))
      (is (= 3 (wget v2 0 2)))
      (is (= 4 (wget v2 1 0)))
      (is (= 5 (wget v2 1 1)))
      (is (= 6 (wget v2 1 2)))
      (is (thrown? IndexOutOfBoundsException (wget v2 0 3)))
      (is (thrown? IndexOutOfBoundsException (wget v2 3 0)))
      (is (thrown? IndexOutOfBoundsException (wget v2 -1 0)))
      (is (thrown? IndexOutOfBoundsException (wget v2 0 -1))))))

(deftest slice-test
  (let [v (vol 3 4 [1  2  3  4
                    5  6  7  8
                    9 10 11 12])
        s (slice v 1 2)]
    (is (= [1 4] (shape s)))
    (is (= [5 6 7 8]
           [(wget s 0 0)
            (wget s 0 1)
            (wget s 0 2)
            (wget s 0 3)]))))

(deftest partition-test
  (let [v (vol 3 2 [1 2
                    3 4
                    5 6])
        p1 (partition v 1)
        p2 (partition v 2)]
    (testing "partition 3 by 1"
      (is (= 3 (count p1)))
      (is (every? #(= [1 2] (shape %)) p1))
      (is (= [1 2] (raw-vec (nth p1 0))))
      (is (= [3 4] (raw-vec (nth p1 1))))
      (is (= [5 6] (raw-vec (nth p1 2)))))
    (testing "partition 3 by 2"
      (is (= 1 (count p2)))
      (is (= [2 2] (shape (first p2))))
      (is (= [1 2
              3 4] (raw-vec (first p2)))))))

(deftest T-test
  (testing "transposed for vector"
    (let [v1 (vol [1 2 3 4 5])
          tv1 (T v1)]
      (is (= [5 1] (shape tv1)))
      (is (= [1 2 3 4 5]
             [(wget tv1 0 0)
              (wget tv1 1 0)
              (wget tv1 2 0)
              (wget tv1 3 0)
              (wget tv1 4 0)]))
      ))
  (testing "transposed for matrix"
    (let [v2 (vol 2 3 [1 2 3
                       4 5 6])
          tv2 (T v2)]
      (is (= [3 2] (shape tv2)))
      (is (= [1 4
              2 5
              3 6]
             [(wget tv2 0 0) (wget tv2 0 1)
              (wget tv2 1 0) (wget tv2 1 1)
              (wget tv2 2 0) (wget tv2 2 1)])))))


(deftest dot-test
  (testing "prod 2x2"
    (let [v1 (vol 2 2 [1 2
                       3 4])
          v2 (vol 2 2 [10 20
                       30 40])
          done (dot v1 v2)]
      (is (= [2 2] (shape done)))
      (is (= [(+ (* 1 10) (* 2 30))  (+ (* 1 20) (* 2 40))
              (+ (* 3 10) (* 4 30))  (+ (* 3 20) (* 4 40))]
             (raw done)))))
  (testing "prod 3x1 1x2 = 3x2"
    (let [v1 (vol 3 1 [1
                       2
                       3])
          v2 (vol 1 2 [10 20])
          done (dot v1 v2)]
      (is (= [3 2] (shape done)))
      (is (= [(* 1 10) (* 1 20)
              (* 2 10) (* 2 20)
              (* 3 10) (* 3 20)]
             (raw done))))))

(deftest w-elm-op-test
  (let [v1 (vol 2 3 [1 2 3
                     4 5 6])
        v2 (vol 2 3 [ 7  8  9
                     10 11 12])]
    (testing "w-elm-op vector"
      (let [done (w-elm-op (fn [a b] [a b]) v1 v2)]
        (is (= [2 3] (shape done)))
        (is (= [[1  7] [2  8] [3  9]
                [4 10] [5 11] [6 12]]
               (raw done)))))))

(deftest repeat-test
  (testing "repeat vector"
    (let [v (vol [1 2 3])
          done (repeat v 2)]
      (is (= [1 3] (shape v)))
      (is (= [2 3] (shape done)))
      (is (= [1 2 3
              1 2 3]
             [(wget done 0 0) (wget done 0 1) (wget done 0 2)
              (wget done 1 0) (wget done 1 1) (wget done 1 2)])))))

(deftest sum-row-test
  (let [v (vol 2 3 [1   2  3
                    0.4 0.5 0.6])
        s (sum-row v)]
    (is (= [1 3] (shape s)))
    (is (= [1.4 2.5 3.6] (raw s)))))


(deftest argmax-test
  (let [v (vol [1 2 3])]
    (is (= [0 2] (argmax v))))
  (let [v (vol 2 3 [1 2 3
                    9 4 5])]
    (is (= [1 0] (argmax v)))))
