(ns neuro.vol-test
  (:require [clojure.test :refer :all]
            [neuro.vol :refer :all]))

(deftest vol-test
  (testing "create vol on default"
    (is (= {:sx 2 :sy 2
            :w [1 2
                3 4]}
           (vol 2 2 [1 2 3 4]))))
  (testing "create vol by x,y"
    (let [{x :sx, y :sy, w :w} (vol 2 3)]
      (is (= 2 x))
      (is (= 3 y))
      (is (= (* 2 3) (count w)))))
  (testing "create vol by vector"
    (let [{x :sx, y :sy, w :w} (vol [1 2 3 4])]
      (is (= 1 x))
      (is (= 4 y))
      (is (= [1 2 3 4] w)))))

(deftest wget-test
  (let [v1 (vol [1 2 3 4 5])
        v2 (vol 2 3 [1 2 3 4 5 6])]
    (testing "wget for vector"
      (is (= 2 (wget v1 0 1))))
    (testing "wget for matrix"
      (is (= 4 (wget v2 1 1))))))

(deftest wset-test
  (let [v1 (vol [1 2 3 4 5])
        v2 (vol 2 3 [1 2 3 4 5 6])]
    (testing "wset for vector"
      (is (= (vol [1 2 100 4 5])
             (wset v1 0 2 100))))
    (testing "wget for matrix"
      (is (= (vol 2 3 [1 2 3 4 100 6])
             (wset v2 0 2 100))))))

(deftest transposed-test
  (let [v1 (vol [1 2 3 4 5])
        v2 (vol 2 3 [1 2
                     3 4
                     5 6])]
    (testing "transposed for vector"
      (let [{x :sx, y :sy, w :w} (transposed v1)]
        (is (= 5 x))
        (is (= 1 y))
        (is (= [1 2 3 4 5] w))))
    (testing "transposed for matrix"
      (let [{x :sx, y :sy, w :w} (transposed v2)]
        (is (= 3 x))
        (is (= 2 y))
        (is (= [1 2 3
                4 5 6])
            w)))))

(deftest w-elm-op-test
  (let [v1 (vol 2 3 [1 2
                     3 4
                     5 6])
        v2 (vol 2 3 [7  8
                     9  10
                     11 12])]
    (testing "w-elm-op vector"
      (let [{x :sx, y :sy, w :w} (w-elm-op (fn [a b] [a b]) v1 v2)]
        (is (= 2 x))
        (is (= 3 y))
        (is (= [[1 7] [2 8]
                [3 9] [4 10]
                [5 11] [6 12]]))))))

(deftest w-prod-test
  (let [v1 (vol 2 1 [1 2])
        v2 (vol 1 2 [10
                     20])
        v3 (vol 2 2 [1 2
                     3 4])
        v4 (vol 2 2 [10 20
                     30 40])]
    (testing "prod 2x1 1x2"
      (let [{x :sx, y :sy, w :w} (w-prod v1 v2)]
        (is (= 1 x))
        (is (= 1 y))
        (is (= [(+ (* 1 10) (* 2 20))]
               w))))
    (testing "prod 2x2"
      (let [{x :sx, y :sy, w :w} (w-prod v3 v4)]
        (is (= 2 x))
        (is (= 2 y))
        (is (= [(+ (* 1 10) (* 2 30))  (+ (* 1 20) (* 2 40))
                (+ (* 3 10) (* 4 30))  (+ (* 3 20) (* 4 40))]
               w))))))
