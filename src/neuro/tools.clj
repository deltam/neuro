(ns neuro.tools
  (:require [neuro.vol :as vl]
            [neuro.layer :as ly]
            [neuro.network :as nw]))


;;; gradient checking (for debug)

(defn add-w-eps
  [net l i o eps]
  (let [layer (nth (:layer net) l)
        v (:w layer)
        we (+ (vl/wget v i o) eps)]
    (assoc net :layer
           (assoc (vec (:layer net)) l
                  (assoc layer :w
                         (vl/wset v i o we))))))

(defn get-dw
  [net l i o]
  (let [layer (nth (:layer net) l)
        v (:dw layer)]
    (vl/wget v i o)))

(defn calc-loss
  [net in-vol train-vol]
  (:loss
   (ly/backward (nw/loss-layer (ly/forward net in-vol))
                train-vol)))

(defn gradient-checking
  "To use when debug of backpropagation.
  See http://ufldl.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization

  (def net (nw/network
            (ly/input 2)
            (ly/fc 2 5)
            (ly/relu 5)
            (ly/fc 5 4)
            (ly/sigmoid 4)
            (ly/fc 4 3)
            (ly/tanh 3)
            (ly/fc 3 2)
            (ly/softmax 2)))

  (gradient-checking net (vl/vol [2 3]) (vl/vol [1 0]) 5 1 0)
  ;> {:result true, :dw -0.11955934238580034, :grad -0.11955934228480292}

  (gradient-checking net (vl/vol [2 3]) (vl/vol [0 1]) 5 1 0)
  ;> {:result true, :dw 0.07476710855760384, :grad 0.07476710852882817}

  (gradient-checking net (vl/vol [2 3]) (vl/vol [0 1]) 7 1 0)
  ;> {:result true, :dw 0.061648897173240895, :grad 0.06164889717413802}"
  [net in-vol train-vol l i o]
  (let [net-bp (backprop net in-vol train-vol (fn [w dw] (- w (* 0.01 dw))))
        dw (get-dw net-bp l i o)
        eps 0.0001
        net1 (add-w-eps net l i o eps)
        net2 (add-w-eps net l i o (- eps))
        loss1 (calc-loss net1 in-vol train-vol)
        loss2 (calc-loss net2 in-vol train-vol)
        grad (/ (- loss1 loss2)
                (* 2 eps))]
    {:result (< (Math/abs (- dw grad))
                eps)
     :dw dw
     :grad grad}))
