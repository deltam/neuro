(defproject neuro "0.1.0-SNAPSHOT"
  :description "Deep Neural Network written in Clojure"
  :url "https://github.com/deltam/neuro"
  :license {:name "MIT License"
            :url "https://opensource.org/licenses/MIT"
            :year 2015
            :key "mit"}
  :plugins [[cider/cider-nrepl "0.18.0-snapshot"]
            [lein-gorilla "0.4.0"]]
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [org.clojure/data.generators "0.1.2"]
                 [com.taoensso/timbre "4.1.1"]])
