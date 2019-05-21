(defproject deltam/neuro "0.1.1-SNAPSHOT"
  :description "Deep Neural Network written in Clojure from scratch"
  :url "https://github.com/deltam/neuro"
  :license {:name "MIT License"
            :url "https://opensource.org/licenses/MIT"
            :year 2015
            :key "mit"}
  :plugins [[cider/cider-nrepl "0.22.0-snapshot"]]
  :dependencies [[org.clojure/clojure "1.10.0"]]
  :profiles {:dev {:dependencies [[com.taoensso/tufte "2.1.0-RC4"]]}})
