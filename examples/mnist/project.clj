(defproject neuro-mnist "0.1.0-SNAPSHOT"
  :description "mnist example for neuro"
  :url "https://github.com/deltam/neuro/tree/master/examples/mnist"
  :license {:name "MIT License"
            :url "https://opensource.org/licenses/MIT"
            :year 2018
            :key "mit"}
  :plugins [[cider/cider-nrepl "0.22.0-snapshot"]
            [lein-jupyter "0.1.16"]]
  :source-paths ["src" "../../src"]
  :dependencies [[org.clojure/clojure "1.10.0"]
                 [funcool/octet "1.1.2"]
;                 [com.taoensso/nippy "2.14.0"]
                 ]
  :profiles {:dev {:dependencies [[com.taoensso/tufte "1.4.0"]]}})
