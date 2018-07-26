(defproject neuro-mnist "0.1.0-SNAPSHOT"
  :description "mnist example for neuro"
  :url "http://example.com/FIXME"
  :license {:name "MIT License"
            :url "https://opensource.org/licenses/MIT"}
  :plugins [[cider/cider-nrepl "0.18.0-snapshot"]
            [lein-gorilla "0.4.0"]]
  :source-paths ["src" "../../src"]
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [org.clojure/data.generators "0.1.2"]
                 [com.taoensso/timbre "4.1.1"]])
