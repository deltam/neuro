(defproject neuro-mnist "0.1.0-SNAPSHOT"
  :description "mnist example for neuro"
  :url "http://example.com/FIXME"
  :license {:name "MIT License"
            :url "https://opensource.org/licenses/MIT"
            :year 2018
            :key "mit"}
  :plugins [[cider/cider-nrepl "0.18.0-snapshot"]
            [lein-gorilla "0.4.0"]]
  :source-paths ["src" "../../src"]
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [org.clojure/data.generators "0.1.2"]
                 [funcool/octet "1.1.1"]
                 [gorilla-plot "0.1.4"]]
  :profiles {:dev {:dependencies [[com.taoensso/tufte "1.4.0"]]}})
