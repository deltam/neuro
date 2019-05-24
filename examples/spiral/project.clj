(defproject neuro-spiral "0.1.0-SNAPSHOT"
  :description "classification spiral data for neuro"
  :url "http://example.com/FIXME"
  :license {:name "MIT License"
            :url "none"
            :year 2019
            :key "mit"}
  :source-paths ["src" "../../src"]
  :dependencies [[org.clojure/clojure "1.10.0"]]
  :profiles {:dev {:dependencies [[com.taoensso/tufte "2.1.0-RC4"]]}}
  :repl-options {:init-ns spiral.core})
