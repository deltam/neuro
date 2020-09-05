(defproject neuro-mnist "0.1.0-SNAPSHOT"
  :description "mnist example for neuro"
  :url "https://github.com/deltam/neuro/tree/master/examples/mnist"
  :license {:name "MIT License"
            :url "https://opensource.org/licenses/MIT"
            :year 2018
            :key "mit"}
  :source-paths ["src" "../../src"]
  :dependencies [[org.clojure/clojure "1.10.0"]
                 [funcool/octet "1.1.2"]]
  :profiles {:dev {:dependencies [[com.taoensso/tufte "2.1.0-RC4"]]}}
  :jvm-opts ["-Xmx2048M"]
  :repl-options {:init-ns mnist.core})
