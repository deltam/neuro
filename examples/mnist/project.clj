(defproject neuro-mnist "0.1.0-SNAPSHOT"
  :description "mnist example for neuro"
  :url "http://example.com/FIXME"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :plugins [[cider/cider-nrepl "0.10.0-SNAPSHOT"]
            [lein-gorilla "0.3.4"]]
  :source-paths ["src" "../../src"]
  :dependencies [[org.clojure/clojure "1.6.0"]
                 [org.clojure/data.generators "0.1.2"]
                 [com.taoensso/timbre "4.1.1"]])
