(ns neuro.mnist)

(def ^:dynamic *train-images-filename* "train-images-idx3-ubyte")
(def ^:dynamic *train-labels-filename* "train-labels-idx1-ubyte")

(defn read-as-byte-array [filename]
  (with-open [fis (java.io.FileInputStream. filename)]
    (let [channel (.getChannel fis)
          byte-buf (java.nio.ByteBuffer/allocate 100000)]
      (.read channel byte-buf)
      byte-buf)))

(defn bytes->int [byte-buf offset]
  (.getInt byte-buf (* offset 4)))

(defn bytes->image-meta [byte-buffer]
  (let [magic (bytes->int byte-buffer 0)
        num (bytes->int byte-buffer 1)
        rows (bytes->int byte-buffer 2)
        cols (bytes->int byte-buffer 3)]
    {:magic magic
     :count num
     :rows rows
     :cols cols}))

(defn byte-range [byte-buf offset len]
  (map #(.get byte-buf (+ offset %))
       (range len)))

(defn bytes->images [byte-buf]
  (let [meta (bytes->image-meta byte-buf)
        offset 16
        len (* (:rows meta) (:cols meta))]
    (for [cnt (range (:count meta))]
      (byte-range byte-buf (+ offset (* cnt len)) len))))


(defn bytes->label-meta [byte-buffer]
  (let [magic (bytes->int byte-buffer 0)
        num (bytes->int byte-buffer 1)]
    {:magic magic
     :count num}))

(defn bytes->labels [byte-buf]
  (let [meta (bytes->label-meta byte-buf)
        offset 8]
    (for [cnt (range (:count meta))]
      (.get byte-buf (+ offset cnt)))))

(defn dataset []
  (let [image-bytes (read-as-byte-array *train-images-filename*)
        images (bytes->images image-bytes)
        label-bytes (read-as-byte-array *train-labels-filename*)
        labels (bytes->labels label-bytes)]
    (for [[image label] (map vector images labels)]
      {:label label
       :image image})))
