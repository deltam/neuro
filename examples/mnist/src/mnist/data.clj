(ns mnist.data
  "Load MNIST binary files"
  (:require [neuro.vol :as vl]
            [octet.core :as buf])
  (:import java.awt.image.BufferedImage
           java.nio.ByteBuffer
           java.nio.channels.FileChannel$MapMode
           java.io.FileInputStream))


(declare read-data)

(defn- load-train-raw []
  (read-data "train-images-idx3-ubyte" "train-labels-idx1-ubyte"))
(def load-train (memoize load-train-raw))

(defn- load-test-raw []
  (read-data "t10k-images-idx3-ubyte" "t10k-labels-idx1-ubyte"))
(def load-test (memoize load-test-raw))



(defn float->byte [f]
  (let [n (int (* 256 f))]
    (byte (if (< n 128) n (- n 256)))))

(defn byte->float [b]
  (float
   (/ (if (neg? b)
        (+ b 256)
        b)
      (float 256))))

(defn- gray->rgb [gf]
  (let [g (float->byte gf)]
    (int
     (bit-or (bit-and (int (bit-shift-left g 8)) 0x00ff0000)
             (bit-and (int (bit-shift-left g 4)) 0x0000ff00)
             (bit-and (int g) 0xff)))))

(defn ^BufferedImage vol->image [v]
  (let [img-buf (BufferedImage. 28 28 BufferedImage/TYPE_BYTE_GRAY)]
    (doseq [x (range 28), y (range 28)]
      (.setRGB ^BufferedImage img-buf x y (gray->rgb (vl/wget v 0 (+ x (* y 28))))))
    img-buf))

(defn vol->digit [v]
  (vl/argmax v))

(defn in-dig
  "`(filter (in-dig 0 1) (:train (mnist.data/dataset)))`"
  [& more]
  (let [dset (set more)]
    (fn [[_ label-vol]] (dset (vol->digit label-vol)))))


;;; read binary files

(def ^{:private true, :const true} len-image-bytes (* 28 28))

(def ^:private image-header-spec
  (buf/spec :magic buf/int32
            :num buf/int32
            :rows buf/int32
            :cols buf/int32))
(def ^:private image-chunk-spec (buf/bytes len-image-bytes))

(def ^:private label-header-spec
  (buf/spec :magic buf/int32
            :num buf/int32))
(def ^:private label-chunk-spec buf/byte)

(defn- read-repeat-chunk [buffer header-spec chunk-spec]
  (let [{num :num} (buf/read buffer header-spec)]
    (buf/read buffer
              (buf/repeat num chunk-spec)
              {:offset (buf/size header-spec)})))

(defn- ^ByteBuffer read-as-byte-buf [^String filename]
  (with-open [f (FileInputStream. ^String filename)]
    (let [ch (.getChannel f)
          mbb (.map ch FileChannel$MapMode/READ_ONLY 0 (.size ch))]
      mbb)))

(defn- read-data [images-filename labels-filename]
  (let [image-buf (read-as-byte-buf images-filename)
        images (read-repeat-chunk image-buf
                                  image-header-spec image-chunk-spec)
        img-vec (mapcat (fn [img] (map #(byte->float (aget ^bytes img %))
                                       (range (* 28 28))))
                        images)
        label-buf (read-as-byte-buf labels-filename)
        label-vec (read-repeat-chunk label-buf
                                     label-header-spec label-chunk-spec)]
    [(vl/vol (count images) (* 28 28) img-vec)
     (vl/one-hot 10 label-vec)]))
