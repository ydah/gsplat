# frozen_string_literal: true

module Gsplat
  module Compression
    # Deterministic Manhattan K-means for SH coefficient vectors.
    module KMeans
      module_function

      def compress(path, array, clusters:, iterations:, quantization:)
        shape = array.shape
        return empty_metadata(array, quantization) if array.empty?

        vectors = array.reshape(shape[0], array.size / shape[0]).to_a
        cluster_count = [clusters, vectors.length, 65_536].min
        centroids, labels = cluster(vectors, cluster_count, iterations)
        quantized, minimum, maximum = quantize_centroids(centroids, quantization)
        IO::Npy.write_npz(
          path,
          {
            centroids: Numo::UInt8.cast(quantized).reshape(cluster_count, vectors.first.length),
            labels: Numo::UInt16.cast(labels)
          }
        )
        {
          "shape" => shape,
          "dtype" => Quantizer.dtype_name(array.class),
          "mins" => minimum,
          "maxs" => maximum,
          "quantization" => quantization,
          "clusters" => cluster_count
        }
      end

      def decompress(path, metadata)
        shape = metadata.fetch("shape")
        return Quantizer.numeric_type(metadata.fetch("dtype")).zeros(*shape) if shape.inject(1, :*).zero?

        archive = IO::Npy.read_npz(path)
        centroids = decode_centroids(archive.fetch("centroids"), shape, metadata)
        values = archive.fetch("labels").to_a.map { |label| centroids.fetch(label) }.flatten
        Quantizer.numeric_type(metadata.fetch("dtype")).cast(values).reshape(*shape)
      end

      def decode_centroids(values, shape, metadata)
        limit = (1 << metadata.fetch("quantization")) - 1
        minimum = metadata.fetch("mins")
        span = metadata.fetch("maxs") - minimum
        feature_count = shape.drop(1).inject(1, :*)
        values.to_a.flatten.each_slice(feature_count).map do |row|
          row.map { |value| minimum + ((value.to_f / limit) * span) }
        end
      end
      private_class_method :decode_centroids

      def cluster(vectors, count, iterations)
        return [vectors.map(&:dup), (0...vectors.length).to_a] if count == vectors.length

        centroids = Array.new(count) do |index|
          vectors[(index * (vectors.length - 1).fdiv([count - 1, 1].max)).round].dup
        end
        labels = Array.new(vectors.length, 0)
        iterations.times do
          changed = assign!(vectors, centroids, labels)
          update!(vectors, centroids, labels)
          break unless changed
        end
        [centroids, labels]
      end
      private_class_method :cluster

      def assign!(vectors, centroids, labels)
        changed = false
        vectors.each_with_index do |vector, index|
          label = centroids.each_index.min_by do |centroid_index|
            vector.zip(centroids[centroid_index]).sum { |left, right| (left - right).abs }
          end
          changed ||= labels[index] != label
          labels[index] = label
        end
        changed
      end
      private_class_method :assign!

      def update!(vectors, centroids, labels)
        sums = centroids.map { |centroid| Array.new(centroid.length, 0.0) }
        counts = Array.new(centroids.length, 0)
        vectors.each_with_index do |vector, index|
          label = labels[index]
          counts[label] += 1
          vector.each_with_index { |value, feature| sums[label][feature] += value }
        end
        centroids.each_index do |index|
          next if counts[index].zero?

          centroids[index] = sums[index].map { |value| value / counts[index] }
        end
      end
      private_class_method :update!

      def quantize_centroids(centroids, bits)
        values = centroids.flatten
        minimum, maximum = values.minmax
        limit = (1 << bits) - 1
        span = maximum - minimum
        quantized = values.map { |value| span.zero? ? 0 : (((value - minimum) / span) * limit).round }
        [quantized, minimum, maximum]
      end
      private_class_method :quantize_centroids

      def empty_metadata(array, quantization)
        {
          "shape" => array.shape,
          "dtype" => Quantizer.dtype_name(array.class),
          "quantization" => quantization,
          "empty" => true
        }
      end
      private_class_method :empty_metadata
    end
  end
end
