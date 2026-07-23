# frozen_string_literal: true

module Gsplat
  module Math
    # SSIM forward and analytic VJP with a grouped Gaussian convolution.
    module Ssim
      # Intermediate tensors retained by {.backward}.
      Cache = Data.define(
        :image_a, :image_b, :mu_a, :mu_b, :ssim_map,
        :numerator_mean, :numerator_variance,
        :denominator_mean, :denominator_variance, :kernel, :layout
      )

      module_function

      # rubocop:disable Metrics/AbcSize
      def forward(image_a, image_b, layout:)
        first, resolved_layout = to_nchw(image_a, layout)
        second, second_layout = to_nchw(image_b, layout)
        validate_inputs!(first, second, resolved_layout, second_layout)
        kernel = gaussian_kernel(first.class)
        mu_a = convolve(first, kernel)
        mu_b = convolve(second, kernel)
        variance_a = convolve(first**2, kernel) - (mu_a**2)
        variance_b = convolve(second**2, kernel) - (mu_b**2)
        covariance = convolve(first * second, kernel) - (mu_a * mu_b)
        numerator_mean = (2 * mu_a * mu_b) + (0.01**2)
        numerator_variance = (2 * covariance) + (0.03**2)
        denominator_mean = (mu_a**2) + (mu_b**2) + (0.01**2)
        denominator_variance = variance_a + variance_b + (0.03**2)
        ssim_map = (numerator_mean * numerator_variance) /
                   (denominator_mean * denominator_variance)
        cache = Cache.new(
          image_a: first, image_b: second, mu_a: mu_a, mu_b: mu_b, ssim_map: ssim_map,
          numerator_mean: numerator_mean, numerator_variance: numerator_variance,
          denominator_mean: denominator_mean, denominator_variance: denominator_variance,
          kernel: kernel, layout: resolved_layout
        )
        [ssim_map.mean.to_f, cache]
      end
      # rubocop:enable Metrics/AbcSize

      # rubocop:disable Metrics/AbcSize
      def backward(cache, grad_output)
        factor = grad_output.to_f / cache.ssim_map.size
        denominator = cache.denominator_mean * cache.denominator_variance
        grad_num_mean = factor * cache.numerator_variance / denominator
        grad_num_variance = factor * cache.numerator_mean / denominator
        grad_den_mean = -factor * cache.ssim_map / cache.denominator_mean
        grad_den_variance = -factor * cache.ssim_map / cache.denominator_variance
        grad_mu_a = (2 * cache.mu_b * grad_num_mean) -
                    (2 * cache.mu_b * grad_num_variance) +
                    (2 * cache.mu_a * grad_den_mean) -
                    (2 * cache.mu_a * grad_den_variance)
        grad_mu_b = (2 * cache.mu_a * grad_num_mean) -
                    (2 * cache.mu_a * grad_num_variance) +
                    (2 * cache.mu_b * grad_den_mean) -
                    (2 * cache.mu_b * grad_den_variance)
        grad_cross = 2 * grad_num_variance
        filtered_variance = convolve(grad_den_variance, cache.kernel)
        filtered_cross = convolve(grad_cross, cache.kernel)
        grad_a = convolve(grad_mu_a, cache.kernel) +
                 (2 * cache.image_a * filtered_variance) +
                 (cache.image_b * filtered_cross)
        grad_b = convolve(grad_mu_b, cache.kernel) +
                 (2 * cache.image_b * filtered_variance) +
                 (cache.image_a * filtered_cross)
        [from_nchw(grad_a, cache.layout), from_nchw(grad_b, cache.layout)]
      end
      # rubocop:enable Metrics/AbcSize

      def gaussian_kernel(type)
        coordinates = (-5..5).map { |value| ::Math.exp(-(value**2) / (2 * (1.5**2))) }
        total = coordinates.sum
        vector = coordinates.map { |value| value / total }
        type.cast(vector.product(vector).map { |left, right| left * right }).reshape(11, 11)
      end
      private_class_method :gaussian_kernel

      def convolve(input, kernel)
        output = input.class.zeros(*input.shape)
        height = input.shape[2]
        width = input.shape[3]
        11.times do |kernel_y|
          next if (kernel_y - 5).abs >= height

          destination_y, source_y = shifted_ranges(height, kernel_y - 5)
          11.times do |kernel_x|
            next if (kernel_x - 5).abs >= width

            destination_x, source_x = shifted_ranges(width, kernel_x - 5)
            destination = [true, true, destination_y, destination_x]
            source = [true, true, source_y, source_x]
            output[*destination] = output[*destination] + (input[*source] * kernel[kernel_y, kernel_x])
          end
        end
        output
      end
      private_class_method :convolve

      def shifted_ranges(length, offset)
        if offset >= 0
          [0...(length - offset), offset...length]
        else
          [(-offset)...length, 0...(length + offset)]
        end
      end
      private_class_method :shifted_ranges

      def to_nchw(image, layout)
        unless [Numo::SFloat, Numo::DFloat].include?(image.class)
          raise ArgumentError, "images must be Numo floating-point arrays"
        end

        resolved = layout == :auto ? infer_layout(image) : layout
        converted = case resolved
                    when :nchw then image
                    when :nhwc then image.transpose(0, 3, 1, 2)
                    when :chw then image.reshape(1, *image.shape)
                    when :hwc then image.transpose(2, 0, 1).reshape(1, image.shape[2], image.shape[0], image.shape[1])
                    else raise ArgumentError, "layout must be :auto, :nchw, :nhwc, :chw, or :hwc"
                    end
        [converted, resolved]
      end
      private_class_method :to_nchw

      def from_nchw(image, layout)
        case layout
        when :nchw then image
        when :nhwc then image.transpose(0, 2, 3, 1)
        when :chw then image[0, true, true, true].dup
        when :hwc then image[0, true, true, true].transpose(1, 2, 0)
        end
      end
      private_class_method :from_nchw

      def infer_layout(image)
        return image.shape[-1] <= 4 && image.shape[1] > 4 ? :nhwc : :nchw if image.ndim == 4
        return image.shape[-1] <= 4 && image.shape[0] > 4 ? :hwc : :chw if image.ndim == 3

        raise ShapeError, "expected a 3D or 4D image, got #{image.shape.inspect}"
      end
      private_class_method :infer_layout

      def validate_inputs!(first, second, first_layout, second_layout)
        return if first.shape == second.shape && first_layout == second_layout && first.ndim == 4

        raise ShapeError, "SSIM image shapes/layouts differ: #{first.shape.inspect} and #{second.shape.inspect}"
      end
      private_class_method :validate_inputs!
    end
  end
end
