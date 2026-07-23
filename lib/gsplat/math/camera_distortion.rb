# frozen_string_literal: true

module Gsplat
  module Math
    # OpenCV pinhole distortion and equidistant fisheye projection.
    module CameraDistortion
      module_function

      # rubocop:disable Metrics/ParameterLists
      def project_camera(means, intrinsics, camera_model, radial: nil, tangential: nil, thin_prism: nil)
        # rubocop:enable Metrics/ParameterLists
        projected = means.class.zeros(means.shape[0], 2)
        jacobians = means.class.zeros(means.shape[0], 2, 3)
        means.shape[0].times do |index|
          point = means[index, true].to_a.map(&:to_f)
          projected[index, true] = project_point(
            point, intrinsics, camera_model, radial, tangential, thin_prism
          )
          jacobians[index, true, true] = numerical_jacobian(
            point, intrinsics, camera_model, radial, tangential, thin_prism, means.class
          )
        end
        [projected, jacobians]
      end

      # rubocop:disable Metrics/AbcSize, Metrics/ParameterLists
      def project_point(point, intrinsics, camera_model, radial, tangential, thin_prism)
        # rubocop:enable Metrics/AbcSize, Metrics/ParameterLists
        x_coord = point[0] / point[2]
        y_coord = point[1] / point[2]
        normalized = if camera_model.to_s == "fisheye"
                       fisheye_normalized = fisheye(x_coord, y_coord, radial)
                       tangential_prism(
                         *fisheye_normalized, *fisheye_normalized, tangential, thin_prism
                       )
                     else
                       pinhole_normalized = pinhole(x_coord, y_coord, radial)
                       tangential_prism(
                         *pinhole_normalized, x_coord, y_coord, tangential, thin_prism
                       )
                     end
        [
          (intrinsics[0, 0].to_f * normalized[0]) +
            (intrinsics[0, 1].to_f * normalized[1]) + intrinsics[0, 2].to_f,
          (intrinsics[1, 0].to_f * normalized[0]) +
            (intrinsics[1, 1].to_f * normalized[1]) + intrinsics[1, 2].to_f
        ]
      end

      def pinhole(x_coord, y_coord, radial)
        coefficients = Array.new(6, 0.0)
        radial&.each_with_index { |value, index| coefficients[index] = value.to_f }
        radius2 = (x_coord * x_coord) + (y_coord * y_coord)
        radius4 = radius2 * radius2
        radius6 = radius4 * radius2
        numerator = 1 + (coefficients[0] * radius2) +
                    (coefficients[1] * radius4) + (coefficients[2] * radius6)
        denominator = 1 + (coefficients[3] * radius2) +
                      (coefficients[4] * radius4) + (coefficients[5] * radius6)
        scale = numerator / denominator
        [x_coord * scale, y_coord * scale]
      end
      private_class_method :pinhole

      def fisheye(x_coord, y_coord, radial)
        radius = ::Math.sqrt((x_coord * x_coord) + (y_coord * y_coord))
        return [x_coord, y_coord] if radius < 1e-12

        theta = ::Math.atan(radius)
        theta2 = theta * theta
        coefficients = Array.new(4, 0.0)
        radial&.each_with_index { |value, index| coefficients[index] = value.to_f }
        polynomial = 1.0
        power = theta2
        coefficients.each do |coefficient|
          polynomial += coefficient * power
          power *= theta2
        end
        scale = theta * polynomial / radius
        [x_coord * scale, y_coord * scale]
      end
      private_class_method :fisheye

      # rubocop:disable Metrics/ParameterLists
      def tangential_prism(base_x, base_y, x_coord, y_coord, tangential, thin_prism)
        # rubocop:enable Metrics/ParameterLists
        p1, p2 = tangential ? tangential.map(&:to_f) : [0.0, 0.0]
        s1, s2, s3, s4 = thin_prism ? thin_prism.map(&:to_f) : [0.0, 0.0, 0.0, 0.0]
        radius2 = (x_coord * x_coord) + (y_coord * y_coord)
        radius4 = radius2 * radius2
        [
          base_x + (2 * p1 * x_coord * y_coord) + (p2 * (radius2 + (2 * x_coord * x_coord))) +
            (s1 * radius2) + (s2 * radius4),
          base_y + (p1 * (radius2 + (2 * y_coord * y_coord))) + (2 * p2 * x_coord * y_coord) +
            (s3 * radius2) + (s4 * radius4)
        ]
      end
      private_class_method :tangential_prism

      # rubocop:disable Metrics/ParameterLists
      def numerical_jacobian(point, intrinsics, camera_model, radial, tangential, thin_prism, type)
        # rubocop:enable Metrics/ParameterLists
        epsilon = type == Numo::DFloat ? 1e-6 : 1e-3
        2.times.map do |output_axis|
          3.times.map do |input_axis|
            positive = point.dup
            negative = point.dup
            positive[input_axis] += epsilon
            negative[input_axis] -= epsilon
            projected_positive = project_point(
              positive, intrinsics, camera_model, radial, tangential, thin_prism
            )
            projected_negative = project_point(
              negative, intrinsics, camera_model, radial, tangential, thin_prism
            )
            (projected_positive[output_axis] - projected_negative[output_axis]) / (2 * epsilon)
          end
        end
      end
      private_class_method :numerical_jacobian
    end
  end
end
