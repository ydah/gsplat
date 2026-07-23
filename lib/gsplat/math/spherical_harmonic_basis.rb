# frozen_string_literal: true

module Gsplat
  module Math
    # Real SH bases and normalized-direction derivatives in gsplat/Inria order.
    module SphericalHarmonicBasis
      C0 = 0.2820947917738781
      C1 = 0.48860251190292
      C2 = [1.092548430592079, 0.3153915652525201, 0.5462742152960395].freeze
      C3 = [0.5900435899266435, 2.890611442640554, 0.4570457994644658,
            0.3731763325901154, 1.445305721320277].freeze
      C4 = [2.5033429417967046, -1.770130769779931, 0.9461746957575601,
            -0.6690465435572892, 0.1057855469152043, 0.47308734787878,
            0.6258357354491763].freeze

      module_function

      # Evaluates bases and derivatives with respect to already-normalized directions.
      #
      # Constants and ordering match gsplat 1.5.3 cuda/_torch_impl.py.
      #
      # @param directions [Numo::NArray] [N,3] normalized directions
      # @param degree [Integer] 0..4
      # @param basis_count [Integer] coefficient basis width
      # @return [Array<Numo::NArray>] bases [N,K], derivatives [N,K,3]
      def evaluate(directions, degree, basis_count)
        count = directions.shape[0]
        bases = directions.class.zeros(count, basis_count)
        derivatives = directions.class.zeros(count, basis_count, 3)
        x_coord = directions[true, 0]
        y_coord = directions[true, 1]
        z_coord = directions[true, 2]
        bases[true, 0] = C0
        fill_degree_one!(bases, derivatives, x_coord, y_coord, z_coord) if degree >= 1
        fill_degree_two!(bases, derivatives, x_coord, y_coord, z_coord) if degree >= 2
        fill_degree_three!(bases, derivatives, x_coord, y_coord, z_coord) if degree >= 3
        fill_degree_four!(bases, derivatives, x_coord, y_coord, z_coord) if degree >= 4
        [bases, derivatives]
      end

      def fill_degree_one!(bases, derivatives, x_coord, y_coord, z_coord)
        bases[true, 1] = -C1 * y_coord
        bases[true, 2] = C1 * z_coord
        bases[true, 3] = -C1 * x_coord
        derivatives[true, 1, 1] = -C1
        derivatives[true, 2, 2] = C1
        derivatives[true, 3, 0] = -C1
      end
      private_class_method :fill_degree_one!

      # rubocop:disable Metrics/AbcSize
      def fill_degree_two!(bases, derivatives, x_coord, y_coord, z_coord)
        bases[true, 4] = C2[0] * x_coord * y_coord
        bases[true, 5] = -C2[0] * y_coord * z_coord
        bases[true, 6] = C2[1] * ((3 * (z_coord**2)) - 1)
        bases[true, 7] = -C2[0] * x_coord * z_coord
        bases[true, 8] = C2[2] * ((x_coord**2) - (y_coord**2))
        derivatives[true, 4, 0] = C2[0] * y_coord
        derivatives[true, 4, 1] = C2[0] * x_coord
        derivatives[true, 5, 1] = -C2[0] * z_coord
        derivatives[true, 5, 2] = -C2[0] * y_coord
        derivatives[true, 6, 2] = 6 * C2[1] * z_coord
        derivatives[true, 7, 0] = -C2[0] * z_coord
        derivatives[true, 7, 2] = -C2[0] * x_coord
        derivatives[true, 8, 0] = 2 * C2[2] * x_coord
        derivatives[true, 8, 1] = -2 * C2[2] * y_coord
      end
      private_class_method :fill_degree_two!

      def fill_degree_three!(bases, derivatives, x_coord, y_coord, z_coord)
        bases[true, 9] = -C3[0] * y_coord * ((3 * (x_coord**2)) - (y_coord**2))
        bases[true, 10] = C3[1] * x_coord * y_coord * z_coord
        bases[true, 11] = -C3[2] * y_coord * ((5 * (z_coord**2)) - 1)
        bases[true, 12] = C3[3] * z_coord * ((5 * (z_coord**2)) - 3)
        bases[true, 13] = -C3[2] * x_coord * ((5 * (z_coord**2)) - 1)
        bases[true, 14] = C3[4] * z_coord * ((x_coord**2) - (y_coord**2))
        bases[true, 15] = -C3[0] * x_coord * ((x_coord**2) - (3 * (y_coord**2)))
        fill_degree_three_derivatives!(derivatives, x_coord, y_coord, z_coord)
      end
      private_class_method :fill_degree_three!

      def fill_degree_three_derivatives!(derivatives, x_coord, y_coord, z_coord)
        derivatives[true, 9, 0] = -6 * C3[0] * x_coord * y_coord
        derivatives[true, 9, 1] = -3 * C3[0] * ((x_coord**2) - (y_coord**2))
        derivatives[true, 10, 0] = C3[1] * y_coord * z_coord
        derivatives[true, 10, 1] = C3[1] * x_coord * z_coord
        derivatives[true, 10, 2] = C3[1] * x_coord * y_coord
        derivatives[true, 11, 1] = -C3[2] * ((5 * (z_coord**2)) - 1)
        derivatives[true, 11, 2] = -10 * C3[2] * y_coord * z_coord
        derivatives[true, 12, 2] = C3[3] * ((15 * (z_coord**2)) - 3)
        derivatives[true, 13, 0] = -C3[2] * ((5 * (z_coord**2)) - 1)
        derivatives[true, 13, 2] = -10 * C3[2] * x_coord * z_coord
        derivatives[true, 14, 0] = 2 * C3[4] * x_coord * z_coord
        derivatives[true, 14, 1] = -2 * C3[4] * y_coord * z_coord
        derivatives[true, 14, 2] = C3[4] * ((x_coord**2) - (y_coord**2))
        derivatives[true, 15, 0] = -3 * C3[0] * ((x_coord**2) - (y_coord**2))
        derivatives[true, 15, 1] = 6 * C3[0] * x_coord * y_coord
      end
      private_class_method :fill_degree_three_derivatives!

      def fill_degree_four!(bases, derivatives, x_coord, y_coord, z_coord)
        x2 = x_coord**2
        y2 = y_coord**2
        z2 = z_coord**2
        bases[true, 16] = C4[0] * x_coord * y_coord * (x2 - y2)
        bases[true, 17] = C4[1] * y_coord * z_coord * ((3 * x2) - y2)
        bases[true, 18] = C4[2] * x_coord * y_coord * ((7 * z2) - 1)
        bases[true, 19] = C4[3] * y_coord * z_coord * ((7 * z2) - 3)
        bases[true, 20] = C4[4] * ((z2 * ((35 * z2) - 30)) + 3)
        bases[true, 21] = C4[3] * x_coord * z_coord * ((7 * z2) - 3)
        bases[true, 22] = C4[5] * (x2 - y2) * ((7 * z2) - 1)
        bases[true, 23] = C4[1] * x_coord * z_coord * (x2 - (3 * y2))
        bases[true, 24] = C4[6] * ((x2**2) - (6 * x2 * y2) + (y2**2))
        fill_degree_four_derivatives!(derivatives, x_coord, y_coord, z_coord)
      end
      private_class_method :fill_degree_four!

      def fill_degree_four_derivatives!(derivatives, x_coord, y_coord, z_coord)
        x_squared = x_coord**2
        y_squared = y_coord**2
        z_squared = z_coord**2
        derivatives[true, 16, 0] = C4[0] * y_coord * ((3 * x_squared) - y_squared)
        derivatives[true, 16, 1] = C4[0] * x_coord * (x_squared - (3 * y_squared))
        derivatives[true, 17, 0] = 6 * C4[1] * x_coord * y_coord * z_coord
        derivatives[true, 17, 1] = 3 * C4[1] * z_coord * (x_squared - y_squared)
        derivatives[true, 17, 2] = C4[1] * y_coord * ((3 * x_squared) - y_squared)
        derivatives[true, 18, 0] = C4[2] * y_coord * ((7 * z_squared) - 1)
        derivatives[true, 18, 1] = C4[2] * x_coord * ((7 * z_squared) - 1)
        derivatives[true, 18, 2] = 14 * C4[2] * x_coord * y_coord * z_coord
        derivatives[true, 19, 1] = C4[3] * z_coord * ((7 * z_squared) - 3)
        derivatives[true, 19, 2] = C4[3] * y_coord * ((21 * z_squared) - 3)
        derivatives[true, 20, 2] = C4[4] * ((140 * (z_coord**3)) - (60 * z_coord))
        derivatives[true, 21, 0] = C4[3] * z_coord * ((7 * z_squared) - 3)
        derivatives[true, 21, 2] = C4[3] * x_coord * ((21 * z_squared) - 3)
        derivatives[true, 22, 0] = 2 * C4[5] * x_coord * ((7 * z_squared) - 1)
        derivatives[true, 22, 1] = -2 * C4[5] * y_coord * ((7 * z_squared) - 1)
        derivatives[true, 22, 2] = 14 * C4[5] * z_coord * (x_squared - y_squared)
        derivatives[true, 23, 0] = 3 * C4[1] * z_coord * (x_squared - y_squared)
        derivatives[true, 23, 1] = -6 * C4[1] * x_coord * y_coord * z_coord
        derivatives[true, 23, 2] = C4[1] * x_coord * (x_squared - (3 * y_squared))
        derivatives[true, 24, 0] = 4 * C4[6] * x_coord * (x_squared - (3 * y_squared))
        derivatives[true, 24, 1] = 4 * C4[6] * y_coord * (y_squared - (3 * x_squared))
      end
      # rubocop:enable Metrics/AbcSize
      private_class_method :fill_degree_four_derivatives!
    end
  end
end
