# frozen_string_literal: true

module Gsplat
  module Training
    # In-memory multi-view training scene.
    class Scene
      attr_reader :colors, :height, :images, :intrinsics, :names, :points,
                  :scene_scale, :viewmats, :width

      # rubocop:disable Metrics/ParameterLists
      def initialize(viewmats:, intrinsics:, images:, points:, colors:, names: nil, scene_scale: nil)
        # rubocop:enable Metrics/ParameterLists
        @viewmats = viewmats
        @intrinsics = intrinsics
        @images = images
        @points = points
        @colors = colors
        @names = names || Array.new(images.shape[0]) { |index| format("camera_%03d", index) }
        @height = images.shape[1]
        @width = images.shape[2]
        validate!
        @scene_scale = scene_scale || infer_scene_scale
      end

      def self.from_colmap(path, data_factor: 1)
        dataset = IO::Colmap.read(path, data_factor: data_factor)
        records = dataset.images.values.sort_by(&:name)
        views = stack(records.map(&:world_to_camera))
        intrinsics = stack(records.map { |record| dataset.cameras.fetch(record.camera_id).intrinsics })
        image_directory = resolve_image_directory(path, data_factor)
        images = stack(records.map { |record| IO::Image.read(File.join(image_directory, record.name)) })
        points = dataset.points3d.values.sort_by(&:id)
        new(
          viewmats: views,
          intrinsics: intrinsics,
          images: images,
          points: stack_rows(points.map(&:xyz)),
          colors: stack_rows(points.map { |point| Numo::DFloat.cast(point.rgb) / 255.0 }),
          names: records.map(&:name)
        )
      end

      def camera_count
        images.shape[0]
      end

      class << self
        private

        def stack(arrays)
          raise ArgumentError, "cannot stack an empty array list" if arrays.empty?

          type = arrays.first.class
          output = type.zeros(*([arrays.length] + arrays.first.shape))
          arrays.each_with_index do |array, index|
            output[*([index] + Array.new(array.ndim, true))] = array
          end
          output
        end

        def stack_rows(rows)
          raise ArgumentError, "COLMAP point cloud is empty" if rows.empty?

          rows.first.class.cast(rows.map(&:to_a))
        end

        def resolve_image_directory(path, factor)
          scaled = File.join(path, "images_#{factor}")
          return scaled if factor != 1 && Dir.exist?(scaled)

          File.join(path, "images")
        end
      end

      private

      def validate!
        cameras = images.shape[0]
        valid = images.ndim == 4 && images.shape[-1] == 3 &&
                viewmats.shape == [cameras, 4, 4] &&
                intrinsics.shape == [cameras, 3, 3] &&
                points.ndim == 2 && points.shape[1] == 3 &&
                colors.shape == [points.shape[0], 3] &&
                names.length == cameras
        raise ShapeError, "inconsistent multi-view scene arrays" unless valid
      end

      def infer_scene_scale
        centers = viewmats.class.zeros(viewmats.shape[0], 3)
        viewmats.shape[0].times do |index|
          rotation = viewmats[index, 0...3, 0...3]
          translation = viewmats[index, 0...3, 3]
          centers[index, true] = -translation.dot(rotation)
        end
        center = centers.mean(axis: 0)
        scale = Numo::NMath.sqrt(((centers - center)**2).sum(axis: 1)).max.to_f
        scale.positive? ? scale : 1.0
      end
    end
  end
end
