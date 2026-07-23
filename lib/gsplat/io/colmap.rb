# frozen_string_literal: true

require_relative "colmap_binary"
require_relative "colmap_text"

module Gsplat
  module IO
    # COLMAP sparse reconstruction reader and camera conversion utilities.
    module Colmap
      Camera = Data.define(:id, :model, :width, :height, :params, :intrinsics, :distortion)
      ImageRecord = Data.define(
        :id, :qvec, :tvec, :camera_id, :name, :points2d, :point3d_ids, :rotation, :world_to_camera
      )
      Point3D = Data.define(:id, :xyz, :rgb, :error, :track)
      Dataset = Data.define(:cameras, :images, :points3d, :path)

      MODELS = {
        0 => "SIMPLE_PINHOLE",
        1 => "PINHOLE",
        2 => "SIMPLE_RADIAL",
        4 => "OPENCV"
      }.freeze

      module_function

      # Reads cameras, registered images, and points from a sparse model directory.
      def read(path, data_factor: 1)
        directory = model_directory(path)
        extension = model_extension(directory)
        Dataset.new(
          cameras: read_cameras(File.join(directory, "cameras.#{extension}"), data_factor: data_factor),
          images: read_images(File.join(directory, "images.#{extension}")),
          points3d: read_points3d(File.join(directory, "points3D.#{extension}")),
          path: directory
        )
      end

      def read_cameras(path, data_factor: 1)
        validate_factor!(data_factor)
        raw_records(path, :cameras).transform_values { |record| camera_record(record, data_factor) }
      end

      def read_images(path)
        raw_records(path, :images).transform_values { |record| image_record(record) }
      end

      def read_points3d(path)
        raw_records(path, :points3d).transform_values do |record|
          Point3D.new(
            id: record.fetch(:id),
            xyz: Numo::DFloat.cast(record.fetch(:xyz)),
            rgb: Numo::UInt8.cast(record.fetch(:rgb)),
            error: record.fetch(:error),
            track: record.fetch(:track).freeze
          )
        end
      end

      # Converts a COLMAP wxyz quaternion to a 3x3 world-to-camera rotation.
      def qvec_to_rotmat(qvec)
        quaternion = Numo::DFloat.cast(qvec).reshape(1, 4)
        Math::Quaternion.to_rotmat(quaternion)[0, true, true].dup
      end

      def camera_record(record, factor)
        model = model_name(record.fetch(:model))
        params = record.fetch(:params).map(&:to_f)
        focal, distortion = intrinsics_for(model, params)
        intrinsics = Numo::DFloat[
          [focal[0] / factor, 0, focal[2] / factor],
          [0, focal[1] / factor, focal[3] / factor],
          [0, 0, 1]
        ]
        Camera.new(
          id: record.fetch(:id),
          model: model,
          width: record.fetch(:width).fdiv(factor).to_i,
          height: record.fetch(:height).fdiv(factor).to_i,
          params: Numo::DFloat.cast(params),
          intrinsics: intrinsics,
          distortion: Numo::DFloat.cast(distortion)
        )
      end
      private_class_method :camera_record

      def image_record(record)
        qvec = Numo::DFloat.cast(record.fetch(:qvec))
        tvec = Numo::DFloat.cast(record.fetch(:tvec))
        rotation = qvec_to_rotmat(qvec)
        view = Numo::DFloat.eye(4)
        view[0...3, 0...3] = rotation
        view[0...3, 3] = tvec
        coordinates = record.fetch(:points2d)
        points2d = coordinates.empty? ? Numo::DFloat.zeros(0, 2) : Numo::DFloat.cast(coordinates)
        ImageRecord.new(
          id: record.fetch(:id),
          qvec: qvec,
          tvec: tvec,
          camera_id: record.fetch(:camera_id),
          name: record.fetch(:name),
          points2d: points2d,
          point3d_ids: Numo::Int64.cast(record.fetch(:point3d_ids)),
          rotation: rotation,
          world_to_camera: view
        )
      end
      private_class_method :image_record

      def intrinsics_for(model, params)
        case model
        when "SIMPLE_PINHOLE" then [[params[0], params[0], params[1], params[2]], []]
        when "PINHOLE" then [params.first(4), []]
        when "SIMPLE_RADIAL" then [[params[0], params[0], params[1], params[2]], [params[3]]]
        when "OPENCV" then [params.first(4), params.drop(4)]
        end
      end
      private_class_method :intrinsics_for

      def model_name(value)
        return value if MODELS.value?(value)

        MODELS.fetch(value) { raise NotSupportedError, "unsupported COLMAP camera model #{value.inspect}" }
      end
      private_class_method :model_name

      def raw_records(path, kind)
        data = File.binread(path)
        reader = File.extname(path) == ".bin" ? ColmapBinary : ColmapText
        reader.public_send(kind, data)
      end
      private_class_method :raw_records

      def model_directory(path)
        candidates = [path, File.join(path, "sparse", "0"), File.join(path, "sparse")]
        directory = candidates.find do |candidate|
          File.file?(File.join(candidate, "cameras.bin")) ||
            File.file?(File.join(candidate, "cameras.txt"))
        end
        directory || raise(Gsplat::Error, "COLMAP sparse model not found under #{path}")
      end
      private_class_method :model_directory

      def model_extension(directory)
        return "bin" if %w[cameras images points3D].all? { |name| File.file?(File.join(directory, "#{name}.bin")) }
        return "txt" if %w[cameras images points3D].all? { |name| File.file?(File.join(directory, "#{name}.txt")) }

        raise Gsplat::Error, "COLMAP sparse model files are incomplete in #{directory}"
      end
      private_class_method :model_extension

      def validate_factor!(factor)
        raise ArgumentError, "data_factor must be positive" unless factor.is_a?(Numeric) && factor.positive?
      end
      private_class_method :validate_factor!
    end
  end
end
