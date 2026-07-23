# frozen_string_literal: true

module Gsplat
  module IO
    # Decoder for COLMAP sparse-model binary files.
    module ColmapBinary
      CAMERA_PARAM_COUNTS = {
        0 => 3, 1 => 4, 2 => 4, 3 => 5, 4 => 8, 5 => 8,
        6 => 12, 7 => 5, 8 => 4, 9 => 5, 10 => 12
      }.freeze

      # Bounds-checked little-endian binary cursor.
      class Cursor
        def initialize(data)
          @data = data
          @offset = 0
        end

        def scalar(format, width)
          bytes = @data.byteslice(@offset, width)
          raise Gsplat::Error, "truncated COLMAP binary file" unless bytes&.bytesize == width

          @offset += width
          bytes.unpack1(format)
        end

        def array(format, width, count)
          Array.new(count) { scalar(format, width) }
        end

        def cstring
          finish = @data.index("\0", @offset)
          raise Gsplat::Error, "unterminated COLMAP image name" unless finish

          value = @data.byteslice(@offset, finish - @offset)
          @offset = finish + 1
          value.force_encoding(Encoding::UTF_8)
        end
      end

      module_function

      def cameras(data)
        cursor = Cursor.new(data)
        Array.new(cursor.scalar("Q<", 8)).to_h do
          id = cursor.scalar("l<", 4)
          model = cursor.scalar("l<", 4)
          count = CAMERA_PARAM_COUNTS.fetch(model) do
            raise NotSupportedError, "unknown COLMAP camera model id #{model}"
          end
          width = cursor.scalar("Q<", 8)
          height = cursor.scalar("Q<", 8)
          [id, { id: id, model: model, width: width, height: height, params: cursor.array("E", 8, count) }]
        end
      end

      def images(data)
        cursor = Cursor.new(data)
        Array.new(cursor.scalar("Q<", 8)).to_h do
          id = cursor.scalar("l<", 4)
          qvec = cursor.array("E", 8, 4)
          tvec = cursor.array("E", 8, 3)
          camera_id = cursor.scalar("l<", 4)
          name = cursor.cstring
          count = cursor.scalar("Q<", 8)
          points2d, point3d_ids = image_points(cursor, count)
          [id, { id: id, qvec: qvec, tvec: tvec, camera_id: camera_id, name: name,
                 points2d: points2d, point3d_ids: point3d_ids }]
        end
      end

      def points3d(data)
        cursor = Cursor.new(data)
        Array.new(cursor.scalar("Q<", 8)).to_h do
          id = cursor.scalar("Q<", 8)
          xyz = cursor.array("E", 8, 3)
          rgb = cursor.array("C", 1, 3)
          error = cursor.scalar("E", 8)
          track_count = cursor.scalar("Q<", 8)
          track = Array.new(track_count) { [cursor.scalar("l<", 4), cursor.scalar("l<", 4)] }
          [id, { id: id, xyz: xyz, rgb: rgb, error: error, track: track }]
        end
      end

      def image_points(cursor, count)
        coordinates = []
        ids = []
        count.times do
          coordinates << cursor.array("E", 8, 2)
          raw_id = cursor.scalar("Q<", 8)
          ids << (raw_id == ((1 << 64) - 1) ? -1 : raw_id)
        end
        [coordinates, ids]
      end
      private_class_method :image_points
    end
  end
end
