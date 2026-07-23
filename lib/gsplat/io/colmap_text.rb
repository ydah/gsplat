# frozen_string_literal: true

module Gsplat
  module IO
    # Decoder for COLMAP sparse-model text files.
    module ColmapText
      module_function

      def cameras(data)
        content_lines(data).to_h do |line|
          fields = line.split
          id = Integer(fields.shift, 10)
          model = fields.shift
          width = Integer(fields.shift, 10)
          height = Integer(fields.shift, 10)
          [id, { id: id, model: model, width: width, height: height, params: fields.map { |item| Float(item) } }]
        end
      end

      def images(data)
        lines = data.lines(chomp: true)
        output = {}
        until lines.empty?
          metadata = next_metadata_line(lines)
          break unless metadata

          fields = metadata.split
          id = Integer(fields.shift, 10)
          qvec = fields.shift(4).map { |item| Float(item) }
          tvec = fields.shift(3).map { |item| Float(item) }
          camera_id = Integer(fields.shift, 10)
          name = fields.join(" ")
          points_line = lines.shift.to_s.strip
          points2d, point3d_ids = parse_image_points(points_line)
          output[id] = {
            id: id, qvec: qvec, tvec: tvec, camera_id: camera_id,
            name: name, points2d: points2d, point3d_ids: point3d_ids
          }
        end
        output
      end

      def points3d(data)
        content_lines(data).to_h do |line|
          fields = line.split
          id = Integer(fields.shift, 10)
          xyz = fields.shift(3).map { |item| Float(item) }
          rgb = fields.shift(3).map { |item| Integer(item, 10) }
          error = Float(fields.shift)
          track = fields.each_slice(2).map { |image_id, point_index| [Integer(image_id, 10), Integer(point_index, 10)] }
          [id, { id: id, xyz: xyz, rgb: rgb, error: error, track: track }]
        end
      end

      def content_lines(data)
        data.lines(chomp: true).map(&:strip).reject { |line| line.empty? || line.start_with?("#") }
      end
      private_class_method :content_lines

      def next_metadata_line(lines)
        until lines.empty?
          line = lines.shift.strip
          return line unless line.empty? || line.start_with?("#")
        end
        nil
      end
      private_class_method :next_metadata_line

      def parse_image_points(line)
        fields = line.split
        coordinates = []
        ids = []
        fields.each_slice(3) do |x_coord, y_coord, point_id|
          raise Gsplat::Error, "invalid COLMAP image point row" unless point_id

          coordinates << [Float(x_coord), Float(y_coord)]
          ids << Integer(point_id, 10)
        end
        [coordinates, ids]
      end
      private_class_method :parse_image_points
    end
  end
end
