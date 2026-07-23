# frozen_string_literal: true

output = File.expand_path("../test/fixtures/fit_image.ppm", __dir__)
size = 128
pixels = Array.new(size * size * 3)
size.times do |row|
  size.times do |column|
    offset = ((row * size) + column) * 3
    x_coord = column.to_f / (size - 1)
    y_coord = row.to_f / (size - 1)
    pixels[offset] = (255 * x_coord).round
    pixels[offset + 1] = (255 * y_coord).round
    pixels[offset + 2] = (255 * (0.25 + (0.5 * x_coord * y_coord))).round
  end
end
File.binwrite(output, "P6\n#{size} #{size}\n255\n" + pixels.pack("C*"))
