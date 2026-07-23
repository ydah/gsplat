# frozen_string_literal: true

require "zlib"

module Gsplat
  module IO
    # Minimal single-disk ZIP reader/writer for NPZ archives.
    module ZipArchive
      LOCAL_SIGNATURE = 0x04034B50
      CENTRAL_SIGNATURE = 0x02014B50
      END_SIGNATURE = 0x06054B50
      UTF8_FLAG = 0x0800
      METHODS = { stored: 0, deflate: 8 }.freeze
      LOCAL_TEMPLATE = "VvvvvvVVVvv"
      CENTRAL_TEMPLATE = "VvvvvvvVVVvvvvvVV"
      END_TEMPLATE = "VvvvvVVv"

      module_function

      # Encodes named binary entries as a ZIP archive.
      #
      # @param entries [Hash{String=>String}]
      # @param compression [Symbol] :stored or :deflate
      # @return [String] ZIP bytes
      def encode(entries, compression: :deflate)
        method = METHODS.fetch(compression) do
          raise ArgumentError, "unknown compression #{compression.inspect}; expected stored or deflate"
        end
        body = +"".b
        central = +"".b

        entries.each do |name, data|
          name = name.to_s.b
          data = data.b
          compressed = method.zero? ? data : deflate(data)
          metadata = entry_metadata(name, data, compressed, method, body.bytesize)
          body << local_header(metadata) << name << compressed
          central << central_header(metadata) << name
        end

        central_offset = body.bytesize
        body << central
        body << end_record(entries.length, central.bytesize, central_offset)
      end

      # Decodes named binary entries from a ZIP archive.
      #
      # @param archive [String] ZIP bytes
      # @return [Hash{String=>String}]
      def decode(archive)
        archive = archive.b
        entry_count, central_offset = end_metadata(archive)
        cursor = central_offset

        entry_count.times.each_with_object({}) do |_, entries|
          metadata, cursor = parse_central_entry(archive, cursor)
          entries[metadata.fetch(:name)] = extract_entry(archive, metadata)
        end
      end

      def entry_metadata(name, data, compressed, method, offset)
        {
          name: name,
          method: method,
          crc: Zlib.crc32(data),
          compressed_size: compressed.bytesize,
          size: data.bytesize,
          offset: offset
        }
      end
      private_class_method :entry_metadata

      def local_header(metadata)
        [
          LOCAL_SIGNATURE, 20, UTF8_FLAG, metadata.fetch(:method), 0, 0,
          metadata.fetch(:crc), metadata.fetch(:compressed_size), metadata.fetch(:size),
          metadata.fetch(:name).bytesize, 0
        ].pack(LOCAL_TEMPLATE)
      end
      private_class_method :local_header

      def central_header(metadata)
        [
          CENTRAL_SIGNATURE, 20, 20, UTF8_FLAG, metadata.fetch(:method), 0, 0,
          metadata.fetch(:crc), metadata.fetch(:compressed_size), metadata.fetch(:size),
          metadata.fetch(:name).bytesize, 0, 0, 0, 0, 0, metadata.fetch(:offset)
        ].pack(CENTRAL_TEMPLATE)
      end
      private_class_method :central_header

      def end_record(entry_count, central_size, central_offset)
        [
          END_SIGNATURE, 0, 0, entry_count, entry_count, central_size, central_offset, 0
        ].pack(END_TEMPLATE)
      end
      private_class_method :end_record

      def end_metadata(archive)
        offset = archive.rindex([END_SIGNATURE].pack("V"))
        raise Gsplat::Error, "invalid ZIP archive: end record not found" unless offset

        fields = byteslice!(archive, offset, 22).unpack(END_TEMPLATE)
        raise Gsplat::Error, "multi-disk ZIP archives are unsupported" unless fields[1].zero? && fields[2].zero?

        [fields[4], fields[6]]
      end
      private_class_method :end_metadata

      def parse_central_entry(archive, cursor)
        fields = byteslice!(archive, cursor, 46).unpack(CENTRAL_TEMPLATE)
        raise Gsplat::Error, "invalid ZIP central directory" unless fields[0] == CENTRAL_SIGNATURE

        name_length, extra_length, comment_length = fields.values_at(10, 11, 12)
        name = byteslice!(archive, cursor + 46, name_length).force_encoding(Encoding::UTF_8)
        metadata = {
          name: name,
          flags: fields[3],
          method: fields[4],
          crc: fields[7],
          compressed_size: fields[8],
          size: fields[9],
          offset: fields[16]
        }
        [metadata, cursor + 46 + name_length + extra_length + comment_length]
      end
      private_class_method :parse_central_entry

      def extract_entry(archive, metadata)
        raise NotSupportedError, "encrypted ZIP entries are unsupported" unless metadata.fetch(:flags).nobits?(1)

        offset = metadata.fetch(:offset)
        fields = byteslice!(archive, offset, 30).unpack(LOCAL_TEMPLATE)
        raise Gsplat::Error, "invalid ZIP local header" unless fields[0] == LOCAL_SIGNATURE

        data_offset = offset + 30 + fields[9] + fields[10]
        compressed = byteslice!(archive, data_offset, metadata.fetch(:compressed_size))
        data = inflate_entry(compressed, metadata.fetch(:method))
        validate_entry!(data, metadata)
        data
      end
      private_class_method :extract_entry

      def inflate_entry(data, method)
        return data if method.zero?
        return inflate(data) if method == METHODS.fetch(:deflate)

        raise NotSupportedError, "ZIP compression method #{method} is unsupported"
      end
      private_class_method :inflate_entry

      def validate_entry!(data, metadata)
        return if data.bytesize == metadata.fetch(:size) && Zlib.crc32(data) == metadata.fetch(:crc)

        raise Gsplat::Error, "corrupt ZIP entry #{metadata.fetch(:name).inspect}"
      end
      private_class_method :validate_entry!

      def deflate(data)
        stream = Zlib::Deflate.new(Zlib::DEFAULT_COMPRESSION, -Zlib::MAX_WBITS)
        stream.deflate(data, Zlib::FINISH)
      ensure
        stream&.close
      end
      private_class_method :deflate

      def inflate(data)
        stream = Zlib::Inflate.new(-Zlib::MAX_WBITS)
        stream.inflate(data)
      ensure
        stream&.close
      end
      private_class_method :inflate

      def byteslice!(data, offset, length)
        slice = data.byteslice(offset, length)
        return slice if slice&.bytesize == length

        raise Gsplat::Error, "truncated ZIP archive"
      end
      private_class_method :byteslice!
    end
  end
end
