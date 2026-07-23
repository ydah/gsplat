# frozen_string_literal: true

module Gsplat
  # Optional C implementation entrypoint.
  module Native
    class << self
      attr_reader :load_error

      def available?
        @available == true
      end

      def mark_available!
        @available = true
      end

      def mark_unavailable!(error)
        @available = false
        @load_error = error
      end
    end
  end
end

begin
  require_relative "gsplat_native"
  Gsplat::Native.mark_available!
rescue LoadError => e
  Gsplat::Native.mark_unavailable!(e)
end
