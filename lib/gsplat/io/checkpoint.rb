# frozen_string_literal: true

require "json"

module Gsplat
  module IO
    # Portable NPZ checkpoint for parameters, Adam moments, step, and config.
    module Checkpoint
      # Current checkpoint schema version.
      VERSION = 1
      # Separator reserved for structured NPZ entry names.
      SEPARATOR = "___"
      # Loaded checkpoint payload.
      #
      # @!attribute params
      #   @return [Hash{Symbol=>Numo::NArray}]
      # @!attribute optimizer_states
      #   @return [Hash] optimizer moments grouped by optimizer and parameter
      # @!attribute step
      #   @return [Integer]
      # @!attribute config
      #   @return [Hash]
      Snapshot = Data.define(:params, :optimizer_states, :step, :config)

      module_function

      def save(target, params:, optimizers:, step:, config: {})
        raise ArgumentError, "step must be a non-negative integer" unless step.is_a?(Integer) && !step.negative?

        arrays = {
          "checkpoint_version" => Numo::Int32[VERSION],
          "checkpoint_step" => Numo::Int64[step],
          "checkpoint_config_json" => encode_json(config)
        }
        params.each do |name, value|
          arrays[key("param", name)] = Ops::TensorOps.data(value)
        end
        optimizers.each do |optimizer_name, optimizer|
          optimizer.groups.each_key do |group_name|
            state = optimizer.state(group_name)
            prefix = key("optimizer", optimizer_name, group_name)
            arrays["#{prefix}#{SEPARATOR}step"] = Numo::Int64[state.step]
            arrays["#{prefix}#{SEPARATOR}exp_avg"] = state.exp_avg
            arrays["#{prefix}#{SEPARATOR}exp_avg_sq"] = state.exp_avg_sq
          end
        end
        Npy.write_npz(target, arrays)
      end

      # Loads parameters and optimizer state without mutating live objects.
      #
      # @param source [String, #read] NPZ checkpoint path or IO
      # @return [Snapshot]
      def load(source)
        arrays = Npy.read_npz(source)
        version = arrays.fetch("checkpoint_version")[0].to_i
        raise NotSupportedError, "unsupported checkpoint version #{version}" unless version == VERSION

        Snapshot.new(
          params: extract_params(arrays),
          optimizer_states: extract_optimizer_states(arrays),
          step: arrays.fetch("checkpoint_step")[0].to_i,
          config: decode_json(arrays.fetch("checkpoint_config_json"))
        )
      rescue KeyError => e
        raise Gsplat::Error, "invalid checkpoint: #{e.message}"
      end

      # Loads a checkpoint into existing Variables and optimizers.
      #
      # @param source [String, #read] NPZ checkpoint path or IO
      # @param params [Hash{Symbol=>Autograd::Variable}]
      # @param optimizers [Hash{Symbol=>Optim::Adam}]
      # @return [Snapshot]
      def restore!(source, params:, optimizers:)
        snapshot = load(source)
        snapshot.params.each do |name, value|
          params.fetch(name).replace_data!(value.dup)
        end
        snapshot.optimizer_states.each do |optimizer_name, groups|
          optimizer = optimizers.fetch(optimizer_name)
          groups.each do |group_name, state|
            optimizer.load_state!(
              group_name,
              step: state.fetch(:step),
              exp_avg: state.fetch(:exp_avg),
              exp_avg_sq: state.fetch(:exp_avg_sq)
            )
          end
        end
        snapshot
      rescue KeyError => e
        raise Gsplat::Error, "checkpoint target mismatch: #{e.message}"
      end

      def key(*parts)
        values = parts.map(&:to_s)
        invalid = values.find { |value| value.empty? || value.include?(SEPARATOR) || !value.match?(/\A\w+\z/) }
        raise ArgumentError, "invalid checkpoint name #{invalid.inspect}" if invalid

        values.join(SEPARATOR)
      end
      private_class_method :key

      def encode_json(config)
        value = config.respond_to?(:to_h) ? config.to_h : config
        Numo::UInt8.cast(JSON.generate(value).bytes)
      rescue JSON::GeneratorError => e
        raise ArgumentError, "config is not JSON serializable: #{e.message}"
      end
      private_class_method :encode_json

      def decode_json(bytes)
        JSON.parse(bytes.to_a.pack("C*"), symbolize_names: true)
      rescue JSON::ParserError => e
        raise Gsplat::Error, "invalid checkpoint config JSON: #{e.message}"
      end
      private_class_method :decode_json

      def extract_params(arrays)
        prefix = "param#{SEPARATOR}"
        arrays.filter_map do |name, value|
          next unless name.start_with?(prefix)

          [name.delete_prefix(prefix).to_sym, value]
        end.to_h
      end
      private_class_method :extract_params

      def extract_optimizer_states(arrays)
        output = Hash.new { |hash, name| hash[name] = Hash.new { |groups, group| groups[group] = {} } }
        pattern = /\Aoptimizer#{SEPARATOR}(\w+)#{SEPARATOR}(\w+)#{SEPARATOR}(step|exp_avg|exp_avg_sq)\z/
        arrays.each do |name, value|
          match = pattern.match(name)
          next unless match

          state_value = match[3] == "step" ? value[0].to_i : value
          output[match[1].to_sym][match[2].to_sym][match[3].to_sym] = state_value
        end
        output.to_h { |name, groups| [name, groups.to_h] }
      end
      private_class_method :extract_optimizer_states
    end
  end
end
