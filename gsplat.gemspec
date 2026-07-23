# frozen_string_literal: true

require_relative "lib/gsplat/version"

Gem::Specification.new do |spec|
  spec.name = "gsplat"
  spec.version = Gsplat::VERSION
  spec.authors = ["Yudai Takada"]
  spec.email = ["t.yudai92@gmail.com"]

  spec.summary = "Differentiable 3D Gaussian splatting for Ruby"
  spec.description = "A CPU differentiable 3D Gaussian splatting renderer, trainer, and IO toolkit " \
                     "with Numo::NArray and optional OpenMP acceleration."
  spec.homepage = "https://github.com/ydah/gsplat"
  spec.license = "Apache-2.0"
  spec.required_ruby_version = ">= 3.2.0"
  spec.metadata["rubygems_mfa_required"] = "true"
  spec.metadata["source_code_uri"] = spec.homepage
  spec.metadata["documentation_uri"] = "https://rubydoc.info/gems/gsplat/#{spec.version}"

  release_files = %w[LICENSE.txt README.md gsplat.gemspec]
  release_roots = %w[docs/ examples/ ext/ lib/]
  spec.files = IO.popen(%w[git ls-files -z], chdir: __dir__, err: IO::NULL) do |ls|
    ls.readlines("\x0", chomp: true).select do |file|
      release_files.include?(file) || file.start_with?(*release_roots)
    end
  end
  spec.files |= release_files.select { |file| File.file?(File.join(__dir__, file)) }
  spec.extra_rdoc_files = %w[README.md LICENSE.txt]
  spec.bindir = "exe"
  spec.executables = spec.files.grep(%r{\Aexe/}) { |f| File.basename(f) }
  spec.extensions = ["ext/gsplat_native/extconf.rb"]
  spec.require_paths = ["lib"]

  spec.add_dependency "numo-narray", ">= 0.9", "< 1.0"
end
