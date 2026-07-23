# frozen_string_literal: true

require_relative "lib/gsplat/version"

Gem::Specification.new do |spec|
  spec.name = "gsplat"
  spec.version = Gsplat::VERSION
  spec.authors = ["Yudai Takada"]
  spec.email = ["t.yudai92@gmail.com"]

  spec.summary = "Differentiable 3D Gaussian splatting for Ruby"
  spec.description = "A Numo::NArray implementation of differentiable 3D Gaussian splatting."
  spec.license = "MIT"
  spec.required_ruby_version = ">= 3.2.0"
  spec.metadata["rubygems_mfa_required"] = "true"

  # Specify which files should be added to the gem when it is released.
  # The `git ls-files -z` loads the files in the RubyGem that have been added into git.
  gemspec = File.basename(__FILE__)
  spec.files = IO.popen(%w[git ls-files -z], chdir: __dir__, err: IO::NULL) do |ls|
    ls.readlines("\x0", chomp: true).reject do |f|
      (f == gemspec) ||
        f.start_with?(*%w[bin/ Gemfile .gitignore .rspec spec/ test/ .github/])
    end
  end
  spec.bindir = "exe"
  spec.executables = spec.files.grep(%r{\Aexe/}) { |f| File.basename(f) }
  spec.extensions = ["ext/gsplat_native/extconf.rb"]
  spec.require_paths = ["lib"]

  spec.add_dependency "numo-narray", ">= 0.9", "< 1.0"
end
