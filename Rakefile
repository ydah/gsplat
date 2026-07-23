# frozen_string_literal: true

require "bundler/gem_tasks"
require "fileutils"
require "rake/testtask"
require "rbconfig"

Rake::TestTask.new(:test) do |task|
  task.libs << "lib"
  task.libs << "test"
  task.pattern = "test/**/*_test.rb"
  task.warning = true
end

task default: :test

desc "Build the optional native extension"
task :compile do
  extension_dir = File.expand_path("ext/gsplat_native", __dir__)
  ruby = RbConfig.ruby
  Dir.chdir(extension_dir) do
    system(ruby, "extconf.rb", exception: true)
    system("make", exception: true)
  end
  extension = Dir[File.join(extension_dir, "gsplat_native.{bundle,so,dll}")].first
  raise "native extension output was not produced" unless extension

  FileUtils.cp(extension, File.expand_path("lib/gsplat", __dir__))
end
