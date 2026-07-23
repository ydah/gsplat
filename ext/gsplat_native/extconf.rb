# frozen_string_literal: true

require "mkmf"
require "rubygems"

# mkmf configures extension builds through these documented globals.
# rubocop:disable Style/GlobalVars
numo_spec = Gem::Specification.find_by_name("numo-narray")
numo_include = File.join(numo_spec.full_gem_path, "lib", "numo")
$INCFLAGS = "-I#{numo_include} #{$INCFLAGS}"
$CFLAGS = "#{$CFLAGS} -O3"

abort "numo/narray.h is required" unless have_header("numo/narray.h")

openmp_enabled = try_compile("int main(void) { return 0; }", "-fopenmp")
if openmp_enabled
  $CFLAGS = "#{$CFLAGS} -fopenmp"
  $LDFLAGS = "#{$LDFLAGS} -fopenmp"
elsif RUBY_PLATFORM.include?("darwin")
  prefixes = [
    ENV.fetch("LIBOMP_PREFIX", nil),
    "/opt/homebrew/opt/libomp",
    "/usr/local/opt/libomp"
  ].compact
  prefix = prefixes.find { |candidate| File.file?(File.join(candidate, "include", "omp.h")) }
  if prefix
    original_flags = [$INCFLAGS, $CFLAGS, $LDFLAGS]
    $INCFLAGS = "-I#{File.join(prefix, 'include')} #{$INCFLAGS}"
    $CFLAGS = "#{$CFLAGS} -Xpreprocessor -fopenmp"
    $LDFLAGS = "#{$LDFLAGS} -L#{File.join(prefix, 'lib')} -lomp"
    openmp_enabled = try_link("#include <omp.h>\nint main(void) { return omp_get_max_threads() < 1; }")
    $INCFLAGS, $CFLAGS, $LDFLAGS = original_flags unless openmp_enabled
  end
end

$defs << "-DGSPLAT_OPENMP=1" if openmp_enabled
# rubocop:enable Style/GlobalVars

create_makefile("gsplat/gsplat_native")
