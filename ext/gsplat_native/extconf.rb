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

if try_compile("int main(void) { return 0; }", "-fopenmp")
  $CFLAGS = "#{$CFLAGS} -fopenmp"
  $LDFLAGS = "#{$LDFLAGS} -fopenmp"
  $defs << "-DGSPLAT_OPENMP=1"
end
# rubocop:enable Style/GlobalVars

create_makefile("gsplat/gsplat_native")
