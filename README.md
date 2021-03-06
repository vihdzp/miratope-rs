A renderer for polytopes, spinned off from [Miratope JS](https://github.com/OfficialURL/miratope).
Still in alpha development.

## What can Miratope do now?
Miratope can already load some polytopes and find out various properties
about them, and it can operate on them via various methods. We're still in
the early stages of porting the original Miratope's functionality, though.

## What are Miratope's goals?
We plan to eventually support all of the original Miratope's features,
as well as the following:

* Various families of polytopes to build and render
  * All [regular polytopes](https://polytope.miraheze.org/wiki/Regular_polytope)
  * All 3D and 4D known [uniform polytopes](https://polytope.miraheze.org/wiki/Uniform_polytope)
  * Many of the known [CRFs](https://polytope.miraheze.org/wiki/Convex_regular-faced_polytope)
* Many operations to apply to these polytopes
  * [Duals](https://polytope.miraheze.org/wiki/Dual)
  * [Petrials](https://polytope.miraheze.org/wiki/Petrial)
  * [Prism products](https://polytope.miraheze.org/wiki/Prism_product)
  * [Tegum products](https://polytope.miraheze.org/wiki/Tegum_product)
  * [Pyramid products](https://polytope.miraheze.org/wiki/Pyramid_product)
  * [Convex hulls](https://polytope.miraheze.org/wiki/Convex_hull)
* Loading and saving into various formats
  * Support for the [Stella OFF format](https://www.software3d.com/StellaManual.php?prod=stella4D#import)
  * Support for the [GeoGebra GGB format](https://wiki.geogebra.org/en/Reference:File_Format)
* Localization
  * Automatic name generation in various languages for many shapes

## How do I use Miratope?
Miratope doesn't have an interface yet, so you'll have to download the source code to do much of anything.

## Where do I get these "OFF files"?
The OFF file format is a format for storing certain kinds of geometric shapes.
Although not in widespread use, it has become the standard format for those who investigate polyhedra and polytopes.
It was initially meant for the [Geomview software](https://people.sc.fsu.edu/~jburkardt/data/off/off.html),
and was later adapted for the [Stella software](https://www.software3d.com/StellaManual.php?prod=stella4D#import).
Miratope uses a further generalization of the Stella OFF format for any amount of dimensions.

Miratope does not yet include a library of OFF files. Nevertheless, many of them can be downloaded from
[OfficialURL's personal collection](https://drive.google.com/drive/u/0/folders/1nQZ-QVVBfgYSck4pkZ7he0djF82T9MVy).
Eventually, they'll be browsable from Miratope itself.

## Why is the rendering buggy?
Proper rendering, even in 3D is a work in progress.