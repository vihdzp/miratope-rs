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
  * All known 3D and 4D [uniform polytopes](https://polytope.miraheze.org/wiki/Uniform_polytope)
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
Miratope doesn't have a completed interface yet and is in the alpha stage, so you'll have to download the source code to do much of anything.
Miratope is written in Rust, so if you don't already have the latest version and its Visual Studio C++ Build tools downloaded then you should do that first. Instructions for downloading can be found here: https://www.rust-lang.org/tools/install.
**You may have to restart your computer for Rust to fully install**.
1. Once you have Rust setup you will have to click the green button here on Github that says "Code".
   * If you already have Github Desktop, you can just click "Open with Github Desktop".
   * If you don't, click "Download ZIP", and once it's done downloading extract the `.zip` file.
2. Next, open a command line. On Windows you can do this by opening Run with `Win+R` and typing `cmd` in the search box.
3. In the command line, first type `cd [FILE PATH]`. If you don't know how to get the file path, in your files go and open the unzipped Miratope file folder, and click on the address bar at the top. Copy the highlighted file path and paste it into the command line in place of `[FILE PATH]`. The last name in the command header should be the name of the folder Miratope is in.
4. Finally, type `cargo run`. It will take a while for the computer to open Miratope for the first time, but after that opening it should be a lot faster. A window should appear, if the version of Miratope you downloaded was a stable one. If it wasn't you'll get an error, and you should wait until the devs have fixed what ever they broke. Once you have completed all the steps you will only need to do step 4 to run Miratope from start up (but if the `[FILE PATH]` changes, you'll need to do step 3 again).

These steps are in place because it would be too costly now to update the application each time a bug is fixed or feature is added. Once Miratope leaves the alpha stage, a simple `.exe` for Version 1 will be provided.

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
