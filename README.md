A tool for building and visualizing polytopes. Still in alpha development. Fork of [vihdzp/miratope-rs](https://github.com/vihdzp/miratope-rs) focused on building concrete polytopes like uniforms and nobles.

## Library

The default library contains:
* All [regular polytopes](https://polytope.miraheze.org/wiki/Regular_polytope)
* All 3D [uniform polytopes](https://polytope.miraheze.org/wiki/Uniform_polytope)
* Some 4D and 5D uniform polytopes
* Some [Johnson solids](https://polytope.miraheze.org/wiki/Johnson_solid)

The library is customizable, you can add your own `.off` files. Sometimes you may need to delete or modify the `.folder` files though.

## Features

* Building polytopes
  * Regular polygons, polygonal prisms/antiprisms/duoprisms
  * [Pyramid](https://polytope.miraheze.org/wiki/Pyramid_product), [prism](https://polytope.miraheze.org/wiki/Prism_product), [tegum](https://polytope.miraheze.org/wiki/Tegum_product), [comb](https://polytope.miraheze.org/wiki/Honeycomb_product), and [star](https://en.wikipedia.org/wiki/Star_product) products
  * [Faceting](https://en.wikipedia.org/wiki/Faceting)
* Operations on polytopes
  * [Dual](https://polytope.miraheze.org/wiki/Dual)
  * [Petrial](https://polytope.miraheze.org/wiki/Petrial)
  * [Antiprism](https://polytope.miraheze.org/wiki/Antiprism)
  * [Truncation](https://polytope.miraheze.org/wiki/Wythoffian_operation)
  * [Cross-section](https://polytope.miraheze.org/wiki/Cross-section)
* Analyzing polytopes
  * Miratope can compute various properties of polytopes, such as [flag](https://polytope.miraheze.org/wiki/Flag) count, [orientability](https://polytope.miraheze.org/wiki/Orientability), [circumsphere](https://polytope.miraheze.org/wiki/Circumscribable_polytope), volume, and symmetry group.
  * It can display a list of all [elements](https://polytope.miraheze.org/wiki/Element) of a polytope, grouped by symmetry equivalence.
  * It can split [compounds](https://en.wikipedia.org/wiki/Polytope_compound) into their components.
* Rendering polytopes
  * Miratope can render wireframes and faces (both toggleable) of polytopes in arbitrary dimension, though it can currently only rotate in 3 dimensions. It can render in perspective and orthogonal projection. It can also interactively render cross-sections of polytopes.
* Importing and exporting polytopes in the [`.off` format](https://www.software3d.com/StellaManual.php?prod=stella4D#import)

## How to use

### If you're using 64-bit Windows
Just download the latest release on the right side of the github page, extract the zip, and run `miratope.exe`.

### If you're using a different OS, or you want to modify the source code
Miratope is written in Rust, so if you don't already have the latest version and its Visual Studio C++ Build tools downloaded then you should do that first. Instructions for downloading can be found here: https://www.rust-lang.org/tools/install. **You may have to restart your computer for Rust to fully install**.
1. Once you have Rust setup click the green button here on Github that says "Code".
   * If you already have Github Desktop, you can just click "Open with Github Desktop".
   * If you don't, click "Download ZIP" and once it's done downloading, extract the `.zip` file.
2. Next, open a command line. On Windows you can do this by opening Run with `Win+R` and typing `cmd` in the search box.
3. In the command line, first type `cd [FILE PATH]`. If you don't know how to get the file path, in your files go open the unzipped Miratope file folder, and click on the address bar at the top. Copy the highlighted file path and paste it into the command line in place of `[FILE PATH]`, and press Enter. The last name in the command header should now be the name of the folder Miratope is in.
4. Finally, type `cargo run --release` and hit Enter. It will take a while for the computer to open Miratope for the first time, but after that, opening it should be a lot faster. A window should appear, if the version of Miratope you downloaded was a stable one. If it wasn't, you'll get an error, and you should wait until the devs have fixed whatever they broke.

Once you have completed all the steps you will only need to do step 4 to run Miratope from startup (but if the `[FILE PATH]` changes, you'll need to do step 3 again).

If you have downloaded Miratope previously, updated to the most recent version, and are getting an error like "`error[E0710]: an unknown tool name found in scoped lint`" in the console, this means a crate that Miratope uses has gone out of date. Don't worry about what that means, just make sure your command line has the header pointed at Miratope (like in step 3), and type `rustup update` in the console. Cargo, Rust's built-in file handler, will automatically update all the crates Miratope uses which should fix the issue. If this still doesn't fix it, contact the devs in the `#miratope` channel on [Polytope Discord](https://discord.gg/zMRu7T4).
