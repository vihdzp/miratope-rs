A renderer for polytopes. Still in alpha.

## What can Miratope do now?
Miratope can generate these classes of polytopes, among others:
* Simplexes
* Hypercubes
* Orthoplexes
* Product prisms
* Polyhedral antiprisms
* Cupolae, cuploids and cupolaic blends

Miratope can also read and export OFF files and GGB files.

## FAQ
### How do I use Miratope?
Miratope doesn't have an interface yet, so you'll have to use the Console to write down JavaScript commands.

Most of the cool generating commands are on the `Build` class. For example, to generate a uniform octagrammic antiprism and render it to the screen, you can use `Build.uniformAntiprism(8, 3).renderTo(mainScene);`.

Here's some other commands to try out:
```javascript
//Renders a cube on screen.
Build.hypercube(3).renderTo(mainScene);

//OFF file for a pentagon-pentagram duoprism.
Product.prism(Build.regularPolygon(5), Build.regularPolygon(5, 2)).saveAsOFF();

//Exports a hexadecachoral prism as a GeoGebra file.
Build.cross(4).extrudeToPrism(1).saveAsGGB();
```

### Where do I get these "OFF files"?
The OFF file format is a format for storing certain kinds of geometric shapes. Although not in widespread use, it has become the standard format for those who investigate polyhedra and polytopes. It was initially meant for the [Geomview software](https://people.sc.fsu.edu/~jburkardt/data/off/off.html), and was later adapted for the [Stella software](https://www.software3d.com/StellaManual.php?prod=stella4D#import). Miratope uses a further generalization of the Stella OFF format for any amount of dimensions.

Miratope does not yet include a library of OFF files. Nevertheless, many of them can be downloaded from [OfficialURL's personal collection](https://drive.google.com/drive/u/0/folders/1nQZ-QVVBfgYSck4pkZ7he0djF82T9MVy). Eventually, they'll be browsable from Miratope itself.

### Why does my OFF file not render?
Provisionally, your OFF file is being loaded into the variable `P`. You have to manually render it using the command `P.renderTo(mainScene);`.

Note that at the moment, this works only for 3D OFF files, and can be somewhat buggy.

### How do I clear the scene?
Use `mainScene.clear();`.

## What's next?
There are lots of planned features for Miratope, some more ambitious than others. You can look at the complete list, along with some ideas on how to implement them [here](https://docs.google.com/document/d/1IEoXR4vmOPELFKosRMIDfDN_M4oaUGWDExdqqDpCwfU/edit?usp=sharing).

The most immediate changes will probably be the following:
* Greater camera control
* Vertex and edge toggling
* Projection type options

Longer term but more substantial changes include:
* Localization
* A minimal working interface
* 4D+ rendering
* Different fill types for faces
* Creation of a dedicated file format and a polytope library
* More operations on polytopes
