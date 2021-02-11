mod polytope;

/// Creates an [[https://polytope.miraheze.org/wiki/Antiprism | antiprism]].
pub fn antiprism_with_height(n:u32, d:u32, h:f64) -> PolytopeC {

}

pub fn antiprism(n:u32, d:u32) -> PolytopeC {
    antiprism_with_height(n, d, 1)
}
