use petgraph::{graph::Graph, Undirected};
use petgraph::graph::{NodeIndex, node_index, EdgeIndex, edge_index};
use std::f64;
use regex::Regex;

///Possible types of CD
enum CDTypes {
    Single{graph: Graph<NodeVal, EdgeVal, Undirected>},
    Compound{count: u32, graphs: Vec<Graph<NodeVal, EdgeVal, Undirected>>},
    LaceSimp{lace_len: f64, count: u32, graph: Vec<Graph<NodeVal, EdgeVal, Undirected>>},
    LaceTower{lace_len: f64, count: u32, graphs: Vec<Graph<NodeVal, EdgeVal, Undirected>>},
    LaceRing{lace_len: f64, count: u32, graphs: Vec<Graph<NodeVal, EdgeVal, Undirected>>},
}

///Possible types of Edge Values
enum EdgeVal {
    //Real Numbers 3, 5, 3/4, etc.
    Rational(i64, i64),
    //Infinity ∞
    //Bool marks retrograde ∞
    Inf(bool),
    //No intersection Ø
    Non,
}

///Possible types of Node Values
enum NodeVal {
    ///Unringed Nodes (different from Ringed(0))
    Unringed,
    ///Ringed Nodes, can hold any float
    Ringed(f64),
    ///Snub Nodes, should definitely make this hold a float
    ///TODO: Agree on a way to specify the length in a snub node
    Snub,
}

///Main function for parsing CDs from strings to CDTypes
fn cd_parse(input: &str) -> Option<CDTypes> {
    use CDTypes::*;
    let input = input.replace("-", "");
    match single_cd_parse(&input[..]) {
        Some(graph) => graph,
        None => None,
    }
}

///Parses singleton cds
fn single_cd_parse(diagram: &str) -> Option<Graph<NodeVal, EdgeVal, Undirected>> {
    let caret = Caret {
        diagram: diagram,
        graph: Graph::new_undirected(),
        index: 0,
        edgemem: (None, None, None),
    };
    //Initial node
    caret.create_node();
    //The rest
    while caret.index < caret.diagram.len() {
        caret.read_edge();
        caret.create_node();
        caret.make_edge();
    };
    Some(caret.graph)
}

/*
///Parses compound and lace cds
fn multi_cd_parse(diagram: &str) -> Option<Vec<Graph<EdgeVal, EdgeVal, Undirected>>> {
    if input.contains("&")||input.contains("#") {
        if Regex::new(r#"&#([a-zA-z]|\(([a-zA-z]|\d+(/\d+)?\)))[tr]?$"#).unwrap().is_match(&input) {
            match &input[input.len()-1..] {
                "t" => /*Lace Tower*/ ,
                "r" => /*Lace Ring*/,
                _ => /*Lace Simplex*/,
            }
        } else {
            /*Invalid Lace*/
            None
        }
    } else {
        /*Compound*/
    
    }
}
*/

///Packages important information needed to interpret CDs
struct Caret {
    diagram: Vec<&str>,
    graph: Graph<NodeVal, EdgeVal, Undirected>,
    index: usize,
    edgemem: (Option<NodeIndex>, Option<NodeIndex>, Option<EdgeVal>),
}

///Operations that are commonly done to parse CDs
impl Caret {
    ///Reads and creates a node
    fn create_node(&self) -> Option<NodeIndex> {
        match &self.diagram[self.index..self.index+1] {
            "(" => {
                match self.diagram[self.index..].find(")") {
                    Some(ind) => close = ind,
                    None => None,
                }
            },
            _ => close = self.index+1,
        };
        let node: &str = &self.diagram[self.index..close+1];
        self.index += node.len();
        match node_to_val(node) {
            Some(val) => Some(self.graph.add_node(val)),
            None => None //Invalid Node!,
        }
    }

    ///Reads an edge from a CD and stores into edgemem
    fn read_edge(&self) -> Option<Caret> {

    }

    ///Creates an edge from edgemem
    fn make_edge(&self) -> Option<EdgeIndex> {

    }

    ///Reads a virtual node
    fn read_virt(&self) -> Option<Caret> {

    }

    ///Reads a lace suffix
    fn read_suff(&self) -> Option<Caret> {

    }
}

///Converts string slices of cd edges to wrapped EdgeVals
fn edge_to_val(c: &str) -> Option<EdgeVal> {
    use EdgeVal::*;
    if Regex::new(r#"^\d+/\d+$"#).unwrap().is_match(c) {
        let bar = c.find("/").unwrap();
        return Some(Rational(c[..bar].parse::<i64>().unwrap(), c[bar+1..].parse::<i64>().unwrap()))
    } else if Regex::new(r#"^\d+$"#).unwrap().is_match(c) {
        return Some(Rational(c.parse::<i64>().unwrap(), 1i64))
    } else {
        match c {
            "∞" => return Some(Inf(false)),
            "∞'" => return Some(Inf(true)),
            "Ø" => return Some(Non),
            _ => None
        };
    };
}

//Converts string slices of cd node values to wrapped NodeVals
fn node_to_val(c: &str) -> Option<NodeVal> {
    use NodeVal::*;
    if (c.len() == 3 || c.len() == 1) & !(Regex::new(r#"([^oxqfvhkuwFe]|\([^oxqfvhkuwFe]\))"#).unwrap().is_match(c)) {
        //For established letter-values
        let c = c.replace("(", "");
        let c = c.replace(")", "");
        match &c[..] {
            "o" => return Some(Unringed),
            "v" => return Some(Ringed((5f64.sqrt()-1f64)/2f64)),
            "x" => return Some(Ringed(1f64)),
            "q" => return Some(Ringed(2f64.sqrt())),
            "f" => return Some(Ringed((5f64.sqrt()+1f64)/2f64)),
            "h" => return Some(Ringed(3f64.sqrt())),
            "k" => return Some(Ringed((2f64.sqrt()+2f64).sqrt())),
            "u" => return Some(Ringed(2f64)),
            "w" => return Some(Ringed(2f64.sqrt()+1f64)),
            "F" => return Some(Ringed((5f64.sqrt()+3f64)/2f64)),
            "e" => return Some(Ringed(3f64.sqrt()+1f64)),
            "Q" => return Some(Ringed(2f64.sqrt()*2f64)),
            "d" => return Some(Ringed(3f64)),
            "V" => return Some(Ringed(5f64.sqrt()+1f64)),
            "U" => return Some(Ringed(2f64.sqrt()+2f64)),
            "A" => return Some(Ringed((5f64.sqrt()+5f64)/4f64)),
            "X" => return Some(Ringed(2f64.sqrt()*2f64+1f64)),
            "B" => return Some(Ringed(5f64.sqrt()+2f64)),
            "s" => return Some(Snub),
            _ => return None
        };
    } else if Regex::new(r#"^\(\d(\.\d+)?\)$"#).unwrap().is_match(c) {
        //For custom lengths
        let c = c.replace("(", "");
        let c = c.replace(")", "");    
        return Some(Ringed(c.parse::<f64>().unwrap()))
    } else {
        return None
    };
}

//Inverts the value held by a EdgeVal
fn num_retro(val: EdgeVal) -> EdgeVal {
    use EdgeVal::*;
    match val {
        Rational(n, d) => Rational(n, n-d),
        Inf(dir) => {let ret = !dir; Inf(ret)},
        Non => Non,
    }
}

//(ONLY FOR MEANT FOR DEBUGGING), prints the contexts of a graph to the console
fn cd_inspect(graph: &Graph<NodeVal, EdgeVal, Undirected>) {
    let ncount = &graph.node_count();
    let ecount = &graph.edge_count();
    // Print node and edge count
    println!("{} Nodes", ncount);
    println!("{} Edges", ecount);
    for (i, n) in graph.node_indices().enumerate() {
        node_inspect(&graph, i, n);
    }
    for (i, n) in graph.edge_indices().enumerate() {
        edge_inspect(&graph, i, n);
    }

}

//(ONLY FOR MEANT FOR DEBUGGING), prints the contents of one node to the console
fn node_inspect(graph: &Graph<NodeVal, EdgeVal, Undirected>, i: usize, n: NodeIndex) {
    match &graph.node_weight(n) {
        Option::None => println!("Node {} carries nothing", i),
        Option::Some(c) => {
            match c {
                NodeVal::Unringed => println!("Node {} is unringed", i),
                NodeVal::Ringed(n) => println!("Node {} carries {}", i, n),
                NodeVal::Snub => println!("Node {} is snub", i),
            };
        },
    };
}

//(ONLY FOR MEANT FOR DEBUGGING), prints the contents of one edge to the console
fn edge_inspect(graph: &Graph<EdgeVal, EdgeVal, Undirected>, i: usize, n: EdgeIndex) {
    match &graph.edge_endpoints(n) {
        Option::None => println!("What? Edge {} doesn't connect any nodes", i),
        Option::Some((e1, e2)) => println!("Edge {} connects Nodes {} and {}", i, e1.index(), e2.index()),
    };
    match &graph.edge_weight(n) {
        Option::None => println!("Edge {} carries nothing", i),
        Option::Some(c) => {
            match c {
                EdgeVal::Rational(n, d) => println!("Edge {} carries {}/{}", i, n, d),
                EdgeVal::Inf(false) => println!("Edge {} carries prograde ∞", i),
                EdgeVal::Inf(true) => println!("Edge {} carries retrograde ∞", i),
                EdgeVal::Non => println!("Edge {} carries Ø", i),
            };
        },
    };
}