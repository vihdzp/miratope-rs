use petgraph::{graph::Graph, Undirected};
use petgraph::graph::{NodeIndex, EdgeIndex};
use std::f64;
use regex::Regex;

//Possible numbers and non-numbers that could appear in CD edges
enum EdgeVal {
    //Real Numbers 3, 5, 3/4, etc.
    Rational(i64, i64),
    //Infinity ∞
    //Bool marks retrograde ∞
    Inf(bool),
    //No intersection Ø
    Non,
}

enum NodeVal {
    Unringed,
    Ringed(f64),
    Snub,
}

//Types of Graphs for polytopes
enum CDGraph {
    //Classic
    Single{graph: Graph<NodeVal, EdgeVal, Undirected>},
    Compound{count: u32, graphs: Vec<Graph<NodeVal, EdgeVal, Undirected>>},
    LaceTower{lace_len: f64, count: u32, graphs: Vec<Graph<NodeVal, EdgeVal, Undirected>>},
    LaceRing{lace_len: f64, count: u32, graphs: Vec<Graph<NodeVal, EdgeVal, Undirected>>},
}

//Main function for parsing CDs from strings to CDGraphs
fn cd_parse(input: &String) -> Option<CDGraph> {
    let input = input.replace("-", "");
    if Regex::new(r#"[a-zA-Z][a-zA-Z]"#).unwrap().is_match(&input) {
        if input.contains("&")||input.contains("#") {
            if Regex::new(r#"&#([a-zA-z]|\(([a-zA-z]|\d+(/\d+)?\)))[tr]?$"#).unwrap().is_match(&input) {
                match &input[input.len()-1..] {
                    "t" => {//Lace Tower
                    return Some(CDGraph::Single{graph: Graph::new_undirected()})},
                    "r" => {//Lace Ring
                    return Some(CDGraph::Single{graph: Graph::new_undirected()})},
                    _ => {//Lace Prism
                    return Some(CDGraph::Single{graph: Graph::new_undirected()})},
                };
            } else {
                //Invalid Lace
                return None
            };
        } else {
            //Compound
            return Some(CDGraph::Single{graph: Graph::new_undirected()})
        };
    } else {
        //Normal
        return Some(CDGraph::Single{graph: Graph::new_undirected()})
    };
}

//Reads and creates a node
fn create_node(graph: Graph<NodeVal, EdgeVal, Undirected>, diagram: &str, caret: usize) -> Option<NodeIndex> {
    let close: usize = 0;
    match &diagram[caret..caret+1] {
        "(" => {
            match diagram[caret..].find(")") {
                Some(ind) => close = ind,
                None => return None
            }
        },
        _ => close = caret+1,
    }
    let node: &str = &diagram[caret..close+1];
    caret += node.len();
    match node_to_val(node) {
        Some(val) => graph.add_node(val),
        None => None //Invalid Node!,
}

//Creates an edge from mem
fn make_edge(graph: Graph<NodeVal, EdgeVal, Undirected>, diagram: &str, caret: u32) -> Option<EdgeIndex> {

}

//Reads an edge from a CD and stores into mem
fn flag_edge(diagram: &str, caret: u32) -> Option<(NodeIndex, NodeIndex, EdgeValue)> {
    
}

//Parses singleton cds
fn single_cd_parse(diagram: &str) -> Option<Graph<NodeVal, EdgeVal, Undirected>> {
    let mut graph: Graph<NodeVal, EdgeVal, Undirected> = Graph::new_undirected();
    let mut caret: usize = 0;
    let mut mem: Option<(u32, u32, EdgeVal)> = None;
    //Initial node

    //Make Node
    let close = diagram.find(")");
    if close == None {
      //Invalid Diagram!
    };
    
    //The rest
    while caret < diagram.len() {
        //Read Edge
        //Make Node
        //Make Edge
    };
    Some(graph)
}

/*
//Parses compound and lace cds
fn multi_cd_parse(diagram: &str) -> Option<Vec<Graph<EdgeVal, EdgeVal, Undirected>>> {

}
*/

//Converts string slices of cd edges to wrapped EdgeVals
fn edge_to_val(c: &str) -> Option<EdgeVal> {
    use EdgeVal::*;
    if Regex::new(r#"^\d+/\d+$"#).unwrap().is_match(c) {
        let bar = c.find("/").unwrap();
        return Some(Rational((c[..bar].parse::<i64>().unwrap()),(c[bar+1..].parse::<i64>().unwrap())))
    } else if Regex::new(r#"^\d+$"#).unwrap().is_match(c) {
        return Some(Rational(c.parse::<f64>().unwrap(), 1i64))
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
fn cd_inspect(graph: &Graph<EdgeVal, EdgeVal, Undirected>) {
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
fn node_inspect(graph: &Graph<EdgeVal, EdgeVal, Undirected>, i: usize, n: NodeIndex) {
    match &graph.node_weight(n) {
        Option::None => println!("Node {} carries nothing", i),
        Option::Some(c) => {
            match c {
                EdgeVal::Rational(n, d) => println!("Node {} carries {}/{}", i, n, d),
                EdgeVal::Inf(false) => println!("Node {} carries prograde ∞", i),
                EdgeVal::Inf(true) => println!("Node {} carries retrograde ∞", i),
                EdgeVal::Non => println!("Node {} carries Ø", i),
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