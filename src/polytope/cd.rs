use petgraph::{graph::Graph, Undirected};
use petgraph::graph::{NodeIndex, EdgeIndex};
use std::f64;
use regex::Regex;

//Possible numbers and non-numbers that could appear in CD edges
enum NumExt {
    //Real Numbers 3, 5, 3/4, etc.
    Rational(i64, i64),
    //Infinity ∞
    //Bool marks retrograde ∞
    Inf(bool),
    //No intersection Ø
    Non,
}

enum Node {
    Unringed,
    Ringed(f64),
    Snub(f64),
}

//Types of Graphs for polytopes
enum CDGraph {
    //Classic
    Single{graph: Graph<NumExt, NumExt, Undirected>},
    Compound{count: u32, graphs: Vec<Graph<NumExt, NumExt, Undirected>>},
    LaceTower{lace_len: NumExt, count: u32, graphs: Vec<Graph<NumExt, NumExt, Undirected>>},
    LaceRing{lace_len: NumExt, count: u32, graphs: Vec<Graph<NumExt, NumExt, Undirected>>},
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

fn create_link(diagram: &str, caret: u32) -> Option<NumExt> {
    let node = diagram[caret..diagram.find(")")+1];
    if caret = 0 {
        caret += node.len();
        match node_to_val(node) {
            Some(val) => graph.add_node(val),
            None => return None,
        }
    }
    //TODO: implement edge creation
    match node_to_val(node) {
        Some(val) => graph.add_node(val),
        None => return None,
    }
}

fn flag_edge(diagram: &str, caret: u32) -> Option<NumExt> {
    
}

fn single_cd_parse(diagram: &str) -> Option<Graph<NumExt, NumExt, Undirected>> {
    use NumExt::*;
    let mut graph = Graph::new_undirected();
    let mut caret: u32 = 0;
    let mut mem: (u32, u32, NumExt) = (0, 0, Non);
    //Initial node
    /*
    match create_link(diagram, caret) {

    }
    */
    //The rest
    while caret < diagram.len() {
        /*
        match flag_edge(diagram, caret) {

        }
        */
    };
    graph
}

/*
fn multi_cd_parse(diagram: &str) -> Option<Vec<Graph<NumExt, NumExt, Undirected>>> {

}
*/

fn edge_to_val(c: &str) -> Option<NumExt> {
    use NumExt::*;
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
//Converts string slices of cd node values to wrapped floats
fn node_to_val(c: &str) -> Option<f64> {
    if (c.len() == 3 || c.len() == 1) & !(Regex::new(r#"([^oxqfvhkuwFe]|\([^oxqfvhkuwFe]\))"#).unwrap().is_match(c)) {
        //For established letter-values
        let c = c.replace("(", "");
        let c = c.replace(")", "");
        match &c[..] {
            "o" => return Some(0f64),
            "x" => return Some(1f64),
            "q" => return Some(2f64.sqrt()),
            "f" => return Some((5f64.sqrt()+1f64)/2f64),
            "v" => return Some((5f64.sqrt()-1f64)/2f64),
            "h" => return Some(3f64.sqrt()),
            "k" => return Some((2f64.sqrt()+2f64).sqrt()),
            "u" => return Some(2f64),
            "w" => return Some(2f64.sqrt()+1f64),
            "F" => return Some((5f64.sqrt()+3f64)/2f64),
            "e" => return Some(3f64.sqrt()+1f64),
            _ => return None
        };
    } else if Regex::new(r#"^\(\d(\.\d+)?\)$"#).unwrap().is_match(c) {
        //For custom lengths
        let c = c.replace("(", "");
        let c = c.replace(")", "");    
        return Some(c.parse::<f64>().unwrap())
    } else {
        return None
    };
}

//Inverts the value held by a NumExt
fn num_retro(val: NumExt) -> NumExt {
    use NumExt::*;
    match val {
        Rational(n, d) => Rational(n, n-d),
        Inf(dir) => {let ret = !dir; Inf(ret)},
        Non => Non,
    }
}

//(ONLY FOR MEANT FOR DEBUGGING), prints the contexts of a graph to the console
fn cd_inspect(graph: &Graph<NumExt, NumExt, Undirected>) {
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
fn node_inspect(graph: &Graph<NumExt, NumExt, Undirected>, i: usize, n: NodeIndex) {
    match &graph.node_weight(n) {
        Option::None => println!("Node {} carries nothing", i),
        Option::Some(c) => {
            match c {
                NumExt::Rational(n, d) => println!("Node {} carries {}/{}", i, n, d),
                NumExt::Inf(false) => println!("Node {} carries prograde ∞", i),
                NumExt::Inf(true) => println!("Node {} carries retrograde ∞", i),
                NumExt::Non => println!("Node {} carries Ø", i),
            };
        },
    };
}

//(ONLY FOR MEANT FOR DEBUGGING), prints the contents of one edge to the console
fn edge_inspect(graph: &Graph<NumExt, NumExt, Undirected>, i: usize, n: EdgeIndex) {
    match &graph.edge_endpoints(n) {
        Option::None => println!("What? Edge {} doesn't connect any nodes", i),
        Option::Some((e1, e2)) => println!("Edge {} connects Nodes {} and {}", i, e1.index(), e2.index()),
    };
    match &graph.edge_weight(n) {
        Option::None => println!("Edge {} carries nothing", i),
        Option::Some(c) => {
            match c {
                NumExt::Rational(n, d) => println!("Edge {} carries {}/{}", i, n, d),
                NumExt::Inf(false) => println!("Edge {} carries prograde ∞", i),
                NumExt::Inf(true) => println!("Edge {} carries retrograde ∞", i),
                NumExt::Non => println!("Edge {} carries Ø", i),
            };
        },
    };
}