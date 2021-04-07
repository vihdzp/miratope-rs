use petgraph::graph::NodeIndex;
use petgraph::{graph::Graph, Undirected};
use regex::Regex;
use std::f64;
use std::{fmt::Display, str::Chars};

/// Possible types of CD
struct Cd(
    // Single {
    Graph<NodeVal, EdgeVal, Undirected>,
    // },
    /*
    Compound{count: u32, graphs: Vec<Graph<NodeVal, EdgeVal, Undirected>>},
    LaceSimp{lace_len: f64, count: u32, graph: Vec<Graph<NodeVal, EdgeVal, Undirected>>},
    LaceTower{lace_len: f64, count: u32, graphs: Vec<Graph<NodeVal, EdgeVal, Undirected>>},
    LaceRing{lace_len: f64, count: u32, graphs: Vec<Graph<NodeVal, EdgeVal, Undirected>>},
    */
);

impl Cd {
    pub fn dimension(&self) -> usize {
        self.0.node_count()
    }
}

impl Display for Cd {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Prints node and edge counts.
        writeln!(f, "{} Nodes", self.0.node_count())?;
        writeln!(f, "{} Edges", self.0.edge_count())?;

        // Prints out nodes.
        for (i, n) in self.0.raw_nodes().iter().enumerate() {
            write!(f, "Node {}: ", i)?;
            NodeVal::fmt(&n.weight, f)?;
        }

        // Prints out edges.
        for (i, e) in self.0.raw_edges().iter().enumerate() {
            write!(f, "Edge {}: ", i)?;
            EdgeVal::fmt(&e.weight, f)?;
        }

        Ok(())
    }
}

/// Possible types of Edge Values
#[derive(Clone, Copy)]
enum EdgeVal {
    //Real Numbers 3, 5, 3/4, etc.
    Rational(i64, i64),
    //Infinity ∞
    //Bool marks retrograde ∞
    Inf(bool),
    //No intersection Ø
    Non,
}

impl Display for EdgeVal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EdgeVal::Rational(n, d) => writeln!(f, "Edge carries {}/{}", n, d),
            EdgeVal::Inf(false) => writeln!(f, "Edge carries prograde ∞"),
            EdgeVal::Inf(true) => writeln!(f, "Edge carries retrograde ∞"),
            EdgeVal::Non => writeln!(f, "Edge carries Ø"),
        }
    }
}

/// Possible types of node values.
enum NodeVal {
    ///Unringed Nodes (different from Ringed(0))
    Unringed,
    ///Ringed Nodes, can hold any float
    Ringed(f64),
    ///Snub Nodes, should definitely make this hold a float
    ///TODO: Agree on a way to specify the length in a snub node
    Snub,
}

impl Display for NodeVal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NodeVal::Unringed => writeln!(f, "Node is unringed"),
            NodeVal::Ringed(x) => writeln!(f, "Node carries {}", x),
            NodeVal::Snub => writeln!(f, "Node is snub"),
        }
    }
}

/// Main function for parsing CDs from strings.
fn cd_parse(input: &str) -> Option<Cd> {
    let mut caret = Caret {
        diagram: input.chars(),
        graph: Graph::new_undirected(),
        edge_mem: EdgeMem {
            node: None,
            edge: None,
        },
    };

    // Reads through the diagram.
    loop {
        caret.create_node()?;

        if caret.make_edge().is_none() {
            return Some(Cd(caret.graph));
        }
    }
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

/// Packages important information needed to interpret CDs
struct Caret<'a> {
    diagram: Chars<'a>,
    graph: Graph<NodeVal, EdgeVal, Undirected>,
    edge_mem: EdgeMem,
}

/// Stores the indices of the node and edge of the next edge in the graph. This
/// is used in order to handle virtual nodes. A new edge will be added to the
/// graph only when both fields of the `EdgeMem` are full, and we're reading a
/// new node.
struct EdgeMem {
    node: Option<NodeIndex>,
    edge: Option<EdgeVal>,
}

/// Operations that are commonly done to parse CDs
impl<'a> Caret<'a> {
    /// Reads the next node in the diagram. Returns `Some(())` if succesful, and
    /// `None` otherwise.
    fn create_node(&mut self) -> Option<()> {
        let mut chars = Vec::new();
        let c = self.diagram.next()?;
        chars.push(c);

        // The index of the new node.
        let mut new_node = NodeIndex::new(self.graph.node_count() - 1);

        match c {
            // If the node is various characters inside parentheses.
            '(' => {
                // We read through the diagram until we find ')'.
                while let Some(c) = self.diagram.next() {
                    chars.push(c);
                    if c == ')' {
                        // Converts the read characters into a value and adds the node to the graph.
                        self.graph
                            .add_node(node_to_val(&chars.into_iter().collect::<String>())?);
                        break;
                    }
                }

                // If the parenthesis isn't closed.
                return None;
            }

            // If the node is a virtual node.
            '*' => {
                // Reads the index the virtual node refers to.
                let idx = NodeIndex::new(
                    match u8::from_str_radix(&self.diagram.next()?.to_string(), 36) {
                        // *0 to *9 aren't valid syntax.
                        Ok(0..=9) => return None,

                        // A virtual node, from *a to *z.
                        Ok(idx) => (idx - 10) as usize,

                        // Something else.
                        Err(_) => return None,
                    },
                );
                // Sets the index of the new node to be where the virtual node is refering to.
                new_node = idx
            }

            // If the node is a single character.
            _ => {
                // Converts the read characters into a value and adds the node to the graph.
                self.graph
                    .add_node(node_to_val(&chars.into_iter().collect::<String>())?);
            }
        }

        // If the EdgeMem is full, we add a new edge to the graph.
        if let EdgeMem {
            node: Some(prev_node),
            edge: Some(edge),
        } = &self.edge_mem
        {
            self.graph.add_edge(*prev_node, new_node, *edge);
        };

        // Resets the EdgeMem so that it only has the node that was just found.
        self.edge_mem = EdgeMem {
            node: Some(new_node),
            edge: None,
        };

        Some(())
    }

    /// Reads an edge from a CD and stores into edgemem
    fn read_edge(&self) -> Option<()> {
        todo!()
    }

    /// Creates an edge from edgemem
    fn make_edge(&self) -> Option<()> {
        todo!()
    }

    /*
    ///Reads a lace suffix
    fn read_suff(&self) -> Option<Caret> {}
    */
}

///Converts string slices of cd edges to wrapped EdgeVals
fn edge_to_val(_: &str) -> Option<EdgeVal> {
    /* use EdgeVal::*;
    if Regex::new(r#"^\d+/\d+$"#).unwrap().is_match(c) {
        let bar = c.find("/").unwrap();
        return Some(Rational(
            c[..bar].parse::<i64>().unwrap(),
            c[bar + 1..].parse::<i64>().unwrap(),
        ));
    } else if Regex::new(r#"^\d+$"#).unwrap().is_match(c) {
        return Some(Rational(c.parse::<i64>().unwrap(), 1i64));
    } else {
        match c {
            "∞" => return Some(Inf(false)),
            "∞'" => return Some(Inf(true)),
            "Ø" => return Some(Non),
            _ => None,
        };
    };*/

    todo!()
}

//Converts string slices of cd node values to wrapped NodeVals
fn node_to_val(c: &str) -> Option<NodeVal> {
    use NodeVal::*;
    if (c.len() == 3 || c.len() == 1)
        & !(Regex::new(r#"([^oxqfvhkuwFe]|\([^oxqfvhkuwFe]\))"#)
            .unwrap()
            .is_match(c))
    {
        //For established letter-values
        let c = c.replace("(", "");
        let c = c.replace(")", "");
        match &c[..] {
            "o" => Some(Unringed),
            "v" => Some(Ringed((5f64.sqrt() - 1f64) / 2f64)),
            "x" => Some(Ringed(1f64)),
            "q" => Some(Ringed(2f64.sqrt())),
            "f" => Some(Ringed((5f64.sqrt() + 1f64) / 2f64)),
            "h" => Some(Ringed(3f64.sqrt())),
            "k" => Some(Ringed((2f64.sqrt() + 2f64).sqrt())),
            "u" => Some(Ringed(2f64)),
            "w" => Some(Ringed(2f64.sqrt() + 1f64)),
            "F" => Some(Ringed((5f64.sqrt() + 3f64) / 2f64)),
            "e" => Some(Ringed(3f64.sqrt() + 1f64)),
            "Q" => Some(Ringed(2f64.sqrt() * 2f64)),
            "d" => Some(Ringed(3f64)),
            "V" => Some(Ringed(5f64.sqrt() + 1f64)),
            "U" => Some(Ringed(2f64.sqrt() + 2f64)),
            "A" => Some(Ringed((5f64.sqrt() + 5f64) / 4f64)),
            "X" => Some(Ringed(2f64.sqrt() * 2f64 + 1f64)),
            "B" => Some(Ringed(5f64.sqrt() + 2f64)),
            "s" => Some(Snub),
            _ => None,
        }
    } else if Regex::new(r#"^\(\d(\.\d+)?\)$"#).unwrap().is_match(c) {
        //For custom lengths
        let c = c.replace("(", "").replace(")", "");

        Some(Ringed(c.parse::<f64>().unwrap()))
    } else {
        None
    }
}

/// Inverts the value held by a EdgeVal
fn num_retro(val: EdgeVal) -> EdgeVal {
    use EdgeVal::*;
    match val {
        Rational(n, d) => Rational(n, n - d),
        Inf(dir) => {
            let ret = !dir;
            Inf(ret)
        }
        Non => Non,
    }
}
