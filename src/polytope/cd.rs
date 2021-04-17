use petgraph::graph::NodeIndex;
use petgraph::{graph::Graph, Undirected};
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
        caret.read_edge()?;

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
                        self.graph.add_node(node_to_val(&chars)?);
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
                self.graph.add_node(node_to_val(&chars)?);
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

///Converts Vecs of chars to wrapped EdgeVals
fn edge_to_val(raw: Vec<char>) -> Option<EdgeVal> {
    use EdgeVal::*;
    let mut raw_iter = raw.iter();
    let mut edge = Vec::new();
    let mut number: Option<i64> = None;
    let mut rat = false;
    let c = *raw_iter.next()?;
    //Starting character
    edge.push(c);

    //If the value is Rational or an Integer
    if c.is_digit(10) {
        for &c in raw_iter {
            //If the "/" is encountered
            if c == '/' {
                //Set the flag for Rationals
                rat = true;

                //Parse and save the numerator
                number = match edge.into_iter().collect::<String>().parse::<i64>() {
                    Ok(number) => Some(number),
                    _ => return None,
                };

                //Reset what's being read
                edge = Vec::new();
            };

            //Wasn't a special character, can continue
            edge.push(c);
        }
        //When you're at the end
        //Parse the end value
        let val = match edge.into_iter().collect::<String>().parse::<i64>() {
            Ok(number) => number,
            _ => return None,
        };

        //If this was a Rational edge, the end value would be the denominator
        //If this wasn't a Rational edge, the end value would be the numerator
        if rat {
            Some(Rational(number?, val))
        } else {
            Some(Rational(val, 1i64))
        }
    } else {
        //For miscellaneous edge symbols,
        //just read the whole thing as a string
        let c = edge.into_iter().collect::<String>();

        match &c[..] {
            "∞" => Some(Inf(false)),
            "∞'" => Some(Inf(true)),
            "'∞" => Some(Inf(true)),
            "Ø" => Some(Non),
            _ => None,
        }
    }
}

///Converts Vecs of chars to wrapped NodeVals
fn node_to_val(raw: &[char]) -> Option<NodeVal> {
    use NodeVal::*;
    let mut raw_iter = raw.iter();
    let mut node = Vec::new();
    let mut c = *raw_iter.next()?;
    //Skips to the next character
    //if the first one an opening parenthesis
    if c == '(' {
        c = *raw_iter.next()?
    }
    //Starting character
    node.push(c);

    //If the node has a custom value
    if c.is_digit(10) {
        for &c in raw_iter {
            //When you're at the end
            if c == ')' {
                //Parse the value
                let val = match node.into_iter().collect::<String>().parse::<f64>() {
                    Ok(number) => number,
                    _ => return None,
                };

                return Some(Ringed(val));
            }

            //This character was normal, can continue
            node.push(c);
        }

        //Something's wrong, return None
        None
    } else {
        //Check shortchord values
        Some(Ringed(match c {
            'o' => return Some(Unringed),
            's' => return Some(Snub),
            'v' => (5f64.sqrt() - 1f64) / 2f64,
            'x' => 1f64,
            'q' => 2f64.sqrt(),
            'f' => (5f64.sqrt() + 1f64) / 2f64,
            'h' => 3f64.sqrt(),
            'k' => (2f64.sqrt() + 2f64).sqrt(),
            'u' => 2f64,
            'w' => 2f64.sqrt() + 1f64,
            'F' => (5f64.sqrt() + 3f64) / 2f64,
            'e' => 3f64.sqrt() + 1f64,
            'Q' => 2f64.sqrt() * 2f64,
            'd' => 3f64,
            'V' => 5f64.sqrt() + 1f64,
            'U' => 2f64.sqrt() + 2f64,
            'A' => (5f64.sqrt() + 5f64) / 4f64,
            'X' => 2f64.sqrt() * 2f64 + 1f64,
            'B' => 5f64.sqrt() + 2f64,
            _ => return None,
        }))
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
