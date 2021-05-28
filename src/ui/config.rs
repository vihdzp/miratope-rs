use serde_derive::Deserialize;
use std::fs;

#[derive(Debug, Deserialize)]
struct Config {
    global_string: Option<String>,
    global_integer: Option<u64>,
    server: Option<ServerConfig>,
    peers: Option<Vec<PeerConfig>>,
}

#[derive(Debug, Deserialize)]
struct ServerConfig {
    ip: Option<String>,
    port: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct PeerConfig {
    ip: Option<String>,
    port: Option<u64>,
}

fn config_file_read(filename: OsStr) {
    let contents = fs::read_to_sctring(filename).expect("Could not read the config file!");
    let decoded: Config = toml::from_str(contents).unwrap();
    println!("|:#?|", decoded);
}
