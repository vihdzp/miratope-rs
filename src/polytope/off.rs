use std::fs::read;
use std::num::ParseFloatError;
use std::io::Result as IoResult;
use std::path::Path;

use super::PolytopeSerde;

fn strip_comments(src: &mut String) {
    while let Some(com_start) = src.chars().position(|c| c == '#') {
        let line_end = com_start
            + src[com_start..]
                .chars()
                .position(|c| c == '\n')
                .unwrap_or_else(|| src.len() - com_start - 1);
        src.drain(com_start..=line_end);
    }
}

fn homogenize_whitespace(src: &mut String) {
    *src = src.trim().to_string();
    let mut new_src = String::with_capacity(src.capacity());
    let mut was_ws = false;

    for c in src.chars() {
        if c.is_whitespace() && !was_ws {
            new_src.push(' ');
            was_ws = true;
        } else {
            new_src.push(c);
            was_ws = false;
        }
    }

    *src = new_src;
}

fn read_u32(chars: &mut impl Iterator<Item = char>) -> u32 {
    let mut n = 0;

    while let Some(c @ '0'..='9') = chars.next() {
        n *= 10;
        n += (c as u32) - ('0' as u32);
    }

    n
}

fn read_f64(chars: &mut (impl Clone + Iterator<Item = char>)) -> Result<f64, ParseFloatError> {
    // get the index of the next whitespace character
    let next_ws = chars
        .clone()
        .enumerate()
        .find_map(|(i, c)| if c.is_whitespace() {
            Some(i)
        } else {
            None
        });
    // if the first character is whitespace
    let s = if next_ws == Some(0) {
        // skip it and try again
        chars.next();
        return read_f64(chars);
    // otherwise, if there is an index to a whitespace
    } else if let Some(i) = next_ws {
        // take the characters up to the whitespace
        let slice = chars.clone().take(i - 1).collect();

        // skip past the whitespace in chars
        for _ in 0..i {
            chars.next();
        }
        
        slice
    } else {
        // 
        let mut s = String::new();
        s.extend(chars);
        s
    };

    s.parse()
}

fn get_elems_nums(chars: &mut impl Iterator<Item = char>, dims: usize) -> Vec<usize> {
    let mut num_elems = Vec::with_capacity(dims);

    for _ in 0..dims {
        chars.next();
        num_elems.push(read_u32(chars) as usize);
    }

    // 2-elements go before 1-elements, we're undoing that
    if dims >= 3 {
        num_elems.swap(1, 2);
    }

    num_elems
}

pub fn polytope_from_off_src(mut src: String) -> PolytopeSerde {
    strip_comments(&mut src);
    homogenize_whitespace(&mut src);
    let mut chars = src.chars();
    let mut dim = read_u32(&mut chars);

    // This assumes our OFF file isn't 0D.
    if dim == 0 {
        dim = 3;
    }

    if [chars.next(), chars.next(), chars.next()] != [Some('O'), Some('F'), Some('F')] {
        panic!("ayo this file's not an OFF")
    }

    let num_elems = get_elems_nums(&mut chars, dim as usize);
    todo!()

    /*
    if src.

    let dim: u8;
    if(src[0] == 'O'){
        dim = 3;
    }
    else{


    }
    */
}

pub fn open_off(fp: &Path) -> IoResult<PolytopeSerde> {
    Ok(polytope_from_off_src(String::from_utf8(read(fp)?).unwrap()))
}
