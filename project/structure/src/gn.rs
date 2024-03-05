use std::{error::Error, fs::File, io::BufReader};

use serde::{Deserialize, Serialize};


#[derive(Debug, Deserialize)]
struct UniProtCoordinates {
    #[serde(rename = "gnCoordinate")]
    gn_coordinate: Vec<UniProtGnCoordinate>,
    sequence: String,
}

#[derive(Debug, Deserialize)]
struct UniProtGnCoordinate {
    #[serde(rename = "genomicLocation")]
    genomic_location: UniProtGenomicLocation,
}

#[derive(Debug, Deserialize)]
struct UniProtGenomicLocation {
    exon: Vec<UniProtExon>,
}

#[derive(Debug, Deserialize)]
struct UniProtExon {
    #[serde(rename = "genomeLocation")]
    genomic_location: UniProtGNLocation,

    #[serde(rename = "proteinLocation")]
    protein_location: UniProtGNLocation,
}

#[derive(Debug, Deserialize)]
struct UniProtGNLocation {
    begin: UniProtGNPosition,
    end: UniProtGNPosition,
}

#[derive(Debug, Deserialize)]
struct UniProtGNPosition {
    position: usize,
}


#[derive(Debug, Serialize)]
pub struct GenomicProtein {
    cum_positions: Vec<usize>,
    pub exons: Vec<Exon>,
    pub sequence: Vec<char>,
}

impl GenomicProtein {
    fn new(sequence: Vec<char>, exons: Vec<Exon>) -> Self {
        let mut cum_positions = Vec::with_capacity(exons.len());
        let mut position = 0;

        for exon in &exons {
            cum_positions.push(position);
            position += exon.protein_range.1 - exon.protein_range.0;
        }

        Self {
            cum_positions,
            exons,
            sequence,
        }
    }

    pub fn find_exon(&self, position: usize) -> Option<&Exon> {
        let index = self.cum_positions.binary_search(&position).unwrap_or_else(|x| x - 1);
        self.exons.get(index)
    }
}

#[derive(Debug, Serialize)]
pub struct Exon {
    pub index: usize,
    pub genomic_range: (usize, usize),
    pub protein_range: (usize, usize),
}

pub fn process_coordinates(path: &str) -> Result<GenomicProtein, Box<dyn Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let coords: UniProtCoordinates = serde_json::from_reader(reader)?;

    let exons = coords.gn_coordinate.first().ok_or("No coordinates found")?.genomic_location.exon.iter().enumerate().map(|(index, raw_exon)| {
        Exon {
            index,
            genomic_range: (
                raw_exon.genomic_location.begin.position - 1,
                raw_exon.genomic_location.end.position - 1
            ),
            protein_range: (
                raw_exon.protein_location.begin.position - 1,
                raw_exon.protein_location.end.position - 1
            )
        }
    }).collect::<Vec<_>>();

    Ok(GenomicProtein::new(coords.sequence.chars().collect(), exons))
}
