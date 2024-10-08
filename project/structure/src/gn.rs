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

    id: String,

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


#[derive(Debug)]
pub struct ProteinData {
    pub exons: Vec<Exon>,
    pub sequence: Vec<char>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Exon {
    pub name: String,
    pub number: usize,

    pub genomic_start_position: usize,
    pub genomic_end_position: usize,

    pub protein_start_position: usize,
    pub protein_end_position: usize,
}

pub fn process_coordinates(path: &str) -> Result<ProteinData, Box<dyn Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let coords: UniProtCoordinates = serde_json::from_reader(reader)?;

    let exons = coords.gn_coordinate.first().ok_or("No coordinates found")?.genomic_location.exon.iter().enumerate().map(|(index, raw_exon)| {
        Exon {
            name: raw_exon.id.clone(),
            number: (index + 1),

            genomic_start_position: raw_exon.genomic_location.begin.position,
            genomic_end_position: raw_exon.genomic_location.end.position,

            protein_start_position: raw_exon.protein_location.begin.position,
            protein_end_position: raw_exon.protein_location.end.position,
        }
    }).collect::<Vec<_>>();

    Ok(ProteinData {
        exons,
        sequence: coords.sequence.chars().collect(),
    })
}
