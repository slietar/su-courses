use std::{borrow::BorrowMut, error::Error, path::Path};
use pdbtbx::{Atom, Residue};
use wgpu::util::DeviceExt;


/* async fn run() -> Result<(), Box<dyn Error>> {
    let cutoff = 20.0f32;

    let filename = "../drive/FBN1_AlphaFold.pdb";
    let filename = "2h1l.pdb";
    // let filename = "5j7o.pdb";
    let (structure, _) = pdbtbx::open(filename, pdbtbx::StrictnessLevel::Loose)
        .map_err(|e| format!("Failed to open PDB: {:?}", e))?;

    let mut atoms_data = vec![0f32; structure.atom_count() * 4];
    let mut residues_data = vec![0u32; structure.residue_count() * 2];

    let max_residue_atom_count = structure.residues()
        .map(|residue| residue.atom_count())
        .max()
        .ok_or("Failed to get max residue atom count")?;

    eprintln!("Total atom count: {}", structure.atom_count());
    eprintln!("Max atom count per residue: {}", max_residue_atom_count);

    let mut current_atom_offset = 0;

    for (residue_index, residue) in structure.residues().enumerate() {
        residues_data[residue_index * 2 + 0] = residue.atom_count() as u32;
        residues_data[residue_index * 2 + 1] = current_atom_offset as u32;

        for atom in residue.atoms() {
            atoms_data[current_atom_offset * 4 + 0] = atom.x() as f32;
            atoms_data[current_atom_offset * 4 + 1] = atom.y() as f32;
            atoms_data[current_atom_offset * 4 + 2] = atom.z() as f32;
            current_atom_offset += 1;
        }
    }
} */

    // eprintln!(">> {:?}", &atoms_data[(atoms_data.len() - 10)..]);
    // eprintln!("{:?}", &atoms_data[0..10]);
    // eprintln!("{:?}", &residues_data[0..10]);
    // eprintln!("{:?}", &settings_data);

    // return Ok(());


enum BufferInfo<'a> {
    Data(&'a [u8]),
    Size(usize),
}

fn write_buffer(device: &wgpu::Device, queue: &wgpu::Queue, buffer_source: &mut Option<wgpu::Buffer>, info: BufferInfo, usage: wgpu::BufferUsages) {
    *buffer_source = None;

    let size = match info {
        BufferInfo::Data(data) => data.len(),
        BufferInfo::Size(size) => size,
    };

    let buffer = match buffer_source {
        Some(ref buffer) if buffer.size() >= size as u64 => buffer,
        _ => {
            let atoms_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                mapped_at_creation: false /* match info {
                    BufferInfo::Data(_) => true,
                    BufferInfo::Size(_) => false,
                } */,
                size: ((size as f32) * 1.0) as u64,
                usage,
            });

            *buffer_source = Some(atoms_buffer);
            buffer_source.as_ref().unwrap()
        }
    };

    if let BufferInfo::Data(data) = info {
        queue.write_buffer(&buffer, 0, data);

        // {
        //     let mut buffer_mut = buffer.slice(..).get_mapped_range_mut();
        //     buffer_mut[..data.len()].copy_from_slice(data);
        // }

        // buffer.unmap();
    }

}


struct Engine {
    bind_group_layout: wgpu::BindGroupLayout,
    device: wgpu::Device,
    pipeline: wgpu::ComputePipeline,
    queue: wgpu::Queue,

    atom_count: Option<usize>,
    residue_count: Option<usize>,

    atoms_buffer: Option<wgpu::Buffer>,
    output_buffer: Option<wgpu::Buffer>,
    read_buffer: Option<wgpu::Buffer>,
    residues_buffer: Option<wgpu::Buffer>,
    settings_buffer: wgpu::Buffer,
}

impl Engine {
    pub async fn new() -> Result<Self, Box<dyn Error>> {
        // Request instance and adapter

        let instance = wgpu::Instance::default();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .ok_or("Failed to request adapter")?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults(),
                },
                None,
            )
            .await
            .unwrap();

        let _info = adapter.get_info();


        // Create pipeline

        let shader_text = std::fs::read_to_string("./src/shader.wgsl").unwrap();
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(shader_text.into()),
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader_module,
            entry_point: "main",
        });

        let bind_group_layout = pipeline.get_bind_group_layout(0);


        // Create buffers

        // let atoms_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        //     label: Some("Atoms buffer"),
        //     contents: bytemuck::cast_slice(&atoms_data),
        //     usage: wgpu::BufferUsages::STORAGE,
        // });

        // let residues_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        //     label: Some("Residues buffer"),
        //     contents: bytemuck::cast_slice(&residues_data),
        //     usage: wgpu::BufferUsages::STORAGE,
        // });

        // let settings_data = [
        //     &(structure.atom_count() as u32).to_le_bytes()[..],
        //     &cutoff.to_le_bytes(),
        // ].concat();

        let settings_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Settings buffer"),
            mapped_at_creation: false,
            size: 8,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        //     label: None,
        //     size: (structure.residue_count() * 4) as u64,
        //     usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        //     mapped_at_creation: false,
        // });

        // let read_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        //     label: None,
        //     mapped_at_creation: false,
        //     size: output_buffer.size(),
        //     usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        // });

        Ok(Self {
            bind_group_layout,
            device,
            pipeline,
            queue,

            atom_count: None,
            residue_count: None,

            read_buffer: None,
            output_buffer: None,
            settings_buffer,
            atoms_buffer: None,
            residues_buffer: None,
        })
    }

    pub fn set_residues(&mut self, residues: &[&Residue]) {
        let atom_count: usize = residues.iter().map(|residue| residue.atom_count()).sum();

        let mut atoms_data = vec![0f32; atom_count * 4];
        let mut residues_data = vec![0u32; residues.len() * 2];

        let mut current_atom_offset = 0;

        for (residue_index, residue) in residues.iter().enumerate() {
            residues_data[residue_index * 2 + 0] = residue.atom_count() as u32;
            residues_data[residue_index * 2 + 1] = current_atom_offset as u32;

            for atom in residue.atoms() {
                atoms_data[current_atom_offset * 4 + 0] = atom.x() as f32;
                atoms_data[current_atom_offset * 4 + 1] = atom.y() as f32;
                atoms_data[current_atom_offset * 4 + 2] = atom.z() as f32;
                current_atom_offset += 1;
            }
        }

        write_buffer(&self.device, &self.queue, &mut self.atoms_buffer, BufferInfo::Data(bytemuck::cast_slice(&atoms_data)), wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
        write_buffer(&self.device, &self.queue, &mut self.residues_buffer, BufferInfo::Data(bytemuck::cast_slice(&residues_data)), wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
        write_buffer(&self.device, &self.queue, &mut self.output_buffer, BufferInfo::Size(residues.len() * 4), wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
        write_buffer(&self.device, &self.queue, &mut self.read_buffer, BufferInfo::Size(residues.len() * 4), wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);

        self.atom_count = Some(atom_count);
        self.residue_count = Some(residues.len());

        // let settings_data = [
        //     &(atom_count as u32).to_le_bytes()[..],
        //     &cutoff.to_le_bytes(),
        // ].concat();

/*         let atoms_buffer = match self.atoms_buffer {
            Some(ref buffer) if buffer.size() >= (atoms_data.len() * 4) as u64 => buffer,
            _ => {
                let atoms_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Atoms buffer"),
                    mapped_at_creation: true,
                    size: (atoms_data.len() * 4) as u64,
                    usage: wgpu::BufferUsages::STORAGE,
                });

                self.atoms_buffer = Some(atoms_buffer);
                self.atoms_buffer.as_ref().unwrap()
            }
        };

        // atoms_buffer.write(bytemuck::cast_slice(&atoms_data));

        let mut buffer_mut = atoms_buffer.slice(..).get_mapped_range_mut();
        let target_data = bytemuck::cast_slice(&atoms_data);
        buffer_mut[..target_data.len()].copy_from_slice(target_data);

        // self.atoms_buffer = Some(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        //     label: Some("Atoms buffer"),
        //     contents: bytemuck::cast_slice(&atoms_data),
        //     usage: wgpu::BufferUsages::STORAGE,
        // })); */
    }

    pub async fn run(&mut self, cutoff_distance: f64) -> Result<Vec<f32>, Box<dyn Error>> {
        // Write settings

        let settings_data = [
            &(self.atom_count.unwrap() as u32).to_le_bytes()[..],
            &(cutoff_distance as f32).to_le_bytes(),
        ].concat();

        self.queue.write_buffer(&self.settings_buffer, 0, &settings_data);


        // Create bind group

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.atoms_buffer.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.residues_buffer.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.settings_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.output_buffer.as_ref().unwrap().as_entire_binding(),
                },
            ],
        });


        // Encode commands

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(
                self.residue_count.unwrap() as u32,
                1,
                1
            );
        }

        let read_buffer = self.read_buffer.as_ref().unwrap();

        encoder.copy_buffer_to_buffer(&self.output_buffer.as_ref().unwrap(), 0, read_buffer, 0, read_buffer.size());

        // queue.write_buffer(&settings_buffer, 4, bytemuck::cast_slice(&[current_y as u32]));
        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = read_buffer.slice(..);
        let (sender, receiver) = flume::bounded(1);

        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
        self.device.poll(wgpu::Maintain::wait()).panic_on_timeout();

        if let Ok(Ok(())) = receiver.recv_async().await {
            let data = buffer_slice.get_mapped_range();
            let cast_data = bytemuck::cast_slice::<u8, f32>(&data);

            let result = cast_data.to_vec();
            drop(data);

            // eprintln!("{:?}", &cast_data[0..12]);

            // let mut output_structure = structure.clone();
            // let mut current_residue_offset = 0;

            // // let i = 5;
            // // eprintln!("{}", cast_data[5]);
            // // eprintln!("{}", f32::from_le_bytes(data[(i * 4)..(i * 4 + 4)].try_into().unwrap()));

            // for (residue_index, residue) in output_structure.residues_mut().enumerate() {
            //     for atom in residue.atoms_mut() {
            //         // if cast_data[residue_index] < 0.0 {
            //         //     eprintln!("{:?}", &cast_data[(residue_index - 2)..(residue_index + 2)]);
            //         // }

            //         atom.set_b_factor(cast_data[residue_index] as f64)?;
            //         current_residue_offset += 1;
            //     }
            // }

            // pdbtbx::save(&output_structure, "output.pdb", pdbtbx::StrictnessLevel::Medium)
            //     .map_err(|e| format!("Failed to save PDB: {:?}", e))?;

            // drop(data);
            read_buffer.unmap();

            Ok(result)
        } else {
            Err("Failed to run compute on gpu!")?
        }
    }
}


async fn run() -> Result<(), Box<dyn Error>> {
    let mut engine = Engine::new().await?;

    for filename in [
        "../drive/FBN1_AlphaFold.pdb",
        // "2h1l.pdb",
        // "5j7o.pdb"
    ] {
        let path = Path::new(filename);
        let (mut structure, _) = pdbtbx::open(filename, pdbtbx::StrictnessLevel::Loose)
            .map_err(|e| format!("Failed to open PDB: {:?}", e))
            .unwrap();

        let residues = structure.residues().collect::<Vec<_>>();
        engine.set_residues(&residues);

        for cutoff in [
            10.0,
            20.0,
            30.0,
            40.0,
            50.0,
            60.0,
            70.0,
            80.0,
            90.0,
            100.0,
        ] {
            eprintln!("Processing {} with cutoff {} Å", filename, cutoff);

            let cv = engine.run(cutoff).await?;


            // let mut output_structure = structure.clone();

            for (residue_index, residue) in structure.residues_mut().enumerate() {
                for atom in residue.atoms_mut() {
                    atom.set_b_factor(cv[residue_index] as f64)?;
                }
            }

            pdbtbx::save(&structure, format!("output/{}_{}.pdb", path.file_stem().unwrap().to_str().unwrap(), (cutoff * 10.0) as u32), pdbtbx::StrictnessLevel::Medium)
                .map_err(|e| format!("Failed to save PDB: {:?}", e))?;
        }
    }

    // eprintln!("{:?}", &res[0..10]);

    Ok(())
}

fn main() {
    pollster::block_on(run());
}
