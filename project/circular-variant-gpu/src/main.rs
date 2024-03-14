use std::error::Error;
use wgpu::util::DeviceExt;


async fn run() -> Result<(), Box<dyn Error>> {
    let filename = "../drive/FBN1_AlphaFold.pdb";
    let filename = "2h1l.pdb";
    // let filename = "5j7o.pdb";
    let (structure, _) = pdbtbx::open(filename, pdbtbx::StrictnessLevel::Loose)
        .map_err(|e| format!("Failed to open PDB: {:?}", e))?;

    let mut atoms_data = vec![0f32; structure.atom_count() * 4];
    let mut residues_data = vec![0u32; structure.residue_count() * 2];

    let settings_data = [
        &(structure.atom_count() as u32).to_le_bytes()[..],
        &(100.0f32).to_le_bytes(),
    ].concat();

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

    // eprintln!(">> {:?}", &atoms_data[(atoms_data.len() - 10)..]);
    // eprintln!("{:?}", &atoms_data[0..10]);
    // eprintln!("{:?}", &residues_data[0..10]);
    // eprintln!("{:?}", &settings_data);

    // return Ok(());


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

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &shader_module,
        entry_point: "main",
    });

    let bind_group_layout = compute_pipeline.get_bind_group_layout(0);


    // Create buffers

    let atoms_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Atoms buffer"),
        contents: bytemuck::cast_slice(&atoms_data),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let residues_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Residues buffer"),
        contents: bytemuck::cast_slice(&residues_data),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let settings_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Settings buffer"),
        contents: &settings_data,
        usage: wgpu::BufferUsages::STORAGE,
    });

    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (structure.residue_count() * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let read_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: output_buffer.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });


    // Create bind group

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: atoms_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: residues_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: settings_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: output_buffer.as_entire_binding(),
            },
        ],
    });


    // Encode commands

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });

        pass.set_pipeline(&compute_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(
            structure.residue_count() as u32,
            1,
            1
        );
    }

    encoder.copy_buffer_to_buffer(&output_buffer, 0, &read_buffer, 0, read_buffer.size());

    // queue.write_buffer(&settings_buffer, 4, bytemuck::cast_slice(&[current_y as u32]));
    queue.submit(Some(encoder.finish()));

    let buffer_slice = read_buffer.slice(..);
    let (sender, receiver) = flume::bounded(1);

    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
    device.poll(wgpu::Maintain::wait()).panic_on_timeout();

    if let Ok(Ok(())) = receiver.recv_async().await {
        let data = buffer_slice.get_mapped_range();
        let cast_data = bytemuck::cast_slice::<u8, f32>(&data);

        // eprintln!("{:?}", &cast_data[0..12]);

        let mut output_structure = structure.clone();
        let mut current_residue_offset = 0;

        // let i = 5;
        // eprintln!("{}", cast_data[5]);
        // eprintln!("{}", f32::from_le_bytes(data[(i * 4)..(i * 4 + 4)].try_into().unwrap()));

        for (residue_index, residue) in output_structure.residues_mut().enumerate() {
            for atom in residue.atoms_mut() {
                // if cast_data[residue_index] < 0.0 {
                //     eprintln!("{:?}", &cast_data[(residue_index - 2)..(residue_index + 2)]);
                // }

                atom.set_b_factor(cast_data[residue_index] as f64)?;
                current_residue_offset += 1;
            }
        }

        pdbtbx::save(&output_structure, "output.pdb", pdbtbx::StrictnessLevel::Medium)
            .map_err(|e| format!("Failed to save PDB: {:?}", e))?;

        drop(data);
        read_buffer.unmap();
    } else {
        panic!("Failed to run compute on gpu!");
    }

    Ok(())
}

fn main() {
    pollster::block_on(run()).unwrap();
}
