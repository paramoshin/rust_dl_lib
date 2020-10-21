mod loss;
mod metrics;
mod optimizers;
mod tensor;

use std::convert::TryFrom;
use std::fs;
use std::io;
use std::io::Read;
use tensor::tensor::Tensor;

const IMAGES_MAGIC_NUMBER: usize = 2051;
const LABELS_MAGIC_NUMBER: usize = 2049;
const NUM_TRAIN_IMAGES: usize = 60_000;
const NUM_TEST_IMAGES: usize = 10_000;
const IMAGE_ROWS: usize = 28;
const IMAGE_COLUMNS: usize = 28;

fn parse_images(
    filename: &str,
    n_images: usize,
) -> io::Result<(
    usize,
    usize,
    usize,
    usize,
    Vec<[u8; IMAGE_ROWS * IMAGE_COLUMNS]>,
)> {
    // Open the file.
    let images_data_bytes = fs::File::open(filename)?;
    let images_data_bytes = io::BufReader::new(images_data_bytes);
    let mut buffer_32: [u8; 4] = [0; 4];

    // Get the magic number.
    images_data_bytes
        .get_ref()
        .take(4)
        .read(&mut buffer_32)
        .unwrap();
    let magic_number = usize::try_from(u32::from_be_bytes(buffer_32)).unwrap();

    // Get number of images.
    images_data_bytes
        .get_ref()
        .take(4)
        .read(&mut buffer_32)
        .unwrap();
    let num_images = usize::try_from(u32::from_be_bytes(buffer_32)).unwrap();

    // Get number or rows per image.
    images_data_bytes
        .get_ref()
        .take(4)
        .read(&mut buffer_32)
        .unwrap();
    let num_rows = usize::try_from(u32::from_be_bytes(buffer_32)).unwrap();

    // Get number or columns per image.
    images_data_bytes
        .get_ref()
        .take(4)
        .read(&mut buffer_32)
        .unwrap();
    let num_cols = usize::try_from(u32::from_be_bytes(buffer_32)).unwrap();

    // Buffer for holding image pixels.
    let mut image_buffer: [u8; IMAGE_ROWS * IMAGE_COLUMNS] = [0; IMAGE_ROWS * IMAGE_COLUMNS];

    // Vector to hold all images in the file.
    let mut images: Vec<[u8; IMAGE_ROWS * IMAGE_COLUMNS]> = Vec::new();

    // Get images from file.
    for _image in 0..n_images {
        images_data_bytes
            .get_ref()
            .take(u64::try_from(num_rows * num_cols).unwrap())
            .read(&mut image_buffer)
            .unwrap();
        images.push(image_buffer.clone());
    }

    Ok((magic_number, num_images, num_rows, num_cols, images))
}

fn main() {
    let n_images = NUM_TRAIN_IMAGES;
    let (_, _, _, _, images) = parse_images(&"mnist/t10k-images-idx3-ubyte", n_images).unwrap();
    let mut train_data: Vec<u8> = Vec::new();
    for img in images {
        train_data.extend_from_slice(&img);
    }
    let train_data: Vec<f64> = train_data.iter().map(|x| *x as f64).collect();

    let train = Tensor::new(train_data, &[n_images, IMAGE_ROWS, IMAGE_COLUMNS]);
    let train = match train {
        Ok(train) => train,
        Err(err) => panic!("{:?}", err),
    };
    println!("{:?}", train.shape)
}
