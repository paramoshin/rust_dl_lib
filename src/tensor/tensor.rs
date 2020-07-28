use std::cmp::Ordering;
use std::ops;

use rand::distributions::Uniform;
use rand::thread_rng;
use rand_distr::{Distribution, Normal};

use crate::tensor::errors::TensorError;

pub struct Tensor {
    pub data: Vec<f64>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
}

pub enum TensorOperations {
    Add,
    Sub,
    Mul,
    Div,
}

pub enum TensorAggregations {
    Sum,
    Mean,
    Max,
    Min,
    ArgMax,
    ArgMin,
}

impl Tensor {
    fn len_from_shape(shape: &[usize]) -> usize {
        let mut len = 1;
        for l in shape {
            len *= l
        }
        len
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    fn strides_from_shape(shape: &[usize]) -> Vec<usize> {
        let mut strides: Vec<usize> = Vec::new();
        let mut current_stride = 1;
        for dim in shape.iter().rev() {
            strides.insert(0, current_stride);
            current_stride *= dim;
        }
        strides
    }

    fn parse_slice_idxs(&self, slice_idxs: &[[isize; 2]]) -> Result<Vec<[usize; 2]>, TensorError> {
        if slice_idxs.is_empty() || slice_idxs.len() > 4 {
            return Err(TensorError::IndexError);
        };

        let mut idxs = slice_idxs.to_vec().clone();

        if self.shape.len() > slice_idxs.len() {
            for _ in 0..(self.shape.len() - slice_idxs.len()) {
                idxs.push([0, -1]);
            }
        }

        let mut idxs_usize: Vec<[usize; 2]> = Vec::with_capacity(idxs.len());

        for (i, idx) in idxs.iter().enumerate() {
            let from = idx[0] as usize;
            let to = if idx[1] == -1 {
                self.shape[i] as usize
            } else {
                idx[1] as usize
            };
            if from >= to || to > self.shape[i] {
                return Err(TensorError::IndexError);
            }
            idxs_usize.push([from, to]);
        }

        Ok(idxs_usize)
    }

    fn shape_from_slice(slice_idxs: &[[usize; 2]]) -> Vec<usize> {
        let mut shape = Vec::with_capacity(slice_idxs.len());

        for idx in slice_idxs.iter() {
            if idx[1] - idx[0] > 1 {
                shape.push(idx[1] - idx[0]);
            }
        }

        if shape.is_empty() {
            shape.push(1)
        }

        shape
    }

    fn physical_idx(logical_idxs: &[usize], strides: &[usize]) -> usize {
        logical_idxs
            .iter()
            .zip(strides)
            .map(|(&idx, stride)| idx * stride)
            .sum()
    }

    pub fn new(data: Vec<f64>, shape: &[usize]) -> Result<Self, TensorError> {
        if data.len() == Tensor::len_from_shape(shape) && !shape.is_empty() && shape.len() <= 4 {
            Ok(Tensor {
                data,
                shape: shape.to_vec(),
                strides: Tensor::strides_from_shape(&shape),
            })
        } else {
            Err(TensorError::InvalidShapeError)
        }
    }

    pub fn zeroes(shape: &[usize]) -> Result<Self, TensorError> {
        if !shape.is_empty() && shape.len() <= 4 {
            let data = vec![0.0; Tensor::len_from_shape(shape)];
            let shape = shape.clone();
            Ok(Tensor::new(data, shape)?)
        } else {
            Err(TensorError::InvalidShapeError)
        }
    }

    pub fn ones(shape: &[usize]) -> Result<Self, TensorError> {
        if !shape.is_empty() && shape.len() <= 4 {
            let data = vec![1.0; Tensor::len_from_shape(shape)];
            let shape = shape.clone();
            Ok(Tensor::new(data, shape)?)
        } else {
            Err(TensorError::InvalidShapeError)
        }
    }

    pub fn normal(shape: &[usize], mean: f64, std_dev: f64) -> Result<Self, TensorError> {
        let len = Tensor::len_from_shape(shape);
        let norm = Normal::new(mean, std_dev).unwrap();
        let mut rng = thread_rng();
        let v: Vec<f64> = norm.sample_iter(&mut rng).take(len).collect();
        Tensor::new(v, shape)
    }

    pub fn uniform(shape: &[usize], low: f64, high: f64) -> Result<Self, TensorError> {
        let len = Tensor::len_from_shape(shape);
        let uni = Uniform::new(low, high);
        let mut rng = thread_rng();
        let v: Vec<f64> = uni.sample_iter(&mut rng).take(len).collect();
        Tensor::new(v, shape)
    }

    pub fn get(&self, logical_idxs: &[usize], strides: &[usize]) -> &f64 {
        &self.data[Tensor::physical_idx(logical_idxs, strides)]
    }

    pub fn reshape(&self, shape: &[isize]) -> Result<Self, TensorError> {
        let num_of_neg = shape.iter().filter(|x| **x == -1).count();
        match num_of_neg {
            0 => {
                if shape.iter().product::<isize>() != self.len() as isize {
                    Err(TensorError::InvalidShapeError)
                } else {
                    let mut new_shape = Vec::new();
                    for v in shape.iter() {
                        new_shape.push(*v as usize);
                    }
                    Tensor::new(self.data.clone(), &new_shape)
                }
            }
            1 => {
                let idx_of_neg = shape.iter().position(|x| *x == -1).unwrap();
                let len_of_pos: isize = shape.iter().filter(|x| **x != -1).product();
                let remainder = self.len() / len_of_pos as usize;
                let mut new_shape = Vec::new();
                for (i, v) in shape.iter().enumerate() {
                    if i != idx_of_neg {
                        new_shape.push(*v as usize);
                    } else {
                        new_shape.push(remainder);
                    }
                }
                Tensor::new(self.data.clone(), &new_shape)
            }
            _ => Err(TensorError::InvalidShapeError),
        }
    }

    fn broadcast(
        letf_shape: &Vec<usize>,
        right_shape: &Vec<usize>,
    ) -> Result<(Vec<usize>, Vec<usize>, Vec<usize>), TensorError> {
        let mut left_shape = letf_shape.clone();
        let mut right_shape = right_shape.clone();
        if letf_shape.len() > right_shape.len() {
            let ones = vec![1; letf_shape.len() - right_shape.len()];
            right_shape = [&ones[..], &right_shape[..]].concat();
        } else {
            let ones = vec![1; right_shape.len() - letf_shape.len()];
            left_shape = [&ones[..], &letf_shape[..]].concat();
        };

        let mut new_shape = Vec::with_capacity(left_shape.len());
        let mut left_strides = Tensor::strides_from_shape(&left_shape);
        let mut right_strides = Tensor::strides_from_shape(&right_shape);

        for i in 0..left_shape.len() {
            if left_shape[i] == right_shape[i] {
                new_shape.push(left_shape[i]);
            } else if left_shape[i] == 1 {
                new_shape.push(right_shape[i]);
                left_strides[i] = 0;
            } else if right_shape[i] == 1 {
                new_shape.push(left_shape[i]);
                right_strides[i] = 0;
            } else {
                return Err(TensorError::BroadcastError);
            }
        }

        Ok((new_shape, left_strides, right_strides))
    }

    fn element_operation(
        left: &f64,
        right: &f64,
        operation: &TensorOperations,
    ) -> Result<f64, TensorError> {
        match operation {
            TensorOperations::Add => Ok(left + right),
            TensorOperations::Sub => Ok(left - right),
            TensorOperations::Mul => Ok(left * right),
            TensorOperations::Div => Ok(left / right),
        }
    }

    fn tensor_operation(
        &self,
        other: &Tensor,
        operation: TensorOperations,
    ) -> Result<Tensor, TensorError> {
        let (new_shape, l_strides, r_strides) = Tensor::broadcast(&self.shape, &other.shape)?;

        let mut new_data: Vec<f64> = Vec::with_capacity(Tensor::len_from_shape(&new_shape));

        match new_shape.len() {
            1 => {
                for i in 0..new_shape[0] {
                    let res = Tensor::element_operation(
                        self.get(&[i], &l_strides),
                        other.get(&[i], &r_strides),
                        &operation,
                    )?;
                    new_data.push(res);
                }
                Tensor::new(new_data, &new_shape)
            }
            2 => {
                for i in 0..new_shape[0] {
                    for j in 0..new_shape[1] {
                        let res = Tensor::element_operation(
                            self.get(&[i, j], &l_strides),
                            other.get(&[i, j], &r_strides),
                            &operation,
                        )?;
                        new_data.push(res);
                    }
                }
                Tensor::new(new_data, &new_shape)
            }
            3 => {
                for i in 0..new_shape[0] {
                    for j in 0..new_shape[1] {
                        for k in 0..new_shape[2] {
                            let res = Tensor::element_operation(
                                self.get(&[i, j, k], &l_strides),
                                other.get(&[i, j, k], &r_strides),
                                &operation,
                            )?;
                            new_data.push(res);
                        }
                    }
                }
                Tensor::new(new_data, &new_shape)
            }
            4 => {
                for i in 0..new_shape[0] {
                    for j in 0..new_shape[1] {
                        for k in 0..new_shape[2] {
                            for l in 0..new_shape[3] {
                                let res = Tensor::element_operation(
                                    self.get(&[i, j, k, l], &l_strides),
                                    other.get(&[i, j, k, l], &r_strides),
                                    &operation,
                                )?;
                                new_data.push(res);
                            }
                        }
                    }
                }
                Tensor::new(new_data, &new_shape)
            }
            _ => Err(TensorError::OperationError),
        }
    }

    fn scalar_operation(
        &self,
        scalar: f64,
        operation: TensorOperations,
    ) -> Result<Tensor, TensorError> {
        let mut new_data: Vec<f64> = Vec::with_capacity(Tensor::len_from_shape(&self.shape));

        let strides = Tensor::strides_from_shape(&self.shape);

        match self.shape.len() {
            1 => {
                for i in 0..self.shape[0] {
                    let res =
                        Tensor::element_operation(self.get(&[i], &strides), &scalar, &operation)?;
                    new_data.push(res);
                }
                Tensor::new(new_data, &self.shape)
            }
            2 => {
                for i in 0..self.shape[0] {
                    for j in 0..self.shape[1] {
                        let res = Tensor::element_operation(
                            self.get(&[i, j], &strides),
                            &scalar,
                            &operation,
                        )?;
                        new_data.push(res);
                    }
                }
                Tensor::new(new_data, &self.shape)
            }
            3 => {
                for i in 0..self.shape[0] {
                    for j in 0..self.shape[1] {
                        for k in 0..self.shape[2] {
                            let res = Tensor::element_operation(
                                self.get(&[i, j, k], &strides),
                                &scalar,
                                &operation,
                            )?;
                            new_data.push(res);
                        }
                    }
                }
                Tensor::new(new_data, &self.shape)
            }
            4 => {
                for i in 0..self.shape[0] {
                    for j in 0..self.shape[1] {
                        for k in 0..self.shape[2] {
                            for l in 0..self.shape[3] {
                                let res = Tensor::element_operation(
                                    self.get(&[i, j, k, l], &strides),
                                    &scalar,
                                    &operation,
                                )?;
                                new_data.push(res);
                            }
                        }
                    }
                }
                Tensor::new(new_data, &self.shape)
            }
            _ => Err(TensorError::OperationError),
        }
    }

    pub fn slice(&self, idxs: &[[isize; 2]]) -> Result<Self, TensorError> {
        let idxs = self.parse_slice_idxs(idxs)?;

        let new_shape = Tensor::shape_from_slice(&idxs);
        let mut new_data = Vec::with_capacity(Tensor::len_from_shape(&new_shape));

        let strides = Tensor::strides_from_shape(&self.shape);

        match idxs.len() {
            1 => {
                for i in idxs[0][0]..idxs[0][1] {
                    new_data.push(*self.get(&[i], &strides));
                }
                Tensor::new(new_data, &new_shape)
            }
            2 => {
                for i in idxs[0][0]..idxs[0][1] {
                    for j in idxs[1][0]..idxs[1][1] {
                        new_data.push(*self.get(&[i, j], &strides));
                    }
                }
                Tensor::new(new_data, &new_shape)
            }
            3 => {
                for i in idxs[0][0]..idxs[0][1] {
                    for j in idxs[1][0]..idxs[1][1] {
                        for k in idxs[2][0]..idxs[2][1] {
                            new_data.push(*self.get(&[i, j, k], &strides));
                        }
                    }
                }
                Tensor::new(new_data, &new_shape)
            }
            4 => {
                for i in idxs[0][0]..idxs[0][1] {
                    for j in idxs[1][0]..idxs[1][1] {
                        for k in idxs[2][0]..idxs[2][1] {
                            for l in idxs[3][0]..idxs[3][1] {
                                new_data.push(*self.get(&[i, j, k, l], &strides));
                            }
                        }
                    }
                }
                Tensor::new(new_data, &new_shape)
            }
            _ => Err(TensorError::IndexError),
        }
    }

    fn aggregation_operation(
        t: &Tensor,
        operation: &TensorAggregations,
    ) -> Result<f64, TensorError> {
        match operation {
            TensorAggregations::Sum => Ok(t.data.iter().sum()),
            TensorAggregations::Mean => Ok(t.data.iter().sum::<f64>() / t.data.len() as f64),
            TensorAggregations::Max => Ok(t.data.iter().cloned().fold(0. / 0., f64::max)),
            TensorAggregations::Min => Ok(t.data.iter().cloned().fold(0. / 0., f64::min)),
            TensorAggregations::ArgMax => {
                let max_arg = t
                    .data
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                    .map(|(i, _)| i);
                match max_arg {
                    Some(val) => Ok(val as f64),
                    None => Err(TensorError::OperationError),
                }
            }
            TensorAggregations::ArgMin => {
                let min_arg = t
                    .data
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                    .map(|(i, _)| i);
                match min_arg {
                    Some(val) => Ok(val as f64),
                    None => Err(TensorError::OperationError),
                }
            }
        }
    }

    fn process_dims(
        idxs: &mut Vec<[isize; 2]>,
        dim: usize,
        current_dim: usize,
        current_idx: usize,
    ) {
        if dim == current_dim {
            idxs[current_dim] = [0, -1];
        } else {
            idxs[current_dim] = [current_idx as isize, current_idx as isize + 1];
        }
    }

    fn aggregation(
        &self,
        dimension: isize,
        aggregation: TensorAggregations,
    ) -> Result<Tensor, TensorError> {
        if self.shape.is_empty() || (self.shape.len() > 4) {
            return Err(TensorError::InvalidShapeError);
        }

        let dim = if dimension == -1 {
            self.shape.len() - 1
        } else if (dimension >= 0) && (dimension < self.shape.len() as isize) {
            dimension as usize
        } else {
            return Err(TensorError::DimensionError);
        };

        let mut new_shape = Vec::new();
        for i in 0..self.shape.len() {
            if i != dim {
                new_shape.push(self.shape[i]);
            }
        }
        if new_shape.is_empty() {
            new_shape.push(1);
        }

        let mut new_data = Vec::with_capacity(Tensor::len_from_shape(&new_shape));

        let mut idxs: Vec<[isize; 2]> = vec![[0, 0]; self.shape.len()];

        let bound_0 = if dim == 0 { 1 } else { self.shape[0] };
        for i in 0..bound_0 {
            Tensor::process_dims(&mut idxs, dim, 0, i);

            if self.shape.len() == 1 {
                new_data.push(Tensor::aggregation_operation(
                    &self.slice(&idxs)?,
                    &aggregation,
                )?);
            } else {
                let bound_1 = if dim == 1 { 1 } else { self.shape[1] };
                for j in 0..bound_1 {
                    Tensor::process_dims(&mut idxs, dim, 1, j);

                    if self.shape.len() == 2 {
                        new_data.push(Tensor::aggregation_operation(
                            &self.slice(&idxs)?,
                            &aggregation,
                        )?);
                    } else {
                        let bound_2 = if dim == 2 { 1 } else { self.shape[2] };
                        for k in 0..bound_2 {
                            Tensor::process_dims(&mut idxs, dim, 2, k);

                            if self.shape.len() == 3 {
                                new_data.push(Tensor::aggregation_operation(
                                    &self.slice(&idxs)?,
                                    &aggregation,
                                )?);
                            } else {
                                let bound_3 = if dim == 3 { 1 } else { self.shape[3] };
                                for l in 0..bound_3 {
                                    Tensor::process_dims(&mut idxs, dim, 3, l);

                                    new_data.push(Tensor::aggregation_operation(
                                        &self.slice(&idxs)?,
                                        &aggregation,
                                    )?);
                                }
                            }
                        }
                    }
                }
            }
        }
        Tensor::new(new_data, &new_shape)
    }

    pub fn pow(&self, n: f64) -> Self {
        let data: Vec<f64> = self.data.iter().map(|x| x.powf(n)).collect();
        let shape = self.shape.clone();
        Tensor::new(data, &shape).unwrap()
    }

    pub fn sqrt(&self) -> Self {
        let data: Vec<f64> = self.data.iter().map(|x| x.sqrt()).collect();
        let shape = self.shape.clone();
        Tensor::new(data, &shape).unwrap()
    }

    pub fn exp(&self) -> Self {
        let data: Vec<f64> = self.data.iter().map(|x| x.exp()).collect();
        let shape = self.shape.clone();
        Tensor::new(data, &shape).unwrap()
    }

    pub fn ln(&self) -> Self {
        let data: Vec<f64> = self.data.iter().map(|x| x.ln()).collect();
        let shape = self.shape.clone();
        Tensor::new(data, &shape).unwrap()
    }

    pub fn log(&self, base: f64) -> Self {
        let data: Vec<f64> = self.data.iter().map(|x| x.log(base)).collect();
        let shape = self.shape.clone();
        Tensor::new(data, &shape).unwrap()
    }

    pub fn log2(&self) -> Self {
        let data: Vec<f64> = self.data.iter().map(|x| x.log2()).collect();
        let shape = self.shape.clone();
        Tensor::new(data, &shape).unwrap()
    }

    pub fn log10(&self) -> Self {
        let data: Vec<f64> = self.data.iter().map(|x| x.log10()).collect();
        let shape = self.shape.clone();
        Tensor::new(data, &shape).unwrap()
    }

    pub fn abs(&self) -> Self {
        let data: Vec<f64> = self.data.iter().map(|x| x.abs()).collect();
        let shape = self.shape.clone();
        Tensor::new(data, &shape).unwrap()
    }

    pub fn sin(&self) -> Self {
        let data: Vec<f64> = self.data.iter().map(|x| x.sin()).collect();
        let shape = self.shape.clone();
        Tensor::new(data, &shape).unwrap()
    }

    pub fn cos(&self) -> Self {
        let data: Vec<f64> = self.data.iter().map(|x| x.cos()).collect();
        let shape = self.shape.clone();
        Tensor::new(data, &shape).unwrap()
    }

    pub fn tan(&self) -> Self {
        let data: Vec<f64> = self.data.iter().map(|x| x.tan()).collect();
        let shape = self.shape.clone();
        Tensor::new(data, &shape).unwrap()
    }

    pub fn tanh(&self) -> Self {
        let data: Vec<f64> = self.data.iter().map(|x| x.tanh()).collect();
        let shape = self.shape.clone();
        Tensor::new(data, &shape).unwrap()
    }

    pub fn transpose(&self) -> Result<Tensor, TensorError> {
        let mut transposed_shape = self.shape.clone();
        transposed_shape.reverse();
        let mut transposed_strides = self.strides.clone();
        transposed_strides.reverse();

        let mut new_data = Vec::with_capacity(Tensor::len_from_shape(&transposed_shape));

        for i in 0..transposed_shape[0] {
            if transposed_shape.len() == 1 {
                new_data.push(*self.get(&[i], &transposed_strides));
            } else {
                for j in 0..transposed_shape[1] {
                    if transposed_shape.len() == 2 {
                        new_data.push(*self.get(&[i, j], &transposed_strides));
                    } else {
                        for k in 0..transposed_shape[2] {
                            if transposed_shape.len() == 3 {
                                new_data.push(*self.get(&[i, j, k], &transposed_strides));
                            } else {
                                for l in 0..transposed_shape[3] {
                                    new_data.push(*self.get(&[i, j, k, l], &transposed_strides));
                                }
                            }
                        }
                    }
                }
            }
        }

        Tensor::new(new_data, &transposed_shape)
    }

    pub fn matmul(&self, right: &Tensor) -> Result<Tensor, TensorError> {
        if self.shape.len() != 2 || right.shape.len() != 2 || self.shape[1] != right.shape[0] {
            return Err(TensorError::InvalidShapeError);
        };

        let new_shape = vec![self.shape[0], right.shape[1]];
        let mut new_data: Vec<f64> = vec![0.0; Tensor::len_from_shape(&new_shape)];
        let new_strides = Tensor::strides_from_shape(&new_shape);

        for i in 0..new_shape[0] {
            for j in 0..new_shape[1] {
                for k in 0..right.shape[0] {
                    let idx = Tensor::physical_idx(&[i, j], &new_strides);
                    new_data[idx] +=
                        *self.get(&[i, k], &self.strides) * *right.get(&[k, j], &right.strides);
                }
            }
        }

        Tensor::new(new_data, &new_shape)
    }

    pub fn sum(&self, dim: isize) -> Result<Tensor, TensorError> {
        self.aggregation(dim, TensorAggregations::Sum)
    }

    pub fn mean(&self, dim: isize) -> Result<Tensor, TensorError> {
        self.aggregation(dim, TensorAggregations::Mean)
    }

    pub fn max(&self, dim: isize) -> Result<Tensor, TensorError> {
        self.aggregation(dim, TensorAggregations::Max)
    }

    pub fn min(&self, dim: isize) -> Result<Tensor, TensorError> {
        self.aggregation(dim, TensorAggregations::Min)
    }

    pub fn argmax(&self, dim: isize) -> Result<Tensor, TensorError> {
        self.aggregation(dim, TensorAggregations::ArgMax)
    }

    pub fn argmin(&self, dim: isize) -> Result<Tensor, TensorError> {
        self.aggregation(dim, TensorAggregations::ArgMin)
    }
}

impl ops::Add<&Tensor> for &Tensor {
    type Output = Tensor;

    fn add(self, other: &Tensor) -> Tensor {
        match self.tensor_operation(other, TensorOperations::Add) {
            Ok(t) => t,
            Err(e) => panic!("{}", e),
        }
    }
}

impl ops::Sub<&Tensor> for &Tensor {
    type Output = Tensor;

    fn sub(self, other: &Tensor) -> Tensor {
        match self.tensor_operation(other, TensorOperations::Sub) {
            Ok(t) => t,
            Err(e) => panic!("{}", e),
        }
    }
}

impl ops::Mul<&Tensor> for &Tensor {
    type Output = Tensor;

    fn mul(self, other: &Tensor) -> Tensor {
        match self.tensor_operation(other, TensorOperations::Mul) {
            Ok(t) => t,
            Err(e) => panic!("{}", e),
        }
    }
}

impl ops::Div<&Tensor> for &Tensor {
    type Output = Tensor;

    fn div(self, other: &Tensor) -> Tensor {
        match self.tensor_operation(other, TensorOperations::Div) {
            Ok(t) => t,
            Err(e) => panic!("{}", e),
        }
    }
}

impl ops::Add<f64> for &Tensor {
    type Output = Tensor;

    fn add(self, other: f64) -> Tensor {
        match self.scalar_operation(other, TensorOperations::Add) {
            Ok(t) => t,
            Err(e) => panic!("{}", e),
        }
    }
}

impl ops::Sub<f64> for &Tensor {
    type Output = Tensor;

    fn sub(self, other: f64) -> Tensor {
        match self.scalar_operation(other, TensorOperations::Sub) {
            Ok(t) => t,
            Err(e) => panic!("{}", e),
        }
    }
}

impl ops::Mul<f64> for &Tensor {
    type Output = Tensor;

    fn mul(self, other: f64) -> Tensor {
        match self.scalar_operation(other, TensorOperations::Mul) {
            Ok(t) => t,
            Err(e) => panic!("{}", e),
        }
    }
}

impl ops::Div<f64> for &Tensor {
    type Output = Tensor;

    fn div(self, other: f64) -> Tensor {
        match self.scalar_operation(other, TensorOperations::Div) {
            Ok(t) => t,
            Err(e) => panic!("{}", e),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_new() {
        let data: Vec<f64> = vec![1.0; 12];
        let t = Tensor::new(data.clone(), &[3, 4]).unwrap();
        assert_eq!(t.data, data);
        assert_eq!(t.len(), data.len());
        assert_eq!(t.shape, vec![3, 4]);
    }

    #[test]
    #[should_panic]
    fn test_wrong_shape() {
        let data: Vec<f64> = vec![1.0; 12];
        let _t = Tensor::new(data.clone(), &[15, 8]).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_no_shape() {
        let data: Vec<f64> = vec![1.0; 12];
        let _t = Tensor::new(data.clone(), &[]).unwrap();
    }

    #[test]
    fn test_zeores() {
        let t = Tensor::zeroes(&[3, 4]).unwrap();
        assert_eq!(t.data, vec![0.0; 12]);
    }

    #[test]
    fn test_ones() {
        let t = Tensor::ones(&[3, 4]).unwrap();
        assert_eq!(t.data, vec![1.0; 12]);
    }

    #[test]
    fn test_pow() {
        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0];
        let t = Tensor::new(data.clone(), &[3, 4]).unwrap();
        assert_eq!(
            t.pow(2.0).data,
            vec![1.0, 4.0, 9.0, 16.0, 1.0, 4.0, 9.0, 16.0, 1.0, 4.0, 9.0, 16.0]
        );
    }

    #[test]
    fn test_strides() {
        assert_eq!(Tensor::strides_from_shape(&[10, 30, 4]), vec![120, 4, 1]);
    }

    #[test]
    fn test_physical_idx() {
        let data: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
        let t = Tensor::new(data.clone(), &[2, 2, 3]).unwrap();
        let mut id = 0;
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..3 {
                    assert_eq!(id, Tensor::physical_idx(&[i, j, k], &t.strides));
                    assert_eq!(
                        id as f64,
                        t.data[Tensor::physical_idx(&[i, j, k], &t.strides)]
                    );
                    id += 1;
                }
            }
        }
    }

    #[test]
    #[should_panic]
    fn test_view_wrong_shape() {
        let data: Vec<f64> = vec![1.0; 27];
        let t = Tensor::new(data, &[3, 3, 3]).unwrap();
        let _v = t.reshape(&[4, 5, 2]).unwrap();
    }

    #[test]
    fn test_view() {
        let data: Vec<f64> = vec![1.0; 27];
        let t = Tensor::new(data, &[3, 3, 3]).unwrap();
        let v = t.reshape(&[1, 3, -1]).unwrap();
        assert_eq!(t.data, v.data);
        assert_eq!(t.len(), v.len());
        assert_eq!(v.shape, [1, 3, 9]);
    }

    #[test]
    fn test_broadcast() {
        let t1 = Tensor::new(vec![2.0, 3.0, 4.0, 5.0], &[2, 2]).unwrap();
        let t2 = Tensor::new(vec![2.0, 3.0], &[2]).unwrap();
        let (new_shape, _left_strides, _right_strides) =
            Tensor::broadcast(&t1.shape, &t2.shape).unwrap();
        assert_eq!(new_shape, vec![2, 2]);

        let t1 = Tensor::new(vec![2.0, 3.0, 4.0, 5.0], &[1, 2, 2]).unwrap();
        let t2 = Tensor::new(vec![2.0, 3.0], &[2]).unwrap();
        let (new_shape, _left_strides, _right_strides) =
            Tensor::broadcast(&t1.shape, &t2.shape).unwrap();
        assert_eq!(new_shape, vec![1, 2, 2]);
    }
    #[test]
    fn test_view_neg() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let _x = t.reshape(&[-1]).unwrap();
        let _x = t.reshape(&[2, -1]).unwrap();
        let _x = t.reshape(&[1, -1]).unwrap();
        let _x = t.reshape(&[4, -1]).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_view_wrong_neg() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let _x = t.reshape(&[-1, -1]).unwrap();
    }

    #[test]
    fn test_tensor_add() {
        let a = Tensor::new(vec![2.0, 3.0, 4.0, 5.0], &[4]).unwrap();
        let b = Tensor::new(vec![2.0, 3.0, 4.0, 5.0], &[4]).unwrap();

        let c = &a + &b;

        assert!((c.data == vec![4.0, 6.0, 8.0, 10.0]) && (c.shape == vec![4]))
    }

    #[test]
    fn test_tensor_sub() {
        let a = Tensor::new(vec![2.0, 3.0, 4.0, 5.0], &[2, 2]).unwrap();
        let b = Tensor::new(vec![2.0, 3.0], &[2]).unwrap();

        let c = &a - &b;

        assert!((c.data == vec![0.0, 0.0, 2.0, 2.0]) && (c.shape == vec![2, 2]))
    }

    #[test]
    fn test_tensor_mul() {
        let a = Tensor::new(vec![2.0, 3.0, 4.0, 5.0], &[2, 2]).unwrap();
        let b = Tensor::new(vec![2.0, 3.0], &[2]).unwrap();

        let c = &a * &b;

        assert!((c.data == vec![4.0, 9.0, 8.0, 15.0]) && (c.shape == vec![2, 2]))
    }

    #[test]
    fn test_tensor_div() {
        let a = Tensor::new(vec![2.0, 4.0, 6.0, 8.0], &[2, 2]).unwrap();
        let b = Tensor::new(vec![2.0, 4.0], &[2]).unwrap();

        let c = &a / &b;

        assert!((c.data == vec![1.0, 1.0, 3.0, 2.0]) && (c.shape == vec![2, 2]))
    }

    #[test]
    fn test_add_broadcast() {
        let a = Tensor::new(vec![2.0, 3.0, 4.0, 5.0], &[1, 4]).unwrap();
        let b = Tensor::new(vec![2.0, 3.0, 4.0, 5.0], &[4]).unwrap();

        let c = &a + &b;

        assert!((c.data == vec![4.0, 6.0, 8.0, 10.0]) && (c.shape == vec![1, 4]));
    }

    #[test]
    fn test_add_broadcast_dims() {
        let a = Tensor::new(vec![2.0, 3.0, 4.0, 5.0], &[2, 2]).unwrap();
        let b = Tensor::new(vec![2.0, 3.0], &[2]).unwrap();

        let c = &a + &b;

        assert!((c.data == vec![4.0, 6.0, 6.0, 8.0]) && (c.shape == vec![2, 2]));
    }

    #[test]
    fn test_add_broadcast_shapes_and_dims() {
        let a = Tensor::new(vec![2.0, 3.0, 4.0, 5.0], &[1, 2, 2]).unwrap();
        let b = Tensor::new(vec![2.0, 3.0], &[2]).unwrap();

        let c = &a + &b;

        assert!((c.data == vec![4.0, 6.0, 6.0, 8.0]) && (c.shape == vec![1, 2, 2]));
    }

    #[test]
    #[should_panic]
    fn test_broadcast_wrong_shape() {
        let a = Tensor::new(vec![2.0, 3.0, 4.0, 5.0], &[4]).unwrap();
        let b = Tensor::new(vec![2.0, 3.0, 4.0, 5.0], &[2, 2]).unwrap();

        let _c = &a + &b;
    }

    #[test]
    fn test_add_scalar() {
        let a = Tensor::new(vec![2.0, 3.0, 4.0, 5.0], &[2, 2]).unwrap();
        let b = &a + 5.0;
        assert_eq!(b.data, vec![7.0, 8.0, 9.0, 10.0]);
    }

    #[test]
    fn test_sub_scalar() {
        let a = Tensor::new(vec![2.0, 3.0, 4.0, 5.0], &[2, 2]).unwrap();
        let b = &a - 5.0;
        assert_eq!(b.data, vec![-3.0, -2.0, -1.0, 0.0]);
    }

    #[test]
    fn test_mul_scalar() {
        let a = Tensor::new(vec![2.0, 3.0, 4.0, 5.0], &[2, 2]).unwrap();
        let b = &a * 2.0;
        assert_eq!(b.data, vec![4.0, 6.0, 8.0, 10.0]);
    }

    #[test]
    fn test_div_scalar() {
        let a = Tensor::new(vec![2.0, 3.0, 4.0, 5.0], &[2, 2]).unwrap();
        let b = &a / 2.0;
        assert_eq!(b.data, vec![1.0, 1.5, 2.0, 2.5]);
    }

    #[test]
    #[should_panic]
    fn test_slice_too_large_slice() {
        let t = Tensor::zeroes(&[2, 2]).unwrap();
        let _x = t.slice(&[[0, 1], [0, 1], [0, 1]]).unwrap();
    }

    #[test]
    fn test_slice_negative_1() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let x = t.slice(&[[0, -1], [0, 1]]).unwrap();

        assert!((x.data == vec![1.0, 3.0]) && (*x.shape == [2]));
    }

    #[test]
    fn test_slice_fewer_dims() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let x = t.slice(&[[0, 1]]).unwrap();

        assert!((x.data == vec![1.0, 2.0]) && (x.shape == vec![2]));
    }

    #[test]
    #[should_panic]
    fn test_slice_start_greater_than_stop() {
        let t = Tensor::zeroes(&[2, 2]).unwrap();
        let _x = t.slice(&[[0, 1], [1, 0]]).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_slice_stop_greater_than_shape() {
        let t = Tensor::zeroes(&[2, 2]).unwrap();
        let _x = t.slice(&[[0, 1], [0, 3]]).unwrap();
    }

    #[test]
    fn test_slice_tensor_1d_element() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();

        let x = t.slice(&[[0, 1]]).unwrap();

        assert!((x.data == vec![1.0]) && (x.shape == vec![1]) && (x.strides == vec![1]))
    }
    #[test]
    fn test_slice_tensor_2d_element() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let x = t.slice(&[[0, 1], [0, 1]]).unwrap();

        assert!((x.data == vec![1.0]) && (x.shape == vec![1]) && (x.strides == vec![1]))
    }

    #[test]
    fn test_slice_tensor_2d_row() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let x = t.slice(&[[0, 1], [0, 2]]).unwrap();

        assert!((x.data == vec![1.0, 2.0]) && (x.shape == vec![2]) && (x.strides == vec![1]))
    }

    #[test]
    fn test_slice_tensor_2d_col() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let x = t.slice(&[[0, 2], [0, 1]]).unwrap();

        assert!((x.data == vec![1.0, 3.0]) && (x.shape == vec![2]) && (x.strides == vec![1]))
    }

    #[test]
    fn test_slice_tensor_3d_element() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]).unwrap();

        let x = t.slice(&[[0, 1], [0, 1], [0, 1]]).unwrap();

        assert!((x.data == vec![1.0]) && (x.shape == vec![1]) && (x.strides == vec![1]))
    }

    #[test]
    fn test_slice_tensor_3d_row() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]).unwrap();

        let x = t.slice(&[[0, 1], [0, 1], [0, 2]]).unwrap();

        assert!((x.data == vec![1.0, 2.0]) && (x.shape == vec![2]) && (x.strides == vec![1]))
    }

    #[test]
    fn test_slice_tensor_3d_col() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]).unwrap();

        let x = t.slice(&[[0, 1], [0, 2], [0, 1]]).unwrap();

        assert!((x.data == vec![1.0, 3.0]) && (x.shape == vec![2]) && (x.strides == vec![1]))
    }

    #[test]
    fn test_slice_tensor_3d_channel() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]).unwrap();

        let x = t.slice(&[[0, 2], [0, 1], [0, 1]]).unwrap();

        assert!((x.data == vec![1.0, 5.0]) && (x.shape == vec![2]) && (x.strides == vec![1]))
    }

    #[test]
    fn test_slice_tensor_3d_chunk() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]).unwrap();

        let x = t.slice(&[[0, 2], [0, 2], [0, 1]]).unwrap();

        assert!(
            (x.data == vec![1.0, 3.0, 5.0, 7.0])
                && (*x.shape == [2, 2])
                && (x.strides == vec![2, 1])
        )
    }
    #[test]
    fn test_slice_tensor_4d_element() {
        let t = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[2, 2, 2, 2],
        )
        .unwrap();

        let x = t.slice(&[[0, 1], [0, 1], [0, 1], [0, 1]]).unwrap();

        assert!((x.data == vec![1.0]) && (x.shape == vec![1]) && (x.strides == vec![1]))
    }

    #[test]
    fn test_slice_tensor_4d_row() {
        let t = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[2, 2, 2, 2],
        )
        .unwrap();

        let x = t.slice(&[[0, 1], [0, 1], [0, 1], [0, 2]]).unwrap();

        assert!((x.data == vec![1.0, 2.0]) && (x.shape == vec![2]) && (x.strides == vec![1]))
    }

    #[test]
    fn test_slice_tensor_4d_col() {
        let t = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[2, 2, 2, 2],
        )
        .unwrap();

        let x = t.slice(&[[0, 1], [0, 1], [0, 2], [0, 1]]).unwrap();

        assert!((x.data == vec![1.0, 3.0]) && (x.shape == vec![2]) && (x.strides == vec![1]))
    }

    #[test]
    fn test_slice_tensor_4d_channel() {
        let t = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[2, 2, 2, 2],
        )
        .unwrap();

        let x = t.slice(&[[0, 1], [0, 2], [0, 1], [0, 1]]).unwrap();

        assert!((x.data == vec![1.0, 5.0]) && (x.shape == vec![2]) && (x.strides == vec![1]))
    }

    #[test]
    fn test_slice_tensor_4d_batch() {
        let t = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[2, 2, 2, 2],
        )
        .unwrap();

        let x = t.slice(&[[0, 2], [0, 1], [0, 1], [0, 1]]).unwrap();

        assert!((x.data == vec![1.0, 9.0]) && (x.shape == vec![2]) && (x.strides == vec![1]))
    }

    #[test]
    fn test_slice_tensor_4d_chunk() {
        let t = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[2, 2, 2, 2],
        )
        .unwrap();

        let x = t.slice(&[[0, 2], [0, 2], [0, 1], [0, 1]]).unwrap();

        assert!(
            (x.data == vec![1.0, 5.0, 9.0, 13.0])
                && (*x.shape == [2, 2])
                && (x.strides == vec![2, 1])
        )
    }

    #[test]
    fn test_sum_1d() {
        let x = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[16],
        )
        .unwrap();

        let y = x.sum(0).unwrap();

        assert!((y.data == vec![136.0]) && (y.shape == vec![1]))
    }

    #[test]
    fn test_sum_2d() {
        let x = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[2, 8],
        )
        .unwrap();

        let y = x.sum(1).unwrap();

        assert!((y.data == vec![36.0, 100.0]) && (y.shape == vec![2]))
    }

    #[test]
    fn test_sum_3d() {
        let x = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[2, 2, 4],
        )
        .unwrap();

        let y = x.sum(1).unwrap();

        assert!(
            (y.data == vec![6.0, 8.0, 10.0, 12.0, 22.0, 24.0, 26.0, 28.0])
                && (y.shape == vec![2, 4])
        )
    }

    #[test]
    fn test_sum_4d() {
        let x = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[2, 2, 2, 2],
        )
        .unwrap();

        let y = x.sum(-1).unwrap();

        assert!(
            (y.data == vec![3.0, 7.0, 11.0, 15.0, 19.0, 23.0, 27.0, 31.0])
                && (y.shape == vec![2, 2, 2])
        )
    }

    #[test]
    fn test_mean() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

        let y = x.mean(-1).unwrap();

        assert!((y.data == vec![2.0, 5.0]) && (y.shape == vec![2]))
    }

    #[test]
    fn test_max() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

        let y = x.max(-1).unwrap();

        assert!((y.data == vec![3.0, 6.0]) && (y.shape == vec![2]))
    }

    #[test]
    fn test_min() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

        let y = x.min(-1).unwrap();

        assert!((y.data == vec![1.0, 4.0]) && (y.shape == vec![2]))
    }

    #[test]
    fn test_argmax() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

        let y = x.argmax(-1).unwrap();

        assert!((y.data == vec![2.0, 2.0]) && (y.shape == vec![2]))
    }

    #[test]
    fn test_argmin() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

        let y = x.argmin(-1).unwrap();

        assert!((y.data == vec![0.0, 0.0]) && (y.shape == vec![2]))
    }

    #[test]
    fn test_transpose_1d() {
        let x = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[16],
        )
        .unwrap();

        let z = x.transpose().unwrap();

        assert!(
            (z.data
                == vec![
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                    15.0, 16.0,
                ])
                && (z.shape == vec![16])
        )
    }

    #[test]
    fn test_transpose_2d() {
        let x = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[2, 8],
        )
        .unwrap();

        let z = x.transpose().unwrap();

        assert!(
            (z.data
                == vec![
                    1.0, 9.0, 2.0, 10.0, 3.0, 11.0, 4.0, 12.0, 5.0, 13.0, 6.0, 14.0, 7.0, 15.0,
                    8.0, 16.0
                ])
                && (z.shape == vec![8, 2])
        )
    }

    #[test]
    fn test_transpose_3d() {
        let x = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[2, 2, 4],
        )
        .unwrap();

        let z = x.transpose().unwrap();

        assert!(
            (z.data
                == vec![
                    1.0, 9.0, 5.0, 13.0, 2.0, 10.0, 6.0, 14.0, 3.0, 11.0, 7.0, 15.0, 4.0, 12.0,
                    8.0, 16.0
                ])
                && (z.shape == vec![4, 2, 2])
        )
    }

    #[test]
    fn test_transpose_4d() {
        let x = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[2, 2, 2, 2],
        )
        .unwrap();

        let z = x.transpose().unwrap();

        assert!(
            (z.data
                == vec![
                    1.0, 9.0, 5.0, 13.0, 3.0, 11.0, 7.0, 15.0, 2.0, 10.0, 6.0, 14.0, 4.0, 12.0,
                    8.0, 16.0
                ])
                && (z.shape == vec![2, 2, 2, 2])
        )
    }

    #[test]
    fn test_matmul() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let y = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3]).unwrap();

        let z = x.matmul(&y).unwrap();

        assert!((z.data == vec![30.0, 36.0, 42.0, 66.0, 81.0, 96.0]) && (z.shape == vec![2, 3]))
    }
}
