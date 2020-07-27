use std::ops;
use rand::thread_rng;
use rand::distributions::Uniform;
use rand_distr::{Distribution, Normal};

use crate::errors::TensorError;


pub struct Tensor {
    pub data: Vec<f64>,
    pub shape: Vec<usize>,
}


pub enum TensorOperations {
    Add,
    Sub,
    Mul,
    Div,
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

    fn physical_idx(&self, logical_idxs: &[usize], strides: &[usize]) -> usize {
        logical_idxs.iter().zip(strides).map(|(&idx, stride)| idx * stride).sum()
    }

    pub fn new(data: Vec<f64>, shape: &[usize]) -> Result<Self, TensorError> {
        if data.len() == Tensor::len_from_shape(shape)
            && !shape.is_empty()
            && shape.len() <= 4
        {
            Ok(Tensor { data, shape: shape.to_vec() })
        } else {
            Err(TensorError::InvalidShapeError)
        }
    }

    pub fn zeroes(shape: &[usize]) -> Result<Self, TensorError> {
        if !shape.is_empty() && shape.len() <= 4 {
            let data = vec![0.0; Tensor::len_from_shape(shape)];
            Ok(Tensor { data: data, shape: shape.to_vec() })
        } else {
            Err(TensorError::InvalidShapeError)
        }
    }

    pub fn ones(shape: &[usize]) -> Result<Self, TensorError> {
        if !shape.is_empty() && shape.len() <= 4 {
            let data = vec![1.0; Tensor::len_from_shape(shape)];
            Ok(Tensor { data, shape: shape.to_vec() })
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
        &self.data[self.physical_idx(logical_idxs, strides)]
    }

    pub fn view(&self, shape: &[isize]) -> Result<Self, TensorError> {
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
            _ => {
                Err(TensorError::InvalidShapeError)
            }
        }
    }

    fn broadcast(
        letf_shape: &Vec<usize>, right_shape: &Vec<usize>
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
        };

        Ok((new_shape, left_strides, right_strides))
    }

    fn element_operation(left: &f64, right: &f64, operation: &TensorOperations) -> Result<f64, TensorError> {
        match operation {
            TensorOperations::Add => Ok(left + right),
            TensorOperations::Sub => Ok(left - right),
            TensorOperations::Mul => Ok(left * right),
            TensorOperations::Div => Ok(left / right),
        }
    }

    fn tensor_operation(&self, other: &Tensor, operation: TensorOperations) -> Result<Tensor, TensorError> {
        let (new_shape, l_strides, r_strides) = Tensor::broadcast(&self.shape, &other.shape)?;

        let mut new_data: Vec<f64> = Vec::with_capacity(Tensor::len_from_shape(&new_shape));

        match new_shape.len() {
            1 => {
                for i in 0..new_shape[0] {
                    let res = Tensor::element_operation(
                        self.get(&[i], &l_strides),
                        other.get(&[i], &r_strides),
                        &operation
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
                            &operation
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
                                &operation
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
                                    &operation
                                )?;
                                new_data.push(res);
                            }
                        }
                    }
                }
                Tensor::new(new_data, &new_shape)
            }
            _ => Err(TensorError::OperationError)
        }
    }

    fn scalar_operation(&self, scalar: f64, operation: TensorOperations) -> Result<Tensor, TensorError> {
        let mut new_data: Vec<f64> = Vec::with_capacity(Tensor::len_from_shape(&self.shape));

        let strides = Tensor::strides_from_shape(&self.shape);

        match self.shape.len() {
            1 => {
                for i in 0..self.shape[0] {
                    let res = Tensor::element_operation(
                        self.get(&[i], &strides),
                        &scalar,
                        &operation
                    )?;
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
                            &operation
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
                                &operation
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
                                    &operation
                                )?;
                                new_data.push(res);
                            }
                        }
                    }
                }
                Tensor::new(new_data, &self.shape)
            }
            _ => Err(TensorError::OperationError)
        }
    }

    pub fn pow(&self, n: f64) -> Self {
        let data: Vec<f64> = self.data.iter().map(|x| x.powf(n)).collect();
        Tensor { data, shape: self.shape.clone() }
    }

    pub fn sqrt(&self) -> Self {
        let data: Vec<f64> = self.data.iter().map(|x| x.sqrt()).collect();
        Tensor { data, shape: self.shape.clone() }
    }

    pub fn exp(&self) -> Self {
        let data: Vec<f64> = self.data.iter().map(|x| x.exp()).collect();
        Tensor { data, shape: self.shape.clone() }
    }

    pub fn ln(&self) -> Self {
        let data: Vec<f64> = self.data.iter().map(|x| x.ln()).collect();
        Tensor { data, shape: self.shape.clone() }
    }

    pub fn log(&self, base: f64) -> Self {
        let data: Vec<f64> = self.data.iter().map(|x| x.log(base)).collect();
        Tensor { data, shape: self.shape.clone() }
    }

    pub fn log2(&self) -> Self {
        let data: Vec<f64> = self.data.iter().map(|x| x.log2()).collect();
        Tensor { data, shape: self.shape.clone() }
    }

    pub fn log10(&self) -> Self {
        let data: Vec<f64> = self.data.iter().map(|x| x.log10()).collect();
        Tensor { data, shape: self.shape.clone() }
    }

    pub fn abs(&self) -> Self {
        let data: Vec<f64> = self.data.iter().map(|x| x.abs()).collect();
        Tensor { data, shape: self.shape.clone() }
    }

    pub fn sin(&self) -> Self {
        let data: Vec<f64> = self.data.iter().map(|x| x.sin()).collect();
        Tensor { data, shape: self.shape.clone() }
    }

    pub fn cos(&self) -> Self {
        let data: Vec<f64> = self.data.iter().map(|x| x.cos()).collect();
        Tensor { data, shape: self.shape.clone() }
    }

    pub fn tan(&self) -> Self {
        let data: Vec<f64> = self.data.iter().map(|x| x.tan()).collect();
        Tensor { data, shape: self.shape.clone() }
    }

    pub fn tanh(&self) -> Self {
        let data: Vec<f64> = self.data.iter().map(|x| x.tanh()).collect();
        Tensor { data, shape: self.shape.clone() }
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
        let t = Tensor::new(data.clone(),&[3, 4]).unwrap();
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
                    assert_eq!(id, t.physical_idx(&[i, j, k], &Tensor::strides_from_shape(&t.shape)));
                    assert_eq!(id as f64, t.data[t.physical_idx(&[i, j, k], &Tensor::strides_from_shape(&t.shape))]);
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
        let _v = t.view(&[4, 5, 2]).unwrap();
    }

    #[test]
    fn test_view() {
        let data: Vec<f64> = vec![1.0; 27];
        let t = Tensor::new(data, &[3, 3, 3]).unwrap();
        let v = t.view(&[1, 3, -1]).unwrap();
        assert_eq!(t.data, v.data);
        assert_eq!(t.len(), v.len());
        assert_eq!(v.shape, [1, 3, 9]);
    }

    #[test]
    fn test_broadcast() {
        let t1 = Tensor::new(vec![2.0, 3.0, 4.0, 5.0], &[2, 2]).unwrap();
        let t2 = Tensor::new(vec![2.0, 3.0], &[2]).unwrap();
        let (new_shape, _left_strides, _right_strides) = Tensor::broadcast(&t1.shape, &t2.shape).unwrap();
        assert_eq!(new_shape, vec![2, 2]);

        let t1 = Tensor::new(vec![2.0, 3.0, 4.0, 5.0], &[1, 2, 2]).unwrap();
        let t2 = Tensor::new(vec![2.0, 3.0], &[2]).unwrap();
        let (new_shape, _left_strides, _right_strides) = Tensor::broadcast(&t1.shape, &t2.shape).unwrap();
        assert_eq!(new_shape, vec![1, 2, 2]);

    }
    #[test]
    fn test_view_neg() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let _x = t.view(&[-1]).unwrap();
        let _x = t.view(&[2, -1]).unwrap();
        let _x = t.view(&[1, -1]).unwrap();
        let _x = t.view(&[4, -1]).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_view_wrong_neg() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let _x = t.view(&[-1, -1]).unwrap();
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
}
