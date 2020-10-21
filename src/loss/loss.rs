use crate::tensor::tensor::Tensor;

pub trait Loss<'a> {
    fn loss(actual: &'a Tensor, prediction: &'a Tensor) -> Tensor<'a>;
    fn grad(actual: &'a Tensor, prediction: &'a Tensor) -> Tensor<'a>;
}

pub struct MSE {}
pub struct CrossEntropy {}

pub enum Losses {
    MSE(MSE),
    CrossEntropy(CrossEntropy),
}

impl<'a> Loss<'a> for MSE {
    fn loss(actual: &'a Tensor, prediction: &'a Tensor) -> Tensor<'a> {
        (prediction - actual).pow(2.0)
    }

    fn grad(actual: &'a Tensor, prediction: &'a Tensor) -> Tensor<'a> {
        let t = prediction - actual;
        let t = &t * 2.0;
        Tensor::new(t.data, &t.shape).unwrap()
    }
}

impl<'a> Loss<'a> for CrossEntropy {
    fn loss(actual: &'a Tensor, prediction: &'a Tensor) -> Tensor<'a> {
        // actual -> Tensor with shape [BS,]
        // prediction -> Tensor with shape [BS, n_classes] containing raw scores

        let mut x_class: Vec<f64> = Vec::new();

        for i in 0..actual.shape[0] {
            let p = *actual.get(&[i], &actual.strides) as usize;
            x_class.push(*prediction.get(&[i, p], &prediction.strides));
        }
        let x_class = Tensor::new(x_class, &[actual.shape[0]]).unwrap(); // we get tensor with shape [BS, 1]

        let l = prediction.exp();
        let l = l.sum(1).unwrap().ln(); // we get tensor with shape [BS, 1]

        let loss_t = &l - &x_class;
        Tensor::new(loss_t.data, &loss_t.shape).unwrap()
    }

    fn grad(actual: &'a Tensor, prediction: &'a Tensor) -> Tensor<'a> {
        // (actual / prediction) - ((1.0 - actual) / (1.0 - prediction))
        let numerator = prediction.exp();
        let denominator = numerator.sum(-1).unwrap();
        let mut softmax = &numerator / &denominator;
        for i in 0..actual.shape[0] {
            let p = *actual.get(&[i], &actual.strides) as usize;
            let logical_idxs: [usize; 2] = [i, p];
            let curr_val = *softmax.get(&logical_idxs, &softmax.strides);
            softmax.set(&logical_idxs, curr_val - 1.0);
        }
        let derivative = &softmax / actual.shape[0] as f64;
        Tensor::new(derivative.data, &derivative.shape).unwrap()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_mse_loss() {
        let pred_data = vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0];
        let pred = Tensor::new(pred_data, &[6]).unwrap();

        let actual_data = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let actual = Tensor::new(actual_data, &[6]).unwrap();

        let loss = MSE::loss(&actual, &pred);
        let mean_loss = loss.sum(-1).unwrap().data[0];

        assert_eq!(mean_loss, 14.0)
    }

    #[test]
    fn test_cross_entropy_loss() {
        let actual = Tensor::new(vec![1.0, 1.0, 1.0, 1.0], &[4]).unwrap();
        let scores = Tensor::new(vec![0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0], &[4, 2]).unwrap();

        let loss = CrossEntropy::loss(&actual, &scores);

        let truth = vec![0.3133, 1.3133, 0.3133, 1.3133];

        assert_eq!(loss.shape, actual.shape);

        for (l, t) in loss.data.iter().zip(truth.iter()) {
            let delta = (t - l).abs();
            assert!(delta < 0.1);
        }
    }
}
