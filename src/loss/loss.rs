// use crate::tensor::tensor::Tensor;
//
// pub trait Loss {
//     fn loss(actual: &Tensor, prediction: &Tensor) -> f64;
//     fn grad(actual: &Tensor, prediction: &Tensor) -> Tensor;
// }
//
// pub struct MSE {}
// pub struct CrossEntropy {}
//
// pub enum Losses {
//     MSE(MSE),
//     CrossEntropy(CrossEntropy),
// }
//
// impl Loss for MSE {
//     fn loss(actual: &Tensor, prediction: &Tensor) -> f64 {
//         (prediction - actual).pow(2.0).sum(0).unwrap().data[0]
//     }
//
//     fn grad(actual: &Tensor, prediction: &Tensor) -> Tensor {
//         &(prediction - actual) * 2.0
//     }
// }
//
// impl Loss for CrossEntropy {
//     fn loss(actual: &Tensor, prediction: &Tensor) -> f64 {
//         let t = - (actual * prediction.log2().unwrap() + (1.0 - actual) * (1.0 - prediction).log2().unwrap()).mean();
//         t.data[0]
//     }
//
//     fn grad(actual: &Tensor, prediction: &Tensor) -> Tensor {
//         (actual / prediction) - ((1.0 - actual) / (1.0 - prediction))
//     }
// }
//
// #[cfg(test)]
// mod test {
//     use super::*;
//
//     #[test]
//     fn test_mse_loss() {
//         let pred_data = vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0];
//         let pred = Tensor::new(pred_data, &[6]).unwrap();
//         let actual_data = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
//         let actual = Tensor::new(actual_data, &[6]).unwrap();
//         let loss = MSE::loss(&actual, &pred);
//         assert_eq!(loss, 14.0)
//     }
// }