module {
    func.func @linear_no_contract(%arg0: tensor<4x32xf32>) -> tensor<4x32xf32> {
        %c0 = arith.constant 1.0 : f32
        %init = tensor.empty() : tensor<4x32xf32>
        %fill = linalg.fill ins(%c0 : f32) outs(%init : tensor<4x32xf32>) -> tensor<4x32xf32>
        return %fill : tensor<4x32xf32>
    }
}
