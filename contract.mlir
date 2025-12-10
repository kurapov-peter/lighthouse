module {
    func.func @linear_contract(%arg0: tensor<4x16xf32>, %arg1: tensor<32x16xf32>) -> tensor<4x32xf32> {
        %c0 = arith.constant 0.0 : f32
        %init = tensor.empty() : tensor<4x32xf32>
        %fill = linalg.fill ins(%c0 : f32) outs(%init : tensor<4x32xf32>) -> tensor<4x32xf32>
        %result = "linalg.contract"(%arg0, %arg1, %fill) <{
            indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
            iterator_types = ["parallel", "parallel", "reduction"],
            operandSegmentSizes = array<i32: 2, 1>
        }> ({
        ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
            %mul = arith.mulf %lhs, %rhs : f32
            %sum = arith.addf %acc, %mul : f32
            linalg.yield %sum : f32
        }) : (tensor<4x16xf32>, tensor<32x16xf32>, tensor<4x32xf32>) -> tensor<4x32xf32>
        return %result : tensor<4x32xf32>
    }
}
