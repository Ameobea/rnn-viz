from tinygrad.tensor import Tensor, Function
from tinygrad.lazy import LazyBuffer, Device, create_lazybuffer, LazyOp
from tinygrad.ops import BinaryOps, ASTRunner, LoadOps
from tinygrad.helpers import dtypes, prod


def where_raw(cond: LazyBuffer, input_: LazyBuffer, other: LazyBuffer):
    inv_cond = cond.binary_op(BinaryOps.CMPEQ, cond.const_like(0.0))
    y = inv_cond.binary_op(BinaryOps.MUL, other)
    y = y.binary_op(
        BinaryOps.ADD,
        input_.binary_op(
            BinaryOps.MUL,
            cond.const_like(1.0).binary_op(BinaryOps.SUB, inv_cond),
        ),
    )


def ameo_forward(x: Tensor):
    x = x.mul(0.5)
    x = x.sub(0.5)
    y = (x <= -1.0).where(x.add(2.0).maximum(0.0), (x <= 0.0).where(x.mul(-1.0), x.minimum(1.0)))
    y = y.sub(0.5).mul(2.0)
    return y


def ameo_grad(x: Tensor):
    (x <= -1.0) * (x > -2.0) + ((x <= 1.0) * (x > 0.0)).where(
        1.0, (x <= 0.0) * (x > -1.0).where(-1.0, 0.0)
    )


def ameo_gpu(ret: LazyBuffer, x: LazyBuffer):
    assert x.device == "GPU", "gpu function requires GPUBuffers"
    assert x.dtype == dtypes.float32, "gpu function only supports float32"
    ret.realized = Device[ret.device].buffer(prod(ret.shape), ret.dtype)
    ASTRunner(
        "ameo_gpu",
        """
    __kernel void ameo_gpu(global float *c, global float *a) {
      int idx = get_global_id(0);
      float x = a[idx];
      float y = 1.0;
      if (x <= -3.0) {
        y = -1.0;
      } else if (x <= -1.0) {
        y = x + 2.0;
      } else if (x <= 1.0) {
        y = -x;
      } else if (x <= 3.0) {
        y = x - 2.0;
      } else {
        y = 1.0;
      }
      c[idx] = y;
    }
    """,
        global_size=[prod(ret.shape)],
    ).build(Device[ret.device].runtime).exec([ret, x])
    return ret.realized


def ameo_grad_gpu(ret: LazyBuffer, x: LazyBuffer, grad_output: LazyBuffer):
    assert x.device == "GPU" and grad_output.device == "GPU", "gpu function requires GPUBuffers"
    assert (
        x.dtype == dtypes.float32 and grad_output.dtype == dtypes.float32
    ), "gpu function only supports float32"
    ret.realized = Device[ret.device].buffer(prod(ret.shape), ret.dtype)
    ASTRunner(
        "ameo_grad_gpu",
        """
    __kernel void ameo_grad_gpu(global float *outbuf, global float *a, global float *grad_output) {
      int idx = get_global_id(0);
      float x = a[idx];
      float y = 0.0;
      if (x <= -3.0) {
        y = 0.0;
      } else if (x <= -1.0) {
        y = 1.0;
      } else if (x <= 1.0) {
        y = -1.0;
      } else if (x <= 3.0) {
        y = 1.0;
      }
      outbuf[idx] = y * grad_output[idx];
    }
    """,
        global_size=[prod(ret.shape)],
    ).build(Device[ret.device].runtime).exec([ret, x, grad_output])
    return ret.realized


class Ameo(Function):
    def forward(self, x: LazyBuffer) -> LazyBuffer:
        self.x = x
        ast = LazyOp(LoadOps.CUSTOM, (x.contiguous(),), {"GPU": ameo_gpu}[x.device])
        return create_lazybuffer(x.device, x.shape, LoadOps, ast, x.dtype)

    def backward(self, grad: LazyBuffer) -> LazyBuffer:
        if not self.needs_input_grad[0]:
            return None

        assert grad.device == self.x.device, "grad and input must be on same device"
        assert grad.dtype == self.x.dtype, "grad and input must be same dtype"
        assert prod(grad.shape) == prod(self.x.shape), "grad and input must be same shape"

        ast = LazyOp(
            LoadOps.CUSTOM,
            (self.x.contiguous(), grad.contiguous()),
            {"GPU": ameo_grad_gpu}[self.x.device],
        )
        return create_lazybuffer(
            self.x.device, self.x.shape, LoadOps, ast, max(self.x.dtype, grad.dtype)
        )
