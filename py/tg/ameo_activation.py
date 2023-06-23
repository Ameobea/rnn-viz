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


def mk_leaky_ameo_gpu(leakyness: float):
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
            y = -1.0 + (x + 3.0) * LEAKYNESS;
        } else if (x <= -1.0) {
            y = x + 2.0;
        } else if (x <= 1.0) {
            y = -x;
        } else if (x <= 3.0) {
            y = x - 2.0;
        } else {
            y = 1.0 + (x - 3.0) * LEAKYNESS;
        }
        c[idx] = y;
        }
        """.replace(
                "LEAKYNESS", str(leakyness)
            ),
            global_size=[prod(ret.shape)],
        ).build(Device[ret.device].runtime).exec([ret, x])
        return ret.realized

    return ameo_gpu


def mk_leaky_ameo_grad_gpu(leakyness: float):
    def leaky_ameo_grad_gpu(ret: LazyBuffer, x: LazyBuffer, grad_output: LazyBuffer):
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
          float y = LEAKYNESS;
          if (x <= -3.0) {
            y = LEAKYNESS;
          } else if (x <= -1.0) {
            y = 1.0;
          } else if (x <= 1.0) {
            y = -1.0;
          } else if (x <= 3.0) {
            y = 1.0;
          }
          outbuf[idx] = y * grad_output[idx];
        }
        """.replace(
                "LEAKYNESS", str(leakyness)
            ),
            global_size=[prod(ret.shape)],
        ).build(Device[ret.device].runtime).exec([ret, x, grad_output])
        return ret.realized

    return leaky_ameo_grad_gpu


def mk_leaky_ameo(leakyness: float = 0.1) -> Function:
    class LeakyAmeo(Function):
        def forward(self, x: LazyBuffer) -> LazyBuffer:
            self.x = x
            ast = LazyOp(
                LoadOps.CUSTOM,
                (x.contiguous(),),
                {"GPU": mk_leaky_ameo_gpu(leakyness)}[x.device],
            )
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
                {"GPU": mk_leaky_ameo_grad_gpu(leakyness)}[self.x.device],
            )
            return create_lazybuffer(
                self.x.device, self.x.shape, LoadOps, ast, max(self.x.dtype, grad.dtype)
            )

    return LeakyAmeo


def mk_interpolated_ameo_gpu(factor: float, leakyness: float):
    def interpolated_ameo_gpu(ret: LazyBuffer, x: LazyBuffer):
        assert x.device == "GPU", "gpu function requires GPUBuffers"
        assert x.dtype == dtypes.float32, "gpu function only supports float32"
        ret.realized = Device[ret.device].buffer(prod(ret.shape), ret.dtype)
        ASTRunner(
            "interpolated_ameo_gpu",
            """
        __kernel void interpolated_ameo_gpu(global float *c, global float *a) {
            int idx = get_global_id(0);
            float x = a[idx];

            float y0 = 1.0;
            if (x <= -3.0) {
                y0 = -1.0 + (x + 3.0) * LEAKYNESS;
            } else if (x <= -1.0) {
                y0 = x + 2.0;
            } else if (x <= 1.0) {
                y0 = -x;
            } else if (x <= 3.0) {
                y0 = x - 2.0;
            } else {
                y0 = 1.0 + (x - 3.0) * LEAKYNESS;
            }

            x *= 0.5;
            x -= 0.5;
            float y1 = 1.0;

            if (x <= -2.0) {
                y1 = LEAKYNESS * (x + 2.0);
            } else if (x <= -1.5) {
                float xPlus2 = x + 2.0;
                y1 = 8.0 * (xPlus2 * xPlus2 * xPlus2 * xPlus2);
            } else if (x <= -0.5) {
                y1 = -8.0 * (x * x * x * x) - 32.0 * (x * x * x) - 48.0 * (x * x) - 32.0 * x - 7.0;
            } else if (x <= 0.5) {
                y1 = 8.0 * (x * x * x * x);
            } else if (x <= 1.0) {
                y1 = -8.0 * (x * x * x * x) + 32.0 * (x * x * x) - 48.0 * (x * x) + 32.0 * x - 7.0;
            } else {
                y1 = LEAKYNESS * (x - 1.0) + 1.0;
            }

            y1 = (y1 - 0.5) * 2.0;


            c[idx] = y0 * FACTOR + (1.0 - FACTOR) * y1;
        }
        """.replace(
                "LEAKYNESS", str(leakyness)
            ).replace(
                "FACTOR", str(factor)
            ),
            global_size=[prod(ret.shape)],
        ).build(Device[ret.device].runtime).exec([ret, x])
        return ret.realized

    return interpolated_ameo_gpu


def mk_interpolated_ameo_grad_gpu(factor: float, leakyness: float):
    def interpolated_ameo_grad_gpu(ret: LazyBuffer, x: LazyBuffer, grad_output: LazyBuffer):
        assert x.device == "GPU" and grad_output.device == "GPU", "gpu function requires GPUBuffers"
        assert (
            x.dtype == dtypes.float32 and grad_output.dtype == dtypes.float32
        ), "gpu function only supports float32"
        ret.realized = Device[ret.device].buffer(prod(ret.shape), ret.dtype)
        ASTRunner(
            "interpolated_ameo_grad_gpu",
            """
        __kernel void interpolated_ameo_grad_gpu(global float *outbuf, global float *a, global float *grad_output) {
            int idx = get_global_id(0);
            float x = a[idx];
            float y0 = LEAKYNESS;
            if (x <= -3.0) {
            y0 = LEAKYNESS;
            } else if (x <= -1.0) {
            y0 = 1.0;
            } else if (x <= 1.0) {
            y0 = -1.0;
            } else if (x <= 3.0) {
            y0 = 1.0;
            }

            x *= 0.5;
            x -= 0.5;
            float y1 = 1.0;

            if (x <= -2.0 || x >= 1.0) {
                y1 = LEAKYNESS;
            } else if (x <= -1.5) {
                float xPlus2 = x + 2.0;
                y1 = 32.0 * (xPlus2 * xPlus2 * xPlus2);
            } else if (x <= -0.5) {
                y1 = -32.0 * (x * x * x) - 96.0 * (x * x) - 96.0 * x - 32.0;
            } else if (x <= 0.5) {
                y1 = 32.0 * (x * x * x);
            } else if (x <= 1.0) {
                y1 = -32.0 * (x * x * x) + 96.0 * (x * x) - 96.0 * x + 32.0;
            }

            outbuf[idx] = (y0 * FACTOR + (1.0 - FACTOR) * y1) * grad_output[idx];
        }
        """.replace(
                "LEAKYNESS", str(leakyness)
            ).replace(
                "FACTOR", str(factor)
            ),
            global_size=[prod(ret.shape)],
        ).build(Device[ret.device].runtime).exec([ret, x, grad_output])
        return ret.realized

    return interpolated_ameo_grad_gpu


def mk_interpolated_ameo(factor: float, leakyness: float = 0.1) -> Function:
    class InterpolatedAmeo(Function):
        def forward(self, x: LazyBuffer) -> LazyBuffer:
            self.x = x
            ast = LazyOp(
                LoadOps.CUSTOM,
                (x.contiguous(),),
                {"GPU": mk_interpolated_ameo_gpu(factor, leakyness)}[x.device],
            )
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
                {"GPU": mk_interpolated_ameo_grad_gpu(factor, leakyness)}[self.x.device],
            )
            return create_lazybuffer(
                self.x.device, self.x.shape, LoadOps, ast, max(self.x.dtype, grad.dtype)
            )

    return InterpolatedAmeo
