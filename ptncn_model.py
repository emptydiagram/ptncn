import torch
import torch.nn as nn
import torch.nn.functional as F

def init_weight_matrix(shape, dtype):
    stddev = 0.05
    return torch.nn.Parameter(torch.normal(0, stddev, size=shape).type(dtype))

class PTNCN(nn.Module):
    def __init__(self, input_size, hidden_size, lambda_=0.1, beta=0.5, gamma=0.5, weight_dtype=None):
        super(PTNCN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lambda_ = lambda_
        self.beta = beta
        self.gamma = gamma

        # this doesn't appear in the paper, but the implementation actually doesn't use the
        # "temporal" error calculation found in the paper. instead, it uses something reminiscent
        # of the LRA-E update rule
        self.use_temporal_error = False

        # scale factor non-temporal error rule
        self.alpha = 1.0

        if weight_dtype is None:
            weight_dtype = torch.float32

        # top-down state-update weights
        self.U1 = init_weight_matrix([hidden_size, hidden_size], weight_dtype)

        # recurrent state-update weights
        self.V1 = init_weight_matrix([hidden_size, hidden_size], weight_dtype)
        self.V2 = init_weight_matrix([hidden_size, hidden_size], weight_dtype)

        # bottom-up state-update weights
        self.M1 = init_weight_matrix([input_size, hidden_size], weight_dtype)
        self.M2 = init_weight_matrix([hidden_size, hidden_size], weight_dtype)

        # (bottom-up) error weights
        self.R1 = init_weight_matrix([input_size, hidden_size], weight_dtype)
        self.R2 = init_weight_matrix([hidden_size, hidden_size], weight_dtype)

        # top-down prediction weights
        self.W1 = init_weight_matrix([hidden_size, input_size], weight_dtype)
        self.W2 = init_weight_matrix([hidden_size, hidden_size], weight_dtype)

        # update function
        self.fn_update = torch.tanh

        # predict function
        self.fn_predict = F.softmax

        self.state_initialized = False

    def forward(self, x, padding_mask):
        batch_size = x.shape[0]
        if not self.state_initialized:
            self.z0 = torch.zeros_like(x)
            self.z1 = torch.zeros((batch_size, self.hidden_size))
            self.z2 = torch.zeros((batch_size, self.hidden_size))
            self.y1 = torch.zeros_like(self.z1)
            self.y2 = torch.zeros_like(self.z2)
            self.d1 = torch.zeros_like(self.z1)
            self.d2 = torch.zeros_like(self.z2)
            self.state_initialized = True

        ### update ###
        self.z0_prev = self.z0
        self.z1_prev = self.z1
        self.z2_prev = self.z2

        self.a2 = self.y2 @ self.V2 + self.y1 @ self.M2
        self.z2 = self.fn_update(self.a2)

        self.a1 = self.y2 @ self.U1 + self.y1 @ self.V1 + self.z0_prev @ self.M1
        self.z1 = self.fn_update(self.a1)

        self.z0 = x

        ### predict ###

        # contrary to the paper, the official code does not apply a function to the z1 prediction, only to z0_pred
        # self.z1_pred = self.fn_predict(self.z2 @ self.W2, dim=1)
        self.z1_pred = self.z2 @ self.W2
        self.z0_pred = self.fn_predict(self.z1 @ self.W1, dim=1)

        ### pre-correction ###
        self.d1_prev = self.d1
        self.d2_prev = self.d2

        # to zero out gradients for padding inputs, we simply need to mask the error values
        padding_mask = padding_mask.unsqueeze(-1)

        self.e0 = (self.z0_pred - self.z0) * padding_mask
        self.e1 = (self.z1_pred - self.z1) * padding_mask

        self.y1 = self.fn_update(self.a1 - (self.beta * (self.e0 @ self.R1) - self.gamma * self.e1 - self.lambda_ * torch.sign(self.z1)))
        self.y2 = self.fn_update(self.a2 - (self.beta * (self.e1 @ self.R2) - self.lambda_ * torch.sign(self.z2)))

        self.d1 = (self.z1 - self.y1) * padding_mask
        self.d2 = (self.z2 - self.y2) * padding_mask

    def compute_correction(self):
        batch_size = 1.0 * self.z0.shape[0]

        dW1 = self.z1.T @ self.e0
        dW2 = self.z2.T @ self.e1

        if self.use_temporal_error:
            dR1 = self.e0.T @ (self.d1 - self.d1_prev)
            dR2 = self.e1.T @ (self.d2 - self.d2_prev)
        else:
            dR1 = self.alpha * dW1.T
            dR2 = self.alpha * dW2.T

        dM1 = self.z0_prev.T @ self.d1
        dM2 = self.z1_prev.T @ self.d2

        dV1 = self.z1_prev.T @ self.d1
        dV2 = self.z2_prev.T @ self.d2

        dU1 = self.z2_prev.T @ self.d1

        self.W1.grad = dW1 / batch_size
        self.W2.grad = dW2 / batch_size
        self.R1.grad = dR1 / batch_size
        self.R2.grad = dR2 / batch_size
        self.M1.grad = dM1 / batch_size
        self.M2.grad = dM2 / batch_size
        self.V1.grad = dV1 / batch_size
        self.V2.grad = dV2 / batch_size
        self.U1.grad = dU1 / batch_size
