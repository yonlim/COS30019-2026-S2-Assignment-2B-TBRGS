"""GRU (Gated Recurrent Unit) traffic flow model.

Owner: Jonathan Yu Yung Jing (105815665)

A GRU cell (Cho et al., 2014) merges the LSTM's cell state and hidden state
and uses two gates instead of three:

    z_t = sigmoid(W_z [h_(t-1), x_t] + b_z)          update gate
    r_t = sigmoid(W_r [h_(t-1), x_t] + b_r)          reset gate
    h~_t = tanh(W_h [r_t * h_(t-1), x_t] + b_h)      candidate state
    h_t = (1 - z_t) * h_(t-1) + z_t * h~_t

With the same number of units it has about a quarter fewer weights than an
LSTM, so it usually trains faster; whether it predicts as well on the
Boroondara data is one of the questions the evaluation answers. The layers
and training settings are shared with the LSTM (recurrent.py).
"""

from .recurrent import RecurrentFlowModel


class GRUModel(RecurrentFlowModel):
    name = "gru"
    cell = "GRU"
