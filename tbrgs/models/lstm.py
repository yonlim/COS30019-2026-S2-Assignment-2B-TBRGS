"""LSTM (Long Short-Term Memory) traffic flow model.

Owner: Lim Yon (105811100)

An LSTM cell keeps a cell state c_t that runs along the sequence and three
gates that decide what to forget, what to add and what to output at each
15-minute step (Hochreiter and Schmidhuber, 1997):

    f_t = sigmoid(W_f [h_(t-1), x_t] + b_f)        forget gate
    i_t = sigmoid(W_i [h_(t-1), x_t] + b_i)        input gate
    o_t = sigmoid(W_o [h_(t-1), x_t] + b_o)        output gate
    c_t = f_t * c_(t-1) + i_t * tanh(W_c [h_(t-1), x_t] + b_c)
    h_t = o_t * tanh(c_t)

The separate cell state lets the network carry information, such as "flows
have been rising for the last hour", across the whole 3-hour input window.
The layers and training settings are shared with the GRU (recurrent.py).
"""

from .recurrent import RecurrentFlowModel


class LSTMModel(RecurrentFlowModel):
    name = "lstm"
    cell = "LSTM"
