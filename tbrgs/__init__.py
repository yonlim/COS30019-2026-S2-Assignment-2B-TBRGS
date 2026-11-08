"""Traffic-Based Route Guidance System (COS30019 Assignment 2, Part B).

Package layout (owner in brackets; see README.txt for the full split):

    config.py        reads and writes config.ini                      (Lau Ka Meng)
    data.py          Task 1: data processing                          (Lim Yon)
    metrics.py       error measures used to compare the models        (Lim Yon)
    models/          Task 2: LSTM (Lim), GRU (Jonathan), XGBoost (Lau),
                     plus two simple baselines used in the evaluation
    predictor.py     turns a trained model into flow predictions      (Lim Yon)
    travel_time.py   Task 3: flow -> speed -> travel time             (Jonathan Yu)
    network.py       Task 3: the Boroondara road network              (Jonathan Yu)
    routing.py       Task 4: top-k routes with the Part A search code (Lau Ka Meng)
    system.py        joins everything together for the CLI and GUI   (Lau Ka Meng)
    mapview.py       OpenStreetMap route map (research)              (Jonathan Yu)
"""

__version__ = "1.0"
