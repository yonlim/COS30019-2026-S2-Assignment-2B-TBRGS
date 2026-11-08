COS30019 Assignment 2B: Traffic-Based Route Guidance System (TBRGS)
===================================================================

Team: Lim Yon (105811100), Jonathan Yu Yung Jing (105815665), Lau Ka Meng (105814507)

The TBRGS predicts traffic flow at 40 SCATS intersections in Boroondara
(VicRoads data, October 2006) with three machine learning models (LSTM, GRU
and XGBoost), turns the predicted flows into travel times, and uses the
team's Part A search methods to find up to five routes between any two
intersections.


1  Setting up (Windows, Command Prompt, in this folder)
--------------------------------------------------------

    py -3.13 -m venv .venv            (Python 3.11, 3.12 or 3.13)
    .venv\Scripts\activate
    pip install -r requirements.txt

Trained models are already in saved_models\, so the system can be used
straight away. Retraining is only needed after changing a model setting.


2  Running the program
----------------------

    python gui.py                      the graphical interface (recommended)

    python main.py                     one route query from the command line,
                                       using the defaults in config.ini
    python main.py -o 2000 -d 3002 --date 2006-10-25 --time 08:00 --model lstm --method AS -k 5
    python main.py --sites             list the 40 SCATS sites

    --model   lstm, gru or xgboost (trained models), historical (historical
              average) or observed (the measured flows, for comparison)
    --method  a Part A search method: AS, CUS1, BFS, DFS, GBFS or CUS2
              (only AS and CUS1 guarantee the fastest routes)
    -k        number of routes, 1 to 5

Dates outside 1-31 October 2006 are mapped to the same weekday and time in
the last week of the data, and the program says so.

GUI tips: left-click a site on the map to make it the origin, right-click
for the destination. "Open map in browser" draws the routes on
OpenStreetMap (needs folium and an internet connection for the map tiles).


3  Training, evaluation, tests and research
-------------------------------------------

    python train.py --model all        train LSTM, GRU and XGBoost (about 40 min on a 2-core CPU)
    python train.py --model xgboost    one model; --epochs N overrides config.ini; --quick for a check
    python evaluate.py                 compare all models on the test week -> results\
    python run_test_cases.py           the 25 documented system test cases -> tests\TEST_CASES.txt
    python -m unittest discover -v     the 79 unit tests (a few minutes; the GUI test
                                       is skipped when there is no display)
    python research.py all             the research experiments -> results\research_*.csv

Data split (by date): training 1-21 Oct 2006, validation (early stopping)
22-24 Oct, test 25-31 Oct. No model ever sees the test week while training.


4  Files and who owns them
--------------------------

    gui.py                  graphical interface                       Lau Ka Meng
    main.py                 command-line route finder                 Lau Ka Meng
    train.py                trains and saves the models               Lim Yon
    evaluate.py             model comparison (+ route impact: Lau)    Lim Yon
    run_test_cases.py       documented system test cases              Lau Ka Meng (cases by owner)
    research.py             research experiments                      all (one study each)
    config.ini              default settings (GUI can save new ones)  Lau Ka Meng

    tbrgs\data.py           Task 1: data processing                   Lim Yon
    tbrgs\metrics.py        error measures                            Lim Yon
    tbrgs\training.py       shared training code                      Lim Yon
    tbrgs\predictor.py      model -> flow predictions for a time      Lim Yon
    tbrgs\models\base.py    common model interface                    Lim Yon
    tbrgs\models\lstm.py    LSTM                                      Lim Yon
    tbrgs\models\gru.py     GRU                                       Jonathan Yu
    tbrgs\models\recurrent.py  layers/training shared by LSTM and GRU Lim Yon, Jonathan Yu
    tbrgs\models\xgboost_model.py  XGBoost (third algorithm)          Lau Ka Meng
    tbrgs\models\baselines.py  persistence and historical average     Lim Yon
    tbrgs\travel_time.py    Task 3: flow -> speed -> travel time      Jonathan Yu
    tbrgs\network.py        Task 3: the road network                  Jonathan Yu
    tbrgs\mapview.py        OpenStreetMap route map (research)        Jonathan Yu
    tbrgs\routing.py        Task 4: Part A methods + Yen's top-k      Lau Ka Meng
    tbrgs\system.py         the end-to-end pipeline                   Lau Ka Meng
    part_a\                 Part A search code, unchanged             (whole team, from A2A)

    data\Scats Data October 2006.xls            the VicRoads dataset (unit resource)
    data\SCATSSiteListingSpreadsheet_VicRoads.xls  site descriptions (unit resource)
    data\Traffic_Count_Locations_with_LONG_LAT.csv used for the coordinate correction
    data\manual_links.csv   three road links added by hand, with reasons
    saved_models\           trained models and the scaler
    results\                evaluation and research outputs
    tests\                  unit tests (test_*.py) and TEST_CASES.txt


5  Known limitations
--------------------

* The data covers one month (October 2006) at 40 sites, so the models
  learn October 2006 traffic only. Other dates are mapped onto it.
* Link lengths are straight lines between sites and every link has the
  same 60 km/h limit and capacity, as the brief allows.
* The flow-to-speed curve assumes no road is over capacity, so predicted
  speeds never drop below 32 km/h: a real jam would be underestimated.
* 6 of the 104 directed links have no detector on the exact approach and
  use another detector at the same site instead (see the GUI's "Flow
  detector rule" column); 6 more use detectors named in manual_links.csv.
* Predictions are one step (15 minutes) ahead and every link uses the
  prediction for the departure time, even on long trips.
