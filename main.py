"""Command-line route finder for the TBRGS.

Usage (from this folder):
    python main.py --origin 2000 --destination 3002
    python main.py -o 2000 -d 3002 --date 2006-10-25 --time 08:00 --model gru --method AS -k 5
    python main.py --sites                      list the 40 SCATS sites

Options left out take their default from config.ini.
--model is lstm, gru or xgboost (trained models), historical (the historical
average) or observed (the measured flows, for comparison).
--method is a Part A search method: AS, CUS1, BFS, DFS, GBFS or CUS2.
"""

import argparse
import sys

from tbrgs.config import load_config


def main(argv=None) -> int:
    config = load_config()
    parser = argparse.ArgumentParser(description="Traffic-based route guidance for Boroondara.")
    parser.add_argument("-o", "--origin", type=int, default=config.getint("gui", "origin"))
    parser.add_argument("-d", "--destination", type=int, default=config.getint("gui", "destination"))
    parser.add_argument("--date", default=config.get("gui", "date"), help="YYYY-MM-DD")
    parser.add_argument("--time", default=config.get("gui", "time"), help="HH:MM (rounded down to 15 min)")
    parser.add_argument("--model", default=config.get("gui", "model"))
    parser.add_argument("--method", default=config.get("routing", "method"))
    parser.add_argument("-k", type=int, default=config.getint("routing", "k"), help="number of routes (1-5)")
    parser.add_argument("--flow-site", choices=["destination", "origin"],
                        default=config.get("travel_time", "flow_site"))
    parser.add_argument("--sites", action="store_true", help="list the SCATS sites and exit")
    args = parser.parse_args(argv)

    from tbrgs.system import TBRGS  # imported here so --help is instant
    system = TBRGS(config)
    if args.sites:
        for site, row in system.data.sites.iterrows():
            print(f"{site:5d}  {row['name']:<32} {row['description']}")
        return 0
    try:
        result = system.find_routes(args.origin, args.destination, f"{args.date} {args.time}",
                                    model=args.model, method=args.method, k=args.k,
                                    flow_site=args.flow_site)
    except (ValueError, FileNotFoundError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print(system.describe(result))
    return 0 if result.routes else 1


if __name__ == "__main__":
    sys.exit(main())
