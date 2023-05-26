# !/bin/bash

set -euxo pipefail

prefix=
. parse_options.sh

[ -z $prefix ] && echo "Error: Missing --prefix" && exit 1

python MLSW/filter_data.py --lang en --word-count-min 4500 --word-count-max 5000 --prefix $prefix
python MLSW/filter_data.py --lang de --word-count-min 4000 --word-count-max 5000 --prefix $prefix
python MLSW/filter_data.py --lang fr --word-count-min 4000 --word-count-max 5000 --prefix $prefix
python MLSW/filter_data.py --lang ru --word-count-min 1000 --word-count-max 5000 --prefix $prefix
