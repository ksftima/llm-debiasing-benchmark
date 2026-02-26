#! /bin/bash

if [ ! -f ~/.Renviron ]; then
    echo "Set your .Renviron first"
    exit 1
fi

echo "---------------------"
echo "-- Loading modules --"
echo "---------------------"

module purge
module load rpy2 scikit-learn/1.4.2-gfbf-2023a

echo "-----------------------"
echo "-- Installing R libs --"
echo "-----------------------"

R -e "install.packages(\"devtools\", repos=\"https://cloud.r-project.org/\")"
R -e "library(devtools); install_github(\"naoki-egami/dsl\", dependencies=TRUE)"

echo "-------------------------"
echo "-- Installing the venv --"
echo "-------------------------"

virtualenv --system-site-packages /mimer/NOBACKUP/groups/ci-nlp-alvis/dsl-use/venv/vera
source /mimer/NOBACKUP/groups/ci-nlp-alvis/dsl-use/venv/vera
python -m pip install --no-cache-dir -r requirements_cluster.txt

echo "-------------------------"
echo "-- Running test script --"
echo "-------------------------"

python ./lib/test-dsl.py
