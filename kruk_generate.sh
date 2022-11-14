#!/bin/bash
export CFLAGS=-I"/opt/anaconda3/lib/python3.9/site-packages/numpy/core/include/"
BASEDIR=$(dirname "$0")
PARAMETERS_PATH=$PWD
echo "$BASEDIR"
export BASEDIR="$BASEDIR"
if [ "x$@" == "x" ]
then
  FILES=`find $PARAMETERS_PATH -name parameters*.py`
else
  FILES="$@"
fi
rm ~/repo_lattice/lattice_cython/sym_dr_010/*.c
rm ~/repo_lattice/lattice_cython/sym_dr_010/*.so
for FILE in $FILES
do
    echo "Preparing for processing" $FILE"."
    export FILE="$FILE"
    qsub -N $FILE -l mem=2gb -V $BASEDIR/pbs_generate.job
    sleep 20
done
