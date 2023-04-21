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
rm ~/repo_lattice_new/*.c
rm ~/repo_lattice_new/*.so
for FILE in $FILES
do
    echo "Preparing for processing" $FILE"."
    export FILE="$FILE"
    qsub -N $FILE -l mem=16gb -V $BASEDIR/pbs_generate.job
    sleep 2
done
