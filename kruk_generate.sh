#!/bin/bash

BASEDIR=$(dirname "$0")
PARAMETERS_PATH=$PWD

if [ "x$@" == "x" ]
then
  FILES=`find $PARAMETERS_PATH -name parameters*.py`
else
  FILES="$@"
fi

for FILE in $FILES
do
    echo "Preparing for processing" $FILE"."
    export FILE="$FILE"
    qsub -N $FILE -l mem=2gb -V $BASEDIR/pbs_generate.job
done
