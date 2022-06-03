#!/bin/bash
if [ "x$@" == "x" ]
then
  FILES=`find . -maxdepth 1 -name 'parameters*.py'`
else
  FILES="$@"
fi

for FILE in $FILES;
do
  FILENAME=$(basename "$FILE")
  DIRECTORY=`echo $(basename "$FILE" .py) | sed 's/,/./g'`
  mkdir -p "$DIRECTORY"
  mv "$FILE" ""$DIRECTORY"/"$FILENAME""
done
