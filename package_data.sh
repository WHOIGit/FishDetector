#!/bin/bash

rm -f data.tar
tar cf data.tar train.txt test.txt
tar rf data.tar -T train.txt
tar rf data.tar -T test.txt
tar rf data.tar -T <(sed -e 's/\.jpg/\.txt/' train.txt)
tar rf data.tar -T <(sed -e 's/\.jpg/\.txt/' test.txt)
