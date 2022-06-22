aria2c -x 10 -j 10 http://images.cocodataset.org/zips/train2017.zip
aria2c -x 10 -j 10 http://images.cocodataset.org/zips/val2017.zip
aria2c -x 10 -j 10 http://images.cocodataset.org/zips/test2017.zip
unzip *.zip
rm *.zip