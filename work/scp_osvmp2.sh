#for inode in nitrogen01 nitrogen02 oxygen01 oxygen02;do
for inode in oxygen02;do
    scp -r osvmp2/ qjliang@$inode:/data1/qjliang/work/
    scp -r osvmp2/ qjliang@$inode:/data1/qjliang/work2/
done
