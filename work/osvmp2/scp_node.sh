file_list="*.py test_functions.sh"
for inode in oxygen01 oxygen02 nitrogen01;do
#for inode in oxygen02;do
    scp  $file_list qjliang@"$inode":./OSV-BOMD-MPI
    scp  $file_list qjliang@"$inode":/data1/qjliang/OSV-BOMD-MPI
    scp  $file_list qjliang@"$inode":/data1/qjliang/OSV-BOMD-TEST
done
