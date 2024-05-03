(
echo d
echo 1
echo d
echo 2
echo d
echo 3
echo d
echo n
echo p
echo 1
echo 
echo 
echo Y
echo a
echo w
) | sudo fdisk /dev/sda


head -n -1 /etc/fstab > temp_fstab
sudo mv temp_fstab /etc/fstab

echo "@reboot sudo resize2fs /dev/sda1" | crontab -
sudo reboot
