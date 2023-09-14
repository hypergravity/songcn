#!/usr/bin/env bash
sudo mkdir /tmp/ramdisk
sudo chmod 777 /tmp/ramdisk
free -m
sudo mount -t tmpfs -o size=1024M tmpfs /tmp/ramdisk
# sudo umount -v /tmp/ramdisk

# to make sure the ram disk persists over reboot, add this to /etc/fstab:
# tmpfs   /tmp/ramdisk tmpfs   nodev,nosuid,noexec,nodiratime,size=1024M   0 0
# conf. https://www.jamescoyle.net/how-to/943-create-a-ram-disk-in-linux