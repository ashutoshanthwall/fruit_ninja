#!/bin/sh
# Lets the logged-in user create a virtual mouse via /dev/uinput (needed on Wayland).
#   sudo ./setup_wayland_mouse.sh           install
#   sudo ./setup_wayland_mouse.sh --remove  undo
set -e
RULE=/etc/udev/rules.d/70-uinput.rules

if [ "$(id -u)" -ne 0 ]; then
    echo "Run with sudo: sudo $0 $*" >&2
    exit 1
fi

if [ "$1" = "--remove" ]; then
    rm -f "$RULE"
    udevadm control --reload
    udevadm trigger --name-match=uinput
    echo "Removed $RULE"
    exit 0
fi

modprobe uinput
echo 'KERNEL=="uinput", SUBSYSTEM=="misc", TAG+="uaccess", OPTIONS+="static_node=uinput"' > "$RULE"
udevadm control --reload
udevadm trigger --name-match=uinput
udevadm settle
echo "Installed $RULE"
getfacl -p /dev/uinput 2>/dev/null | grep '^user:' || echo "If access is still missing, log out and back in."
