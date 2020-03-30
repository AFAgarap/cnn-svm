#!/bin/bash
if command -v python3 &>/dev/null; then
	echo Python 3 is installed
	if command -v pip &>/dev/null; then
		echo pip is installed
		sudo pip install -r requirements.txt
	else
		sudo apt-get install python3-pip python3-dev build-essential
		sudo pip install -r requirements.txt
	fi
else
	echo Python3 is not installed
	sudo apt-get update
	sudo apt-get install python3.6 python3-pip python3-dev build-essential
	sudo pip install --upgrade pip virtualenv
	sudo pip install -r requirements.txt
fi
