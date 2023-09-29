#!/bin/bash

service nginx start
nginx -t
service nginx restart

fwatchdog