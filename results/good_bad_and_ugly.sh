#!/bin/bash

pyopt aietes -odc Baseline.aietes.conf -T Baseline
pyopt aietes -odc Bad_Bravo.aietes.conf -T Bad_Bravo
pyopt bounos -s Baseline.aietes.npz Bad_Bravo.aietes.npz -c -D
