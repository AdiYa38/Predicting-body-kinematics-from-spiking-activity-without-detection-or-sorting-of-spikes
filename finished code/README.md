This directory contains all the code files for this project.

This project wad made in 3 parts:
  -general
  - Unit activity
  - MUA

*General- 
contains files that are the base of the project:
  -data.py: contains functions to import the data from the given data files.
  -heatmaps.py: contains functions that handle all operations done on the maps.
  -prediction.py: contains functions that needed in the location prediction process.
  -test.py: a run file to visualize rate map creation process and test prediction using one unit.
  -test_prediction.py: a run file to test prediction using multiple units.

  *Unit activity-



  *MUA-
  Contains files used in the analysis of MUA.
    -MUA.py: contains function to create and analyze MUA signals.
    - mua_test.py: a run file to test MUA creation process, and test consistant areas between differnt maps
    - mua_predictions.py: a run file to test predictions using MUA signals.
    - MUA_corr.py: a run file that creates an xl file of correlation matrix between MUA channels
    - mua_ripple.py: a run file that removes ripple events from channels and prints the new rate maps.
