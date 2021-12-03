#!/bin/bash

TIMESTEPS='0.00001' # 0.00005' # 0.0001 0.00001 0.000001'
GAMMAS='0 10 30 100 300'
DYS='0' #0.1 0.3 0.5'
LMPS=../../../lammps/build/lmp

for TIMESTEP in $TIMESTEPS
do
    for GAMMA_N in $GAMMAS
    do
        for DY in $DYS
        do
            FBASENAME=${TIMESTEP}_${GAMMA_N}_${DY}
            ${LMPS} -in impact_damped.liggghts -var TIMESTEP $TIMESTEP -var GAMMA_N $GAMMA_N -var DY ${DY}
            ./extract_thermo.awk log_$FBASENAME.lammps > impact_$FBASENAME.thermo
            ./extract_atom_one.awk lammps_$FBASENAME.dump > atom_one_$FBASENAME.dump
        done
    done
done
