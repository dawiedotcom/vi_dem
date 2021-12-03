#!/bin/bash

TIMESTEPS='0.001 0.0003 0.0001 0.00003 0.00001 0.000003 0.000001'
LMPS=../../../lammps/build/lmp

for TIMESTEP in $TIMESTEPS
do
    ${LMPS} -in impact.liggghts -var TIMESTEP $TIMESTEP
    ./extract_thermo.awk log_$TIMESTEP.lammps > impact_$TIMESTEP.thermo
    ./extract_atom_one.awk lammps_$TIMESTEP.dump > atom_one_$TIMESTEP.dump
done
