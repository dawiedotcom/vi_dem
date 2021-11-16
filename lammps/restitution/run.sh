#!/bin/bash

#TIMESTEPS='0.00005 0.00001'
TIMESTEPS='7.2360125455826764e-06 8.962517674617792e-06 1.1100965146456642e-05 1.3749643979151248e-05 1.7030294849070365e-05 2.1093705632382182e-05 2.6126642036962752e-05 3.236043187592832e-05 4.008159753997746e-05 4.964502536666592e-05 6.14902772076058e-05 7.616179391877173e-05 9.43339193827535e-05 0.00011684189523690456 0.00014472025091165353 0.00014472025091165353 0.00019042138277849152 0.00023612251464532946 0.00028182364651216737 0.00032752477837900534 0.00037322591024584336 0.00041892704211268127 0.0004646281739795192 0.0005103293058463572 0.0005560304377131951 0.000601731569580033 0.0006474327014468712 0.000693133833313709 0.000738834965180547 0.000784536097047385 0.0008302372289142229 0.0008759383607810608 0.0009216394926478987 0.0009673406245147367 0.0010130417563815747'
THETAS='0' # 10'
#GAMMAS='0 10 30 100'
GAMMAS='0' # 10 20 30'
OMEGAS='0' # 3.14'
RUN_STYLE='verlet'
LMPS=lmp 
IN_SCRIPT='restitution.lammps'

DUMP_DIR=dump_$RUN_STYLE
if [ ! -d $DUMP_DIR ]; then
    mkdir $DUMP_DIR
else
    rm $DUMP_DIR/*
fi

for TIMESTEP in $TIMESTEPS
do
    for GAMMA_N in $GAMMAS
    do
        for OMEGAZ in $OMEGAS
        do
            for THETA in $THETAS
            do
                FBASENAME=${TIMESTEP}_${GAMMA_N}_${OMEGAZ}_${THETA}
                ${LMPS} -in ${IN_SCRIPT} -var TIMESTEP $TIMESTEP -var GAMMA_N $GAMMA_N -var OMEGAZ ${OMEGAZ} -var THETA ${THETA}
                ./extract_thermo.awk $DUMP_DIR/log_$FBASENAME.lammps > $DUMP_DIR/restitution_$FBASENAME.thermo
                ./extract_atom_one.awk $DUMP_DIR/lammps_$FBASENAME.dump > $DUMP_DIR/atom_one_$FBASENAME.dump
            done
        done
    done
done
