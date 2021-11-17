#!/bin/bash

LMPS=lmp
RUN_SCRIPT='box.lammps'
GEN_SCRIPT='box_generate.lammps'

#TIMESTEPS='0.001 0.00075 0.0005 0.0001 0.00001' #'0.0005 0.0001 0.00005 0.00001'
TIMESTEPS='0.0001  0.00001' #'0.0005 0.0001 0.00005 0.00001'
#DIAMETERS='1.0 0.8 0.6' # diameters of approx 0.5^(1/3) and 0.25^(1/3) should double the particle count
N_PARTS='50 100' 

#for N in $N_PARTS
#do
#    ${LMPS} -in ${GEN_SCRIPT} -var N_PART $N
#done

for N in $N_PARTS
do
    for TIMESTEP in $TIMESTEPS
    do
        DUMP_DIR="dump_${N}_${TIMESTEP}"
        echo "$DUMP_DIR"
        if [ ! -d "$DUMP_DIR" ]; then
            echo "mkdir $DUMP_DIR"
            mkdir "$DUMP_DIR"
        fi
        ${LMPS} -in ${RUN_SCRIPT} -var DT $TIMESTEP -var N_PART $N

        #${LMPS} -in ${RUN_SCRIPT} -var TIMESTEP $TIMESTEP -var GAMMA_N $GAMMA_N -var OMEGAZ ${OMEGAZ}
        #./extract_thermo.awk log_$FBASENAME.lammps > restitution_$FBASENAME.thermo
        #./extract_atom_one.awk lammps_$FBASENAME.dump > atom_one_$FBASENAME.dump
    done
done
