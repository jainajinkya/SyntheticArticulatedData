#!/bin/bash
DATADIR=$1
CUR_DIR=`pwd`
OBJ_TYPE=$2

# change .obj files to .stl:
for filename in ${DATADIR}textured_objs/*.obj; do
  ctmconv "$filename" "${filename/.obj/.stl}" > obj2res.txt;
done

# Correct mesh if needed
for filename in ${DATADIR}textured_objs/*.stl; do
  python ${CUR_DIR}/sapien_dataset/dataset_tools.py -i "$filename" --correct-mesh
done

# Change mesh file names in the urdf
cp ${DATADIR}mobility.urdf ${DATADIR}mobility_mujoco.urdf
sed -i -e 's/.obj"/.stl"/g' ${DATADIR}mobility_mujoco.urdf
xmlstarlet ed -L -s /robot -t elem -n mujoco -v "" -s /robot/mujoco -t elem -n compiler -v "" \
-i /robot/mujoco/compiler -t attr -n meshdir -v textured_objs/ -i /robot/mujoco/compiler -t attr -n convexhull -v true \
-i /robot/mujoco/compiler -t attr -n balanceinertia -v true ${DATADIR}mobility_mujoco.urdf

# Conversion to MuJoCo model
# shellcheck disable=SC2006
file_in=`realpath ${DATADIR}mobility_mujoco.urdf`
file_out=`realpath ${DATADIR}mobility_mujoco.xml`
cd /home/ajinkya/.mujoco/ && ./mujoco200/bin/compile "${file_in}" "${file_out}"; cd "${CUR_DIR}" || return 0

# Add names to mesh geometries and add actuator tags
python ${CUR_DIR}/sapien_dataset/dataset_tools.py -i "${file_in}" -o "${file_out}" -uxt --obj-type "${OBJ_TYPE}"
echo "MUJOCO-Compatible XML file created for ${DATADIR}"
