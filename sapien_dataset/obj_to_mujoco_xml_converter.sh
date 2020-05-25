#!/bin/bash
DATADIR=$1
# change .obj files to .stl:
for filename in $DATADIR/textured_objs/*.obj; do
  ctmconv "$filename" "${filename//.obj}.stl";
done

# Correct mesh if needed
for filename in $DATADIR/textured_objs/*.stl; do
  python ~/Software/generalized_kinematics/SyntheticArticulatedData/sapien_dataset/mesh_utils.py -i "$filename"
done

# Change mesh file names in the urdf
cp $DATADIR/mobility.urdf $DATADIR/mobility_mujoco.urdf
sed -i -e 's/.obj"/.stl"/g' $DATADIR/mobility_mujoco.urdf
xmlstarlet ed -L -s /robot -t elem -n mujoco -v "" -s /robot/mujoco -t elem -n compiler -v "" \
-i /robot/mujoco/compiler -t attr -n meshdir -v textured_objs/ -i /robot/mujoco/compiler -t attr -n convexhull -v true \
-i /robot/mujoco/compiler -t attr -n balanceinertia -v true  $DATADIR/mobility_mujoco.urdf

# Conversion to MuJoCo model
file_in=`realpath $DATADIR/mobility_mujoco.urdf`; file_out=`realpath $DATADIR/mobility_mujoco.xml`; orig_dir=`pwd`; \
cd /home/ajinkya/.mujoco/ && ./mujoco200/bin/compile "$file_in" "$file_out"; cd "$orig_dir"
