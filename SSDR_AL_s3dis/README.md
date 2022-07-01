# Environment
*1.* Setup python environment
```
conda create -n ssdr python=3.5
source activate ssdr
pip install -r helper_requirements.txt
sh compile_op.sh
```

*2.* Install additional Python packages:
```
pip install future python-igraph tqdm transforms3d pynvrtc fastrlock cupy h5py sklearn plyfile scipy pandas
```

*3.* Install Boost (1.63.0 or newer) and Eigen3, in Conda:<br>
```
conda install -c anaconda boost; conda install -c omnia eigen3; conda install eigen; conda install -c r libiconv
```

*4.* Compile the ```libply_c``` and ```libcp``` libraries:
```
CONDAENV=YOUR_CONDA_ENVIRONMENT_LOCATION
cd partition/ply_c
cmake . -DPYTHON_LIBRARY=$CONDAENV/lib/libpython3.6m.so -DPYTHON_INCLUDE_DIR=$CONDAENV/include/python3.6m -DBOOST_INCLUDEDIR=$CONDAENV/include -DEIGEN3_INCLUDE_DIR=$CONDAENV/include/eigen3
make
cd ..
cd cut-pursuit
mkdir build
cd build
cmake .. -DPYTHON_LIBRARY=$CONDAENV/lib/libpython3.6m.so -DPYTHON_INCLUDE_DIR=$CONDAENV/include/python3.6m -DBOOST_INCLUDEDIR=$CONDAENV/include -DEIGEN3_INCLUDE_DIR=$CONDAENV/include/eigen3
make
```

# Run
S3DIS dataset. Download the files named "Stanford3dDataset_v1.2_Aligned_Version.zip". Uncompress the folder and move it to 
`/data/S3DIS`.

- Preparing the dataset:
```
python utils/data_prepare_s3dis.py
```
- Run:
```
./run_sota_comparison.sh
./run_graph_reasoning_analysis.sh
./run_threshold_analysis.sh
```
