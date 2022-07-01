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
*5.* build chamfer3D
```
cd chamefer3D
python setup.py install
cd ..
```

# Run
7zip is required to uncompress the raw data in this dataset, to install p7zip:
```
sudo apt-get install p7zip-full
```
- Download and extract the dataset. First, please specify the path of the dataset by changing the `BASE_DIR` in "download_semantic3d.sh"    
```
sh utils/download_semantic3d.sh
```
- Preparing the dataset:
```
python utils/data_prepare_semantic3d_no_ignore.py
```
- Run:
```
./run_semantic3d_0.012.sh
```
