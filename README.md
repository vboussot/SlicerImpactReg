## 🚀 Installation

1. Install **3D Slicer ≥ 5.6**

2. Clone this repository:
   ```bash
   git clone https://github.com/vboussot/SlicerImpactReg.git
   ```

3. Build the module:
   ```bash
   mkdir build
   cd build

   cmake \
     -DCMAKE_BUILD_TYPE:STRING=Debug \
     -DSlicer_DIR:PATH=/path/to/Slicer-SuperBuild-Debug/Slicer-build \
     ..

   cmake --build . --config Release -j$(nproc)
   ```

4. In Slicer, open:  
   **Edit → Application Settings → Modules → Additional Module Paths**  
   and add the folder:
   ```
   SlicerImpactReg/lib/Slicer-5.11/qt-scripted-modules
   ```

5. Restart Slicer and open the **IMPACT-Reg** module.
