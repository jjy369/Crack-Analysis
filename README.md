# Crack Analysis Toolkit

This repository contains Python source code for computing pavement crack morphological
features. These metrics support quantitative evaluation of
crack geometry, directional anisotropy, and pavement condition.

---

## 📦 Features

The toolkit provides automated computation of the following crack metrics:

### **Geometric Metrics**
- **Crack Length** (pixel scale and physical scale)
- **Crack Width** (pixel and millimeters)
- **Bending Degree (BD)**
- **Boundary Roughness (R)**

### **Directional Anisotropy Metrics**
- **ABI — Anisotropic Bias Index**  
  Positive ABI indicates longitudinal cracking tendency;  
  Negative ABI indicates transverse cracking tendency.

- **D_nonT — Non-transverse Crack Density**  
  Total length of non-transverse cracks divided by image area.

### **Composite Index**
- **CCI — Crack Condition Index**  
  Weighted combination of normalized metrics for comprehensive crack condition evaluation.

### **Processing Capabilities**
- Batch processing of crack mask images
- Automatic export of results to Excel (.xlsx)
- Fully reproducible and compatible with academic workflows

---

▶️ Usage
1. Prepare Input Crack Masks
Store your binary crack mask images (0 and 1) in a folder, e.g.:

input_folder/
│── img001.png
│── img002.png
│── img003.png
2. Configure Paths
In crack_analysis.py, modify:

python
input_folder = r"your_input_folder_path"
output_excel_path = r"your_output_path\crack_results.xlsx"


3. Run the Script
bash
python crack_analysis.py

