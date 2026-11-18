# UAV Landing-Site Optimization
### Team Codex

**Perception, Search, and Knowledge-Driven Landing-Site Optimization**

An AI-based system for identifying and ranking suitable UAV landing locations from a single downward-facing aerial image.

## 1. Project Overview

Selecting a safe landing location is an important challenge for Unmanned Aerial Vehicles (UAVs), particularly when the surrounding terrain contains different surface types and environmental conditions.

This project develops an intelligent landing-site optimization system that combines visual perception, geometric analysis, candidate search, and knowledge-graph-based semantic reasoning.

The system analyses an aerial image, identifies potential landing regions, evaluates their terrain characteristics, and ranks them according to a combined cost function.

An important feature of the proposed approach is the incorporation of package properties into landing-site selection. Properties such as Fragile, Heavy, Valuable, and Biohazard influence terrain suitability through knowledge-graph reasoning.

## 2. System Workflow

The system follows the workflow below:

1. **Input:** Provide a downward-facing aerial image.
2. **Target Selection:** Select the desired target location within the image.
3. **Package Configuration:** Specify package properties and dimensions.
4. **Semantic Segmentation:** Uses a U-Net architecture with a ResNet34 encoder, implemented using the `segmentation_models_pytorch` library. The model processes 256 × 256 RGB images and predicts pixel-level labels across 24 semantic classes. Trained model weights are loaded from `src/weights/best_unet_r34.pth`.
5. **Depth Estimation:** Estimate relative depth using monocular depth estimation.
6. **Geometric Analysis:** Estimate terrain slope and roughness.
7. **Candidate Generation:** Search for potential landing locations satisfying the configured safety constraints.
8. **Knowledge Graph Reasoning:** Evaluate terrain suitability based on the package properties.
9. **Cost Calculation:** Calculate the total cost for each candidate.
10. **Ranking and Selection:** Rank candidates and identify the recommended landing location.
11. **Explanation:** Display candidate metrics and knowledge-graph reasoning.


## 3. Project Structure

```text
UAV-Landing-Site-Optimization/
│
├── .streamlit/
│   └── config.toml
│
├── checkpoints/
│   └── Model checkpoints
│
├── models/
│   └── Model architecture and supporting files
│
├── src/
│   ├── app.py
│   ├── pipeline.py
│   ├── preprocessing.py
│   ├── segment.py
│   ├── depth_module_v2.py
│   ├── candidates.py
│   ├── class_policy.py
│   ├── class_safety.py
│   ├── cost.py
│   ├── semantic_brain.py
│   ├── dataset.py
│   ├── inference.py
│   ├── train.py
│   │
│   ├── test_images/
│   ├── test_masks/
│   ├── weights/
│   ├── candidates/
│   ├── midas_outputs/
│   ├── full_pipeline_result/
│   └── predicted_vs_gt_cand/
│
├── tests/
│   └── Test scripts
│
├── requirements.txt
└── README.md
```

*Note: Generated result folders and optional development files may be excluded from the final GitHub repository. The structure should reflect the files actually committed.*

## 4. Installation and Setup

### Clone the repository

```bash
git clone https://github.com/anjaliiz27/Perception-Search-and-Knowledge-Driven-Landing-Site-Optimization.git

cd Perception-Search-and-Knowledge-Driven-Landing-Site-Optimization
```

### Install dependencies

```bash
pip install -r requirements.txt
```

It is recommended to use a Python virtual environment.

### Dataset

The project uses aerial imagery from the TU Graz Semantic Drone Dataset for semantic segmentation and evaluation.

Download the dataset separately and configure the appropriate dataset path in the project configuration or preprocessing code.

Large datasets are not required to be committed directly to the GitHub repository.

## 5. Running the Application

Run the following command from the project root:

```bash
python -m streamlit run src/app.py
```

Open the local URL displayed in the terminal to access the interactive application.

## 6. Running Tests

Run the test suite from the project root:

```bash
python -m pytest tests -q
```

The tests are used to check different components of the system, including model behaviour, preprocessing, and pipeline functionality.

## 7. Application Outputs

The application provides:

- Semantic segmentation visualizations.
- Estimated depth maps.
- Terrain slope and roughness information.
- Potential landing-site candidates.
- Ranked candidate tables.
- Distance, roughness, semantic penalty, slope, and total cost values.
- Recommended landing location.
- Knowledge-graph-based explanations.
- Package-property comparisons.

## 8. Technologies Used

- Python
- PyTorch
- OpenCV
- NumPy
- Streamlit
- MiDaS
- Semantic Segmentation
- Monocular Depth Estimation
- Knowledge Graphs
- Graph-Based Semantic Reasoning

## 9. Team
**Team Codex**
| S. No. | Name | ID Number |
|:---:|---|---|
| 1 | D. Sree Geetanjali | 2023B4AD0609H |
| 2 | K. Manobhiram Reddy | 2024APS0240H |
| 3 | Sri Sai Tanuj B | 2024A3PS0451H |
| 4 | J. Bhanu Srikar | 2024APS0241H | 
| 5 | B. Jayanth Reddy | 2024AAPS0222H |
| 6 | Sushanth Reddy Kompally | 2024A8PS0459H |
| 7 | Gundla Sreejith | 2024A3PS0427H |
| 8 | Brijesh Dasari | 2024AAPS0270H |