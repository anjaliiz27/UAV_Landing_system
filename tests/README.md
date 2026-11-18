# UAV Landing-Site Optimization (Team Codex)

Perception, search and knowledge-driven landing-site selection from a single downward-facing image.

## Layout
- `models/`  encoder.py (4 blocks), decoder.py (4 blocks), segmentation_pipeline.py (SegNet = encoder + decoder)
- `src/`     preprocessing, dataset, train, inference, roughness, candidate_search, knowledge_graph, cost_function
- `checkpoints/` best_model.pth (committed, small)
- `tests/`   shape and preprocessing tests

## Run (always from the repo root)
```bash
pip install -r requirements.txt
export DATA_ROOT=/path/to/semantic_drone_dataset      # Kaggle: /kaggle/input/<dataset-name>
python -m src.preprocessing check      # then set NUM_CLASSES in src/preprocessing.py
python -m src.preprocessing prep
python -m src.train --epochs 60
python -m src.inference some_image.jpg
python -m pytest tests -q
```
