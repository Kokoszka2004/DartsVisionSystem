# Automatic Darts Scorer

Automatic darts scoring system based on computer vision.
The project detects darts on a dartboard, estimates their positions,
and calculates the score automatically.

## How it works
- Detects darts on the dartboard using a trained vision model
- Maps detected positions to dartboard scoring zones
- Calculates the final score automatically

## Files
- main.py – main application logic
- DART_MASTER_4K.pt – trained model used for dart detection
