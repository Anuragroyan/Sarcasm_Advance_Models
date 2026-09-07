🤖 Sarcasm Detector — ONNX Model

Sarcasm Detector is a Python-based machine learning project designed to detect sarcasm in user-provided text. The project uses an ONNX model for efficient inference and includes trained model resources, configuration files, and supporting text data required for model processing and prediction.

✨ Features

* 🤖 Sarcasm detection using machine learning
* 📝 Text-based sarcasm classification
* 🐍 Python-based implementation
* 🔄 ONNX model integration
* ⚡ Efficient model inference
* 📦 Support for trained model/checkpoint files
* 🔤 Text and configuration resources
* 🧠 Model-ready architecture for application integration

📂 Project Structure

The repository contains the Python environment, required dependencies, trained machine learning models, and supporting files used for sarcasm detection.

* venv/ – Python virtual environment used to run the project.
* requirements.txt – Contains the Python libraries required by the project.
* .onnx – Trained ONNX model files used for prediction.
* .safetensors – Stores trained model weights.
* .json – Contains model configuration, metadata, or supporting data.
* .txt – Contains supporting text data such as vocabulary or labels.
* .pt – PyTorch model files used during model development.
* .pth – PyTorch checkpoint files containing saved model weights or training states.

🔄 Model Workflow

<img width="1222" height="1287" alt="image12" src="https://github.com/user-attachments/assets/6caabd9e-7ac3-4ebf-b21e-6d832d22eefb" />

🧠 Model Integration

<img width="1536" height="1024" alt="image13" src="https://github.com/user-attachments/assets/282dd2e2-8676-4576-8f84-9de95211ae2e" />


📦 Model Resources

The repository contains multiple model-related formats used during development and integration:
* .onnx — ONNX models prepared for inference
* .safetensors — Serialized model weights
* .pt — PyTorch model files
* .pth — PyTorch checkpoints
* .json — Configuration and metadata
* .txt — Supporting text resources
* requirements.txt — Python dependency list

🛠️ Tech Stack

* Python
* ONNX
* ONNX Runtime
* PyTorch
* SafeTensors
* Natural Language Processing
* Machine Learning

🚀 Setup

Create and activate a Python virtual environment:

python -m venv venv

Activate it:

Windows

venv\Scripts\activate

macOS / Linux

source venv/bin/activate

Install dependencies:

pip install -r requirements.txt

🎯 Project Purpose

This repository focuses on the machine learning side of sarcasm detection, keeping the model, weights, configuration, and supporting resources organized separately from the application layer. The ONNX model provides a portable format that can be used for efficient inference and integration into different applications.
