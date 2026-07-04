# AIBC BootCamp

## From Pixels to Prompts: Practical Computer Vision and Vision-Language Model Bootcamp

This repository contains learning materials, hands-on exercises, and practical examples for the **AIBC BootCamp**, focusing on the evolution of computer vision from classical image processing to modern Vision-Language Models (VLMs).

The bootcamp is designed for both **academic learning** and **practical implementation**, especially for participants who want to understand how computer vision models work and how they can be applied in real-world AI applications.

---

## Overview

Computer vision has evolved from manually designed features such as PCA, Haar features, and convolution filters into deep learning models and multimodal Vision-Language Models.

This project introduces the complete learning path:

```text
Pixels → Features → CNN / ViT → CLIP → VLM → Prompting → RAG → Fine-tuning → Deployment
```

The goal is to help participants understand not only the theory, but also how to use models practically for image analysis, visual inspection, document understanding, and AI agent applications.

---

## Learning Objectives

After completing this bootcamp, participants should be able to:

* Understand how computers represent images as pixels and tensors
* Explain the role of classical computer vision methods such as PCA and Haar features
* Understand how CNNs and Vision Transformers learn visual features
* Explain how CLIP connects images and text in a shared embedding space
* Understand the architecture of modern Vision-Language Models
* Use pre-trained computer vision and VLM models for practical tasks
* Analyze images using local VLM models through Ollama
* Generate structured outputs such as JSON from image analysis
* Build simple applications using VLM models
* Understand when to use prompting, RAG, LoRA, or fine-tuning
* Design practical AI applications for inspection, monitoring, and reporting

---

## Main Topics

### 1. Image as Pixels

Introduction to how images are represented by computers.

Topics include:

* Image tensor representation
* RGB and grayscale images
* Pixel intensity values
* Image resolution and dimensionality
* Why raw pixels are difficult for machine learning

---

### 2. Classical Computer Vision Features

Introduction to feature engineering before deep learning.

Topics include:

* PCA for dimensionality reduction
* Eigenfaces for face recognition
* Haar features
* Viola-Jones face detection
* Feature extraction and classification

---

### 3. Deep Learning for Computer Vision

Understanding how deep models learn features automatically.

Topics include:

* Convolution operation
* CNN architecture
* Feature hierarchy
* AlexNet, ResNet, and Vision Transformer
* Learned features vs manually designed features

---

### 4. CLIP and Cross-Modal Learning

Understanding how vision and language are connected.

Topics include:

* Image encoder
* Text encoder
* Shared embedding space
* Contrastive learning
* Zero-shot image classification

---

### 5. Vision-Language Models

Introduction to modern VLM architecture.

Topics include:

* Vision encoder
* Projector layer
* Large Language Model integration
* Image tokens and text tokens
* Visual Question Answering
* Document and chart understanding

---

### 6. Practical Use of VLM Models

Hands-on usage of VLM models for real-world tasks.

Examples include:

* Image description
* Object identification
* Safety inspection
* Maintenance inspection
* Document extraction
* Chart and diagram explanation
* CCTV image analysis

---

### 7. Using Ollama for Local VLM

This project includes practical examples for using VLM models locally with Ollama.

Recommended models:

```bash
ollama pull qwen2.5vl
ollama pull llama3.2-vision
```

Example use cases:

* Image question answering
* OCR from image
* Visual inspection report
* JSON extraction from image
* Local AI assistant for computer vision

---

### 8. VLM Application Workflow

The practical computer vision workflow covered in this bootcamp:

```text
1. Load a pre-trained model
2. Use the model for inference
3. Check whether the object is recognized correctly
4. Identify if the object is new or unseen
5. Collect custom data if needed
6. Fine-tune the model when accuracy is not enough
7. Evaluate and deploy the improved model
```

---

## How to Use a Computer Vision Model

A practical workflow:

### Step 1: Load a Pre-trained Model

Use a ready-to-use computer vision or VLM model.

Example:

```bash
ollama run qwen2.5vl
```

Use this approach when:

* The object category is already known by the model
* You only need general image understanding
* You want fast prototyping
* You want to analyze common objects, scenes, or documents

---

### Step 2: Test the Model with Your Object

Run inference using your own images.

Ask questions such as:

```text
What objects are visible in this image?
```

```text
Is there any safety issue in this image?
```

```text
Return the detected issue in JSON format.
```

Important question:

```text
Was this object likely available during the model training?
```

If the object is new, domain-specific, or visually different from common images, the pre-trained model may not give accurate results.

---

### Step 3: Fine-tune When Accuracy Is Not Enough

Fine-tuning is recommended when:

* The model gives inconsistent results
* The model cannot recognize your domain-specific object
* The model output is not accurate enough
* The model needs to follow a fixed format
* The model needs to understand local or industry-specific visual conditions

Fine-tuning requires:

* Domain-specific images
* Correct labels or instruction-answer pairs
* Training and validation dataset
* Evaluation metrics
* Deployment testing

---

## Example VLM Prompt

```text
You are a visual inspection assistant.

Analyze this image and return only JSON:

{
  "image_type": "",
  "objects_detected": [],
  "main_issue": "",
  "risk_level": "Low / Medium / High",
  "recommended_action": "",
  "short_report": ""
}
```

---

## Example Use Cases

### Industrial Inspection

* Detect damaged road
* Identify blocked drainage
* Analyze smoke or fire indication
* Read gauge or meter panels
* Inspect PPE compliance

### Campus and Facility Monitoring

* Classroom condition monitoring
* Facility damage reporting
* Parking and traffic observation
* Safety inspection

### Document AI

* Extract information from invoices
* Read forms and receipts
* Analyze scanned documents
* Convert document image into structured data

### AI Agent Application

* Visual inspection agent
* Maintenance report agent
* Security monitoring agent
* Tenant complaint analysis agent
* Daily reporting agent

---

## Suggested Folder Structure

```text
AIBC_BootCamp/
│
├── README.md
├── slides/
│   └── From_Pixels_to_Prompts_VLMs.pptx
│
├── images/
│   └── sample_images/
│
├── notebooks/
│   └── computer_vision_intro.ipynb
│
├── apps/
│   └── vlm_inspection_app/
│       ├── app.py
│       └── requirements.txt
│
├── prompts/
│   └── vlm_inspection_prompt.txt
│
├── datasets/
│   └── sample_dataset/
│
└── docs/
    └── bootcamp_notes.md
```

---

## Simple Ollama VLM Example

Install the required Python package:

```bash
pip install ollama
```

Example Python script:

```python
import ollama

response = ollama.chat(
    model="qwen2.5vl",
    messages=[
        {
            "role": "user",
            "content": "Describe this image and identify any possible safety issue.",
            "images": ["sample_image.jpg"]
        }
    ]
)

print(response["message"]["content"])
```

---

## Streamlit Application Example

Install dependencies:

```bash
pip install streamlit ollama pillow
```

Run the app:

```bash
streamlit run app.py
```

Example features:

* Upload image
* Select analysis type
* Analyze image using local VLM
* Generate structured report
* Export result for documentation

---

## Fine-Tuning Roadmap

Fine-tuning should not be the first step. The recommended roadmap is:

```text
Prompt Engineering
→ Structured Output
→ RAG with Domain Knowledge
→ Few-shot Examples
→ LoRA Fine-tuning
→ Full Fine-tuning
```

Use fine-tuning only when prompt engineering and RAG are not enough.

---

## Evaluation Checklist

Before deploying a computer vision or VLM application, evaluate:

* Accuracy on real images
* Consistency of output format
* Ability to handle unseen objects
* Performance on blurry or low-quality images
* False positive and false negative cases
* Human review requirement
* Latency and hardware requirements
* Data privacy and security

---

## Recommended Tools

| Tool             | Purpose                              |
| ---------------- | ------------------------------------ |
| Ollama           | Run local LLM and VLM models         |
| Python           | Build scripts and applications       |
| Streamlit        | Create simple AI web applications    |
| OpenCV           | Classical computer vision processing |
| PyTorch          | Model training and fine-tuning       |
| Roboflow         | Dataset annotation and management    |
| Label Studio     | Custom data labeling                 |
| ChromaDB / FAISS | Vector database for RAG              |
| GitHub           | Version control and collaboration    |

---

## Target Participants

This bootcamp is suitable for:

* Computer science students
* AI and data science learners
* Lecturers and researchers
* IT developers
* Industrial automation engineers
* Facility and maintenance teams
* AI practitioners who want practical VLM implementation

---

## Expected Output

At the end of this bootcamp, participants are expected to produce:

* A basic understanding of computer vision evolution
* A working local VLM model using Ollama
* A simple image analysis application
* A structured visual inspection prompt
* A prototype for image-based reporting
* A roadmap for fine-tuning or deployment

---

## Author

**Dr. Eng. Handri Santoso**
Faculty of Science and Technology
Pradita University

Focus areas:

* Artificial Intelligence
* Computer Vision
* Vision-Language Models
* Industrial IoT
* Control Systems
* AI for Industrial and Academic Applications

---

## License

This project is intended for educational and research purposes.

You may adapt the materials for academic training, internal workshops, and practical AI implementation projects.

---

## Acknowledgement

This project is part of the AIBC BootCamp learning initiative to help participants understand and apply modern AI, computer vision, and Vision-Language Models in real-world scenarios.
