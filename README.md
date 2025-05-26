# 📸 Comparative Analysis of Image Captioning Models: DenseNet+LSTM vs. BLIP

This project compares two image captioning models—**DenseNet+LSTM (baseline)** and **BLIP (SOTA model)**—on the Flickr8k dataset. We evaluate the models using human-written captions, BLEU and NLG metrics, and CLIP-based visual-text similarity.

---

## 📦 Project Structure

<pre>
├── 📁 Data
│ ├── Flickr8k Images (/content/drive/MyDrive/flickr8k/Images)
│ └── Captions CSV (/content/drive/MyDrive/flickr8k/captions.txt)
│
├── 📁 Model Files
│ ├── DenseNet+LSTM model (model.keras)
│ ├── Tokenizer (tokenizer.pkl)
│ ├── Extracted Features (features.pkl)
│
├── 📂 Notebooks / Scripts
│ ├── DenseNet+LSTM training and evaluation
│ ├── BLIP caption generation
│ ├── Evaluation metrics (BLEU, NLG-Eval, CLIP)
│ ├── Visualizations (images, captions, CLIP scores, t-SNE)
│
└── 📄 README.md (this file)
</pre>

---

## 🧠 Models Used

### 1️⃣ DenseNet+LSTM Baseline
- **Image encoder**: DenseNet201 (pretrained)
- **Sequence decoder**: LSTM with embedding layers
- **Trained on**: Flickr8k captions

### 2️⃣ BLIP (Bootstrapping Language-Image Pretraining)
- **Pretrained BLIP model** from Hugging Face (`Salesforce/blip-image-captioning-base`)

---

## 📊 Evaluation Metrics

- **BLEU-1 & BLEU-2**: N-gram overlap (via NLTK)
- **NLG-Eval**: METEOR, ROUGE-L, CIDEr (via NLG-Eval)
- **CLIP Similarity**: Measures visual-text alignment using OpenAI's CLIP model

---

## 🚀 Main Findings

- **BLIP outperforms DenseNet+LSTM** in both qualitative (captions) and quantitative metrics.
- **DenseNet+LSTM** struggles with object specificity and context; often predicts generic scenes.
- **BLIP** captures fine-grained details better (e.g., "a lion chasing a buffalo" instead of "dog in grass").
- **CLIP scores** confirm BLIP’s stronger visual-text alignment.

---

## 🖼️ Sample Outputs
2973269132_252bfd0160.jpg
270263570_3160f360d3.jpg
| Image | Ground Truth | DenseNet+LSTM | BLIP | CLIP Scores (GT, Dense, BLIP) |
| :---: | :----------: | :-----------: | :--: | :--------------------------: |
| Image | Ground Truth | DenseNet+LSTM | BLIP | CLIP Scores (GT, Dense, BLIP) |
| :---: | :----------: | :-----------: | :--: | :--------------------------: |
| ![](images/2973269132_252bfd0160.jpg) | "A large wild cat is pursuing a horse across a meadow." | "dog is running through the grass" | "a lion chasing a buffalo" | 0.2878, 0.0000, 0.7124 |
| ![](images/270263570_3160f360d3.jpg) | "Two brown dogs fight on the leafy ground." | "brown dog is running in the grass" | "a dog and a dog playing together" | 0.6367, 0.0007, 0.3628 |

---

## 🛠️ Key Components

- **Feature extraction**: DenseNet201 on Flickr8k images
- **Caption generation**: LSTM decoder (DenseNet) + BLIP pre-trained model
- **Hallucination filtering**: YOLOv5 + keyword filtering
- **Evaluation pipeline**: BLEU, NLG-Eval, CLIP, t-SNE visualization

---

## 🔧 Requirements

- Python == 3.11
- TensorFlow, PyTorch, Hugging Face Transformers
- OpenAI CLIP, NLG-Eval, YOLOv5
- NLTK, Pandas, Matplotlib, Seaborn

---

## 📌 How to Run

1️⃣ Clone the repository  
2️⃣ Install requirements (`pip install -r requirements.txt`)  
3️⃣ Download Flickr8k dataset and prepare files  
4️⃣ Run notebooks/scripts in sequence:
   - **Feature extraction**
   - **DenseNet+LSTM training**
   - **BLIP inference**
   - **Metrics evaluation**
   - **Visualizations**

---

## 💡 Acknowledgements

- **Flickr8k Dataset**: [Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k)
- **BLIP Model**: [Salesforce BLIP on Hugging Face](https://huggingface.co/Salesforce/blip-image-captioning-base)
- **CLIP Model**: [OpenAI CLIP](https://github.com/openai/CLIP)
- **YOLOv5**: [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)

---