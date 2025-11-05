# 🧠 NutriVision – AI-Powered Food Ingredient Analyzer



### 🥗 Overview
**NutriVision** is an AI-powered web application that analyzes food product ingredients from images and provides **health and nutritional insights**.  
It combines the power of **Google Cloud Vision API** (for text extraction) and **OpenAI’s LLM** (for intelligent food analysis) to help users make informed and safer food choices.

🔗 **Live Project:** [https://ingredienthealthcheck.netlify.app](https://ingredienthealthcheck.netlify.app)

---

## 🚀 Features

- 📸 Upload or capture a food product label image  
- 🔍 Extract ingredients automatically using **Google Cloud Vision API**  
- 🤖 Analyze extracted ingredients with **OpenAI LLM** (food specialist role)  
- ⚠️ Identify **harmful or allergenic ingredients** that may cause health issues  
- 📊 Get clear and concise health insights  
- 🌐 Deployed using **Netlify (frontend)** and **Render (backend, free tier)**

---

## 🧩 Tech Stack

| Layer | Technology |
|:------|:------------|
| Frontend | React, HTML5, CSS3, Bootstrap |
| Backend | FastAPI |
| AI & APIs | Google Cloud Vision API, OpenAI REST API |
| Deployment | Netlify (Frontend), Render (Backend) |

---

## 🧠 How It Works

1. **Image Upload** → User uploads a food label photo.  
2. **Text Extraction** → Backend uses **Google Cloud Vision API** to read the ingredient list.  
3. **AI Analysis** → Extracted text is sent to **OpenAI LLM**, which acts as a food specialist.  
4. **Insight Generation** → AI returns detailed health insights and safety analysis.  
5. **Display Results** → Frontend shows a clean, easy-to-read result to the user.

---

## 💡 Real-Life Use Cases

- 🧴 Detect allergens or harmful additives (e.g., gluten, MSG, trans fats)  
- ❤️ Help users with conditions like diabetes, hypertension, or food allergies  
- 🧠 Educate consumers about nutrition and food safety  
- 🛒 Empower smarter and healthier product choices  

---

## 🛠️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/NutriVision.git
cd NutriVision
