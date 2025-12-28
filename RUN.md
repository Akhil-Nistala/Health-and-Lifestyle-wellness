# 🚀 How to Run — Health & Lifestyle AI

## 1️⃣ Prerequisites

Make sure you have:

* Python **3.9+**
* Internet connection
* Groq API Key

---

## 2️⃣ Clone the Repository

```
git clone <your-repo-link>
cd <repo-folder>
```

---

## 3️⃣ Install Dependencies

Create virtual environment (recommended):

```
python -m venv venv
```

Activate it
Windows:

```
venv\Scripts\activate
```

Mac / Linux:

```
source venv/bin/activate
```

Install requirements:

```
pip install -r requirements.txt
```

---

## 4️⃣ Set Groq API Key

Create a `.env` file in the project folder:

```
touch .env
```

Open it and add:

```
GROQ_API_KEY=your_key_here
```

Get your key from:
[https://console.groq.com/keys](https://console.groq.com/keys)

---

## 5️⃣ Run the Application

Run:

```
python lifestyle_final.py
```

---

## 6️⃣ Using the System

The system is **CLI interactive**.
Follow on-screen instructions.

### Setup Flow

1️⃣ Create User Profile
2️⃣ Enter Health Logs
3️⃣ Choose actions:

* 📝 Daily Check-in
* 🍽️ Diet AI Agent
* 💪 Fitness AI Agent
* ⚕️ Health Risk Analysis
* 🍛 Mess Menu Upload
* 🤖 Mess Optimizer
* 📊 Dashboard

Reports can be **saved locally**.

---

## 🗂️ Data Storage

All user data is safely stored locally in:

```
health_ai_data/
```

Includes:

* Profiles
* Logs
* Mess Menus
* AI Reports
* Backups

No cloud storage. No privacy risk.

---

## ⚠️ Troubleshooting

**1️⃣ “GROQ_API_KEY not found”**
– You did not set `.env` correctly
– Restart terminal after creating `.env`

---

**2️⃣ ImportError / Missing Libraries**
Run again:

```
pip install -r requirements.txt
```

---

**3️⃣ Permission Error on Linux/Mac**
Run:

```
chmod +x lifestyle4.py
```

---

## 🛡️ Notes

* This system is **not medical diagnosis**
* Lifestyle + preventive guidance only
* Works best when logs are entered regularly

