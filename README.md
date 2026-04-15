# 🧠 Blog AI Agent
An intelligent multi-agent system that generates high-quality technical blogs using LLMs, structured workflows, and adaptive content generation pipelines.

---

## 🚀 Overview
Blog AI Agent is designed to automate the end-to-end process of technical blog creation — from ideation to structured content generation — using modern AI architectures.

It leverages:
- Multi-agent workflows
- Prompt engineering
- Iterative refinement
- Structured Markdown output

---

## 🏗️ Architecture
The system follows a modular pipeline:

```text
User Input → Planner Agent → Research Agent → Writer Agent → Refiner → Final Blog Output
```

**Key Components:**
- **Planner** → Breaks topic into sections
- **Researcher** → Gathers structured knowledge
- **Writer** → Generates blog content
- **Refiner** → Improves clarity and formatting

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python |
| LLM APIs | OpenAI / Gemini |
| Frontend | Streamlit |
| Orchestration | Custom multi-agent pipeline |
| Data Format | Markdown |

---

## ✨ Features

- 🧩 Multi-agent architecture
- 📄 Automatic blog structuring
- 🔁 Iterative content refinement
- ⚡ Fast inference using OpenAI
- 🧠 Prompt-engineered outputs
- 📊 Notebook-based experimentation

---

## 📂 Project Structure

```
.
├── bwa_backend.py          # Core backend logic
├── bwa_frontend.py         # Streamlit UI
├── main.py                 # Entry point
├── requirements.txt
├── README.md
├── .env.example
├── notebooks/
│   ├── 1_bwa_basic.ipynb
│   ├── 2_bwa_improved_prompting.ipynb
│   ├── 3_bwa_research.ipynb
│   ├── 4_bwa_research_fine_tuned.ipynb
│   ├── 5_bwa_image.ipynb
```

---

## 🔧 Installation

```bash
git clone https://github.com/Rituraj1404/blog_ai_agent.git
cd blog_ai_agent
pip install -r requirements.txt
```

---

## 🔑 Environment Setup

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_openai_key_here
GOOGLE_API_KEY=your_google_key_here
```

> ⚠️ Never commit your `.env` file. It is already listed in `.gitignore`.

---

## ▶️ Running the Project

**Run backend:**
```bash
python bwa_backend.py
```

**Run frontend:**
```bash
streamlit run bwa_frontend.py
```

---

## 📸 Output Example

- Structured technical blogs
- Markdown formatted content
- Section-wise generation

---

## 🧠 Key Concepts Demonstrated

- Multi-agent systems
- Prompt engineering
- LLM orchestration
- Content generation pipelines
- AI-assisted writing systems

---

## 🚀 Future Improvements

- [ ] Diagram generation (Mermaid)
- [ ] Agent memory (long-term context)
- [ ] Web scraping for real-time research
- [ ] Deployment (Docker + cloud)

---

## 👨‍💻 Author

**Rituraj** — [GitHub](https://github.com/Rituraj1404)

---

## ⭐ If you like this project

Give it a star ⭐ and share it with others!
