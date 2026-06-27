# NVIDIA-Inspired Hierarchical Memory Management Implementation

An advanced, production-grade conversational AI memory management system inspired by NVIDIA's modern LLM architecture patterns. This project prevents context window overflow and eliminates critical information loss by utilizing a **4-Layer Hierarchical Memory Architecture**. It operates collectively to capture immediate context, long-term summaries, semantic vectors, and structured entity data.

---

## 🚀 Key Features

| Feature | Description |
| :--- | :--- |
| **4-Layer Architecture** | Seamless coordination between Buffer, Summary, Vector, and Entity layers. |
| **Zero Context Loss** | Information automatically flows through layers, ensuring critical details are never lost. |
| **Dual Interfaces** | Fully functional via both a lightweight Terminal CLI and an interactive Streamlit UI. |
| **Optimized Retrieval** | Semantic search driven by L2-normalized vector embeddings combined with rule-based tracking. |

---

## 🧠 Hierarchical Memory Layers

| Layer | Type | Mechanism & Purpose |
| :--- | :--- | :--- |
| **Layer 1** | **Buffer Memory** | Holds the most recent raw conversation turns ($BUFFER\_LIMIT = 5$) for immediate conversational fluency. |
| **Layer 2** | **Summary Memory** | Triggered periodically ($SUMMARY\_TRIGGER = 10$) to condense older background history into concise rolling summaries. |
| **Layer 3** | **Vector Memory** | Converts past messages into L2-normalized hash embeddings to perform semantic top-K similarity retrieval. |
| **Layer 4** | **Entity Memory** | Extracts and tracks persistent structural entities (e.g., Names, Products, APIs) along with their frequency counts. |

---

## 🛠️ Project Structure & Architecture

| File | Purpose | Key Components |
| :--- | :--- | :--- |
| `on_terminal.py` | Command-line interface for the chatbot. | Session loop, state debugger, raw standard inputs. |
| `on_streamlit.py`| Web UI representation of the chatbot. | Visual state inspectors, dynamic chat layout, sidebar configurations. |
| `README.md` | Documentation. | Setup instructions, usage, and architectural details. |

---

## 💻 Usage

You can run this project using two different interfaces depending on your workflow:

| Interface | Run Command | Description |
| :--- | :--- | :--- |
| **Terminal (CLI)** | `python on_terminal.py` | Standard interactive terminal mode. Type `debug` to inspect the memory layer allocations in real-time, or `exit` to quit. |
| **Streamlit (Web UI)** | `streamlit run on_streamlit.py` | Launches a local browser window showcasing real-time visual tracking of the 4 memory layers side-by-side with the chat. |

---

## 📊 Evaluation & Memory Workflow
