# 🛠️ Corrective RAG

This project implements a **Corrective Retrieval-Augmented Generation (RAG)** pipeline using **LangGraph**, **Streamlit**, and **Groq LLMs**.  
It demonstrates how to build an **intelligent RAG system** that retrieves, grades, and refines context before generating answers — improving reliability compared to standard RAG.

---

## 🚀 Features
- 📄 **Document Retrieval & Grading** – Fetches documents and filters them based on relevance.  
- 🔄 **Query Transformation** – Automatically rewrites questions if retrieved documents are irrelevant.  
- 🌐 **Web Search Fallback** – Uses external search when knowledge base documents are insufficient.  
- 🤖 **LLM-Powered Answer Generation** – Uses Groq-hosted LLMs for efficient response generation.  
- 📊 **Execution Logs in UI** – Displays workflow steps (retrieve → grade → transform → search → generate) directly in the frontend.  

---

## 📂 Project Structure

CorrectiveRAG/                # Root project folder
├── app.py                    # Streamlit app entrypoint
├── main.py                   # CLI stub for graph testing
├── requirements.txt          # Project dependencies
├── README.md                 # Documentation
├── .env.example              # Example environment variables
│
├── src/
│   ├── langgraphCorrectiveAI/    # LangGraph corrective RAG workflows
│   │   └── graph/workflow.py     # Core workflow (retrieve → grade → transform → web search → generate)
│   │
│   ├── nodes/                    # Modular workflow nodes
│   │   ├── retrieve_node.py
│   │   ├── grade_node.py
│   │   ├── transform_node.py
│   │   ├── web_search_node.py
│   │   └── generate_node.py
│   │
│   ├── tools/                    # Tools (retrievers, graders, web search utilities)
│   │   └── search_tool.py
│   │
│   └── state/
│       └── graph_state.py        # Shared state across workflow execution
│
└── UI/streamlitUI/               # Streamlit-based UI layer
    ├── display_result.py
    ├── loadui.py
    └── uiconfigfile.py


---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
git clone https://github.com/daanyal-23/corrective-rag-demo.git
cd corrective-rag-demo
2️⃣ Create a Virtual Environment
python -m venv venv
# On Windows
venv\Scripts\activate
# On Mac/Linux
source venv/bin/activate
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Configure Environment Variables
Create a .env file in the root directory:
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
5️⃣ Run the Streamlit App
streamlit run app.py
🧪 Example Workflow
Enter a question in the Streamlit UI.

System retrieves documents and checks for relevance.

If irrelevant, the query is rewritten and a web search is performed.

The final LLM-generated answer is displayed.

The execution steps are shown in the UI for transparency.

📌 Future Improvements
✅ Add unit tests for workflow nodes.

✅ Improve frontend visualization of execution steps.

✅ Extend support for multiple vector stores (e.g., FAISS, Pinecone).

✅ Containerize using Docker for deployment.

🤝 Contributing
Pull requests are welcome!
For major changes, please open an issue first to discuss what you would like to improve.

Made with ❤️ using LangGraph + Groq + Streamlit


