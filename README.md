# ⚖️ NYAYA
### AI-Powered Situation-Based Legal Assistance
**Making Indian Legal Information More Accessible Through RAG and Multi-Agent Intelligence**

## 🌟 About the Project

NYAYA is an intelligent legal assistance chatbot designed to simplify access to Indian legal information through Artificial Intelligence.

Traditional legal resources often require users to know specific Acts, Sections, and legal terminology. NYAYA addresses this challenge by allowing users to describe their real-life problems in natural language.

Using Retrieval-Augmented Generation (RAG) and a coordinated Multi-Agent Architecture, NYAYA analyzes user situations, retrieves relevant legal provisions from its knowledge base, and generates context-aware responses.

Its multilingual interface supports English, Tamil, and Hindi, making legal information easier to understand and access.

## 💡 What Makes NYAYA Different?

- **Situation-Based Understanding:** Users can explain problems in everyday language without knowing legal terminology.
- **Intelligent Legal Classification:** Identifies the relevant legal domain based on the user's situation.
- **Source-Grounded Responses:** Retrieves relevant information from a collection of Indian legal documents before generating answers.
- **Multi-Agent Coordination:** Divides the legal assistance workflow into specialized tasks.
- **Multilingual Interaction:** Supports English, Tamil, and Hindi.
- **Context-Aware Clarification:** Asks follow-up questions when additional details are needed.
- **Legal Claim Verification:** Checks whether important generated claims are supported by retrieved documents.
- **Transparent Processing:** Displays retrieved legal sources and an agent execution trace.

## 🧠 AI Architecture

NYAYA combines two important AI approaches:

**1. Retrieval-Augmented Generation (RAG)**

Retrieves relevant legal document sections and provides them as context to the language model, helping generate source-grounded responses.

**2. Multi-Agent Architecture**

Uses specialized agent roles to coordinate question clarification, legal analysis, document retrieval, relevance evaluation, response generation, and verification.

### Agent Workflow

1. Clarification Agent – Identifies missing information and asks follow-up questions.
2. Analyzer Agent – Detects the language, legal domain, and urgency.
3. Retrieval Agent – Searches the legal knowledge base.
4. Grader – Evaluates retrieved document relevance.
5. Rewriter – Improves queries when retrieval needs refinement.
6. Generator – Produces context-based legal information.
7. Verifier – Checks important claims against retrieved sources.

## 🔍 Intelligent Document Retrieval

NYAYA uses a hybrid retrieval approach combining:

- **Multilingual E5 Small:** Converts queries and legal document chunks into embeddings.
- **ChromaDB:** Stores and retrieves vector representations.
- **BM25:** Performs keyword-based document retrieval.
- **Reciprocal Rank Fusion (RRF):** Combines search rankings to improve retrieval results.

## 📚 Legal Knowledge Domains

The knowledge base includes documents related to:

- Women's Rights
- Children's Rights
- Consumer Rights
- Criminal Rights
- Cyber Rights
- Fundamental Rights
- Labour Rights
- Legal Rights
- Right to Information (RTI)

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| User Interface | Streamlit |
| LLM Integration | Nexus API – Nova-Micro |
| Embedding Model | Multilingual E5 Small |
| Vector Database | ChromaDB |
| Retrieval Techniques | Vector Search, BM25, RRF |
| AI Architecture | RAG, Multi-Agent System |
| Development Environment | Visual Studio Code |

## 🔄 Application Workflow

User Query → Clarification → Situation Analysis → Legal Domain Identification → Hybrid Document Retrieval → Relevance Evaluation → Response Generation → Claim Verification → Final Response with Legal Sources

## 🎯 Project Objective

To make Indian legal information more approachable by transforming complex legal documents into understandable, situation-based guidance through AI-driven information retrieval and multilingual interaction.

## ⚠️ Disclaimer

NYAYA is an educational legal information assistant. It does not provide professional legal advice, guarantee legal outcomes, or replace consultation with a qualified legal professional. Users should verify applicable laws and seek professional guidance for specific legal matters.

## 👩‍💻 Developer

**Abinaya S**  
Computer Science and Engineering

**Project:** NYAYA – AI-Powered Situation-Based Legal Assistance
