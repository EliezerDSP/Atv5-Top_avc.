# Sistema de Recuperação de Informações com VectorDB e RetrievalQA

## 📝 Descrição do Projeto
Este projeto consiste no desenvolvimento de um sistema de perguntas e respostas baseado em documentos (sistema RAG - *Retrieval-Augmented Generation*). O sistema lê um arquivo de texto local (`.txt`), divide o conteúdo em blocos (*chunks*), gera vetores de características (*embeddings*) e os armazena em um banco de dados vetorial usando a biblioteca **FAISS**.

Quando o usuário faz uma pergunta, o sistema busca os fragmentos de texto mais relevantes no banco vetorial e os envia como contexto para o modelo de linguagem local **Llama 3.2 (versão leve de 1B)** via **Ollama**, gerando uma resposta precisa e fundamentada no documento fornecido.

> **Nota de Otimização:** O projeto foi adaptado para utilizar o modelo `llama3.2:1b`. Essa escolha foi feita para permitir a execução local eficiente do pipeline completo de IA em computadores com recursos de hardware mais modestos (como limitações de memória RAM/VRAM), mantendo uma excelente capacidade de resposta.

---

## 🛠️ Tecnologias e Bibliotecas Utilizadas
* **Python 3.10+**
* **LangChain / LangChain Community / LangChain Ollama**: Framework para construção de aplicações com LLMs.
* **FAISS (Facebook AI Similarity Search)**: Banco de dados vetorial para busca eficiente por similaridade.
* **Ollama**: Ferramenta para execução de LLMs locais.
* **Modelo de Linguagem (LLM)**: `llama3.2:1b` (1 bilhão de parâmetros).
* **Modelo de Embeddings**: `mxbai-embed-large`.

---

## 🚀 Como Instalar e Executar o Projeto

### Pró-requisitos
Certifique-se de ter o **Ollama** instalado na sua máquina. Caso não tenha, baixe em [ollama.com](https://ollama.com).

Com o Ollama ativo no seu sistema, baixe os modelos necessários executando os comandos abaixo no terminal:
```bash
ollama pull llama3.2:1b
ollama pull mxbai-embed-large
