from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA

# 1. CARREGAMENTO DO ARQUIVO .TXT
# Define o caminho do seu arquivo de texto
txt_path = 'dados/meu_texto.txt'

if not os.path.exists(txt_path):
    print(f"Erro: O arquivo {txt_path} não foi encontrado! Crie o arquivo antes de rodar.")
    exit()

print("Carregando o arquivo de texto...")
loader = TextLoader(txt_path, encoding='utf-8')
documents = loader.load()

# 2. DIVISÃO DO TEXTO EM CHUNKS (PEDAÇOS)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=50,
    length_function=len,
)

texts = text_splitter.split_documents(documents)

print(f"Texto dividido com sucesso!")
print(f"Quantidade total de chunks gerados: {len(texts)}")
print(f"Exemplo do primeiro chunk:\n{texts[0].page_content}\n")
print("-" * 50)

# 3. CRIAÇÃO DO BANCO DE DADOS VETORIAL (FAISS)
print("Gerando embeddings e populando o banco vetorial FAISS (isso pode levar alguns instantes)...")
# Nota: Certifique-se de que o Ollama está rodando e que você possui o embedding "mxbai-embed-large" baixado.
# Caso não tenha, rode no terminal: ollama pull mxbai-embed-large
embeddings_model = OllamaEmbeddings(model="mxbai-embed-large")
db = FAISS.from_documents(texts, embeddings_model)
print("Banco vetorial criado com sucesso!")
print("-" * 50)

# 4. CONFIGURAÇÃO DO MODELO LLM (Llama 3.2 1B) E RETRIEVER
print("Configurando o modelo Llama 3.2:1b e o pipeline de QA...")
# Aqui definimos explicitamente a versão leve de 1 bilhão de parâmetros que você possui
model = OllamaLLM(model="llama3.2:1b")

# Configura o buscador para trazer os 5 chunks mais relevantes
retriever = db.as_retriever(search_kwargs={"k": 5})

# Cria a cadeia de Recuperação de Questões (RetrievalQA)
qa_chain = RetrievalQA.from_chain_type(
    llm=model, 
    retriever=retriever, 
    chain_type="stuff"
)

# 5. EXECUÇÃO DE UMA PERGUNTA DE TESTE
# Altere a pergunta abaixo para algo que faça sentido com o conteúdo do seu arquivo .txt
query = "Qual é o tema principal abordado no texto?"

print(f"\nRealizando a pergunta: '{query}'")
response = qa_chain.invoke(query)

print("\n=== Resposta do Sistema ===")
# O retorno do invoke é um dicionário, extraímos o resultado pela chave 'result'
print(response['result'])
