# LangchainProject-1

This project uses UnstructuredURLLoader, RecursiveCharacterTextSplitter, RetrievalQAWithSourcesChain, and OpenAI from Langchain
It also makes use of FAISS to store the vectors. 



In this project, we can enter upto 3 URLs in the text fields provided, from which the data inside the URLs will be loaded, split into chunks, and finally vectorized using OpenAIEmbeddings. Once the URLs have been processes, the user can enter any question and the llm from OpenAI will provide the correct answer and also the correct URL source from which the answer was obtained.
