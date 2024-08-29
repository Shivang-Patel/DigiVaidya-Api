import subprocess
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.replicate import Replicate
from llama_index.core import Settings

# Install all requirements from requirements.txt
subprocess.call(['pip', 'install', '-r', 'requirements.txt'])



llm = Replicate(
    model="meta/llama-2-70b-chat:2796ee9483c3fd7aa2e171d38f4ca12251a30609463dcfd4cd76703f22e96cdf",
    is_chat_model=True,
    additional_kwargs={"max_new_tokens": 512}
)

Settings.llm = llm
Settings.embed_model = "local:BAAI/bge-small-en-v1.5"

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins

KB_DIR = "./knowledge_base"
if not os.path.exists(KB_DIR):
    # knowledgebase is not present, hence perform indexing and store knowledge
    documents = SimpleDirectoryReader("Source").load_data()
    index = VectorStoreIndex.from_documents(documents)
    
    # store it for later use
    index.storage_context.persist(persist_dir=KB_DIR)
else:
    storage_context = StorageContext.from_defaults(persist_dir=KB_DIR)
    index = load_index_from_storage(storage_context)

query_engine = index.as_query_engine()

def serialize_node_with_score(node_with_score):
    # Custom serialization logic for NodeWithScore
    return {
        "node": str(node_with_score.node),  # Convert node to string or dict as needed
        "score": node_with_score.score,
    }

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    query_text = data.get('query')
    if not query_text:
        return jsonify({'error': 'No query provided'}), 400

    response = query_engine.query(query_text)
    # Serialize response if it contains NodeWithScore objects

    # serialized_response = [serialize_node_with_score(node) for node in response.response]
    return jsonify({'response': response.response})

if __name__ == '__main__':
    app.run(debug=True)
