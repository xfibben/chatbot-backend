import psycopg2
import yaml
import torch
from transformers import pipeline
from subprocess import run
import os

def fetch_questions_from_db():
    # Conectar a la base de datos PostgreSQL
    conn = psycopg2.connect(
        dbname="chatbot",
        user="postgres",
        password="root",
        host="localhost",
        port="5432"
    )
    cursor = conn.cursor()
    
    # Ejecutar la consulta para obtener las preguntas
    cursor.execute("SELECT message FROM text")
    questions = cursor.fetchall()
    
    # Cerrar la conexión
    cursor.close()
    conn.close()
    
    return [q[0] for q in questions]

# Cargar el modelo de clasificación preentrenado
classifier = pipeline("zero-shot-classification", model="roberta-large-mnli")

def classify_question(question, labels):
    # Clasificar la pregunta utilizando el modelo preentrenado
    result = classifier(question, candidate_labels=labels)
    return result['labels'][0]

def update_nlu_file(questions, nlu_file):
    with open(nlu_file, 'r') as file:
        nlu_data = yaml.safe_load(file)

    intents = [intent['intent'] for intent in nlu_data['nlu']]
    for question in questions:
        intent = classify_question(question, intents)
        example = f"- {question}"
        intent_found = False
        for nlu_item in nlu_data['nlu']:
            if nlu_item['intent'] == intent:
                if example not in nlu_item['examples']:
                    nlu_item['examples'] = nlu_item['examples'].strip() + f"\n    {example}\n"
                intent_found = True
                break
        if not intent_found:
            nlu_data['nlu'].append({
                'intent': intent,
                'examples': f"{example}\n"
            })

    with open(nlu_file, 'w') as file:
        yaml.dump(nlu_data, file, allow_unicode=True, sort_keys=False, default_flow_style=False)

    print(f"NLU data updated with new examples in {nlu_file}")

def train_rasa_model():
    result = run(["rasa", "train"], capture_output=True, text=True)
    if result.returncode == 0:
        print("Model trained successfully")
    else:
        print("Error training model")
        print(result.stdout)
        print(result.stderr)

def restart_rasa_server():
    result = run(["pkill", "-f", "rasa run"], capture_output=True, text=True)
    if result.returncode == 0:
        print("Rasa server stopped successfully")
        run(["nohup", "rasa", "run", "&"])
        print("Rasa server started successfully")
    else:
        print("Error stopping Rasa server")
        print(result.stdout)
        print(result.stderr)

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    nlu_file = os.path.join(script_dir, 'data', 'nlu.yml')
    
    questions = fetch_questions_from_db()
    update_nlu_file(questions, nlu_file)
    train_rasa_model()
    restart_rasa_server()
